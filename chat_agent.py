import os
import requests
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from bs4 import BeautifulSoup

# ==============================
# 0. Load Environment Variables
# ==============================
load_dotenv()

# ==============================
# 1. Initialize LLM and Embeddings
# ==============================
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("OPENAI_AZURE_ENDPOINT"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version=os.getenv("OPENAI_API_VERSION"),
    model=os.getenv("OPENAI_MODEL_NAME", "gpt-4"),
    temperature=0.7
)

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("OPENAI_AZURE_ENDPOINT"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version=os.getenv("OPENAI_API_VERSION"),
    model="text-embedding-3-large"
)

# ==============================
# 2. Global Variables
# ==============================
vectorstore_docs = None
vectorstore_code = None
qa_docs = None
qa_code = None

BASE_URL = "https://docs.frappe.io/erpnext"
VECTORSTORE_PATH = "erpnext_vectorstore"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # Optional: for higher rate limits

# ==============================
# 3. Load Documentation and Create Vector Store
# ==============================
def get_all_links(base_url):
    """Crawl all internal links under the ERPNext docs site."""
    print("🔍 Crawling ERPNext documentation pages...")
    visited = set()
    to_visit = {base_url}
    all_links = set()

    while to_visit:
        url = to_visit.pop()
        if url in visited or not url.startswith(base_url):
            continue

        visited.add(url)
        try:
            res = requests.get(url, timeout=10, verify=False)
            if res.status_code != 200:
                continue
            soup = BeautifulSoup(res.text, "html.parser")

            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.startswith("/"):
                    href = base_url.rstrip("/") + href
                if href.startswith(base_url) and href not in visited:
                    to_visit.add(href)
                    all_links.add(href)
        except Exception as e:
            print(f"⚠️ Failed to load {url}: {e}")

    print(f"✅ Found {len(all_links)} pages!")
    return list(all_links)

def initialize_vectorstores():
    """Initialize vector stores for docs and code"""
    global vectorstore_docs, vectorstore_code, qa_docs, qa_code

    # Try loading from cache
    if os.path.exists(VECTORSTORE_PATH):
        try:
            print("💾 Loading cached vectorstore from disk...")
            vectorstore_docs = FAISS.load_local(
                VECTORSTORE_PATH, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            vectorstore_code = vectorstore_docs
            print("✅ Vectorstore loaded successfully!")
        except Exception as e:
            print(f"⚠️ Failed to load cached vectorstore: {e}")
            vectorstore_docs = None

    # Build from scratch if needed
    if vectorstore_docs is None:
        print("🔄 Building vectorstore from ERPNext documentation...")
        try:
            all_links = get_all_links(BASE_URL)
            all_docs = []
            
            for link in all_links[:50]:  # Limit for faster testing
                try:
                    loader = WebBaseLoader(link, verify_ssl=False, trust_env=True)
                    docs = loader.load()
                    all_docs.extend(docs)
                    print(f"📄 Loaded: {link}")
                except Exception as e:
                    print(f"⚠️ Skipping {link}: {e}")

            if all_docs:
                print(f"✅ Loaded {len(all_docs)} documents. Creating vectorstore...")
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                split_docs = splitter.split_documents(all_docs)
                
                vectorstore_docs = FAISS.from_documents(split_docs, embeddings)
                vectorstore_code = vectorstore_docs
                
                vectorstore_docs.save_local(VECTORSTORE_PATH)
                print("💾 Saved vectorstore to disk!")
            else:
                raise Exception("No documents loaded")
                
        except Exception as e:
            print(f"⚠️ Creating fallback vectorstore: {e}")
            # Generate fallback content using LLM
            fallback_content = generate_fallback_content()
            vectorstore_docs = FAISS.from_documents(fallback_content, embeddings)
            vectorstore_code = vectorstore_docs

    # Create retrievers
    qa_docs = vectorstore_docs.as_retriever(search_kwargs={"k": 3})
    qa_code = vectorstore_code.as_retriever(search_kwargs={"k": 2})

# ==============================
# 4. Helper Functions
# ==============================
def get_github_contributors(keyword: str, repo: str = "frappe/erpnext") -> dict:
    """Get contributors who worked on a specific topic from commit history."""
    url = f"https://api.github.com/search/commits?q={keyword}+repo:{repo}&per_page=30"
    headers = {"Accept": "application/vnd.github.cloak-preview"}
    
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return {"error": f"GitHub API returned status {response.status_code}"}
        
        data = response.json()
        commits = data.get("items", [])
        
        if not commits:
            return {"error": f"No commits found for: {keyword}"}
        
        # Extract unique contributors
        contributors = {}
        for commit in commits:
            author = commit.get("commit", {}).get("author", {})
            committer = commit.get("author", {})  # GitHub user info
            
            author_name = author.get("name", "Unknown")
            author_email = author.get("email", "")
            github_username = committer.get("login", "") if committer else ""
            github_url = committer.get("html_url", "") if committer else ""
            
            # Use email as unique key
            if author_email and author_email not in contributors:
                contributors[author_email] = {
                    "name": author_name,
                    "email": author_email,
                    "github_username": github_username,
                    "github_url": github_url,
                    "commits": 1
                }
            elif author_email:
                contributors[author_email]["commits"] += 1
        
        # Sort by number of commits
        sorted_contributors = sorted(
            contributors.values(), 
            key=lambda x: x["commits"], 
            reverse=True
        )
        
        return {
            "contributors": sorted_contributors[:10],  # Top 10
            "total_commits": len(commits),
            "keyword": keyword
        }
        
    except Exception as e:
        return {"error": f"Could not fetch contributors: {str(e)}"}

def search_github_code(keyword: str, repo: str = "frappe/erpnext"):
    """Search GitHub code repository for specific keywords - returns structured data."""
    
    # Check if token exists
    if not GITHUB_TOKEN:
        return {
            "error": "GitHub Code Search requires authentication. Please add GITHUB_TOKEN to your .env file.",
            "files": [],
            "help": "Get a token from: https://github.com/settings/tokens (needs 'public_repo' scope)"
        }
    
    url = f"https://api.github.com/search/code?q={keyword}+repo:{repo}&per_page=10"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"token {GITHUB_TOKEN}"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 401:
            return {
                "error": "GitHub authentication failed. Check your GITHUB_TOKEN in .env file.",
                "files": []
            }
        elif response.status_code == 403:
            return {
                "error": "GitHub API rate limit exceeded. Wait a few minutes or check your token.",
                "files": []
            }
        elif response.status_code != 200:
            return {
                "error": f"GitHub API returned status {response.status_code}",
                "files": []
            }
        
        data = response.json()
        items = data.get("items", [])
        
        if not items:
            # Use general repository search as fallback
            return search_github_files_fallback(keyword, repo)
        
        # Return structured data
        files = []
        for item in items:
            files.append({
                "name": item.get('name', 'unknown'),
                "path": item.get('path', ''),
                "url": item.get('html_url', ''),
                "repository": item.get('repository', {}).get('full_name', repo)
            })
        
        return {"files": files, "total": len(files), "keyword": keyword}
        
    except Exception as e:
        return {"error": f"Could not search code: {str(e)}", "files": []}

def search_github_files_fallback(keyword: str, repo: str = "frappe/erpnext"):
    """Fallback method using repository contents search when code search fails."""
    try:
        # Search using repository tree API - doesn't require special permissions
        url = f"https://api.github.com/repos/{repo}/git/trees/develop?recursive=1"
        headers = {"Accept": "application/vnd.github+json"}
        if GITHUB_TOKEN:
            headers["Authorization"] = f"token {GITHUB_TOKEN}"
        
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code != 200:
            return {"error": f"No code found for: {keyword}", "files": []}
        
        data = response.json()
        tree = data.get('tree', [])
        
        # Filter files by keyword in path/name
        matching_files = []
        keyword_lower = keyword.lower()
        
        for item in tree:
            if item.get('type') == 'blob':  # Only files, not directories
                path = item.get('path', '')
                if keyword_lower in path.lower():
                    matching_files.append({
                        "name": path.split('/')[-1],
                        "path": path,
                        "url": f"https://github.com/{repo}/blob/develop/{path}",
                        "repository": repo
                    })
        
        if matching_files:
            # Limit to top 10
            return {"files": matching_files[:10], "total": len(matching_files), "keyword": keyword}
        else:
            return {"error": f"No files found matching: {keyword}", "files": []}
            
    except Exception as e:
        return {"error": f"Fallback search failed: {str(e)}", "files": []}

def fetch_file_content(file_url: str) -> str:
    """Fetch file content from GitHub URL."""
    try:
        # Convert web URL to raw content URL
        raw_url = file_url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')
        headers = {}
        if GITHUB_TOKEN:
            headers['Authorization'] = f'token {GITHUB_TOKEN}'
        
        response = requests.get(raw_url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.text[:5000]  # Limit to first 5000 chars
        return None
    except Exception as e:
        return None

def generate_fallback_content():
    """Generate fallback vectorstore content using LLM when documentation fails."""
    try:
        prompt = """You are an ERPNext expert. Generate comprehensive overview content about ERPNext for documentation purposes.

Provide 10 distinct, informative paragraphs covering:
1. ERPNext overview and purpose
2. Main modules (Accounting, HR, Manufacturing, Sales, Purchase, etc.)
3. Common roles in ERPNext projects
4. Key features and capabilities
5. Integration and customization
6. Workflow and automation
7. Reporting and analytics
8. User management and permissions
9. Implementation best practices
10. Technical architecture overview

Format each paragraph on a separate line. Make each paragraph substantial (100-150 words)."""

        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Split into paragraphs and create documents
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        fallback_docs = [Document(page_content=para, metadata={"source": "llm-generated"}) for para in paragraphs if len(para) > 50]
        
        if fallback_docs:
            print(f"   ✓ Generated {len(fallback_docs)} fallback documents using LLM")
            return fallback_docs
        
    except Exception as e:
        print(f"   ⚠️ LLM fallback generation failed: {e}")
    
    # Ultimate fallback - minimal content
    return [
        Document(page_content="ERPNext is an open-source ERP system with comprehensive modules.", metadata={"source": "minimal-fallback"}),
        Document(page_content="Common ERPNext roles include Developer, Tester, and Analyst.", metadata={"source": "minimal-fallback"})
    ]

def generate_dynamic_list(topic: str, example: str) -> list:
    """Generate a dynamic list using LLM for any topic."""
    try:
        prompt = f"""Generate a comprehensive list of {topic}.

Provide at least 8-10 items as a comma-separated list.
Example format: {example}

Return ONLY the comma-separated list, no explanations."""

        response = llm.invoke(prompt)
        text = response.content if hasattr(response, 'content') else str(response)
        text = text.strip().replace('`', '').replace('*', '')
        if '\n' in text:
            text = text.split('\n')[0]
        
        items = [item.strip() for item in text.split(",") if item.strip()]
        return items if items else [example.split(",")[0].strip()]
        
    except Exception as e:
        print(f"   ⚠️ Dynamic list generation failed: {e}")
        return [item.strip() for item in example.split(",")]

def extract_domains_and_roles():
    """Use LLM to extract domains and roles - roles generated purely from LLM knowledge."""
    print("🔍 Extracting available domains and roles...")
    
    # Get comprehensive docs for DOMAINS only
    try:
        domain_docs = qa_docs.invoke("ERPNext modules domains accounting manufacturing sales HR purchase projects healthcare")
        domain_context = "\n".join([doc.page_content[:600] for doc in domain_docs[:5]])
    except Exception as e:
        print(f"⚠️ Could not fetch docs, using general knowledge: {e}")
        domain_context = "ERPNext documentation covers various business domains and modules."
    
    # Extract domains from documentation
    domain_prompt = f"""Based on this ERPNext documentation context:

{domain_context}

List ALL the main ERPNext modules/domains/functional areas. Consider:
- Core business modules (Accounting, Sales, Purchase, etc.)
- Industry-specific modules (Manufacturing, Healthcare, Education, etc.)
- Support modules (HR, Projects, CRM, etc.)

Return ONLY a comma-separated list of module/domain names.
Example format: Accounting, Sales, Purchase, Manufacturing
Do NOT include any other text, explanations, or preamble."""

    # Extract roles purely from LLM general knowledge (NO documentation)
    role_prompt = """You are an expert in ERPNext and ERP implementation projects.

Based on your knowledge of software development and ERP implementation teams, list ALL common roles involved in ERPNext projects.

Consider these categories:
1. Development roles: Developers, Engineers, etc.
2. Quality Assurance: Testers, QA Engineers, etc.
3. Business Analysis: Analysts, Consultants, etc.
4. Architecture & Design: Architects, Technical Leads, etc.
5. Administration: System Admins, DevOps, etc.
6. Project Management: Project Managers, Coordinators, etc.
7. Functional: Functional Consultants, Domain Experts, etc.

Return ONLY a comma-separated list of role names.
Example format: Developer, Senior Developer, Tester, Business Analyst, Solution Architect
Do NOT include any other text, explanations, or preamble."""

    domains = []
    roles = []
    
    try:
        print("   🤖 Asking LLM for domains (from documentation)...")
        domains_response = llm.invoke(domain_prompt)
        domains_text = domains_response.content if hasattr(domains_response, 'content') else str(domains_response)
        
        # Clean up the response
        domains_text = domains_text.strip()
        domains_text = domains_text.replace('`', '').replace('*', '')
        if '\n' in domains_text:
            domains_text = domains_text.split('\n')[0]
        
        domains = [d.strip() for d in domains_text.split(",") if d.strip()]
        print(f"   ✓ Found {len(domains)} domains from LLM")
        
    except Exception as e:
        print(f"   ⚠️ LLM domain extraction failed: {e}")
    
    try:
        print("   🤖 Asking LLM for roles (from LLM knowledge only)...")
        roles_response = llm.invoke(role_prompt)
        roles_text = roles_response.content if hasattr(roles_response, 'content') else str(roles_response)
        
        # Clean up the response
        roles_text = roles_text.strip()
        roles_text = roles_text.replace('`', '').replace('*', '')
        if '\n' in roles_text:
            roles_text = roles_text.split('\n')[0]
        
        roles = [r.strip() for r in roles_text.split(",") if r.strip()]
        print(f"   ✓ Found {len(roles)} roles from LLM")
        
    except Exception as e:
        print(f"   ⚠️ LLM role extraction failed: {e}")
    
    # Only use fallback if LLM completely failed
    if not domains:
        print("   ⚠️ Generating fallback domains using LLM...")
        domains = generate_dynamic_list("ERPNext modules and domains", "Accounting, Sales, Manufacturing")
    
    if not roles:
        print("   ⚠️ Generating fallback roles using LLM...")
        roles = generate_dynamic_list("Common roles in ERP implementation projects", "Developer, Tester, Business Analyst")
    
    print(f"✅ Final: {len(domains)} domains, {len(roles)} roles\n")
    return domains, roles

# ==============================
# 5. User Interaction Functions
# ==============================
def match_user_input(user_input: str, options: list) -> str:
    """Use LLM to match user input to closest option from list."""
    # First try exact match
    if user_input in options:
        return user_input
    
    # Try case-insensitive match
    for opt in options:
        if user_input.lower() == opt.lower():
            return opt
    
    # Use LLM for fuzzy matching
    try:
        prompt = f"""The user entered: "{user_input}"

Available options:
{chr(10).join([f"- {opt}" for opt in options])}

Which option does the user most likely mean? Return ONLY the exact option text from the list above, nothing else."""

        response = llm.invoke(prompt)
        matched = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        matched = matched.replace('`', '').replace('*', '').replace('-', '').strip()
        
        # Verify the match is in options
        for opt in options:
            if matched.lower() == opt.lower() or matched in opt or opt in matched:
                return opt
        
        return None
        
    except Exception as e:
        # Fallback to simple substring match
        matches = [opt for opt in options if user_input.lower() in opt.lower()]
        return matches[0] if matches else None

def collect_info():
    """Collect user's domain and role."""
    print("\n" + "="*70)
    print("📋 LET'S GET YOU SET UP")
    print("="*70)
    
    domains, roles = extract_domains_and_roles()

    # Select domain
    while True:
        print(f"\n📦 Available domains:")
        for i, d in enumerate(domains, 1):
            print(f"   {i}. {d}")
        
        domain_input = input("\nSelect your domain (name or number): ").strip()
        
        # Try number selection
        if domain_input.isdigit():
            idx = int(domain_input) - 1
            if 0 <= idx < len(domains):
                domain = domains[idx]
                break
        
        # Use LLM for matching
        domain = match_user_input(domain_input, domains)
        if domain:
            if domain != domain_input:
                print(f"✓ Matched to: {domain}")
            break
        
        print("❌ Invalid selection, please try again.")

    # Select role
    while True:
        print(f"\n👤 Available roles:")
        for i, r in enumerate(roles, 1):
            print(f"   {i}. {r}")
        
        role_input = input("\nSelect your role (name or number): ").strip()
        
        # Try number selection
        if role_input.isdigit():
            idx = int(role_input) - 1
            if 0 <= idx < len(roles):
                role = roles[idx]
                break
        
        # Use LLM for matching
        role = match_user_input(role_input, roles)
        if role:
            if role != role_input:
                print(f"✓ Matched to: {role}")
            break
        
        print("❌ Invalid selection, please try again.")

    print(f"\n✅ Great! You're a {role} in the {domain} domain.")
    return {"domain": domain, "role": role}

def present_summary(inputs):
    """Generate and present onboarding summary."""
    print("\n" + "="*70)
    print("📊 GENERATING YOUR ONBOARDING SUMMARY")
    print("="*70)
    print(f"\n🔄 Creating personalized onboarding for {inputs['role']} in {inputs['domain']}...\n")
    
    # Query 1: General domain information
    domain_query = f"{inputs['domain']} module in ERPNext features capabilities overview"
    
    # Query 2: Role-specific information
    role_query = f"{inputs['role']} responsibilities tasks in {inputs['domain']} ERPNext"
    
    try:
        # Get domain-specific documentation
        domain_docs = qa_docs.invoke(domain_query)
        domain_context = "\n\n".join([doc.page_content[:800] for doc in domain_docs[:3]])
        
        # Get role-specific documentation
        role_docs = qa_docs.invoke(role_query)
        role_context = "\n\n".join([doc.page_content[:800] for doc in role_docs[:2]])
        
        # Combine contexts
        combined_context = f"Domain Information:\n{domain_context}\n\nRole Information:\n{role_context}"
        
        prompt = f"""Create a comprehensive onboarding summary for a {inputs['role']} joining the {inputs['domain']} domain in ERPNext.

Documentation context:
{combined_context}

Structure your response with these sections:

🎯 **ROLE OVERVIEW**
What does a {inputs['role']} do in the {inputs['domain']} domain?

📋 **KEY RESPONSIBILITIES**
List 4-6 main responsibilities for this role in this domain.

🔧 **ESSENTIAL FEATURES & TOOLS**
What ERPNext features and tools will they use most?

⚙️ **COMMON WORKFLOWS**
Describe 2-3 typical day-to-day workflows or processes.

🚀 **GETTING STARTED CHECKLIST**
First steps to become productive (prioritized list).

📚 **LEARNING RESOURCES**
What documentation, modules, or topics should they study first?

💡 **PRO TIPS**
2-3 insider tips for success in this role and domain.

Be specific, practical, and encouraging. Use real ERPNext terminology from the documentation."""
        
        response = llm.invoke(prompt)
        summary = response.content if hasattr(response, 'content') else str(response)
        
        # Display the summary with nice formatting
        print(summary)
        print("\n" + "="*70)
        
    except Exception as e:
        print(f"⚠️ Could not generate detailed summary: {e}")
    return {**inputs, "summary": summary}

def search_materials(inputs):
    """Search for learning materials based on keyword/topic only - using semantic search."""
    print("\n" + "="*70)
    print("📚 LEARNING MATERIALS")
    print("="*70)
    
    keyword = input("\n🔍 Enter topic to learn about: ").strip()
    
    if not keyword:
        print("No topic provided.")
        return {**inputs, "materials_doc": "", "materials_code": "", "commits": "", "keyword": ""}
    
    print(f"\n🔎 Performing semantic search for '{keyword}'...")
    print("   📖 Searching documentation with embeddings...")
    
    # Semantic search in documentation using similarity_search_with_score
    docs_found = False
    doc_result = ""
    
    if vectorstore_docs:
        try:
            # Use semantic similarity search with relevance scores
            results = vectorstore_docs.similarity_search_with_score(keyword, k=5)
            
            if results:
                # Filter by relevance - FAISS returns distance (lower is better)
                # Keep results with reasonable distance scores (< 2.0 is usually relevant)
                relevant_docs = [(doc, score) for doc, score in results if score < 2.0]
                
                if relevant_docs:
                    docs_found = True
                    print(f"   ✅ Found {len(relevant_docs)} relevant documentation sections\n")
                    
                    # Build context from top results
                    context = "\n\n".join([doc.page_content[:1000] for doc, score in relevant_docs[:3]])
                    
                    prompt = f"""Explain '{keyword}' for a {inputs['role']} in {inputs['domain']} domain.

Context from documentation:
{context}

Provide:
1. **What it is**: Brief definition
2. **How it works**: Technical/functional explanation
3. **Key points**: Important things to remember

Be concise and role-specific."""
                    
                    response = llm.invoke(prompt)
                    doc_result = response.content if hasattr(response, 'content') else str(response)
                else:
                    print("   ⚠️ No highly relevant documentation found\n")
            else:
                print("   ⚠️ No documentation results\n")
                
        except Exception as e:
            print(f"   ⚠️ Error in semantic search: {e}\n")
            doc_result = f"Error searching docs: {e}"
    
    # If no documentation found, search in GitHub code
    code_result = ""
    if not docs_found:
        print("   🔍 No documentation found. Searching GitHub codebase...")
        code_data = search_github_code(keyword)
        
        if "error" not in code_data:
            files = code_data.get("files", [])
            if files:
                print(f"   ✅ Found {len(files)} code files\n")
                
                # Fetch code snippets from top files
                code_snippets = []
                for file in files[:5]:
                    try:
                        content = fetch_file_content(file['url'])
                        if content:
                            # Extract relevant lines
                            lines = content.split('\n')
                            relevant_lines = [line for line in lines if keyword.lower() in line.lower()]
                            if relevant_lines:
                                code_snippets.append({
                                    'file': file['path'],
                                    'url': file['url'],
                                    'lines': relevant_lines[:5]
                                })
                    except Exception as ex:
                        print(f"   ⚠️ Could not fetch {file['name']}: {ex}")
                        pass
                
                if code_snippets:
                    # Generate high-level analysis with code context
                    code_context = "\n\n".join([
                        f"File: {s['file']}\nCode:\n" + "\n".join(s['lines'])
                        for s in code_snippets[:3]
                    ])
                    
                    prompt = f"""You are an ERPNext expert analyzing code for a {inputs['role']} in the {inputs['domain']} domain.

The user searched for topic: {keyword}

No documentation was found. Here are code files and snippets:

{code_context[:3000]}

Provide HIGH-LEVEL INFORMATION:
1. **What this is**: Based on file names and code, what feature/functionality is this
2. **Purpose**: What problem does it solve in ERPNext
3. **Architecture overview**: How it fits into ERPNext (controllers, models, views, etc.)
4. **Key components**: Main files and their roles
5. **Code insights**: Notable patterns, functions, or classes found
6. **Usage guidance**: How a {inputs['role']} might work with or extend this

Be practical and help them understand the big picture without diving too deep into code details."""
                    
                    try:
                        response = llm.invoke(prompt)
                        code_result = response.content if hasattr(response, 'content') else str(response)
                        
                        # Add file references
                        code_result += "\n\n**📂 Related Files:**\n"
                        for i, file in enumerate(files[:5], 1):
                            code_result += f"{i}. {file['path']}\n   🔗 {file['url']}\n"
                            
                    except Exception as e:
                        code_result = f"Error analyzing code: {e}"
                else:
                    # No code snippets fetched, just list files
                    code_result = f"**📂 Found {len(files)} related files:**\n\n"
                    for i, file in enumerate(files[:10], 1):
                        code_result += f"{i}. {file['path']}\n   🔗 {file['url']}\n"
            else:
                code_result = "No code files found in GitHub repository."
        else:
            # Display helpful error message
            error_msg = code_data.get('error', 'Unknown error')
            code_result = f"⚠️ GitHub Search Error: {error_msg}"
            
            if 'help' in code_data:
                code_result += f"\n\n💡 Help: {code_data['help']}"
            
            # Try to provide alternative guidance using LLM
            print(f"\n   ⚠️ {error_msg}")
            if 'authentication' in error_msg.lower() or 'token' in error_msg.lower():
                print(f"   💡 To enable GitHub code search, add GITHUB_TOKEN to your .env file")
                print(f"   📖 Get token from: https://github.com/settings/tokens")
                
            # Generate guidance without code
            try:
                fallback_prompt = f"""A {inputs['role']} in {inputs['domain']} domain wants to learn about "{keyword}" in ERPNext, but code search is unavailable.

Based on your ERPNext knowledge, provide:
1. What this topic likely involves
2. Where in ERPNext this would typically be found
3. Key concepts to understand
4. Recommended resources or documentation to explore

Be helpful and practical."""
                
                response = llm.invoke(fallback_prompt)
                code_result = response.content if hasattr(response, 'content') else code_result
            except:
                pass
    else:
        # Documentation was found, still check code for additional context
        print("   🔍 Checking GitHub for code references...")
        code_data = search_github_code(keyword)
        
        if "error" not in code_data:
            files = code_data.get("files", [])
            if files:
                code_result = f"\n**📂 Related Code Files ({len(files)} found):**\n\n"
                for i, file in enumerate(files[:5], 1):
                    code_result += f"{i}. {file['path']}\n   🔗 {file['url']}\n\n"
            else:
                code_result = "No code files found."
        else:
            code_result = code_data.get('error', 'Error searching code.')
    
    # Get contributors for this topic
    print("   🔍 Finding team members who worked on this topic...")
    contributors_data = get_github_contributors(keyword)
    
    # Display results
    print("\n" + "─"*70)
    if docs_found:
        print("📖 DOCUMENTATION INSIGHTS:")
        print("─"*70)
        print(doc_result)
    else:
        print("💻 CODE ANALYSIS (No documentation found):")
        print("─"*70)
        print(code_result if code_result else "No results found")
    
    if docs_found and code_result:
        print("\n" + "─"*70)
        print("💻 CODE REFERENCES:")
        print("─"*70)
        print(code_result)
    
    print("\n" + "─"*70)
    print("👥 TEAM MEMBERS WHO WORKED ON THIS:")
    print("─"*70)
    
    if "error" in contributors_data:
        print(f"⚠️ {contributors_data['error']}")
    else:
        contributors = contributors_data.get("contributors", [])
        total_commits = contributors_data.get("total_commits", 0)
        
        if contributors:
            print(f"Found {len(contributors)} contributors with {total_commits} related commits:\n")
            for i, contrib in enumerate(contributors, 1):
                print(f"{i}. {contrib['name']}")
                if contrib['github_username']:
                    print(f"   GitHub: @{contrib['github_username']}")
                    print(f"   Profile: {contrib['github_url']}")
                if contrib['email']:
                    print(f"   Email: {contrib['email']}")
                print(f"   Commits on this topic: {contrib['commits']}")
                print()
        else:
            print("No contributors found for this topic.")
    
    if "No" in doc_result and "No" in code_result:
        print("\n💡 TIP: Limited results found. Try different keywords or contact the team for guidance.")
    
    return {
        **inputs, 
        "materials_doc": doc_result, 
        "materials_code": code_result, 
        "contributors": contributors_data,
        "keyword": keyword
    }

def ask_quiz(inputs):
    """Ask if user wants to take a quiz - with flexible interpretation."""
    print("\n" + "="*70)
    response = input("❓ Would you like to take a quiz on what you've learned? (yes/no): ").strip().lower()
    return interpret_yes_no(response)

def parse_quiz_with_llm(quiz_content: str, inputs: dict) -> list:
    """Use LLM to parse quiz content into structured format."""
    try:
        prompt = f"""Parse this quiz content into a structured JSON format.

Quiz content:
{quiz_content[:3000]}

Extract each question with:
- question: the question text
- options: list of 4 options (A, B, C, D with their text)
- answer: the correct answer letter (A/B/C/D)
- explanation: explanation of the correct answer

Return ONLY a valid JSON array of question objects, no other text.
Example: [{{"question": "Q1: ...", "options": ["A) ...", "B) ...", "C) ...", "D) ..."], "answer": "A", "explanation": "..."}}]"""

        response = llm.invoke(prompt)
        result = response.content if hasattr(response, 'content') else str(response)
        
        # Try to parse JSON
        import json
        result = result.strip()
        if result.startswith('```'):
            result = result.split('```')[1]
            if result.startswith('json'):
                result = result[4:]
        result = result.strip()
        
        quiz_data = json.loads(result)
        return quiz_data if isinstance(quiz_data, list) else []
        
    except Exception as e:
        print(f"   ⚠️ LLM parsing failed, using regex: {e}")
        # Fallback to original parsing
        quiz_data = []
        current_question = {}
        
        for line in quiz_content.split('\n'):
            line = line.strip()
            if line.startswith('Q') and ':' in line:
                if current_question:
                    quiz_data.append(current_question)
                current_question = {'question': line, 'options': [], 'answer': '', 'explanation': ''}
            elif line.startswith(('A)', 'B)', 'C)', 'D)')):
                current_question['options'].append(line)
            elif line.startswith('ANSWER:'):
                current_question['answer'] = line.replace('ANSWER:', '').strip()
            elif line.startswith('EXPLANATION:'):
                current_question['explanation'] = line.replace('EXPLANATION:', '').strip()
        
        if current_question:
            quiz_data.append(current_question)
        
        return quiz_data

def evaluate_quiz_with_llm(correct_count: int, total: int, inputs: dict) -> str:
    """Generate dynamic quiz feedback using LLM."""
    try:
        percentage = (correct_count * 100) // total
        
        prompt = f"""A {inputs['role']} in {inputs['domain']} domain just completed a quiz about ERPNext.

Score: {correct_count}/{total} ({percentage}%)

Generate encouraging and personalized feedback in 2-3 sentences. Include:
- Recognition of their performance
- Actionable advice based on their score
- Motivation to continue learning

Keep it warm, professional, and specific to their role and domain."""

        response = llm.invoke(prompt)
        feedback = response.content if hasattr(response, 'content') else str(response)
        return feedback.strip()
        
    except Exception as e:
        # Fallback to simple feedback
        if correct_count == total:
            return "🌟 Perfect score! Excellent work!"
        elif correct_count >= total * 0.7:
            return "👍 Great job! You have a solid understanding."
        elif correct_count >= total * 0.5:
            return "📚 Good effort! Review the topics you missed."
        else:
            return "💪 Keep learning! Review the materials and try again."

def present_quiz(inputs):
    """Generate and present quiz questions interactively."""
    print("\n" + "="*70)
    print("📝 QUIZ TIME!")
    print("="*70)
    
    prompt = f"""Generate 5 multiple-choice quiz questions for a {inputs['role']} in {inputs['domain']} domain.

Format EXACTLY as follows for each question:
Q1: [Question text]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
ANSWER: [A/B/C/D]
EXPLANATION: [Brief explanation why this is correct]

---

Make questions practical and relevant to their role. Cover different aspects of {inputs['domain']} in ERPNext."""
    
    try:
        response = llm.invoke(prompt)
        quiz_content = response.content if hasattr(response, 'content') else str(response)
        
        # Parse quiz using LLM
        quiz_data = parse_quiz_with_llm(quiz_content, inputs)
        
        # Interactive quiz
        if not quiz_data:
            print("⚠️ Could not parse quiz. Showing raw format:\n")
            print(quiz_content)
            return {**inputs, "quiz": quiz_content, "quiz_data": []}
        
        user_answers = []
        correct_count = 0
        
        for i, q in enumerate(quiz_data, 1):
            print(f"\n{'─'*70}")
            print(f"\n{q.get('question', f'Q{i}: Missing question')}")
            for opt in q.get('options', []):
                print(f"   {opt}")
            
            # Get user answer with flexible interpretation
            while True:
                user_input = input("\nYour answer (A/B/C/D): ").strip().upper()
                if user_input in ['A', 'B', 'C', 'D']:
                    user_answers.append(user_input)
                    break
                print("❌ Please enter A, B, C, or D")
        
        # Show results after all questions
        print("\n" + "="*70)
        print("📊 QUIZ RESULTS")
        print("="*70)
        
        for i, (q, user_ans) in enumerate(zip(quiz_data, user_answers), 1):
            correct_ans = q.get('answer', '').strip().upper()
            is_correct = user_ans == correct_ans
            if is_correct:
                correct_count += 1
            
            print(f"\n{'─'*70}")
            print(f"Question {i}: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
            print(f"Your answer: {user_ans} | Correct answer: {correct_ans}")
            if q.get('explanation'):
                print(f"💡 {q['explanation']}")
        
        print(f"\n{'='*70}")
        print(f"🎯 FINAL SCORE: {correct_count}/{len(quiz_data)} ({correct_count*100//len(quiz_data) if len(quiz_data) > 0 else 0}%)")
        
        # Use LLM for personalized feedback
        feedback = evaluate_quiz_with_llm(correct_count, len(quiz_data), inputs)
        print(feedback)
        
        print("="*70)
        
        return {**inputs, "quiz": quiz_content, "quiz_data": quiz_data, "score": f"{correct_count}/{len(quiz_data)}"}
        
    except Exception as e:
        print(f"⚠️ Could not generate quiz: {e}")
        return {**inputs, "quiz": "", "quiz_data": []}

def interpret_user_choice(user_input: str, options: dict) -> str:
    """Use LLM to interpret user's choice flexibly."""
    # First try exact match
    if user_input in options:
        return user_input
    
    # Try case-insensitive match
    for key in options:
        if user_input.lower() == key.lower():
            return key
    
    # Use LLM for flexible interpretation
    try:
        options_str = "\n".join([f"{k}. {v}" for k, v in options.items()])
        
        prompt = f"""User entered: "{user_input}"

Available options:
{options_str}

Which option did the user likely mean? Return ONLY the option number/key (like "1", "2", or "3"), nothing else."""

        response = llm.invoke(prompt)
        interpreted = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        interpreted = interpreted.replace('`', '').replace('*', '').replace('.', '').strip()
        
        if interpreted in options:
            return interpreted
            
    except Exception as e:
        pass
    
    return None

def generate_dynamic_recommendations(contributors: list, keyword: str, inputs: dict) -> str:
    """Generate dynamic team connection recommendations using LLM."""
    try:
        if not contributors or len(contributors) == 0:
            return "No specific recommendations available."
        
        # Build contributor summary
        contrib_summary = "\n".join([
            f"- {c['name']}: {c['commits']} commits, GitHub: @{c.get('github_username', 'N/A')}"
            for c in contributors[:5]
        ])
        
        prompt = f"""A {inputs['role']} in {inputs['domain']} domain needs to connect with team members about "{keyword}".

Here are the top contributors:
{contrib_summary}

Generate 2-3 personalized recommendations for who to reach out to and why. Consider:
- Their contribution level
- Relevance to the role and domain
- Practical next steps

Keep it concise and actionable."""

        response = llm.invoke(prompt)
        recommendations = response.content if hasattr(response, 'content') else str(response)
        return recommendations.strip()
        
    except Exception as e:
        # Fallback
        if contributors and len(contributors) > 0:
            msg = f"• Reach out to {contributors[0]['name']} - most active on this topic"
            if len(contributors) > 1:
                msg += f"\n• Also connect with {contributors[1]['name']} for additional insights"
            return msg
        return "Consider reaching out to the team members listed above."

def ask_continue(inputs):
    """Ask what user wants to do next - with flexible interpretation."""
    print("\n" + "="*70)
    print("🎯 WHAT'S NEXT?")
    print("="*70)
    
    options = {
        "1": "Continue Learning (search more topics)",
        "2": "Connect with Team (get contact info)",
        "3": "End Session"
    }
    
    for key, desc in options.items():
        print(f"{key}. {desc}")
    
    choice = input("\nYour choice (1/2/3 or describe what you want): ").strip()
    
    # Use LLM for flexible interpretation
    interpreted = interpret_user_choice(choice, options)
    if interpreted:
        if interpreted != choice:
            print(f"✓ Understood: {options[interpreted]}")
        return interpreted
    
    # If interpretation fails, default to end
    print("⚠️ Unclear choice, ending session.")
    return "3"

def connect_with_team(inputs):
    """Provide team contact information based on actual GitHub contributors."""
    print("\n" + "="*70)
    print("👥 TEAM CONNECTIONS")
    print("="*70)
    
    # Check if we have contributor data from previous search
    contributors_data = inputs.get("contributors", {})
    keyword = inputs.get("keyword", inputs['domain'])
    
    if not contributors_data or "error" in contributors_data:
        # Fetch contributors for the domain if not available
        print(f"🔍 Finding team members who work on {inputs['domain']}...")
        contributors_data = get_github_contributors(inputs['domain'])
    
    # Display actual team members from GitHub
    if "error" not in contributors_data:
        contributors = contributors_data.get("contributors", [])
        
        if contributors:
            print(f"\n📋 TEAM MEMBERS WORKING ON {keyword.upper()}:\n")
            print("─"*70)
            
            for i, contrib in enumerate(contributors[:5], 1):  # Show top 5
                print(f"\n{i}. **{contrib['name']}**")
                if contrib['github_username']:
                    print(f"   🔗 GitHub: @{contrib['github_username']}")
                    print(f"   📍 Profile: {contrib['github_url']}")
                if contrib['email'] and not contrib['email'].endswith('users.noreply.github.com'):
                    print(f"   📧 Email: {contrib['email']}")
                print(f"   💼 Contributions: {contrib['commits']} commits on this topic")
            
            print("\n" + "─"*70)
            print("\n💡 RECOMMENDATION:")
            
            # Use LLM for dynamic recommendations
            recommendations = generate_dynamic_recommendations(contributors, keyword, inputs)
            print(recommendations)
        else:
            print("⚠️ No specific contributors found for this topic.")
    
    # Generate general guidance using LLM
    print("\n" + "─"*70)
    print("📚 GENERAL GUIDANCE:")
    print("─"*70)
    
    prompt = f"""For a {inputs['role']} in {inputs['domain']} domain working on ERPNext:

Provide practical guidance on:

1. **Communication Channels**:
   - Recommended Slack/Teams channels
   - Discussion forums or groups
   - Regular meetings to attend

2. **Learning Resources**:
   - Documentation to review
   - Code repositories to explore
   - Training materials or videos

3. **Best Practices**:
   - How to effectively collaborate with the team
   - Questions to ask when stuck
   - Tools and workflows to familiarize with

Be specific and actionable. Keep it concise (2-4 bullet points total)."""
    
    try:
        response = llm.invoke(prompt)
        guidance = response.content if hasattr(response, 'content') else str(response)
        print("\n" + guidance)
    except Exception as e:
        print(f"⚠️ Error generating guidance: {e}")
    
    print("\n" + "="*70)

# ==============================
# 6. Main Onboarding Loop
# ==============================
def agentic_onboarding():
    """Main onboarding workflow."""
    # Step 1: Collect user info
    user_info = collect_info()
    
    # Step 2: Present summary
    user_info = present_summary(user_info)
    
    # Step 3: Learning loop
    while True:
        # Search materials
        user_info = search_materials(user_info)
        
        # Offer quiz
        if ask_quiz(user_info):
            user_info = present_quiz(user_info)
        
        # Ask what's next
        choice = ask_continue(user_info)
        
        if choice == "1":
            continue  # Continue learning
        elif choice == "2":
            connect_with_team(user_info)
            # Use LLM to interpret continue response
            continue_response = input("\nContinue learning? (yes/no): ").strip().lower()
            should_continue = interpret_yes_no(continue_response)
            if not should_continue:
                break
        elif choice == "3":
            # Generate dynamic farewell message
            farewell = generate_farewell(user_info)
            print("\n" + "="*70)
            print(farewell)
            print("="*70)
            break
        else:
            print("Ending session.")
            break

def interpret_yes_no(user_input: str) -> bool:
    """Use LLM to interpret yes/no responses flexibly."""
    user_input = user_input.lower().strip()
    
    # Quick checks first
    if user_input in ['yes', 'y', 'yeah', 'yep', 'sure', 'ok', 'okay']:
        return True
    if user_input in ['no', 'n', 'nope', 'nah']:
        return False
    
    # Use LLM for unclear responses
    try:
        prompt = f"""User was asked "Continue learning?" and responded: "{user_input}"

Is this a YES or NO? Reply with ONLY the word YES or NO."""

        response = llm.invoke(prompt)
        result = response.content.strip().upper() if hasattr(response, 'content') else 'NO'
        return 'YES' in result
        
    except:
        return False

def generate_farewell(inputs: dict) -> str:
    """Generate personalized farewell message using LLM."""
    try:
        prompt = f"""Generate a warm farewell message for a {inputs['role']} in {inputs['domain']} domain who just completed ERPNext onboarding.

Keep it brief (2-3 lines), encouraging, and personalized to their role. Include relevant emojis."""

        response = llm.invoke(prompt)
        return response.content.strip() if hasattr(response, 'content') else "👋 Thank you for using ERPNext Onboarding!\n🎓 Keep learning and growing!"
        
    except:
        return "👋 Thank you for using ERPNext Onboarding!\n🎓 Keep learning and growing!"

# ==============================
# 7. Main Entry Point
# ==============================
def display_banner():
    """Display welcome banner."""
    print("\n" + "="*70)
    print("🎓  WELCOME TO ERPNEXT TEAM ONBOARDING")
    print("     Your Interactive Learning Companion")
    print("="*70)
    print("\n💡 I'll guide you through:")
    print("   ✓ Understanding your role and domain")
    print("   ✓ Finding learning materials")
    print("   ✓ Testing your knowledge with quizzes")
    print("   ✓ Connecting with team members")
    print("\n" + "="*70)

if __name__ == "__main__":
    display_banner()
    
    # Initialize vector stores
    initialize_vectorstores()
    
    # Run onboarding
    agentic_onboarding()