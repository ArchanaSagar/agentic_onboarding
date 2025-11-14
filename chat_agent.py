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
            fallback_docs = [
                Document(page_content="ERPNext is an open-source ERP system with modules for Accounting, HR, Manufacturing, Sales, Purchase, Projects, and Healthcare.", metadata={}),
                Document(page_content="Common roles in ERPNext include Developer, Tester, Business Analyst, Solution Architect, and System Administrator.", metadata={})
            ]
            vectorstore_docs = FAISS.from_documents(fallback_docs, embeddings)
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

def search_github_code(keyword: str, repo: str = "frappe/erpnext") -> str:
    """Search GitHub code repository for specific keywords."""
    url = f"https://api.github.com/search/code?q={keyword}+repo:{repo}&per_page=5"
    headers = {"Accept": "application/vnd.github+json"}
    
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return f"GitHub API returned status {response.status_code}. (Consider adding GITHUB_TOKEN to .env for higher rate limits)"
        
        data = response.json()
        items = data.get("items", [])
        
        if not items:
            return f"No code found for: {keyword}"
        
        result = [f"Code files containing '{keyword}':\n"]
        for i, item in enumerate(items[:5], 1):
            result.append(f"{i}. {item['name']} (in {item['path']})")
            result.append(f"   {item['html_url']}\n")
        
        return "\n".join(result)
    except Exception as e:
        return f"Could not search code: {str(e)}"

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
        print("   ⚠️ Using fallback domains")
        domains = ["Accounting", "Sales", "Purchase", "Manufacturing", "HR & Payroll", 
                   "Projects", "Healthcare", "CRM", "Assets", "Stock/Inventory"]
    
    if not roles:
        print("   ⚠️ Using fallback roles")
        roles = ["Developer", "Tester", "Business Analyst", "Solution Architect", 
                 "System Administrator", "Functional Consultant", "Project Manager"]
    
    print(f"✅ Final: {len(domains)} domains, {len(roles)} roles\n")
    return domains, roles

# ==============================
# 5. User Interaction Functions
# ==============================
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
        # Try name matching
        elif domain_input in domains:
            domain = domain_input
            break
        else:
            # Fuzzy match
            matches = [d for d in domains if domain_input.lower() in d.lower()]
            if matches:
                domain = matches[0]
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
        # Try name matching
        elif role_input in roles:
            role = role_input
            break
        else:
            # Fuzzy match
            matches = [r for r in roles if role_input.lower() in r.lower()]
            if matches:
                role = matches[0]
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
        
        # Fallback summary
        summary = f"""
🎯 **ROLE OVERVIEW**
Welcome as a {inputs['role']} in the {inputs['domain']} domain! You'll be working with ERPNext's {inputs['domain']} module.

📋 **KEY RESPONSIBILITIES**
- Understanding {inputs['domain']} business processes
- Working with ERPNext {inputs['domain']} features
- Collaborating with the team on implementations
- Testing and quality assurance

🚀 **GETTING STARTED**
1. Explore the ERPNext {inputs['domain']} module documentation
2. Set up your development environment
3. Review existing workflows and processes
4. Connect with senior team members

💡 **NEXT STEPS**
Start exploring specific features and workflows in the learning section!
"""
        print(summary)
        print("\n" + "="*70)
    
    return {**inputs, "summary": summary}

def search_materials(inputs):
    """Search for learning materials based on keyword/feature."""
    print("\n" + "="*70)
    print("📚 LEARNING MATERIALS")
    print("="*70)
    
    keyword = input("\n🔍 Enter topic/feature ID/PBI to learn about: ").strip()
    
    if not keyword:
        print("No keyword provided.")
        return {**inputs, "materials_doc": "", "materials_code": "", "commits": "", "keyword": ""}
    
    print(f"\n🔎 Searching for '{keyword}'...")
    
    # Search documentation
    try:
        doc_query = f"{keyword} in {inputs['domain']} for {inputs['role']}"
        docs = qa_docs.invoke(doc_query)
        
        if docs:
            context = "\n\n".join([doc.page_content[:1000] for doc in docs])
            
            prompt = f"""Explain '{keyword}' for a {inputs['role']} in {inputs['domain']} domain.

Context from documentation:
{context}

Provide:
1. **What it is**: Brief definition
2. **How it works**: Technical/functional explanation
3. **Practical usage**: Real-world application for {inputs['role']}
4. **Key points**: Important things to remember

Be concise and role-specific."""
            
            response = llm.invoke(prompt)
            doc_result = response.content if hasattr(response, 'content') else str(response)
        else:
            doc_result = "No documentation found in vectorstore."
    except Exception as e:
        doc_result = f"Error searching docs: {e}"
    
    # Search GitHub code
    print("   🔍 Searching GitHub code repository...")
    code_result = search_github_code(keyword)
    
    # Get contributors for this topic
    print("   🔍 Finding team members who worked on this topic...")
    contributors_data = get_github_contributors(keyword)
    
    # Display results
    print("\n" + "─"*70)
    print("📖 DOCUMENTATION:")
    print("─"*70)
    print(doc_result)
    
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
    """Ask if user wants to take a quiz."""
    print("\n" + "="*70)
    response = input("❓ Would you like to take a quiz on what you've learned? (yes/no): ").strip().lower()
    return response in ['yes', 'y']

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
        
        # Parse quiz into questions and answers
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
        
        # Interactive quiz
        if not quiz_data:
            print("⚠️ Could not parse quiz. Showing raw format:\n")
            print(quiz_content)
            return {**inputs, "quiz": quiz_content, "quiz_data": []}
        
        user_answers = []
        correct_count = 0
        
        for i, q in enumerate(quiz_data, 1):
            print(f"\n{'─'*70}")
            print(f"\n{q['question']}")
            for opt in q['options']:
                print(f"   {opt}")
            
            # Get user answer
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
            correct_ans = q['answer']
            is_correct = user_ans == correct_ans
            if is_correct:
                correct_count += 1
            
            print(f"\n{'─'*70}")
            print(f"Question {i}: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
            print(f"Your answer: {user_ans} | Correct answer: {correct_ans}")
            if q['explanation']:
                print(f"💡 {q['explanation']}")
        
        print(f"\n{'='*70}")
        print(f"🎯 FINAL SCORE: {correct_count}/{len(quiz_data)} ({correct_count*100//len(quiz_data)}%)")
        
        if correct_count == len(quiz_data):
            print("🌟 Perfect score! Excellent work!")
        elif correct_count >= len(quiz_data) * 0.7:
            print("👍 Great job! You have a solid understanding.")
        elif correct_count >= len(quiz_data) * 0.5:
            print("📚 Good effort! Review the topics you missed.")
        else:
            print("💪 Keep learning! Review the materials and try again.")
        
        print("="*70)
        
        return {**inputs, "quiz": quiz_content, "quiz_data": quiz_data, "score": f"{correct_count}/{len(quiz_data)}"}
        
    except Exception as e:
        print(f"⚠️ Could not generate quiz: {e}")
        return {**inputs, "quiz": "", "quiz_data": []}

def ask_continue(inputs):
    """Ask what user wants to do next."""
    print("\n" + "="*70)
    print("🎯 WHAT'S NEXT?")
    print("="*70)
    print("1. Continue Learning (search more topics)")
    print("2. Connect with Team (get contact info)")
    print("3. End Session")
    
    choice = input("\nYour choice (1/2/3): ").strip()
    return choice

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
            print(f"   • Reach out to {contributors[0]['name']} - most active on this topic")
            if len(contributors) > 1:
                print(f"   • Also connect with {contributors[1]['name']} for additional insights")
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
            # After team connection, ask again
            if input("\nContinue learning? (yes/no): ").strip().lower() not in ['yes', 'y']:
                break
        elif choice == "3":
            print("\n" + "="*70)
            print("👋 Thank you for using ERPNext Onboarding!")
            print("🎓 Keep learning and growing!")
            print("="*70)
            break
        else:
            print("Invalid choice. Ending session.")
            break

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