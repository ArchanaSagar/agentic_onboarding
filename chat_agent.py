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
def search_github_commits(keyword: str, repo: str = "frappe/erpnext") -> str:
    """Search GitHub commits for specific keywords."""
    url = f"https://api.github.com/search/commits?q={keyword}+repo:{repo}&per_page=5"
    headers = {"Accept": "application/vnd.github.cloak-preview"}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return f"GitHub API returned status {response.status_code}."
        
        data = response.json()
        commits = data.get("items", [])
        
        if not commits:
            return f"No commits found for: {keyword}"
        
        result = [f"Recent commits for '{keyword}':\n"]
        for i, c in enumerate(commits[:5], 1):
            msg = c['commit']['message'].split('\n')[0][:80]
            result.append(f"{i}. {msg}\n   {c['html_url']}")
        
        return "\n".join(result)
    except Exception as e:
        return f"Could not search commits: {str(e)}"

def extract_domains_and_roles():
    """Use LLM to extract domains and roles from documentation."""
    print("🔍 Extracting available domains and roles...")
    
    # Get sample docs for context
    try:
        docs = qa_docs.invoke("ERPNext modules and domains")
        context = "\n".join([doc.page_content[:500] for doc in docs])
    except Exception as e:
        print(f"⚠️ Could not fetch docs: {e}")
        context = "ERPNext has modules for Accounting, HR, Manufacturing, Sales, Purchase, Projects, Healthcare."
    
    domain_prompt = f"""Based on this ERPNext documentation:
{context}

List the main ERPNext modules/domains in comma-separated format.
Include: Accounting, HR, Manufacturing, Sales, Purchase, Projects, Healthcare, etc.
Return ONLY the comma-separated list."""

    role_prompt = """List common roles in ERPNext projects in comma-separated format.
Include: Developer, Tester, Business Analyst, Solution Architect, System Administrator, Functional Consultant, etc.
Return ONLY the comma-separated list."""

    try:
        # Use invoke instead of predict for newer versions
        domains_response = llm.invoke(domain_prompt)
        roles_response = llm.invoke(role_prompt)
        
        # Handle different response types
        domains_text = domains_response.content if hasattr(domains_response, 'content') else str(domains_response)
        roles_text = roles_response.content if hasattr(roles_response, 'content') else str(roles_response)
        
        
        domains = [d.strip() for d in domains_text.split(",") if d.strip()]
        roles = [r.strip() for r in roles_text.split(",") if r.strip()]
        
        # Fallback to defaults if extraction fails
        if not domains:
            domains = ["Accounting", "HR & Payroll", "Manufacturing", "Sales", "Purchase", "Projects", "Healthcare"]
        if not roles:
            roles = ["Developer", "Tester", "Business Analyst", "Solution Architect", "System Administrator"]
            
        return domains, roles
    except Exception as e:
        print(f"⚠️ Using default domains/roles due to error: {e}")
        # Fallback defaults
        return ["Accounting", "HR & Payroll", "Manufacturing", "Sales", "Purchase", "Projects"], \
               ["Developer", "Tester", "Business Analyst", "Solution Architect", "System Administrator"]

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
        return {**inputs, "materials_doc": "", "materials_code": "", "commits": ""}
    
    print(f"\n🔎 Searching for '{keyword}'...")
    
    # Search documentation
    try:
        doc_query = f"Explain '{keyword}' for {inputs['role']} in {inputs['domain']} domain"
        docs = qa_docs.invoke(doc_query)
        
        if docs:
            context = "\n\n".join([doc.page_content[:1000] for doc in docs])
            
            prompt = f"""Explain '{keyword}' for a {inputs['role']} in {inputs['domain']} domain.

Context from documentation:
{context}

Provide:
1. What it is
2. How it works
3. Practical usage
4. Key points to know

Be concise and role-specific."""
            
            response = llm.invoke(prompt)
            doc_result = response.content if hasattr(response, 'content') else str(response)
        else:
            doc_result = "No documentation found."
    except Exception as e:
        doc_result = f"Error searching docs: {e}"
    
    # Search code
    try:
        code_docs = qa_code.invoke(f"Code implementation of {keyword}")
        code_result = "\n\n".join([doc.page_content[:500] for doc in code_docs]) if code_docs else "No code snippets found."
    except Exception as e:
        code_result = f"Error searching code: {e}"
    
    # Search GitHub commits
    commits = search_github_commits(keyword)
    
    # Display results
    print("\n" + "─"*70)
    print("📖 DOCUMENTATION:")
    print("─"*70)
    print(doc_result)
    
    print("\n" + "─"*70)
    print("💻 CODE SNIPPETS:")
    print("─"*70)
    print(code_result[:1500])
    
    print("\n" + "─"*70)
    print("🔄 RELATED COMMITS:")
    print("─"*70)
    print(commits)
    
    if "No" in doc_result and "No" in code_result:
        print("\n💡 TIP: I couldn't find much content. You may want to contact the team for help.")
    
    return {**inputs, "materials_doc": doc_result, "materials_code": code_result, "commits": commits}

def ask_quiz(inputs):
    """Ask if user wants to take a quiz."""
    print("\n" + "="*70)
    response = input("❓ Would you like to take a quiz on what you've learned? (yes/no): ").strip().lower()
    return response in ['yes', 'y']

def present_quiz(inputs):
    """Generate and present quiz questions."""
    print("\n" + "="*70)
    print("📝 QUIZ TIME!")
    print("="*70)
    
    prompt = f"""Generate 5 multiple-choice quiz questions for a {inputs['role']} in {inputs['domain']} domain.

Format each question as:
Q1: [Question text]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
Correct Answer: [A/B/C/D]

Make questions practical and relevant to their role."""
    
    try:
        response = llm.invoke(prompt)
        quiz = response.content if hasattr(response, 'content') else str(response)
        print("\n" + quiz)
        return {**inputs, "quiz": quiz}
    except Exception as e:
        print(f"⚠️ Could not generate quiz: {e}")
        return {**inputs, "quiz": ""}

def evaluate_quiz(inputs):
    """Collect and evaluate quiz answers."""
    print("\n" + "─"*70)
    print("Answer the questions above (format: 1A 2C 3B 4D 5A)")
    answers = input("Your answers: ").strip()
    
    if not answers:
        print("No answers provided.")
        return inputs
    
    prompt = f"""Quiz:
{inputs.get('quiz', '')}

User's answers: {answers}

Evaluate the answers and provide:
1. Score (X/5)
2. Which answers were correct/incorrect
3. Brief explanation for incorrect answers
4. Encouraging feedback"""
    
    try:
        response = llm.invoke(prompt)
        feedback = response.content if hasattr(response, 'content') else str(response)
        print("\n" + "─"*70)
        print("📊 QUIZ RESULTS:")
        print("─"*70)
        print(feedback)
    except Exception as e:
        print(f"⚠️ Could not evaluate quiz: {e}")
    
    return inputs

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
    """Provide team contact information."""
    print("\n" + "="*70)
    print("👥 TEAM CONNECTIONS")
    print("="*70)
    
    prompt = f"""For a {inputs['role']} in {inputs['domain']} domain:

Suggest:
1. Who they should contact (role/team)
2. Relevant communication channels (Slack/Teams)
3. Documentation resources
4. Community forums

Be specific and practical."""
    
    try:
        response = llm.invoke(prompt)
        contact_info = response.content if hasattr(response, 'content') else str(response)
        print("\n" + contact_info)
    except Exception as e:
        print(f"⚠️ Error: {e}")
        print(f"\n💡 Try reaching out to senior {inputs['role']}s in the {inputs['domain']} team!")

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
            user_info = evaluate_quiz(user_info)
        
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