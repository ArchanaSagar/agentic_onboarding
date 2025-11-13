import os
import requests
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List, Literal
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
import json
from bs4 import BeautifulSoup
from langchain_core.documents import Document

# ==============================
# 0. Load Environment Variables
# ==============================
load_dotenv()

# ==============================
# 1. Initialize LLM with Tool Binding
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
# 2. Initialize Vector Stores
# ==============================
vectorstore_docs = None
vectorstore_code = None


BASE_URL = "https://docs.frappe.io/erpnext"

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

            # Collect internal links
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
    global vectorstore_docs, vectorstore_code
    VECTORSTORE_PATH = "erpnext_vectorstore"

    # ✅ Step 1: Load from existing FAISS vectorstore if available
    if os.path.exists(VECTORSTORE_PATH):
        try:
            print("💾 Loading cached vectorstore from disk...")
            vectorstore_docs = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
            vectorstore_code = vectorstore_docs
            print("✅ Vectorstore loaded successfully from cache!")
            return
        except Exception as e:
            print(f"⚠️  Failed to load cached vectorstore: {e}")
            print("Rebuilding from source...")

    # ✅ Step 2: Otherwise, rebuild it from ERPNext documentation
    
    print("🔄 Loading ERPNext documentation...")
    try:
        all_links = get_all_links(BASE_URL)
        all_docs = []
        for link in all_links:
           
            try:
                loader = WebBaseLoader(link,verify_ssl = False,trust_env= True)
                docs = loader.load()
                all_docs.extend(docs)
                print(f"📄 Loaded: {link}")
            except Exception as e:
                print(f"⚠️ Skipping {link}: {e}")

        print(f"✅ Loaded {len(all_docs)} documents.")

        # Split text into manageable chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_docs = splitter.split_documents(all_docs)

        # Create FAISS vectorstore
        vectorstore_docs = FAISS.from_documents(split_docs, embeddings)
        vectorstore_code = vectorstore_docs
        print("✅ Vectorstore created successfully!")

        # Optional: save to disk
        vectorstore_docs.save_local("erpnext_vectorstore")
        print("💾 Saved vectorstore to 'erpnext_vectorstore'")
    except Exception as e:
        print(f"⚠️  Warning: Could not load documentation: {e}")
        print("Creating minimal fallback vectorstore...")
        fallback_docs = [
            Document(page_content="ERPNext is an open-source ERP system with modules for Accounting, HR, Manufacturing, Sales, Purchase, Projects, and Healthcare.", metadata={}),
            Document(page_content="Common roles in ERPNext include Developer, Tester, Business Analyst, Solution Architect, and System Administrator.", metadata={})
        ]
        vectorstore_docs = FAISS.from_documents(fallback_docs, embeddings)
        vectorstore_code = vectorstore_docs
        vectorstore_docs = FAISS.from_documents(fallback_docs, embeddings)
        vectorstore_code = vectorstore_docs

# ==============================
# 3. Define MCP-Style Tools
# ==============================

@tool
def search_documentation(query: str) -> str:
    """Search ERPNext documentation for relevant information.
    
    Args:
        query: The search query or topic to look up
        
    Returns:
        Relevant documentation content
    """
    if vectorstore_docs is None:
        return "Error: Documentation not loaded."
    
    retriever = vectorstore_docs.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(query)
    
    if not docs:
        return "No relevant documentation found."
    
    content = "\n\n---\n\n".join([doc.page_content for doc in docs])
    return content[:3000]

@tool
def search_code_repository(query: str) -> str:
    """Search code repository for relevant code snippets and implementations.
    
    Args:
        query: The feature or code pattern to search for
        
    Returns:
        Relevant code snippets and examples
    """
    if vectorstore_code is None:
        return "Error: Code repository not loaded."
    
    retriever = vectorstore_code.as_retriever(search_kwargs={"k": 2})
    docs = retriever.get_relevant_documents(query)
    
    if not docs:
        return "No relevant code found."
    
    content = "\n\n---\n\n".join([doc.page_content for doc in docs])
    return content[:2000]

@tool
def search_github_commits(keyword: str, repo: str = "frappe/erpnext") -> str:
    """Search GitHub commits for specific keywords to find recent changes.
    
    Args:
        keyword: The keyword to search in commit messages
        repo: GitHub repository in format 'owner/repo' (default: frappe/erpnext)
        
    Returns:
        List of recent commits related to the keyword
    """
    url = f"https://api.github.com/search/commits?q={keyword}+repo:{repo}&per_page=5"
    headers = {"Accept": "application/vnd.github.cloak-preview"}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return f"GitHub API returned status {response.status_code}. Commits search unavailable."
        
        data = response.json()
        commits = data.get("items", [])
        
        if not commits:
            return f"No commits found for keyword: {keyword}"
        
        result = ["Recent commits related to your topic:\n"]
        for i, c in enumerate(commits[:5], 1):
            msg = c['commit']['message'].split('\n')[0][:80]
            result.append(f"{i}. {msg}...\n   {c['html_url']}")
        
        return "\n".join(result)
    except Exception as e:
        return f"Could not search commits: {str(e)}"

@tool
def extract_available_domains() -> str:
    """Extract available ERPNext domains/modules from documentation.
    
    Returns:
        JSON string with list of available domains
    """
    query = "What are all the main modules and domains in ERPNext? List them."
    docs_content = search_documentation.invoke({"query": query})
    
    prompt = f"""Based on this ERPNext documentation:
{docs_content}

Extract and list ALL main domains/modules in ERPNext. Return ONLY a JSON array of strings.
Example format: ["Accounting", "HR", "Manufacturing", "Sales", "Purchase"]

If you can't find specific modules, return the standard ERPNext modules list.
Return ONLY the JSON array, no other text."""
    
    response = llm.invoke(prompt)
    content = response.content.strip()
    
    # Try to parse JSON, fallback to default if needed
    try:
        domains = json.loads(content)
        return json.dumps(domains)
    except:
        # Fallback to known ERPNext domains
        default_domains = [
            "Accounting", "HR & Payroll", "Manufacturing", 
            "Sales & CRM", "Purchase & Inventory", "Projects", 
            "Healthcare", "Assets", "Stock", "Quality"
        ]
        return json.dumps(default_domains)

@tool
def extract_available_roles() -> str:
    """Extract available roles for ERPNext projects.
    
    Returns:
        JSON string with list of available roles
    """
    query = "What are common roles in ERPNext implementation projects?"
    docs_content = search_documentation.invoke({"query": query})
    
    prompt = f"""Based on this documentation:
{docs_content}

Extract and list common roles in ERPNext projects. Return ONLY a JSON array of strings.
Include roles like Developer, Tester, Business Analyst, etc.

Return ONLY the JSON array, no other text."""
    
    response = llm.invoke(prompt)
    content = response.content.strip()
    
    try:
        roles = json.loads(content)
        return json.dumps(roles)
    except:
        # Fallback to common roles
        default_roles = [
            "Developer", "Tester/QA", "Business Analyst", 
            "Solution Architect", "System Administrator", 
            "Functional Consultant", "End User", "Project Manager"
        ]
        return json.dumps(default_roles)

@tool
def select_best_domain(user_description: str, available_domains: str) -> str:
    """Intelligently select the best domain based on user's description or interests.
    
    Args:
        user_description: User's description of their interests or background
        available_domains: JSON string of available domains
        
    Returns:
        Selected domain name
    """
    domains = json.loads(available_domains)
    
    prompt = f"""User said: "{user_description}"

Available ERPNext domains: {', '.join(domains)}

Based on the user's input, select the MOST appropriate domain.
Return ONLY the exact domain name from the list, nothing else."""
    
    response = llm.invoke(prompt)
    selected = response.content.strip()
    
    # Validate selection
    if selected in domains:
        return selected
    
    # Fuzzy match
    for domain in domains:
        if domain.lower() in selected.lower() or selected.lower() in domain.lower():
            return domain
    
    # Default to first domain
    return domains[0]

@tool
def select_best_role(user_description: str, available_roles: str) -> str:
    """Intelligently select the best role based on user's description.
    
    Args:
        user_description: User's description of their background or expertise
        available_roles: JSON string of available roles
        
    Returns:
        Selected role name
    """
    roles = json.loads(available_roles)
    
    prompt = f"""User said: "{user_description}"

Available roles: {', '.join(roles)}

Based on the user's input, select the MOST appropriate role.
Return ONLY the exact role name from the list, nothing else."""
    
    response = llm.invoke(prompt)
    selected = response.content.strip()
    
    # Validate selection
    if selected in roles:
        return selected
    
    # Fuzzy match
    for role in roles:
        if role.lower() in selected.lower() or selected.lower() in role.lower():
            return role
    
    return roles[0]

@tool
def generate_onboarding_summary(domain: str, role: str) -> str:
    """Generate comprehensive onboarding summary for the user.
    
    Args:
        domain: The selected ERPNext domain
        role: The user's role
        
    Returns:
        Detailed onboarding summary
    """
    # Search relevant docs
    query = f"{role} responsibilities in {domain} module ERPNext workflows features"
    docs_content = search_documentation.invoke({"query": query})
    
    prompt = f"""Create a comprehensive onboarding summary for a {role} joining the {domain} domain in ERPNext.

Relevant documentation:
{docs_content}

Structure your summary with:
1. **Welcome & Role Overview**: What this role does in this domain
2. **Key Responsibilities**: 3-5 main responsibilities
3. **Essential Features**: Top features they need to master
4. **Common Workflows**: Typical processes they'll work with
5. **Quick Start Guide**: First steps to get productive
6. **Resources**: What to study first

Make it engaging, specific, and actionable. Use bullet points for clarity."""
    
    response = llm.invoke(prompt)
    return response.content

@tool
def generate_learning_material(topic: str, domain: str, role: str) -> str:
    """Generate comprehensive learning material for a specific topic.
    
    Args:
        topic: The topic or feature to learn
        domain: User's domain
        role: User's role
        
    Returns:
        Structured learning content
    """
    # Gather information from multiple sources
    doc_query = f"{topic} in {domain} for {role}"
    docs = search_documentation.invoke({"query": doc_query})
    code = search_code_repository.invoke({"query": topic})
    commits = search_github_commits.invoke({"keyword": topic})
    
    prompt = f"""Create a comprehensive learning guide for: "{topic}"
For: {role} in {domain} domain

Available information:
=== Documentation ===
{docs}

=== Code Examples ===
{code}

=== Recent Changes ===
{commits}

Create a structured learning guide with:
1. **Concept Overview**: What is this and why it matters
2. **How It Works**: Technical explanation appropriate for a {role}
3. **Practical Examples**: Real-world usage scenarios
4. **Code Snippets** (if applicable): Key implementation details
5. **Recent Updates**: Important changes to be aware of
6. **Pro Tips**: Best practices and gotchas
7. **Next Steps**: Related topics to explore

Make it practical and role-appropriate."""
    
    response = llm.invoke(prompt)
    return response.content

@tool
def generate_quiz(domain: str, role: str, covered_topics: str, num_questions: int = 5) -> str:
    """Generate an interactive quiz based on what the user has learned.
    
    Args:
        domain: User's domain
        role: User's role
        covered_topics: Comma-separated list of topics covered
        num_questions: Number of questions to generate
        
    Returns:
        Quiz in structured JSON format
    """
    prompt = f"""Generate {num_questions} multiple-choice quiz questions for a {role} in {domain} domain.

Topics covered: {covered_topics}

Return a JSON array where each question has this structure:
{{
    "question": "The question text",
    "options": ["A) option1", "B) option2", "C) option3", "D) option4"],
    "correct": "A",
    "explanation": "Why this answer is correct"
}}

Make questions practical, relevant to their role, and test understanding not just memorization.
Return ONLY the JSON array, no markdown or other text."""
    
    response = llm.invoke(prompt)
    return response.content

@tool
def evaluate_quiz_answers(quiz_json: str, user_answers: str) -> str:
    """Evaluate user's quiz answers and provide feedback.
    
    Args:
        quiz_json: The original quiz in JSON format
        user_answers: User's answers as JSON {"1": "A", "2": "C", ...}
        
    Returns:
        Detailed feedback and score
    """
    prompt = f"""Quiz:
{quiz_json}

User's Answers:
{user_answers}

Evaluate the answers and provide:
1. Score (X/Y correct)
2. For each question:
   - Whether they got it right ✓ or wrong ✗
   - The correct answer if they were wrong
   - Brief explanation
3. Overall feedback and suggestions for improvement
4. Topics they should review

Be encouraging and constructive."""
    
    response = llm.invoke(prompt)
    return response.content

@tool
def suggest_next_topic(domain: str, role: str, completed_topics: str) -> str:
    """Intelligently suggest the next topic to learn based on progress.
    
    Args:
        domain: User's domain
        role: User's role
        completed_topics: Comma-separated list of topics already covered
        
    Returns:
        Suggested next topic with reasoning
    """
    query = f"Learning path for {role} in {domain}"
    docs = search_documentation.invoke({"query": query})
    
    prompt = f"""User is a {role} in {domain} domain.

Topics they've already covered: {completed_topics}

Relevant documentation:
{docs}

Suggest the NEXT most logical topic they should learn.
Consider:
- Natural learning progression
- Building on what they know
- Practical importance for their role

Return format:
Topic: [topic name]
Reason: [why this topic next in 1-2 sentences]
Prerequisites: [what they need to know - they should already know this]"""
    
    response = llm.invoke(prompt)
    return response.content

@tool
def identify_team_contact(domain: str, topic: str) -> str:
    """Identify which team member or channel to contact for help.
    
    Args:
        domain: User's domain
        topic: The topic they need help with
        
    Returns:
        Suggested contact or channel
    """
    prompt = f"""For a question about "{topic}" in the {domain} domain of ERPNext:

Who should they contact? Suggest:
1. Most relevant team/role (e.g., "Senior Developer in Accounting module")
2. Slack/communication channel (e.g., "#erpnext-accounting")
3. Documentation resources
4. Community forums or GitHub discussions

Be specific and practical."""
    
    response = llm.invoke(prompt)
    return response.content

# ==============================
# 4. Define Agent State
# ==============================

class AgentState(TypedDict):
    messages: Annotated[List, "Conversation history"]
    domain: str
    role: str
    covered_topics: List[str]
    current_topic: str
    quiz_taken: bool
    session_stage: str  # "init", "onboarding", "learning", "quiz", "complete"

# ==============================
# 5. Create Tools List
# ==============================

tools = [
    search_documentation,
    search_code_repository,
    search_github_commits,
    extract_available_domains,
    extract_available_roles,
    select_best_domain,
    select_best_role,
    generate_onboarding_summary,
    generate_learning_material,
    generate_quiz,
    evaluate_quiz_answers,
    suggest_next_topic,
    identify_team_contact
]

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)

# ==============================
# 6. Define Agent Nodes
# ==============================

def agent_node(state: AgentState) -> AgentState:
    """Main agent reasoning node - decides what to do next"""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": messages + [response]}

def tool_node(state: AgentState) -> AgentState:
    """Execute tools requested by the agent"""
    messages = state["messages"]
    last_message = messages[-1]
    
    tool_calls = last_message.tool_calls if hasattr(last_message, 'tool_calls') else []
    
    tool_results = []
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        # Find and execute the tool
        for tool in tools:
            if tool.name == tool_name:
                result = tool.invoke(tool_args)
                tool_results.append({
                    "tool_call_id": tool_call["id"],
                    "output": result
                })
                break
    
    # Add tool results to messages
    from langchain_core.messages import ToolMessage
    tool_messages = [
        ToolMessage(content=str(res["output"]), tool_call_id=res["tool_call_id"])
        for res in tool_results
    ]
    
    return {"messages": messages + tool_messages}

def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Determine if we should continue to tools or end"""
    messages = state["messages"]
    last_message = messages[-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return "end"

# ==============================
# 7. Build LangGraph Workflow
# ==============================

def create_agent_graph():
    """Create the LangGraph workflow"""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    
    # Tools always go back to agent
    workflow.add_edge("tools", "agent")
    
    # Compile with memory
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

# ==============================
# 8. Main Execution
# ==============================

def run_onboarding_agent():
    """Run the autonomous onboarding agent"""
    # Initialize
    initialize_vectorstores()
    
    # Create graph
    app = create_agent_graph()
    
    # System prompt for the agent
    system_prompt = """You are an intelligent ERPNext onboarding agent. Your goal is to autonomously guide new team members through onboarding.

Your process:
1. **Extract domains**: Use extract_available_domains tool
2. **Extract roles**: Use extract_available_roles tool
3. **Understand user**: Ask them briefly about their background/interests
4. **Select domain**: Use select_best_domain tool based on their input
5. **Select role**: Use select_best_role tool based on their input
6. **Generate summary**: Use generate_onboarding_summary tool
7. **Facilitate learning**: 
   - Ask what they want to learn
   - Use generate_learning_material tool for each topic
   - Use suggest_next_topic to guide them
   - Track covered topics
8. **Quiz them**: After 2-3 topics, offer quiz using generate_quiz and evaluate_quiz_answers
9. **Connect team**: Use identify_team_contact when needed

Be conversational, encouraging, and use tools to automate everything. Don't ask users to manually select from lists - understand their input and use tools to decide.

Keep responses concise and actionable. Use emojis sparingly for friendliness."""

    # Initial state
    config = {"configurable": {"thread_id": "onboarding-session-1"}}
    initial_state = {
        "messages": [
            SystemMessage(content=system_prompt),
            AIMessage(content="""
To start, tell me a bit about yourself:
- What's your technical background? (e.g., "I'm a Python developer", "I do QA testing", "I'm new to ERPs")
- What area interests you? (e.g., "interested in finance modules", "want to work on manufacturing features")

Just describe in your own words, and I'll figure out the rest! 🚀""")
        ],
        "domain": "",
        "role": "",
        "covered_topics": [],
        "current_topic": "",
        "quiz_taken": False,
        "session_stage": "init"
    }
    
    print(initial_state["messages"][-1].content)
    print("\n" + "="*60 + "\n")
    
    # Interactive loop
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("\n👋 Goodbye! Great learning with you!")
            break
        
        if not user_input:
            continue
        
        # Add user message
        initial_state["messages"].append(HumanMessage(content=user_input))
        
        # Run agent
        try:
            result = app.invoke(initial_state, config)
            
            # Update state
            initial_state = result
            
            # Print agent response
            last_message = result["messages"][-1]
            if hasattr(last_message, 'content'):
                print(f"\nAgent: {last_message.content}")
                print("\n" + "="*60 + "\n")
        
        except Exception as e:
            print(f"\n⚠️  Error: {e}")
            print("Let's try again...\n")



# ==============================
# . Interactive Interface
# ==============================

def display_banner():
    """Display welcome banner"""
    print("\n" + "="*70)
    print("🎓  WELCOME TO ERPNEXT TEAM ONBOARDING")
    print("     Your Interactive Learning Companion")
    print("="*70)
    print("\n💡 I'm here to help you get started with ERPNext!")
    print("📚 I'll guide you through learning materials")
    print("🎯 Create personalized learning paths")
    print("✅ Quiz you to reinforce learning")
    print("👥 Connect you with team members")
    print("\nType 'help' anytime for guidance, 'exit' to end session")
    print("="*70 + "\n")
# ==============================
# 9. Entry Point
# ==============================

if __name__ == "__main__":
    display_banner()
    run_onboarding_agent()