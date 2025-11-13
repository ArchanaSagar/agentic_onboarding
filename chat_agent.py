import os
import requests
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List, Literal
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import json
from datetime import datetime

# ==============================
# 0. Load Environment Variables
# ==============================
load_dotenv()

# ==============================
# 1. Initialize LLM
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

# Global storage
vectorstore_docs = None
vectorstore_code = None

# ==============================
# 2. Initialize Knowledge Base
# ==============================
def initialize_knowledge_base():
    """Initialize vector stores for documentation and code"""
    global vectorstore_docs, vectorstore_code
    
    print("\n🔄 Initializing knowledge base...")
    try:
        loader_docs = WebBaseLoader("https://docs.frappe.io/erpnext/introduction")
        docs = loader_docs.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=100
        )
        split_docs = text_splitter.split_documents(docs)
        
        vectorstore_docs = FAISS.from_documents(split_docs, embeddings)
        vectorstore_code = vectorstore_docs  # Placeholder
        print("✅ Knowledge base ready!\n")
    except Exception as e:
        print(f"⚠️  Using fallback knowledge base: {e}\n")
        from langchain_core.documents import Document
        fallback_docs = [
            Document(page_content="""ERPNext is a comprehensive open-source ERP system with the following modules:
            - Accounting: Financial management, invoicing, payments, reports
            - HR & Payroll: Employee management, attendance, payroll processing
            - Manufacturing: Production planning, BOM, work orders
            - Sales & CRM: Customer management, sales orders, quotations
            - Purchase & Inventory: Supplier management, stock, warehouses
            - Projects: Project management, tasks, timesheets
            - Healthcare: Patient management, appointments, medical records"""),
            
            Document(page_content="""Common roles in ERPNext implementation:
            - Developer: Customization, API integration, custom apps
            - Tester/QA: Testing workflows, bug reporting, quality assurance
            - Business Analyst: Requirements gathering, process documentation
            - Solution Architect: System design, technical decisions
            - System Administrator: Server setup, deployment, maintenance
            - Functional Consultant: Business process configuration
            - End User: Daily operations, data entry, report generation"""),
        ]
        vectorstore_docs = FAISS.from_documents(fallback_docs, embeddings)
        vectorstore_code = vectorstore_docs

# ==============================
# 3. Define Tools
# ==============================

@tool
def search_documentation(query: str) -> str:
    """Search ERPNext documentation for relevant information."""
    if vectorstore_docs is None:
        return "Knowledge base not initialized."
    
    retriever = vectorstore_docs.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(query)
    
    if not docs:
        return "No relevant documentation found."
    
    return "\n\n---\n\n".join([doc.page_content for doc in docs])[:3000]

@tool
def search_code_examples(query: str) -> str:
    """Search for code examples and implementation patterns."""
    if vectorstore_code is None:
        return "Code repository not available."
    
    retriever = vectorstore_code.as_retriever(search_kwargs={"k": 2})
    docs = retriever.get_relevant_documents(query)
    
    if not docs:
        return "No code examples found."
    
    return "\n\n".join([doc.page_content for doc in docs])[:2000]

@tool
def get_recent_commits(keyword: str) -> str:
    """Get recent GitHub commits related to a topic."""
    url = f"https://api.github.com/search/commits?q={keyword}+repo:frappe/erpnext&per_page=3"
    headers = {"Accept": "application/vnd.github.cloak-preview"}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return "Recent commits unavailable at the moment."
        
        commits = response.json().get("items", [])
        if not commits:
            return f"No recent commits found for: {keyword}"
        
        result = []
        for i, c in enumerate(commits, 1):
            msg = c['commit']['message'].split('\n')[0][:100]
            result.append(f"{i}. {msg}\n   🔗 {c['html_url']}")
        
        return "\n\n".join(result)
    except:
        return "Could not fetch recent commits."

@tool
def get_available_modules() -> str:
    """Get list of all ERPNext modules."""
    docs = search_documentation.invoke({"query": "ERPNext modules domains features"})
    
    prompt = f"""Based on this documentation:
{docs}

List the main ERPNext modules/domains. Return as JSON array of strings.
If unclear, return: ["Accounting", "HR & Payroll", "Manufacturing", "Sales & CRM", "Purchase & Inventory", "Projects", "Healthcare", "Assets"]

Return ONLY the JSON array."""
    
    response = llm.invoke(prompt)
    try:
        return response.content.strip()
    except:
        return '["Accounting", "HR & Payroll", "Manufacturing", "Sales & CRM", "Purchase & Inventory", "Projects", "Healthcare"]'

@tool
def get_available_roles() -> str:
    """Get list of common team roles."""
    return '["Developer", "QA/Tester", "Business Analyst", "Solution Architect", "System Admin", "Functional Consultant", "Project Manager", "End User"]'

@tool
def match_user_to_module(user_input: str) -> str:
    """Match user's interest/background to best ERPNext module."""
    modules = json.loads(get_available_modules.invoke({}))
    
    prompt = f"""User said: "{user_input}"

Available modules: {', '.join(modules)}

Which module best matches their interest? Return ONLY the exact module name."""
    
    response = llm.invoke(prompt)
    match = response.content.strip()
    
    for module in modules:
        if module.lower() in match.lower():
            return module
    
    return modules[0]

@tool
def match_user_to_role(user_input: str) -> str:
    """Match user's background to best role."""
    roles = json.loads(get_available_roles.invoke({}))
    
    prompt = f"""User said: "{user_input}"

Available roles: {', '.join(roles)}

Which role best matches their background? Return ONLY the exact role name."""
    
    response = llm.invoke(prompt)
    match = response.content.strip()
    
    for role in roles:
        if role.lower() in match.lower():
            return role
    
    return roles[0]

@tool
def create_onboarding_plan(module: str, role: str, user_name: str) -> str:
    """Create personalized onboarding plan."""
    docs = search_documentation.invoke({"query": f"{role} {module} responsibilities workflows"})
    
    prompt = f"""Create a warm, personalized onboarding plan for {user_name}, a new {role} joining the {module} team.

Available information:
{docs}

Structure:
**Welcome to the Team, {user_name}! 👋**

**Your Role**: [Brief description of what a {role} does]

**Your Module**: [What the {module} module does and why it matters]

**Week 1 Goals** 🎯:
- [3-4 specific, achievable goals]

**Key Topics to Learn** 📚:
1. [Topic 1] - [Why it's important]
2. [Topic 2] - [Why it's important]
3. [Topic 3] - [Why it's important]

**Quick Wins** ⚡:
- [2-3 things they can do immediately to contribute]

**Your Learning Path**:
I'll guide you through each topic step-by-step. Let's start whenever you're ready!

Keep it encouraging, specific, and actionable. Use their name to personalize."""
    
    return llm.invoke(prompt).content

@tool
def create_learning_guide(topic: str, module: str, role: str) -> str:
    """Create detailed learning guide for a topic."""
    docs = search_documentation.invoke({"query": f"{topic} {module} {role}"})
    code = search_code_examples.invoke({"query": topic})
    commits = get_recent_commits.invoke({"keyword": topic})
    
    prompt = f"""Create an engaging learning guide for: **{topic}**
For a {role} in {module}.

Resources:
=== Documentation ===
{docs}

=== Code Examples ===
{code}

=== Recent Updates ===
{commits}

Structure:
**📖 What is {topic}?**
[Clear explanation in 2-3 sentences]

**🎯 Why it matters for you**
[Relevance to their role]

**🔧 How it works**
[Step-by-step explanation]

**💡 Key Concepts**
- [Concept 1]: [Brief explanation]
- [Concept 2]: [Brief explanation]

**👨‍💻 Practical Examples**
[Real-world scenarios they'll encounter]

**⚡ Pro Tips**
- [2-3 insider tips]

**🚨 Common Pitfalls**
- [What to watch out for]

**🔗 Related Topics**
[What to learn next]

Make it conversational and practical!"""
    
    return llm.invoke(prompt).content

@tool
def generate_quiz(module: str, role: str, topics_covered: str) -> str:
    """Generate interactive quiz."""
    prompt = f"""Create a 5-question quiz for a {role} in {module}.
Topics covered: {topics_covered}

Return JSON array with this exact structure:
[
  {{
    "question": "Question text?",
    "options": ["A) First option", "B) Second option", "C) Third option", "D) Fourth option"],
    "correct": "A",
    "explanation": "Why A is correct and what it teaches"
  }}
]

Make questions practical and scenario-based. Return ONLY valid JSON."""
    
    return llm.invoke(prompt).content

@tool
def evaluate_answers(quiz_json: str, answers_json: str, user_name: str) -> str:
    """Evaluate quiz and provide encouraging feedback."""
    prompt = f"""Quiz: {quiz_json}
User answers: {answers_json}

Provide warm, encouraging feedback for {user_name}:

**Your Results** 🎯
Score: [X/5 correct]

**Question-by-question:**
1. [✓ or ✗] [If wrong, explain correct answer warmly]
2. ...

**What you mastered** ⭐:
[Topics they understood well]

**Let's review together** 📚:
[Topics to revisit, with encouragement]

**You're doing great!** Keep going! 🚀

Be supportive and constructive, never discouraging."""
    
    return llm.invoke(prompt).content

@tool
def suggest_next_topic(module: str, role: str, completed: str) -> str:
    """Suggest what to learn next."""
    docs = search_documentation.invoke({"query": f"{module} learning path {role}"})
    
    prompt = f"""User is {role} in {module}.
Completed topics: {completed}

Documentation: {docs}

Suggest the next best topic to learn.

Format:
**Next up: [Topic Name]** 🎯

**Why this next:**
[1-2 sentences on why this builds on what they know]

**What you'll learn:**
- [Benefit 1]
- [Benefit 2]

**Time needed:** [Realistic estimate]

Ready to dive in? Just say "yes" or "tell me about [topic]"!"""
    
    return llm.invoke(prompt).content

@tool
def find_team_help(module: str, topic: str) -> str:
    """Find who can help with specific questions."""
    prompt = f"""For questions about "{topic}" in {module}:

Suggest:
**Who can help** 👥:
- Primary: [Role/person who knows this best]
- Backup: [Alternative contact]

**Where to ask** 💬:
- Slack channel: #erpnext-{module.lower().replace(' ', '-')}
- Team email: {module.lower()}@company.com

**Other resources** 📚:
- [Relevant documentation links]
- [Community forum topics]

**Pro tip**: [How to ask effective questions about this topic]"""
    
    return llm.invoke(prompt).content

@tool
def save_progress(user_name: str, module: str, role: str, topics: str) -> str:
    """Save user's learning progress."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    progress = f"""
Progress saved for {user_name}!
Role: {role}
Module: {module}
Topics completed: {topics}
Last updated: {timestamp}

Keep up the great work! 🌟
"""
    # In production, save to database
    return progress

# ==============================
# 4. Agent State
# ==============================

class OnboardingState(TypedDict):
    messages: Annotated[List, "Conversation history"]
    user_name: str
    module: str
    role: str
    completed_topics: List[str]
    current_topic: str
    stage: str  # welcome, profile, planning, learning, quiz, complete
    quiz_data: str
    score: int

# ==============================
# 5. Build Agent
# ==============================

tools = [
    search_documentation,
    search_code_examples,
    get_recent_commits,
    get_available_modules,
    get_available_roles,
    match_user_to_module,
    match_user_to_role,
    create_onboarding_plan,
    create_learning_guide,
    generate_quiz,
    evaluate_answers,
    suggest_next_topic,
    find_team_help,
    save_progress
]

llm_with_tools = llm.bind_tools(tools)

def agent_node(state: OnboardingState) -> OnboardingState:
    """Main agent reasoning"""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": messages + [response]}

def tool_node(state: OnboardingState) -> OnboardingState:
    """Execute tools"""
    from langchain_core.messages import ToolMessage
    
    messages = state["messages"]
    last_message = messages[-1]
    
    tool_calls = getattr(last_message, 'tool_calls', [])
    
    tool_messages = []
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        for tool in tools:
            if tool.name == tool_name:
                result = tool.invoke(tool_args)
                tool_messages.append(
                    ToolMessage(content=str(result), tool_call_id=tool_call["id"])
                )
                break
    
    return {"messages": messages + tool_messages}

def should_continue(state: OnboardingState) -> Literal["tools", "end"]:
    """Route to tools or end"""
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return "end"

def create_onboarding_graph():
    """Build the agent graph"""
    workflow = StateGraph(OnboardingState)
    
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    
    workflow.set_entry_point("agent")
    
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END}
    )
    
    workflow.add_edge("tools", "agent")
    
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

# ==============================
# 6. Interactive Interface
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

def run_interactive_onboarding():
    """Run the interactive onboarding agent"""
    display_banner()
    
    # Initialize knowledge base
    initialize_knowledge_base()
    
    # Create agent
    app = create_onboarding_graph()
    
    # System prompt
    system_prompt = """You are Jamie, a friendly and enthusiastic ERPNext onboarding coach! 🎓

Your personality:
- Warm, encouraging, and supportive
- Use emojis to keep things friendly (but not too many!)
- Celebrate small wins
- Patient and thorough
- Make learning fun and interactive

Your process:
1. **Get to know them**: Ask their name warmly
2. **Understand background**: Ask about their interests and experience (don't list options, just ask naturally)
3. **Match them**: Use tools to match them to module and role
4. **Create plan**: Use create_onboarding_plan with their name
5. **Guide learning**: 
   - Use create_learning_guide for each topic
   - Check understanding frequently
   - Offer quiz after 2-3 topics
6. **Keep them engaged**:
   - Ask if they have questions
   - Suggest next topics with suggest_next_topic
   - Offer to connect them with team using find_team_help
7. **Track progress**: Use save_progress periodically

Important:
- Always use their name when you know it
- Be conversational, not robotic
- Ask open-ended questions, not multiple choice
- Celebrate their progress
- If they seem stuck, offer help proactively
- Keep responses concise unless explaining complex topics

Commands you recognize:
- "quiz me" → Generate quiz
- "what's next" → Suggest next topic
- "need help" → Connect with team
- "save progress" → Save their learning state

Start by warmly greeting them and asking their name!"""

    # Initial state
    config = {"configurable": {"thread_id": f"onboarding-{datetime.now().strftime('%Y%m%d-%H%M%S')}"}}
    
    state = {
        "messages": [
            SystemMessage(content=system_prompt),
            AIMessage(content="Hi there! 👋 Welcome to the ERPNext team! I'm Jamie, and I'm so excited to be your onboarding coach.\n\nBefore we dive in, what's your name?")
        ],
        "user_name": "",
        "module": "",
        "role": "",
        "completed_topics": [],
        "current_topic": "",
        "stage": "welcome",
        "quiz_data": "",
        "score": 0
    }
    
    print("Jamie: " + state["messages"][-1].content)
    print("\n" + "-"*70 + "\n")
    
    # Main loop
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            # Handle special commands
            if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                print("\nJamie: It's been wonderful working with you! 🌟")
                print("Remember, you can always come back to continue learning.")
                print("Good luck on your ERPNext journey! 👋\n")
                break
            
            if user_input.lower() == 'help':
                print("\nJamie: Here's how I can help you:")
                print("  📚 Learn about topics - just ask!")
                print("  🎯 'quiz me' - test your knowledge")
                print("  🔜 'what's next' - see what to learn next")
                print("  👥 'need help' - connect with team members")
                print("  💾 'save progress' - save your learning state")
                print("  🚪 'exit' - end the session\n")
                continue
            
            if not user_input:
                continue
            
            # Add user message
            state["messages"].append(HumanMessage(content=user_input))
            
            # Run agent
            result = app.invoke(state, config)
            state = result
            
            # Display agent response
            last_message = state["messages"][-1]
            if hasattr(last_message, 'content') and last_message.content:
                print(f"\nJamie: {last_message.content}")
                print("\n" + "-"*70 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nJamie: Taking a break? No problem! Your progress is saved.")
            print("Come back anytime to continue! 👋\n")
            break
        except Exception as e:
            print(f"\nJamie: Oops! I encountered an issue: {e}")
            print("Let's try that again! 😊\n")

# ==============================
# 7. Main Entry Point
# ==============================

if __name__ == "__main__":
    run_interactive_onboarding()