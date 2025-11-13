"""
Complete Project Onboarding Agent with LangGraph
Follows the defined user journey with proper state management
"""

from typing import TypedDict, Annotated, List, Optional, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import operator
import os
import json
import re

load_dotenv()

# ============================================================================
# STATE DEFINITION
# ============================================================================

class OnboardingState(TypedDict):
    """Complete state tracking for onboarding journey"""
    # Conversation history
    messages: Annotated[List, operator.add]
    
    # User profile
    domain: Optional[str]
    role: Optional[str]
    
    # Learning context
    learning_topic: Optional[str]  # keyword/feature ID/PBI
    project_summary: Optional[str]
    learning_materials: List[dict]
    
    # Quiz state
    quiz_ready: bool
    quiz_questions: List[dict]
    quiz_answers: List[str]
    feedback: Optional[str]
    
    # Team contacts
    team_contact: Optional[dict]
    
    # Flow control
    next_action: Optional[str]
    user_input: Optional[str]


# ============================================================================
# CONFIGURATION
# ============================================================================

# Mock project data (replace with actual data source)
PROJECT_DATA = {
    "backend": {
        "summary": """
        **Backend Team Overview:**
        Our backend infrastructure powers a microservices architecture handling 10M+ daily requests.
        - Tech Stack: Python (FastAPI), Node.js, PostgreSQL, Redis, Kafka
        - Key Services: User Auth, Payment Processing, Data Pipeline, API Gateway
        - Recent Focus: Migration to event-driven architecture and improved observability
        """,
        "materials": {
            "authentication": {
                "title": "OAuth2 Implementation Guide",
                "url": "https://github.com/company/backend-auth",
                "content": "Step-by-step OAuth2 setup with FastAPI and JWT tokens for secure authentication"
            },
            "kafka": {
                "title": "Event Streaming with Kafka",
                "url": "https://github.com/company/kafka-setup",
                "content": "Producer/Consumer patterns and best practices for event-driven architecture"
            }
        },
        "contact": {
            "name": "Sarah Chen",
            "role": "Backend Team Lead",
            "email": "sarah.chen@company.com",
            "slack": "@sarah"
        }
    },
    "frontend": {
        "summary": """
        **Frontend Team Overview:**
        We build responsive, accessible web applications using modern frameworks.
        - Tech Stack: React 18, TypeScript, Next.js, TailwindCSS, React Query
        - Key Focus: Component library, performance optimization, accessibility (WCAG 2.1)
        - Recent Work: Design system migration and micro-frontend architecture
        """,
        "materials": {
            "components": {
                "title": "Design System Components",
                "url": "https://github.com/company/design-system",
                "content": "Reusable React components with Storybook documentation and accessibility features"
            },
            "state-management": {
                "title": "State Management Patterns",
                "url": "https://github.com/company/react-patterns",
                "content": "Context, Redux Toolkit, and React Query patterns for scalable state management"
            }
        },
        "contact": {
            "name": "Mike Rodriguez",
            "role": "Frontend Team Lead",
            "email": "mike.rodriguez@company.com",
            "slack": "@mike"
        }
    },
    "data": {
        "summary": """
        **Data Science Team Overview:**
        We build ML models and data pipelines for analytics and personalization.
        - Tech Stack: Python, PyTorch, Airflow, Snowflake, DBT, MLflow
        - Key Projects: Recommendation engine, churn prediction, A/B testing platform
        - Recent Focus: MLOps maturity and real-time feature stores
        """,
        "materials": {
            "ml-pipeline": {
                "title": "ML Pipeline Architecture",
                "url": "https://github.com/company/ml-pipelines",
                "content": "End-to-end ML workflow from training to deployment with MLOps best practices"
            },
            "feature-store": {
                "title": "Feature Store Setup",
                "url": "https://github.com/company/feature-store",
                "content": "Real-time and batch feature engineering patterns for ML models"
            }
        },
        "contact": {
            "name": "Dr. Priya Sharma",
            "role": "Data Science Lead",
            "email": "priya.sharma@company.com",
            "slack": "@priya"
        }
    }
}

# ============================================================================
# LLM SETUP WITH PROMPTS
# ============================================================================

def get_llm(temperature=0.7):
    """Initialize LLM with Azure OpenAI configuration"""
    return AzureChatOpenAI(
        azure_endpoint=os.getenv("OPENAI_AZURE_ENDPOINT"),
        api_key=os.getenv("OPENAI_API_KEY"),
        api_version=os.getenv("OPENAI_API_VERSION"),
        model_name=os.getenv("OPENAI_MODEL_NAME"),
        temperature=temperature
    )


# Prompt templates for each stage
COLLECT_INFO_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a friendly onboarding assistant. Your job is to collect the user's domain and role.

Ask about:
1. Domain: backend, frontend, or data
2. Role: junior developer, senior engineer, lead, etc.

Be conversational and warm. If the user provides one piece of info, ask for the other."""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are presenting a project summary to a new team member.

Domain: {domain}
Role: {role}

Here's the summary:
{summary}

Present this in a welcoming way, then ask what specific topic they'd like to learn about.
Suggest they can ask about:
- Specific keywords (e.g., "authentication", "components", "ml-pipeline")
- Feature IDs or PBIs
- General areas of interest"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

MATERIAL_SEARCH_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are helping find learning materials based on user's topic request: {topic}

Available materials for {domain} domain:
{materials_list}

If the topic matches available materials, present them enthusiastically.
If no match, be honest and offer to connect them with the team lead."""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

QUIZ_INTRO_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """The user has finished learning about {topic}. 

Ask if they'd like to take a quick 2-question quiz to test their understanding.
Be encouraging and emphasize it's just for learning, not evaluation."""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

QUIZ_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a quiz generator for technical onboarding.

Generate 2 multiple-choice questions based on the following learning material:

**Domain:** {domain}
**Topic:** {topic}
**Material Content:** {material_content}
**Project Summary:** {project_summary}

Requirements:
- Create 2 questions that test understanding of key concepts
- Each question should have 4 options (A, B, C, D)
- Questions should be practical and relevant to the role
- Include the correct answer and a clear explanation

Return ONLY valid JSON in this exact format (no markdown, no preamble):
[
  {{
    "question": "Question text here?",
    "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
    "correct": "B",
    "explanation": "Explanation of why this is correct and others are wrong"
  }},
  {{
    "question": "Second question text?",
    "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
    "correct": "C",
    "explanation": "Explanation here"
  }}
]"""),
    ("human", "Generate the quiz now.")
])


# ============================================================================
# NODE FUNCTIONS
# ============================================================================

def collect_domain_role(state: OnboardingState) -> OnboardingState:
    """Node 1: Collect domain and role from user"""
    print("\n--- COLLECT DOMAIN & ROLE ---")
    
    # Check if we already have both - if yes, skip to summary
    if state.get("domain") and state.get("role"):
        print(f"Already have domain: {state.get('domain')}, role: {state.get('role')}")
        return {"next_action": "present_summary"}
    
    user_input = state.get("user_input", "")
    
    # Skip LLM call if user_input is empty (first run)
    if not user_input or user_input == "Hello":
        return {
            "messages": [AIMessage(content="Hello! What domain will you be working in (backend, frontend, or data) and what's your role?")],
            "next_action": "collect_info"
        }
    
    # Try to extract domain and role from user input (flexible keyword matching)
    domain_keywords = {"backend", "frontend", "front-end", "front end", "data", "data science"}
    role_keywords = {
        "junior", "senior", "lead", "developer", "engineer", "dev", 
        "architect", "manager", "intern", "principal"
    }
    
    user_lower = user_input.lower()
    extracted_domain = state.get("domain")  # Keep existing if already set
    extracted_role = state.get("role")  # Keep existing if already set
    
    # Extract domain if not already set
    if not extracted_domain:
        for domain in domain_keywords:
            if domain in user_lower:
                # Normalize domain names
                if "front" in domain:
                    extracted_domain = "frontend"
                elif "data" in domain:
                    extracted_domain = "data"
                else:
                    extracted_domain = domain
                break
    
    # Extract role if not already set - look for any role keyword
    if not extracted_role:
        for role in role_keywords:
            if role in user_lower:
                extracted_role = role
                break
    
    print(f"Extracted - Domain: {extracted_domain}, Role: {extracted_role}")
    
    # If we couldn't extract either, provide options
    if not extracted_domain or not extracted_role:
        missing_info = []
        if not extracted_domain:
            missing_info.append("domain")
        if not extracted_role:
            missing_info.append("role")
        
        response = f"I couldn't identify your {' and '.join(missing_info)}. Let me help!\n\n"
        
        if not extracted_domain:
            response += "**Available Domains:**\n"
            response += "1. Backend - Server-side development, APIs, databases\n"
            response += "2. Frontend - User interfaces, web applications\n"
            response += "3. Data - Data science, ML, analytics\n\n"
        
        if not extracted_role:
            response += "**Common Roles:**\n"
            response += "- Junior Developer/Engineer\n"
            response += "- Senior Developer/Engineer\n"
            response += "- Lead/Team Lead\n"
            response += "- Architect\n"
            response += "- Manager\n\n"
        
        response += "Please tell me your domain and role (e.g., 'backend senior developer' or 'frontend lead')."
        
        return {
            "messages": [AIMessage(content=response)],
            "domain": extracted_domain,
            "role": extracted_role,
            "next_action": "collect_info"
        }
    
    # Use LLM for friendly response when we have the info
    llm = get_llm(temperature=0.7)
    chain = COLLECT_INFO_PROMPT | llm | StrOutputParser()
    history = state.get("messages", [])
    
    response = chain.invoke({
        "history": history,
        "input": user_input
    })
    
    # Build updates
    updates = {
        "messages": [AIMessage(content=response)],
        "domain": extracted_domain,
        "role": extracted_role
    }
    
    # Decide next action based on what we have
    if extracted_domain and extracted_role:
        print("Both found! Moving to present_summary")
        updates["next_action"] = "present_summary"
    else:
        print(f"Missing - Domain: {not extracted_domain}, Role: {not extracted_role}")
        updates["next_action"] = "collect_info"
    
    return updates


def present_summary(state: OnboardingState) -> OnboardingState:
    """Node 2: Present project summary based on domain/role"""
    print("\n--- PRESENT SUMMARY ---")
    
    domain = state.get("domain", "backend")
    role = state.get("role", "team member")
    
    # Get domain-specific summary
    summary = PROJECT_DATA.get(domain, PROJECT_DATA["backend"])["summary"]
    
    llm = get_llm(temperature=0.7)
    chain = SUMMARY_PROMPT | llm | StrOutputParser()
    
    response = chain.invoke({
        "domain": domain,
        "role": role,
        "summary": summary,
        "history": state.get("messages", []),
        "input": state.get("user_input", "")
    })
    
    return {
        "project_summary": summary,
        "messages": [AIMessage(content=response)],
        "next_action": "search_materials"
    }


def search_materials(state: OnboardingState) -> OnboardingState:
    """Node 3: Search for learning materials based on topic"""
    print("\n--- SEARCH MATERIALS ---")
    
    user_input = state.get("user_input", "").lower()
    domain = state.get("domain", "backend")
    
    # Get available materials for domain
    domain_data = PROJECT_DATA.get(domain, PROJECT_DATA["backend"])
    available_materials = domain_data.get("materials", {})
    
    # Search for matching material
    found_material = None
    for keyword, material in available_materials.items():
        if keyword in user_input or user_input in keyword:
            found_material = material
            state["learning_topic"] = keyword
            break
    
    # Prepare materials list for LLM
    materials_list = "\n".join([
        f"- {keyword}: {mat['title']}"
        for keyword, mat in available_materials.items()
    ])
    
    llm = get_llm(temperature=0.7)
    chain = MATERIAL_SEARCH_PROMPT | llm | StrOutputParser()
    
    response = chain.invoke({
        "topic": user_input,
        "domain": domain,
        "materials_list": materials_list,
        "history": state.get("messages", []),
        "input": user_input
    })
    
    if found_material:
        # Material found - present it
        material_details = f"\n\n📚 **{found_material['title']}**\n"
        material_details += f"🔗 Repository: {found_material['url']}\n"
        material_details += f"📝 Overview: {found_material['content']}\n"
        
        response += material_details
        
        return {
            "learning_materials": [found_material],
            "messages": [AIMessage(content=response)],
            "next_action": "ask_quiz"
        }
    else:
        # No material found - provide contact
        contact = domain_data["contact"]
        contact_info = f"\n\n👤 **Contact Information:**\n"
        contact_info += f"Name: {contact['name']}\n"
        contact_info += f"Role: {contact['role']}\n"
        contact_info += f"Email: {contact['email']}\n"
        contact_info += f"Slack: {contact['slack']}\n"
        
        response += contact_info
        
        return {
            "team_contact": contact,
            "messages": [AIMessage(content=response)],
            "next_action": "ask_continue"
        }


def ask_quiz(state: OnboardingState) -> OnboardingState:
    """Node 4: Ask if user wants to take a quiz"""
    print("\n--- ASK FOR QUIZ ---")
    
    topic = state.get("learning_topic", "the material")
    
    llm = get_llm(temperature=0.7)
    chain = QUIZ_INTRO_PROMPT | llm | StrOutputParser()
    
    response = chain.invoke({
        "topic": topic,
        "history": state.get("messages", []),
        "input": state.get("user_input", "")
    })
    
    return {
        "messages": [AIMessage(content=response)],
        "next_action": "wait_quiz_response"
    }


def present_quiz(state: OnboardingState) -> OnboardingState:
    """Node 5: Generate and present quiz questions using LLM"""
    print("\n--- GENERATE & PRESENT QUIZ ---")
    
    domain = state.get("domain", "backend")
    topic = state.get("learning_topic", "general concepts")
    learning_materials = state.get("learning_materials", [])
    project_summary = state.get("project_summary", "")
    
    # Get material content
    material_content = ""
    if learning_materials:
        material_content = learning_materials[0].get("content", "")
    
    # Generate quiz using LLM
    llm = get_llm(temperature=0.7)
    chain = QUIZ_GENERATION_PROMPT | llm | StrOutputParser()
    
    try:
        quiz_json = chain.invoke({
            "domain": domain,
            "topic": topic,
            "material_content": material_content,
            "project_summary": project_summary
        })
        
        # Parse JSON response with robust cleaning
        cleaned_json = quiz_json.strip()
        
        # Remove markdown code blocks if present
        if "```" in cleaned_json:
            # Find content between code blocks
            match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', cleaned_json)
            if match:
                cleaned_json = match.group(1).strip()
            else:
                # Just remove all backticks
                cleaned_json = cleaned_json.replace('```json', '').replace('```', '').strip()
        
        # Parse the JSON
        questions = json.loads(cleaned_json)
        
        # Validate the structure
        if not isinstance(questions, list) or len(questions) == 0:
            raise ValueError("Invalid quiz format")
        
        # Format quiz for display
        quiz_text = "\n\n📝 **Quick Knowledge Check**\n\n"
        
        for i, q in enumerate(questions, 1):
            quiz_text += f"**Question {i}:** {q['question']}\n"
            for option in q['options']:
                quiz_text += f"{option}\n"
            quiz_text += "\n"
        
        quiz_text += "Please answer with the letters (e.g., 'B, C' or 'B and C')"
        
        return {
            "quiz_questions": questions,
            "messages": [AIMessage(content=quiz_text)],
            "next_action": "evaluate_quiz"
        }
    
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Raw response: {quiz_json[:200]}")
        error_msg = "I had trouble generating the quiz format. Let's continue with other topics instead!"
        return {
            "messages": [AIMessage(content=error_msg)],
            "next_action": "ask_continue"
        }
    except Exception as e:
        print(f"Error generating quiz: {e}")
        error_msg = "I had trouble generating the quiz. Let's continue with other topics instead!"
        return {
            "messages": [AIMessage(content=error_msg)],
            "next_action": "ask_continue"
        }


def evaluate_quiz(state: OnboardingState) -> OnboardingState:
    """Node 6: Evaluate quiz answers and provide feedback"""
    print("\n--- EVALUATE QUIZ ---")
    
    user_answers = state.get("user_input", "").upper()
    questions = state.get("quiz_questions", [])
    
    # Parse user answers (simple extraction)
    answer_letters = re.findall(r'[A-D]', user_answers)
    
    # Evaluate
    feedback_text = "\n\n✨ **Quiz Results**\n\n"
    correct_count = 0
    
    for i, q in enumerate(questions):
        user_answer = answer_letters[i] if i < len(answer_letters) else "?"
        is_correct = user_answer == q['correct']
        
        if is_correct:
            correct_count += 1
            feedback_text += f"✅ **Question {i+1}: Correct!**\n"
        else:
            feedback_text += f"❌ **Question {i+1}: Incorrect**\n"
            feedback_text += f"   Correct answer: {q['correct']}\n"
        
        feedback_text += f"   💡 {q['explanation']}\n\n"
    
    score = (correct_count / len(questions)) * 100
    feedback_text += f"\n🎯 **Final Score: {correct_count}/{len(questions)} ({score:.0f}%)**\n\n"
    
    if score >= 80:
        feedback_text += "Excellent work! You've got a solid understanding! 🎉"
    elif score >= 60:
        feedback_text += "Good effort! Review the explanations above and you'll master this! 💪"
    else:
        feedback_text += "Keep learning! The explanations above will help clarify these concepts. 📚"
    
    return {
        "feedback": feedback_text,
        "messages": [AIMessage(content=feedback_text)],
        "next_action": "ask_continue"
    }


def ask_continue(state: OnboardingState) -> OnboardingState:
    """Node 7: Ask if user wants to continue learning"""
    print("\n--- ASK CONTINUE ---")
    
    continue_text = "\n\nWould you like to:\n"
    continue_text += "- Learn about another topic? (just tell me what!)\n"
    continue_text += "- Connect with team members?\n"
    continue_text += "- End the onboarding session?\n"
    
    return {
        "messages": [AIMessage(content=continue_text)],
        "next_action": "route_continue"
    }


# ============================================================================
# ROUTING FUNCTIONS
# ============================================================================

def route_from_collect(state: OnboardingState) -> Literal["collect_info", "present_summary"]:
    """Route after collecting info"""
    next_action = state.get("next_action", "collect_info")
    if next_action == "present_summary":
        return "present_summary"
    return "collect_info"


def route_from_summary(state: OnboardingState) -> Literal["search_materials"]:
    """Always go to search after summary"""
    return "search_materials"


def route_from_search(state: OnboardingState) -> Literal["ask_quiz", "ask_continue"]:
    """Route based on whether material was found"""
    next_action = state.get("next_action", "ask_continue")
    if next_action == "ask_quiz":
        return "ask_quiz"
    return "ask_continue"


def route_from_quiz_ask(state: OnboardingState) -> Literal["present_quiz", "ask_continue"]:
    """Check if user wants quiz"""
    user_input = state.get("user_input", "").lower()
    
    # Simple yes/no detection
    if any(word in user_input for word in ["yes", "sure", "ok", "yeah", "yep"]):
        return "present_quiz"
    return "ask_continue"


def route_from_continue(state: OnboardingState) -> Literal["search_materials", "end"]:
    """Check if user wants to continue or end"""
    user_input = state.get("user_input", "").lower()
    
    # Check for end signals
    if any(word in user_input for word in ["end", "finish", "done", "exit", "quit", "bye"]):
        return "end"
    
    # Otherwise continue learning
    return "search_materials"


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def create_onboarding_graph():
    """Build the complete LangGraph workflow"""
    
    workflow = StateGraph(OnboardingState)
    
    # Add all nodes
    workflow.add_node("collect_info", collect_domain_role)
    workflow.add_node("present_summary", present_summary)
    workflow.add_node("search_materials", search_materials)
    workflow.add_node("ask_quiz", ask_quiz)
    workflow.add_node("present_quiz", present_quiz)
    workflow.add_node("evaluate_quiz", evaluate_quiz)
    workflow.add_node("ask_continue", ask_continue)
    
    # Set entry point
    workflow.set_entry_point("collect_info")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "collect_info",
        route_from_collect,
        {
            "collect_info": "collect_info",
            "present_summary": "present_summary"
        }
    )
    
    workflow.add_conditional_edges(
        "present_summary",
        route_from_summary,
        {"search_materials": "search_materials"}
    )
    
    workflow.add_conditional_edges(
        "search_materials",
        route_from_search,
        {
            "ask_quiz": "ask_quiz",
            "ask_continue": "ask_continue"
        }
    )
    
    workflow.add_conditional_edges(
        "ask_quiz",
        route_from_quiz_ask,
        {
            "present_quiz": "present_quiz",
            "ask_continue": "ask_continue"
        }
    )
    
    workflow.add_edge("present_quiz", "evaluate_quiz")
    workflow.add_edge("evaluate_quiz", "ask_continue")
    
    workflow.add_conditional_edges(
        "ask_continue",
        route_from_continue,
        {
            "search_materials": "search_materials",
            "end": END
        }
    )
    
    # Compile with configuration
    return workflow.compile(
        checkpointer=None,
        interrupt_before=None,
        interrupt_after=None,
        debug=False
    )


# ============================================================================
# MAIN EXECUTION - SIMPLE CHAT LOOP
# ============================================================================

def run_onboarding_chat():
    """Run the onboarding agent as an interactive chat"""
    
    print("=" * 60)
    print("🚀 WELCOME TO PROJECT ONBOARDING AGENT")
    print("=" * 60)
    print("\nI'll help you get started with our project!")
    print("Type 'exit' anytime to quit.\n")
    
    # Create graph
    app = create_onboarding_graph()
    
    # Initialize state
    state = {
        "messages": [],
        "domain": None,
        "role": None,
        "learning_topic": None,
        "project_summary": None,
        "learning_materials": [],
        "quiz_ready": False,
        "quiz_questions": [],
        "quiz_answers": [],
        "feedback": None,
        "team_contact": None,
        "next_action": "collect_info",
        "user_input": ""
    }
    
    # Start with initial greeting message
    print("🤖 Agent: Hello! I'm here to help you get onboarded to our project. Let me know what domain you'll be working in (backend, frontend, or data) and your role (e.g., junior developer, senior engineer, lead).\n")
    
    # Chat loop
    while True:
        # Get user input
        user_input = input("👤 You: ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("\n👋 Thanks for onboarding! Good luck with the project!\n")
            break
        
        if not user_input:
            continue
        
        # Update state with user input
        state["user_input"] = user_input
        state["messages"].append(HumanMessage(content=user_input))
        
        # Run graph
        try:
            result = app.invoke(state, config={"recursion_limit": 50})
            
            # Update state for next iteration
            state = result
            
            # Print agent response
            if result["messages"]:
                last_message = result["messages"][-1]
                if isinstance(last_message, AIMessage):
                    print(f"\n🤖 Agent: {last_message.content}\n")
        
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            print("Let's try again!\n")


if __name__ == "__main__":

    run_onboarding_chat()