import os
import streamlit as st
from dotenv import load_dotenv

# Import functions and variables from chat_agent
from chat_agent import (
    initialize_vectorstores,
    extract_domains_and_roles,
    match_user_input,
    get_github_contributors,
    search_github_code,
    llm,
    qa_docs,
    qa_code
)

# ==============================
# Helper Functions - Following chat_agent.py logic
# ==============================
def extract_topic(user_query: str) -> str:
    """Extract main topic from user query using LLM."""
    try:
        prompt = f"""Extract the main topic/concept from this query: "{user_query}"

Return ONLY the topic name as 2-4 words, nothing else.
Examples:
- "How do I create sales invoices?" -> "Sales Invoice"
- "Tell me about purchase orders" -> "Purchase Order"
"""
        response = llm.invoke(prompt)
        topic = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        return topic.replace('"', '').replace("'", "")
    except:
        return user_query

def generate_summary(inputs: dict) -> str:
    """Generate onboarding summary - matches present_summary() from chat_agent.py"""
    domain_query = f"{inputs['domain']} module in ERPNext features capabilities overview"
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
        return response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        return f"⚠️ Could not generate detailed summary: {e}"

def generate_learning_guide(inputs: dict, keyword: str) -> str:
    """Generate learning materials - matches search_materials() from chat_agent.py"""
    try:
        # Semantic search
        query = f"{keyword} in {inputs['domain']} for {inputs['role']}"
        docs = qa_docs.invoke(query)
        doc_context = "\n\n".join([doc.page_content[:600] for doc in docs[:3]])
        
        # Code search
        code_docs = qa_code.invoke(keyword)
        code_context = "\n\n".join([doc.page_content[:500] for doc in code_docs[:2]])
        
        prompt = f"""Create a comprehensive learning guide for a {inputs['role']} in {inputs['domain']} about: {keyword}

Documentation:
{doc_context}

Code Examples:
{code_context}

Provide:
1. **Overview**: What is this and why is it important?
2. **Key Concepts**: Main concepts to understand
3. **How to Use It**: Practical steps and examples
4. **Common Patterns**: Typical use cases in {inputs['domain']}
5. **Tips for {inputs['role']}**: Role-specific advice
6. **Related Topics**: What to explore next

Use markdown formatting and be practical."""
        
        response = llm.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        return f"Could not generate learning guide: {str(e)}"

def generate_quiz(inputs: dict, topic: str) -> dict:
    """Generate quiz - matches present_quiz() from chat_agent.py"""
    try:
        prompt = f"""Generate a 5-question multiple choice quiz about {topic} in {inputs['domain']} for a {inputs['role']}.

Return in this exact JSON format:
{{
    "topic": "{topic}",
    "questions": [
        {{
            "question": "Question text here?",
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "correct": "Option A"
        }}
    ]
}}

Make questions practical and relevant to their role."""
        
        response = llm.invoke(prompt)
        text = response.content if hasattr(response, 'content') else str(response)
        
        import json
        text = text.strip()
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0]
        elif '```' in text:
            text = text.split('```')[1].split('```')[0]
        
        return json.loads(text)
    except:
        return {
            "topic": topic,
            "questions": [
                {
                    "question": f"What is the primary purpose of {topic} in ERPNext?",
                    "options": ["Data Management", "Process Automation", "Reporting", "All of the above"],
                    "correct": "All of the above"
                },
                {
                    "question": f"Which module in ERPNext would you primarily use {topic} with?",
                    "options": [inputs['domain'], "System", "Core", "All modules"],
                    "correct": inputs['domain']
                },
                {
                    "question": f"As a {inputs['role']}, what is your main responsibility regarding {topic}?",
                    "options": ["Configuration", "Testing", "Documentation", "All of the above"],
                    "correct": "All of the above"
                },
                {
                    "question": f"How does {topic} integrate with other ERPNext features?",
                    "options": ["Direct integration", "Through APIs", "Manual sync", "All methods possible"],
                    "correct": "All methods possible"
                },
                {
                    "question": f"What is a best practice when working with {topic}?",
                    "options": ["Regular backups", "User training", "Documentation", "All of the above"],
                    "correct": "All of the above"
                }
            ]
        }

def check_answer(user_answer: str, correct_answer: str) -> bool:
    """Check if answer is correct."""
    return user_answer.strip().lower() == correct_answer.strip().lower()

def generate_feedback(is_correct: bool, question: str, user_answer: str, correct_answer: str) -> str:
    """Generate quiz feedback."""
    if is_correct:
        return f"✅ **Correct!** {user_answer} is the right answer."
    else:
        return f"❌ **Not quite.** The correct answer is: **{correct_answer}**"

def generate_farewell(inputs: dict) -> str:
    """Generate farewell message."""
    try:
        prompt = f"""Generate a warm farewell message for a {inputs['role']} in {inputs['domain']} domain who just completed ERPNext onboarding.

Keep it brief (2-3 lines), encouraging, and personalized. Include relevant emojis."""
        
        response = llm.invoke(prompt)
        return response.content.strip() if hasattr(response, 'content') else "👋 Thank you for completing your onboarding! Keep learning!"
    except:
        return "👋 Thank you for completing your onboarding! Keep learning!"


# ==============================
# Page Configuration
# ==============================
st.set_page_config(
    page_title="ERPNext Team Onboarding",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for purple theme
st.markdown("""
    <style>
    /* Purple theme for buttons */
    .stButton > button {
        background-color: #8B5CF6 !important;
        color: white !important;
        border: none !important;
    }
    
    .stButton > button:hover {
        background-color: #7C3AED !important;
        border: none !important;
    }
    
    .stButton > button:active {
        background-color: #6D28D9 !important;
    }
    
    /* Purple theme for primary buttons */
    .stButton > button[kind="primary"] {
        background-color: #8B5CF6 !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        background-color: #7C3AED !important;
    }
    
    /* Purple theme for selectbox (dropdown) */
    .stSelectbox > div > div > div {
        border-color: #8B5CF6 !important;
    }
    
    .stSelectbox > div > div > div:focus-within {
        border-color: #7C3AED !important;
        box-shadow: 0 0 0 0.2rem rgba(139, 92, 246, 0.25) !important;
    }
    
    /* Purple highlight for dropdown options */
    div[data-baseweb="select"] > div {
        border-color: #8B5CF6 !important;
    }
    
    /* Purple for selected option */
    div[role="option"][aria-selected="true"] {
        background-color: #8B5CF6 !important;
    }
    
    /* Purple for hover on dropdown options */
    div[role="option"]:hover {
        background-color: rgba(139, 92, 246, 0.1) !important;
    }
    
    /* Purple for radio buttons */
    .stRadio > div > div > div > label:hover {
        background-color: rgba(139, 92, 246, 0.1) !important;
    }
    
    /* Purple for selected radio button */
    .stRadio > div > div > div > label > div:first-child > div {
        background-color: #8B5CF6 !important;
        border-color: #8B5CF6 !important;
    }
    
    /* Purple for text input focus */
    .stTextInput > div > div > input:focus {
        border-color: #8B5CF6 !important;
        box-shadow: 0 0 0 0.2rem rgba(139, 92, 246, 0.25) !important;
    }
    
    /* Purple for text area focus */
    .stTextArea > div > div > textarea:focus {
        border-color: #8B5CF6 !important;
        box-shadow: 0 0 0 0.2rem rgba(139, 92, 246, 0.25) !important;
    }
    
    /* Purple accent for expanders */
    .streamlit-expanderHeader:hover {
        color: #8B5CF6 !important;
    }
    
    /* Purple for form submit button */
    .stFormSubmitButton > button {
        background-color: #8B5CF6 !important;
        color: white !important;
    }
    
    .stFormSubmitButton > button:hover {
        background-color: #7C3AED !important;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================
# Session State Initialization - Following chat_agent.py flow
# ==============================
def init_session_state():
    """Initialize session state - strict sequential flow."""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.vectorstore_ready = False
        
    if 'user_info' not in st.session_state:
        st.session_state.user_info = {
            'role': '',
            'domain': '',
            'summary': '',
            'topics_learned': [],
            'quiz_scores': []
        }
    
    # Step tracking - must complete in order
    if 'step' not in st.session_state:
        st.session_state.step = 1  # 1=collect_info, 2=summary, 3=learning_loop
    
    if 'summary_shown' not in st.session_state:
        st.session_state.summary_shown = False
    
    if 'current_topic' not in st.session_state:
        st.session_state.current_topic = None
    
    if 'learning_materials' not in st.session_state:
        st.session_state.learning_materials = None
    
    if 'github_code' not in st.session_state:
        st.session_state.github_code = None
    
    if 'contributors' not in st.session_state:
        st.session_state.contributors = None
    
    if 'quiz_data' not in st.session_state:
        st.session_state.quiz_data = None
    
    if 'show_quiz' not in st.session_state:
        st.session_state.show_quiz = False
    
    if 'quiz_results' not in st.session_state:
        st.session_state.quiz_results = None
    
    if 'domains_roles' not in st.session_state:
        st.session_state.domains_roles = None

# ==============================
# UI Components
# ==============================
def display_header():
    """Display simple header without navigation."""
    st.markdown("""
        <div style='text-align: center; padding: 2rem 0 1rem 0;'>
            <h1>🎓 ERPNext Team Onboarding</h1>
            <p style='font-size: 1.1rem; color: #666;'>Your Interactive Learning Companion</p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

def display_sidebar():
    """Display sidebar with profile and progress."""
    with st.sidebar:
        st.markdown("### 👤 Your Profile")
        
        if st.session_state.user_info['role']:
            st.markdown(f"**Role:** {st.session_state.user_info['role']}")
            st.markdown(f"**Domain:** {st.session_state.user_info['domain']}")
            
            st.markdown("---")
            st.markdown("### 📊 Your Progress")
            st.markdown(f"**Topics Explored:** {len(st.session_state.user_info['topics_learned'])}")
            
            if st.session_state.user_info['quiz_scores']:
                avg_score = sum(st.session_state.user_info['quiz_scores']) / len(st.session_state.user_info['quiz_scores'])
                st.markdown(f"**Quizzes Taken:** {len(st.session_state.user_info['quiz_scores'])}")
                st.markdown(f"**Average Score:** {avg_score:.0f}%")
            
            if st.session_state.user_info['topics_learned']:
                st.markdown("---")
                st.markdown("### 📖 Topics Learned")
                for topic in st.session_state.user_info['topics_learned']:
                    st.markdown(f"- {topic}")
        else:
            st.info("👋 Complete your profile to get started!")
        
        st.markdown("---")
        
        # Reset button
        if st.button("🔄 Start Over", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key not in ['vectorstore_ready']:
                    del st.session_state[key]
            init_session_state()
            st.rerun()

# ==============================
# Step 1: Collect Info - Using dropdowns
# ==============================
def step_collect_info():
    """Collect user's role and domain using dropdowns - matches collect_info() from chat_agent.py"""
    st.markdown("## � Let's Get You Set Up")
    
    # Load domains and roles if not already loaded
    if st.session_state.domains_roles is None:
        with st.spinner("🔍 Loading available domains and roles..."):
            domains, roles = extract_domains_and_roles()
            st.session_state.domains_roles = {'domains': domains, 'roles': roles}
    
    domains = st.session_state.domains_roles['domains']
    roles = st.session_state.domains_roles['roles']
    
    st.markdown("### 📦 Select Your Domain")
    domain = st.selectbox(
        "Choose the ERPNext module/domain you'll be working with:",
        options=domains,
        index=None,
        placeholder="Select a domain..."
    )
    
    st.markdown("### 👤 Select Your Role")
    role = st.selectbox(
        "Choose your role in the project:",
        options=roles,
        index=None,
        placeholder="Select a role..."
    )
    
    if domain and role:
        if st.button("Continue ➡️", use_container_width=True, type="primary"):
            st.session_state.user_info['domain'] = domain
            st.session_state.user_info['role'] = role
            st.session_state.step = 2
            st.success(f"✅ Great! You're a {role} in the {domain} domain.")
            st.rerun()
    else:
        st.info("👆 Please select both your domain and role to continue.")

# ==============================
# Step 2: Present Summary
# ==============================
def step_present_summary():
    """Generate and display onboarding summary - matches present_summary() from chat_agent.py"""
    st.markdown("## � Your Onboarding Summary")
    
    if not st.session_state.summary_shown:
        with st.spinner(f"� Creating personalized onboarding for {st.session_state.user_info['role']} in {st.session_state.user_info['domain']}..."):
            summary = generate_summary(st.session_state.user_info)
            st.session_state.user_info['summary'] = summary
            st.session_state.summary_shown = True
    
    # Display the summary
    st.markdown(st.session_state.user_info['summary'])
    
    st.markdown("---")
    if st.button("Continue to Learning ➡️", use_container_width=True, type="primary"):
        st.session_state.step = 3
        st.rerun()

# ==============================
# Step 3: Learning Loop
# ==============================
def step_learning_loop():
    """Main learning loop - matches the while loop in agentic_onboarding()"""
    st.markdown("## 📚 Learning Materials")
    
    # Topic search
    st.markdown("### 🔍 What would you like to learn about?")
    st.markdown(f"*Ask anything related to {st.session_state.user_info['domain']} in ERPNext*")
    
    keyword = st.text_input(
        "Enter a topic, question, or keyword:",
        placeholder="e.g., How do I create a sales invoice?",
        key="topic_search"
    )
    
    col1, col2 = st.columns([1, 1])
    with col1:
        search_clicked = st.button("🔍 Search Learning Materials", use_container_width=True, type="primary")
    with col2:
        quiz_clicked = st.button("📝 Take a Quiz", use_container_width=True)
    
    # Handle quiz button click - set flag to show quiz
    if quiz_clicked:
        st.session_state.show_quiz = True
        # Clear any existing quiz data to force regeneration
        st.session_state.quiz_data = None
        st.session_state.quiz_results = None
    
    # Handle search
    if search_clicked and keyword:
        with st.spinner("📖 Searching for learning materials..."):
            topic = extract_topic(keyword)
            materials = generate_learning_guide(st.session_state.user_info, topic)
            
            st.session_state.learning_materials = materials
            st.session_state.current_topic = topic
            
            # Track topic
            if topic not in st.session_state.user_info['topics_learned']:
                st.session_state.user_info['topics_learned'].append(topic)
        
        # Search GitHub for code references
        with st.spinner("🔍 Finding relevant code files..."):
            code_results = search_github_code(topic)
            st.session_state.github_code = code_results
        
        # Search for contributors
        with st.spinner("👥 Finding topic experts..."):
            contributors = get_github_contributors(topic)
            st.session_state.contributors = contributors
    
    # Display learning materials if available
    if st.session_state.learning_materials:
        st.markdown("---")
        st.markdown(f"### 📖 Learning Guide: {st.session_state.current_topic}")
        st.markdown(st.session_state.learning_materials)
        
        # Display GitHub code references
        if hasattr(st.session_state, 'github_code') and st.session_state.github_code:
            st.markdown("---")
            st.markdown("### 📂 Related Code Files")
            
            code_results = st.session_state.github_code
            if 'error' in code_results:
                st.info(f"ℹ️ {code_results['error']}")
            elif code_results.get('files'):
                st.success(f"✅ Found {len(code_results['files'])} related files")
                
                for file in code_results['files'][:5]:  # Show top 5
                    with st.expander(f"📄 {file['name']}"):
                        st.markdown(f"**Path:** `{file['path']}`")
                        st.markdown(f"**Repository:** {file['repository']}")
                        st.markdown(f"[View on GitHub]({file['url']})")
        
        # Display contributors/experts
        if hasattr(st.session_state, 'contributors') and st.session_state.contributors:
            st.markdown("---")
            st.markdown("### 👥 Topic Experts")
            
            team = st.session_state.contributors
            if 'error' in team:
                st.info(f"ℹ️ {team['error']}")
            elif team.get('contributors'):
                st.success(f"✅ Found {len(team['contributors'])} contributors for '{team['keyword']}'")
                
                # Show top 3 contributors inline
                cols = st.columns(min(3, len(team['contributors'])))
                for idx, contributor in enumerate(team['contributors'][:3]):
                    with cols[idx]:
                        st.markdown(f"**{contributor['name']}**")
                        st.markdown(f"📧 {contributor['email']}")
                        if contributor['github_username']:
                            st.markdown(f"[@{contributor['github_username']}]({contributor['github_url']})")
                        st.markdown(f"💻 {contributor['commits']} commits")
                
                # Show more contributors in expander
                if len(team['contributors']) > 3:
                    with st.expander(f"View all {len(team['contributors'])} contributors"):
                        for contributor in team['contributors'][3:]:
                            st.markdown(f"**{contributor['name']}** - {contributor['email']} - {contributor['commits']} commits")
                            if contributor['github_url']:
                                st.markdown(f"[GitHub Profile]({contributor['github_url']})")
                            st.markdown("---")
        
        st.markdown("---")
        st.markdown("#### What would you like to do next?")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔍 Search Another Topic", use_container_width=True):
                st.session_state.learning_materials = None
                st.session_state.current_topic = None
                st.rerun()
        
        with col2:
            if st.button("📝 Take Quiz on This Topic", use_container_width=True):
                st.session_state.show_quiz = True
                st.rerun()
        
        with col3:
            if st.button("👥 Find Team Experts", use_container_width=True):
                st.session_state.step = 4
                st.rerun()
    
    # Handle quiz
    if quiz_clicked or st.session_state.show_quiz:
        show_quiz_section()
    
    # Additional options
    st.markdown("---")
    st.markdown("### 🎯 Other Actions")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("👥 Connect with Team", use_container_width=True):
            st.session_state.step = 4
            st.rerun()
    
    with col2:
        if st.button("✅ Complete Onboarding", use_container_width=True):
            st.session_state.step = 5
            st.rerun()

# ==============================
# Quiz Section (within learning loop)
# ==============================
def show_quiz_section():
    """Display quiz section - matches ask_quiz() and present_quiz() from chat_agent.py"""
    st.markdown("---")
    st.markdown("## 📝 Knowledge Check Quiz")
    
    # Generate quiz if needed
    if st.session_state.quiz_data is None:
        # Determine topic - use current topic or fall back to domain
        if st.session_state.current_topic:
            topic = st.session_state.current_topic
        else:
            topic = st.session_state.user_info['domain']
        
        # Ensure we have a valid topic
        if not topic:
            st.error("⚠️ Unable to generate quiz. Please search for a topic first.")
            return
        
        with st.spinner(f"Generating quiz questions about {topic}..."):
            quiz = generate_quiz(st.session_state.user_info, topic)
            st.session_state.quiz_data = quiz
    
    # Check if we have quiz results to display
    if 'quiz_results' in st.session_state and st.session_state.quiz_results is not None:
        # Display results (outside form)
        results = st.session_state.quiz_results
        
        st.markdown("## 🎯 Quiz Results")
        
        score_color = "#28a745" if results['score'] >= 70 else "#ffc107" if results['score'] >= 50 else "#dc3545"
        st.markdown(f"""
            <div style='text-align: center; padding: 2rem; background-color: {score_color}20; border-radius: 10px; border: 2px solid {score_color};'>
                <h1 style='color: {score_color}; margin: 0;'>{results['score']}%</h1>
                <p style='margin: 0.5rem 0 0 0;'>You got {results['correct']} out of {results['total']} questions correct!</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📊 Detailed Feedback")
        for i, fb in enumerate(results['feedback'], 1):
            icon = "✅" if fb['correct'] else "❌"
            with st.expander(f"Question {i} {icon}"):
                st.markdown(f"**{fb['question']}**")
                st.markdown(fb['feedback'])
        
        st.markdown("---")
        if st.button("Continue Learning ➡️", use_container_width=True, type="primary"):
            # Reset quiz state
            st.session_state.quiz_data = None
            st.session_state.show_quiz = False
            st.session_state.quiz_results = None
            st.rerun()
        return
    
    # Show quiz form
    quiz = st.session_state.quiz_data
    
    with st.form("quiz_form"):
        st.markdown(f"### {quiz['topic']}")
        st.markdown("*Select the best answer for each question*")
        st.markdown("---")
        
        answers = {}
        for i, q in enumerate(quiz['questions'], 1):
            st.markdown(f"**Question {i}:** {q['question']}")
            answers[i] = st.radio(
                f"Select your answer for question {i}:",
                q['options'],
                index=None,  # No preselection
                key=f"q{i}",
                label_visibility="collapsed"
            )
            st.markdown("---")
        
        submitted = st.form_submit_button("Submit Answers ✅", use_container_width=True, type="primary")
        
        if submitted:
            # Check if all questions are answered
            unanswered = [i for i, ans in answers.items() if ans is None]
            if unanswered:
                st.error(f"⚠️ Please answer all questions before submitting! Unanswered: Question(s) {', '.join(map(str, unanswered))}")
                st.stop()
            
            # Grade quiz
            correct = 0
            total = len(quiz['questions'])
            feedback_list = []
            
            for i, q in enumerate(quiz['questions'], 1):
                user_answer = answers[i]
                is_correct = check_answer(user_answer, q['correct'])
                
                feedback = generate_feedback(is_correct, q['question'], user_answer, q['correct'])
                
                if is_correct:
                    correct += 1
                
                feedback_list.append({
                    'question': q['question'],
                    'correct': is_correct,
                    'feedback': feedback
                })
            
            score = int((correct / total) * 100)
            st.session_state.user_info['quiz_scores'].append(score)
            
            # Store results to display outside form
            st.session_state.quiz_results = {
                'score': score,
                'correct': correct,
                'total': total,
                'feedback': feedback_list
            }
            
            st.rerun()

# ==============================
# Step 4: Team Connection
# ==============================
def step_team_connection():
    """Connect with team members - matches connect_with_team() from chat_agent.py"""
    st.markdown("## 👥 Connect with the Team")
    
    st.markdown("Find team members who are experts in specific topics!")
    st.markdown(f"*Search for contributors in the {st.session_state.user_info['domain']} domain*")
    
    keyword = st.text_input(
        "What topic do you need help with?",
        placeholder="e.g., Sales Invoice, Purchase Order, Manufacturing",
        key="team_search"
    )
    
    if st.button("🔍 Find Experts", use_container_width=True, type="primary"):
        if keyword:
            with st.spinner("Searching for team members..."):
                topic_key = extract_topic(keyword)
                team_info = get_github_contributors(topic_key)
                
                if 'error' in team_info:
                    st.warning(f"⚠️ {team_info['error']}")
                else:
                    st.success(f"✅ Found {len(team_info['contributors'])} contributors for '{team_info['keyword']}'")
                    
                    st.markdown("### 👥 Top Contributors")
                    
                    for contributor in team_info['contributors']:
                        with st.expander(f"👤 {contributor['name']} ({contributor['commits']} commits)"):
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.markdown(f"**Name:** {contributor['name']}")
                                st.markdown(f"**Email:** {contributor['email']}")
                                if contributor['github_username']:
                                    st.markdown(f"**GitHub:** @{contributor['github_username']}")
                                st.markdown(f"**Commits:** {contributor['commits']}")
                            
                            with col2:
                                if contributor['github_url']:
                                    st.markdown(f"[View Profile]({contributor['github_url']})")
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📚 Back to Learning", use_container_width=True):
            st.session_state.step = 3
            st.rerun()
    
    with col2:
        if st.button("✅ Complete Onboarding", use_container_width=True, type="primary"):
            st.session_state.step = 5
            st.rerun()

# ==============================
# Step 5: Complete
# ==============================
def step_complete():
    """Show completion screen - matches the farewell in agentic_onboarding()"""
    farewell = generate_farewell(st.session_state.user_info)
    
    st.markdown(f"""
        <div style='text-align: center; padding: 3rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; color: white;'>
            <h1>🎉 Congratulations!</h1>
            <p style='font-size: 1.3rem; margin: 2rem 0;'>{farewell}</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("## 📊 Your Onboarding Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Topics Explored", len(st.session_state.user_info['topics_learned']))
    
    with col2:
        st.metric("Quizzes Taken", len(st.session_state.user_info['quiz_scores']))
    
    with col3:
        if st.session_state.user_info['quiz_scores']:
            avg_score = sum(st.session_state.user_info['quiz_scores']) / len(st.session_state.user_info['quiz_scores'])
            st.metric("Average Score", f"{avg_score:.0f}%")
        else:
            st.metric("Average Score", "N/A")
    
    st.markdown("---")
    
    if st.button("🏠 Start New Onboarding", use_container_width=True, type="primary"):
        for key in list(st.session_state.keys()):
            if key not in ['vectorstore_ready']:
                del st.session_state[key]
        init_session_state()
        st.rerun()

# ==============================
# Main App - Following chat_agent.py flow exactly
# ==============================
def main():
    """Main application - sequential steps, no jumping."""
    init_session_state()
    
    # Initialize vectorstore on first run
    if not st.session_state.vectorstore_ready:
        with st.spinner("🔄 Initializing knowledge base... This may take a moment."):
            try:
                initialize_vectorstores()
                st.session_state.vectorstore_ready = True
            except Exception as e:
                st.error(f"⚠️ Error initializing vectorstore: {e}")
                return
    
    # Display header and sidebar
    display_header()
    display_sidebar()
    
    # Sequential step flow - matches agentic_onboarding() logic
    if st.session_state.step == 1:
        step_collect_info()
    elif st.session_state.step == 2:
        step_present_summary()
    elif st.session_state.step == 3:
        step_learning_loop()
    elif st.session_state.step == 4:
        step_team_connection()
    elif st.session_state.step == 5:
        step_complete()

if __name__ == "__main__":
    main()
