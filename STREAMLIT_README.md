# Streamlit UI for ERPNext Onboarding Agent

## Overview

This project provides an interactive Streamlit web interface for the ERPNext Team Onboarding Agent. The UI makes it easy to:

- Collect user information (role and domain)
- Search and explore learning materials
- Take interactive quizzes
- Connect with team members who are experts in specific topics

## Features

### 🎯 Main Stages

1. **User Profile** - Enter your name, role, and domain of interest
2. **Learning** - Search for topics and get personalized learning materials
3. **Quiz** - Test your knowledge with AI-generated quizzes
4. **Team Connection** - Find GitHub contributors who are experts in specific areas
5. **Completion** - View your progress and achievements

### 💡 Key Capabilities

- **AI-Powered Search**: Uses LangChain and FAISS vector store to find relevant documentation
- **Personalized Learning**: Content tailored to your specific role and domain
- **Interactive Quizzes**: Dynamically generated questions with instant feedback
- **Team Discovery**: Connects you with GitHub contributors based on topic expertise
- **Progress Tracking**: Monitors topics learned and quiz scores

## Installation

### Prerequisites

- Python 3.11 or higher
- Poetry (for dependency management)
- Azure OpenAI API credentials

### Setup Steps

1. **Clone the repository** (if not already done)

2. **Install dependencies**:

   ```bash
   poetry install
   poetry run pip install streamlit
   ```

3. **Set up environment variables**:
   Create a `.env` file in the project root with:
   ```
   OPENAI_AZURE_ENDPOINT=your_azure_endpoint
   OPENAI_API_KEY=your_api_key
   OPENAI_API_VERSION=your_api_version
   OPENAI_MODEL_NAME=gpt-4
   GITHUB_TOKEN=your_github_token  # Optional: for team connection features
   ```

## Running the Application

### Start the Streamlit App

```bash
poetry run streamlit run streamlit_app.py
```

The app will open in your default browser at `http://localhost:8501`

### Using the CLI Version

If you prefer the command-line interface:

```bash
poetry run python chat_agent.py
```

## Usage Guide

### 1. Getting Started

When you first launch the app:

- Enter your name
- Specify your role (e.g., Developer, Business Analyst, QA Engineer)
- Choose your domain of interest (e.g., Accounting, Sales, HR, Manufacturing)

### 2. Learning Materials

- Enter any topic or question in the search box
- The AI will find relevant documentation and create a personalized learning guide
- Topics are tracked in your progress sidebar

### 3. Taking Quizzes

- Click "Take a Quiz" to test your knowledge
- Answer multiple-choice questions
- Get instant feedback with explanations
- View your cumulative quiz scores

### 4. Connecting with the Team

- Enter a topic you need help with
- The system searches GitHub for contributors who have worked on that topic
- View contributor profiles, commit history, and contact information

### 5. Tracking Progress

The sidebar shows:

- Your profile information
- Number of topics explored
- Quizzes taken and average score
- List of all topics you've learned

## Project Structure

```
.
├── streamlit_app.py          # Streamlit web interface
├── chat_agent.py             # Core onboarding logic and CLI interface
├── pyproject.toml            # Poetry dependencies
├── .env                      # Environment variables (not in repo)
└── erpnext_vectorstore/      # Cached vector store (generated on first run)
```

## Technical Details

### Architecture

- **Frontend**: Streamlit for interactive UI
- **Backend**: LangChain for LLM orchestration
- **Vector Store**: FAISS for document search
- **LLM**: Azure OpenAI (GPT-4)
- **Embeddings**: Azure OpenAI (text-embedding-3-large)

### Key Components

- **Session State Management**: Maintains user progress across interactions
- **Vector Store Initialization**: Loads or builds ERPNext documentation index
- **Dynamic Content Generation**: Creates personalized learning materials and quizzes
- **GitHub Integration**: Searches commit history for topic experts

## Troubleshooting

### Vector Store Issues

If the vector store fails to load:

- The app will build a new one from ERPNext documentation
- This may take several minutes on first run
- The vector store is cached in `erpnext_vectorstore/` for future use

### API Rate Limits

If you see GitHub API errors:

- Add a `GITHUB_TOKEN` to your `.env` file
- Get a token from: https://github.com/settings/tokens
- Requires 'public_repo' scope

### Module Import Errors

If you see import errors:

```bash
poetry install
poetry run pip install streamlit
```

## Development

### Adding New Features

The Streamlit app is modular:

- Add new stages by creating new functions like `new_stage()`
- Add stages to the navigation in `main()`
- Update `current_stage` session state to navigate

### Customizing the UI

- Modify `display_header()` to change the header
- Update `display_sidebar()` to add sidebar widgets
- Adjust colors and styling with Streamlit's theming

## Contributing

To contribute to this project:

1. Create a new branch
2. Make your changes
3. Test thoroughly with `poetry run streamlit run streamlit_app.py`
4. Submit a pull request

## License

This project follows the same license as the parent repository.

## Support

For issues or questions:

- Check the troubleshooting section above
- Review the code comments in `streamlit_app.py`
- Refer to the original `chat_agent.py` for CLI usage

---

**Happy Learning! 🎓**
