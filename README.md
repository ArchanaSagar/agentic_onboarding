# Agentic Onboarding

AI-powered project onboarding agent using LangGraph and LangChain.

## Setup

This project uses Poetry for dependency management.

### Install Dependencies

```bash
poetry install
```

### Configure Environment

The `.env` file is included in the repository with Azure OpenAI credentials.

Make sure it contains:

- `OPENAI_AZURE_ENDPOINT`
- `OPENAI_API_KEY`
- `OPENAI_API_VERSION`
- `OPENAI_MODEL_NAME`

### Run the Agent

```bash
poetry run python chat_agent.py
```

## Project Structure

- `chat_agent.py` - Main onboarding agent with LangGraph workflow
- `.env` - Environment configuration (committed to repo)
- `pyproject.toml` - Poetry dependencies and project metadata
- `poetry.lock` - Locked dependency versions for reproducible installs

## Dependencies

- **langgraph**: Workflow orchestration
- **langchain**: LLM framework
- **langchain-openai**: Azure OpenAI integration
- **python-dotenv**: Environment variable management
