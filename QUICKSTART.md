# 🚀 Quick Start Guide - Streamlit UI

## Launch Commands

```bash
# Easy way - use the script
./run_streamlit.sh

# Manual way
poetry run streamlit run streamlit_app.py
```

## First Time Setup

1. **Check your .env file**:

   ```
   OPENAI_AZURE_ENDPOINT=your_endpoint
   OPENAI_API_KEY=your_key
   OPENAI_API_VERSION=your_version
   OPENAI_MODEL_NAME=gpt-4
   GITHUB_TOKEN=optional_token
   ```

2. **Install dependencies** (if needed):

   ```bash
   poetry install
   poetry run pip install streamlit
   ```

3. **Run the app**:

   ```bash
   ./run_streamlit.sh
   ```

4. **Open browser**: http://localhost:8501

## Usage Flow

1. **Enter Profile** → Name, Role, Domain
2. **Learn Topics** → Search and explore
3. **Take Quizzes** → Test knowledge
4. **Find Experts** → GitHub contributors
5. **Complete** → View progress

## Keyboard Shortcuts

- `Ctrl+C` in terminal - Stop the server
- `R` in browser - Rerun the app
- `C` in browser - Clear cache

## Common Issues

**Port already in use?**

```bash
# Kill existing Streamlit process
pkill -f streamlit
# Then rerun
./run_streamlit.sh
```

**Vector store errors?**

- Delete `erpnext_vectorstore/` folder
- Restart app to rebuild

**Import errors?**

```bash
poetry install
poetry run pip install streamlit
```

## Features at a Glance

- ✅ Interactive web interface
- ✅ Progress tracking
- ✅ AI-generated learning materials
- ✅ Dynamic quizzes
- ✅ Team member discovery
- ✅ Mobile-friendly design

## Pro Tips

💡 **Use the sidebar** - See your progress anytime
💡 **Take multiple quizzes** - Build your knowledge
💡 **Explore topics freely** - No linear path required
💡 **Connect early** - Find experts when stuck
💡 **Check the docs** - STREAMLIT_README.md has details

---

**Ready? Run: `./run_streamlit.sh` 🎓**
