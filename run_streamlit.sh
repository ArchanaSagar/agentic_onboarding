#!/bin/bash

# ERPNext Onboarding - Streamlit UI Launcher
# This script starts the Streamlit web interface

echo "🚀 Starting ERPNext Onboarding Streamlit UI..."
echo ""
echo "The app will open in your browser at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

poetry run streamlit run streamlit_app.py
