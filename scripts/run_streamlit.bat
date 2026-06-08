@echo off
cd /d "%~dp0\.."
streamlit run app\streamlit_app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false

