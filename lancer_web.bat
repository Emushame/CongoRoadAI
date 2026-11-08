@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -X utf8 -m streamlit run streamlit_app.py
if errorlevel 1 pause
