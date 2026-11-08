@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Installez les dependances selon README.md avant le lancement.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" app.py
if errorlevel 1 pause
