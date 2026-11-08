@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  py -3.11 -m venv .venv
  if errorlevel 1 (
    echo Python 3.11 est requis. Installez-le puis relancez ce fichier.
    pause
    exit /b 1
  )
)
".venv\Scripts\python.exe" -X utf8 -m pip install --timeout 60 --upgrade pip
if errorlevel 1 (
  echo La mise a jour de pip a echoue. Verifiez la connexion puis relancez.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -X utf8 -m pip install --timeout 60 -r requirements.txt
if errorlevel 1 (
  echo Installation incomplete. Verifiez la connexion puis relancez installer.bat.
  pause
  exit /b 1
)
echo Installation terminee. Utilisez lancer.bat pour ouvrir CongoRoadAI.
pause
