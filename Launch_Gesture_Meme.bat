@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo First run — setting up ^(this takes a minute^)...
    python -m venv .venv
)

".venv\Scripts\python.exe" -m pip install --quiet --upgrade pip
".venv\Scripts\python.exe" -m pip install --quiet --upgrade -r requirements.txt
".venv\Scripts\python.exe" gesture_meme.py