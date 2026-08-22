@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo First run — setting up ^(this takes a minute^)...
    python -m venv .venv
    ".venv\Scripts\pip.exe" install --quiet --upgrade pip
    ".venv\Scripts\pip.exe" install --quiet -r requirements.txt
)

".venv\Scripts\python.exe" gesture_meme.py