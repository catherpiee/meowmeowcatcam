@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo First run - creating the Python environment...
    py -3 -m venv .venv
    if errorlevel 1 (
        echo.
        echo Could not create the virtual environment. Make sure Python 3 is installed.
        pause
        exit /b 1
    )

    echo Installing dependencies...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo Dependency installation failed.
        pause
        exit /b 1
    )
)

".venv\Scripts\python.exe" gesture_meme.py
if errorlevel 1 (
    echo.
    echo Gesture Meme exited with an error.
    pause
)
