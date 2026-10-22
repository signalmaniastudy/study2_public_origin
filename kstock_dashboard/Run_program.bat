@echo off
REM Korean stock short-term dashboard launcher (Windows)
REM First run: creates .venv and installs packages. Next runs: starts immediately.
setlocal
cd /d "%~dp0"

REM ---- find Python 3.10+
set "PY="
py -3 --version >nul 2>&1 && set "PY=py -3"
if not defined PY (
    python --version >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo [ERROR] Python not found. Install Python 3.10+ from https://www.python.org/downloads/
    echo         and check "Add python.exe to PATH" during installation.
    pause
    exit /b 1
)

REM ---- create virtual environment (first run only)
if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creating virtual environment...
    %PY% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

REM ---- install packages (first run, or after deleting .venv\installed.txt)
if not exist ".venv\installed.txt" (
    echo [2/3] Installing packages. This may take a few minutes...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Package installation failed. Check your internet connection.
        pause
        exit /b 1
    )
    echo installed> ".venv\installed.txt"
)

REM ---- run dashboard (browser opens at http://localhost:8501)
echo [3/3] Starting dashboard... Close this window or press Ctrl+C to stop.
".venv\Scripts\python.exe" -m streamlit run app.py --browser.gatherUsageStats false
if errorlevel 1 pause
endlocal
