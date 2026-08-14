@echo off
setlocal
title VoiceRAG - Low-Latency Voice-Enabled Dense RAG System

cd /d "%~dp0"

echo =====================================================================
echo   VoiceRAG: Low-Latency Voice-Enabled Dense RAG System
echo   Hacker House Goa - Target Latency: ^< 200ms - Port: 8000
echo =====================================================================
echo.

:: 1. Virtual Environment Activation
if exist ".venv\Scripts\activate.bat" (
    echo [*] Activating virtual environment from .venv
    call .venv\Scripts\activate.bat
    goto :python_check
)

if exist "venv\Scripts\activate.bat" (
    echo [*] Activating virtual environment from venv
    call venv\Scripts\activate.bat
    goto :python_check
)

echo [*] Using system Python environment.

:python_check
:: 2. Check Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python was not found in your PATH.
    echo Please install Python 3.10+ and add it to your system PATH.
    echo.
    pause
    exit /b 1
)

echo [*] Python detected. Checking dependencies...

:: 3. Fast Dependency Check
python -c "import fastapi, uvicorn" >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Installing required dependencies from requirements.txt...
    python -m pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] Dependency installation failed.
        pause
        exit /b 1
    )
)

echo [*] Dependencies verified.
echo.
echo =====================================================================
echo   VoiceRAG Server Starting
echo   Local Dashboard: http://localhost:8000
echo   API Documentation: http://localhost:8000/docs
echo =====================================================================
echo.

:: 4. Open browser
start http://localhost:8000

:: 5. Launch FastAPI server via Uvicorn
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] VoiceRAG server stopped unexpectedly.
    pause
)
