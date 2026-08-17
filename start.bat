@echo off
setlocal
title VoiceRAG - Low-Latency Voice-Enabled Multilingual Dense RAG

cd /d "%~dp0"

echo =====================================================================
echo   VoiceRAG: Low-Latency Multilingual Voice RAG System (MSMARCO-XI)
echo   Reasoning Engine : Groq LPU (Llama-3.3-70B / 3.1-8B)
echo   Vector Storage   : FAISS IVF-PQ + LMDB Zero-Copy
echo   Target Latency   : Sub-200ms Retrieval Bound
echo =====================================================================
echo.

REM 1. Initialize .env if missing
if not exist ".env" if exist ".env.example" copy /y ".env.example" ".env" >nul

REM 2. Activate virtual environment if present
if exist ".venv\Scripts\activate.bat" (
    echo [*] Activating virtual environment .venv
    call ".venv\Scripts\activate.bat"
    goto :check_python
)
if exist "venv\Scripts\activate.bat" (
    echo [*] Activating virtual environment venv
    call "venv\Scripts\activate.bat"
    goto :check_python
)

echo [*] Using system Python environment

:check_python
REM 3. Check Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python was not found in your PATH.
    echo Please install Python 3.10+ and add it to your system PATH.
    echo.
    pause
    exit /b 1
)

echo [*] Python detected. Checking dependencies...

REM 4. Verify core dependencies
python -c "import fastapi, uvicorn, httpx, faiss, lmdb" >nul 2>&1
if %errorlevel% neq 0 (
    echo [*] Installing missing dependencies from requirements.txt...
    python -m pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install dependencies. Please run 'pip install -r requirements.txt' manually.
        pause
        exit /b 1
    )
)

echo [*] All core dependencies verified.
echo.
echo =====================================================================
echo   VoiceRAG Server Starting...
echo   Local Dashboard   : http://localhost:8000
echo   API Documentation : http://localhost:8000/docs
echo   Architecture View : http://localhost:8000/architecture
echo =====================================================================
echo.

REM 5. Open Web Dashboard in default browser in background
start "" http://localhost:8000

REM 6. Launch FastAPI Server with uvicorn
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] VoiceRAG server exited with code %errorlevel%.
    pause
)
