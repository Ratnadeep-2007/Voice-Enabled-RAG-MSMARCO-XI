@echo off
setlocal enabledelayedexpansion
title VoiceRAG - Low-Latency Voice-Enabled Multilingual Dense RAG

cd /d "%~dp0"

REM 1. Initialize .env if missing
if not exist ".env" (
    if exist ".env.example" (
        echo [*] Initializing .env configuration from .env.example...
        copy /y ".env.example" ".env" >nul
    )
)

REM 2. Identify best Python binary
set "PY_BIN="
if exist ".venv\Scripts\python.exe" (
    set "PY_BIN=.venv\Scripts\python.exe"
    echo [*] Found virtual environment at .venv
    goto :found_python
)
if exist "venv\Scripts\python.exe" (
    set "PY_BIN=venv\Scripts\python.exe"
    echo [*] Found virtual environment at venv
    goto :found_python
)

where python >nul 2>&1
if %errorlevel% equ 0 (
    set "PY_BIN=python"
    goto :found_python
)

where py >nul 2>&1
if %errorlevel% equ 0 (
    set "PY_BIN=py -3"
    goto :found_python
)

echo [ERROR] Python 3.10+ was not found in your PATH or virtual environment.
echo Please install Python from https://python.org and check 'Add Python to PATH'.
echo.
pause
exit /b 1

:found_python

REM 3. Verify core dependencies
%PY_BIN% -c "import fastapi, uvicorn, httpx, faiss, lmdb" >nul 2>&1
if %errorlevel% neq 0 (
    echo [*] Installing dependencies from requirements.txt...
    %PY_BIN% -m pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install dependencies automatically.
        echo Please run: %PY_BIN% -m pip install -r requirements.txt
        pause
        exit /b 1
    )
)

REM 4. Launch with smart runner (handles ports, browser opening, and active instance detection)
%PY_BIN% run_server.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] VoiceRAG server exited with code %errorlevel%.
    pause
)
