@echo off
setlocal EnableDelayedExpansion
title VibeStudio
set OLLAMA_GPU_OVERHEAD=2147483648
set OLLAMA_MAX_LOADED_MODELS=1
set OLLAMA_NUM_PARALLEL=1
set OLLAMA_FLASH_ATTENTION=1
set OLLAMA_KV_CACHE_TYPE=q8_0
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=!SCRIPT_DIR:~0,-1!"
tasklist | findstr /I "ollama.exe" >nul
if errorlevel 1 (
    start "" /B ollama serve
    timeout /t 4 /nobreak >nul
)
"!SCRIPT_DIR!\venv\Scripts\python.exe" "!SCRIPT_DIR!\main.py"
