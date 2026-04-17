@echo off
setlocal EnableDelayedExpansion
title VibeStudio Setup

:: ============================================================
::  VIBRESTUDIO SETUP
::  Installs all dependencies, pulls models, launches the app.
::  Run this ONCE from the VibeStudio folder.
:: ============================================================

set "SETUP_DIR=%~dp0"
set "SETUP_DIR=!SETUP_DIR:~0,-1!"
set "VENV_DIR=!SETUP_DIR!\venv"

echo.
echo  =====================================================
echo   VibeStudio Setup
echo   Qwen3 14B + DeepSeek-R1 8B  //  RTX 4080 16GB
echo  =====================================================
echo.
echo  Folder : !SETUP_DIR!
echo.

:: ============================================================
:: STEP 1 - Check Python
:: ============================================================
echo [1/9] Checking Python...
where python >nul 2>&1
if errorlevel 1 (
    echo.
    echo  [ERROR] Python not found on PATH.
    echo  Download from https://python.org  ^(tick "Add Python to PATH"^)
    echo.
    pause
    exit /b 1
)
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo  [OK] Python !PYVER!

:: ============================================================
:: STEP 2 - Check / install Ollama
:: ============================================================
echo.
echo [2/9] Checking Ollama...
where ollama >nul 2>&1
if not errorlevel 1 goto :ollama_found
if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
    set "PATH=%LOCALAPPDATA%\Programs\Ollama;%PATH%"
    goto :ollama_found
)

echo  Ollama not found. Downloading installer...
curl -L --progress-bar "https://ollama.com/download/OllamaSetup.exe" -o "%TEMP%\OllamaSetup.exe"
if errorlevel 1 (
    echo  [ERROR] Download failed. Check your internet connection.
    pause
    exit /b 1
)
echo  Running installer — complete it then return here...
"%TEMP%\OllamaSetup.exe"
echo  Waiting for install to finalise...
timeout /t 15 /nobreak >nul
set "PATH=%LOCALAPPDATA%\Programs\Ollama;%PATH%"
where ollama >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Ollama still not found after install.
    echo  Close this window, open a new terminal, and re-run setup.bat.
    pause
    exit /b 1
)

:ollama_found
echo  [OK] Ollama found.

:: ============================================================
:: STEP 3 - Set VRAM environment variables
:: ============================================================
echo.
echo [3/9] Setting VRAM environment variables...

:: Permanent (survives reboots)
setx OLLAMA_GPU_OVERHEAD    2147483648 >nul
setx OLLAMA_MAX_LOADED_MODELS 1        >nul
setx OLLAMA_NUM_PARALLEL    1          >nul
setx OLLAMA_KEEP_ALIVE      10m        >nul
setx OLLAMA_FLASH_ATTENTION 1          >nul
setx OLLAMA_KV_CACHE_TYPE   q8_0       >nul

:: Current session
set OLLAMA_GPU_OVERHEAD=2147483648
set OLLAMA_MAX_LOADED_MODELS=1
set OLLAMA_NUM_PARALLEL=1
set OLLAMA_KEEP_ALIVE=10m
set OLLAMA_FLASH_ATTENTION=1
set OLLAMA_KV_CACHE_TYPE=q8_0

echo  [OK] VRAM vars set.

:: ============================================================
:: STEP 4 - Start Ollama and wait for it to be ready
:: ============================================================
echo.
echo [4/9] Starting Ollama service...
start "" /B ollama serve

:wait_ollama
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    timeout /t 2 /nobreak >nul
    goto wait_ollama
)
echo  [OK] Ollama ready.

:: ============================================================
:: STEP 5 - Pull models (graceful failure on each)
:: ============================================================
echo.
echo [5/9] Pulling models ^(this will take a while — ~15GB total^)...
echo.

echo  Pulling qwen3:14b ^(~9GB^)...
ollama pull qwen3:14b
if errorlevel 1 echo  [WARNING] qwen3:14b pull failed. App will use fallback if available.

echo.
echo  Pulling deepseek-r1:8b ^(~5GB^)...
ollama pull deepseek-r1:8b
if errorlevel 1 echo  [WARNING] deepseek-r1:8b pull failed. Planner stage will use fallback.

echo.
echo  Pulling nomic-embed-text ^(for RAG^)...
ollama pull nomic-embed-text
if errorlevel 1 echo  [WARNING] nomic-embed-text pull failed. RAG indexing will not work.

:: ============================================================
:: STEP 6+7 - Write Modelfiles via Python (no echo redirection)
:: ============================================================
echo.
echo [6/9] Writing Modelfiles...

python -c "
content = '''FROM qwen3:14b
PARAMETER num_ctx 16384
PARAMETER temperature 0.6
PARAMETER top_p 0.95
PARAMETER num_gpu 99
SYSTEM \"\"\"You are an expert software engineer. You write complete, production-quality code. You make minimal, focused changes to existing code. You explain what you changed and why. You do not introduce unnecessary dependencies.\"\"\"
'''
import os
path = os.path.join(os.environ['SETUP_DIR_PY'], 'Modelfile.qwen3-coder')
open(path, 'w').write(content)
print('  Modelfile.qwen3-coder written.')
"
if errorlevel 1 echo  [WARNING] Failed to write Modelfile.qwen3-coder

python -c "
content = '''FROM deepseek-r1:8b
PARAMETER num_ctx 8192
PARAMETER temperature 0.6
PARAMETER num_gpu 99
SYSTEM \"\"\"You are a software architecture and planning specialist. You analyse tasks, decompose them into clear numbered steps, and identify risks. You are precise, concise, and your plans are designed to be executed literally by another AI model.\"\"\"
'''
import os
path = os.path.join(os.environ['SETUP_DIR_PY'], 'Modelfile.deepseek-reasoner')
open(path, 'w').write(content)
print('  Modelfile.deepseek-reasoner written.')
"
if errorlevel 1 echo  [WARNING] Failed to write Modelfile.deepseek-reasoner

:: ============================================================
:: STEP 8 - Create tuned models
:: ============================================================
echo.
echo [7/9] Creating tuned models...
set "SETUP_DIR_PY=!SETUP_DIR!"

ollama create qwen3-coder -f "!SETUP_DIR!\Modelfile.qwen3-coder"
if errorlevel 1 (
    echo  [WARNING] qwen3-coder creation failed. App will fall back to qwen3:14b.
) else (
    echo  [OK] qwen3-coder ready.
)

ollama create deepseek-reasoner -f "!SETUP_DIR!\Modelfile.deepseek-reasoner"
if errorlevel 1 (
    echo  [WARNING] deepseek-reasoner creation failed. App will fall back to qwen3:14b.
) else (
    echo  [OK] deepseek-reasoner ready.
)

:: ============================================================
:: STEP 9 - Python venv + packages
:: ============================================================
echo.
echo [8/9] Creating Python virtual environment...
python -m venv "!VENV_DIR!"
if errorlevel 1 (
    echo  [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)

echo  Installing packages ^(PySide6, requests, chromadb, rich^)...
"!VENV_DIR!\Scripts\pip.exe" install --quiet PySide6 requests chromadb rich
if errorlevel 1 (
    echo  [ERROR] Package installation failed.
    pause
    exit /b 1
)
echo  [OK] Packages installed.

:: ============================================================
:: Ensure logs\ directory exists
:: ============================================================
if not exist "!SETUP_DIR!\logs" mkdir "!SETUP_DIR!\logs"

:: ============================================================
:: Write launch.bat via Python (no echo redirection)
:: ============================================================
echo.
echo [9/9] Writing launch.bat...

python -c "
import os
setup_dir = os.environ['SETUP_DIR_PY']
content = r'''@echo off
setlocal EnableDelayedExpansion
title VibeStudio
set OLLAMA_GPU_OVERHEAD=2147483648
set OLLAMA_MAX_LOADED_MODELS=1
set OLLAMA_NUM_PARALLEL=1
set OLLAMA_FLASH_ATTENTION=1
set OLLAMA_KV_CACHE_TYPE=q8_0
set \"SCRIPT_DIR=%~dp0\"
set \"SCRIPT_DIR=!SCRIPT_DIR:~0,-1!\"
tasklist | findstr /I \"ollama.exe\" >nul
if errorlevel 1 (
    start \"\" /B ollama serve
    timeout /t 4 /nobreak >nul
)
\"!SCRIPT_DIR!\venv\Scripts\python.exe\" \"!SCRIPT_DIR!\main.py\"
'''
path = os.path.join(setup_dir, 'launch.bat')
open(path, 'w').write(content)
print('  launch.bat written.')
"
if errorlevel 1 (
    echo  [ERROR] Failed to write launch.bat.
    pause
    exit /b 1
)

:: ============================================================
:: ALL DONE — launch the app
:: ============================================================
echo.
echo  =====================================================
echo   Setup complete!
echo  =====================================================
echo.
echo  Models  : qwen3-coder ^(Qwen3 14B^), deepseek-reasoner ^(DeepSeek-R1 8B^)
echo  VRAM    : capped at 13GB, one model at a time
echo  Launch  : double-click launch.bat from now on
echo.
echo  Launching VibeStudio now...
echo.
set "SETUP_DIR_PY=!SETUP_DIR!"
"!VENV_DIR!\Scripts\python.exe" "!SETUP_DIR!\main.py"
