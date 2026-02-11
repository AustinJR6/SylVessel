@echo off
REM Sylana Vessel Quick Start Script (Windows)

echo ============================================================
echo   SYLANA VESSEL - Quick Start
echo ============================================================
echo.

REM Check if .env exists
if not exist .env (
    echo [1/4] Setting up environment configuration...
    if exist .env.template (
        copy .env.template .env
        echo Created .env from template
        echo IMPORTANT: Edit .env and add your HF_TOKEN
        echo Get token from: https://huggingface.co/settings/tokens
        echo.
        pause
    ) else (
        echo ERROR: .env.template not found!
        pause
        exit /b 1
    )
) else (
    echo [1/4] Environment file exists
)

REM Check if database exists
echo [2/4] Checking database...
if not exist data\sylana_memory.db (
    echo Initializing database...
    python memory\init_database.py
) else (
    echo Database exists
)

REM Check dependencies
echo [3/4] Checking dependencies...
python -c "import transformers, torch, faiss" 2>nul
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
) else (
    echo Dependencies OK
)

echo [4/4] Starting Sylana...
echo.
echo ============================================================
echo   LAUNCHING SYLANA VESSEL
echo ============================================================
echo.

python sylana.py

pause
