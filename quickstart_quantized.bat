@echo off
REM Sylana Vessel - Quantized Model Quick Start (Windows)

echo ============================================================
echo   SYLANA VESSEL - Quantized Model Setup
echo   Optimized for ancient laptops with limited disk space!
echo ============================================================
echo.

REM Always use quantized config for this script
echo [1/5] Setting up quantized configuration...
if exist .env.quantized (
    copy /Y .env.quantized .env >nul
    echo Using quantized model configuration
) else (
    echo ERROR: .env.quantized template not found!
    pause
    exit /b 1
)

REM Check if database exists
echo [2/5] Checking database...
if not exist data\sylana_memory.db (
    echo Initializing database...
    python memory\init_database.py
) else (
    echo Database exists
)

REM Check dependencies
echo [3/5] Checking dependencies...
pip show ctransformers >nul 2>&1
if errorlevel 1 (
    echo Installing ctransformers - has pre-built wheels
    pip install ctransformers
) else (
    echo ctransformers OK
)

pip show transformers >nul 2>&1
if errorlevel 1 (
    echo Installing other dependencies
    pip install -r requirements.txt
) else (
    echo Other dependencies OK
)

REM Check if quantized model exists
echo [4/5] Checking quantized model...
if exist models\llama-2-7b-chat.Q4_K_M.gguf (
    echo Quantized model found!
    goto launch
)

echo.
echo ============================================================
echo   QUANTIZED MODEL NOT FOUND
echo ============================================================
echo.
echo Model file not found: models\llama-2-7b-chat.Q4_K_M.gguf
echo This file is about 6GB (much smaller than the 13.5GB full model)
echo.
echo OPTION 1: Download automatically
echo   The script will download it for you
echo.
echo OPTION 2: Download manually
echo   - Go to: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF
echo   - Download: llama-2-7b-chat.Q4_K_M.gguf
echo   - Save to: models\llama-2-7b-chat.Q4_K_M.gguf
echo.
echo ============================================================
echo.
set /p download="Would you like to download it now? (y/n): "
if /i "%download%"=="y" (
    echo.
    echo Downloading model - This will take some time (about 6GB)
    python download_model.py
    if errorlevel 1 (
        echo.
        echo Download failed. Please try manual download.
        pause
        exit /b 1
    )
    echo.
    echo Download complete! Model ready to use.
    echo.
) else (
    echo.
    echo Please download the model manually and run this script again.
    pause
    exit /b 1
)

:launch
echo [5/5] Starting Sylana (Quantized Edition)...
echo.
echo ============================================================
echo   LAUNCHING SYLANA VESSEL - QUANTIZED
echo ============================================================
echo.

python sylana_quantized.py

pause

