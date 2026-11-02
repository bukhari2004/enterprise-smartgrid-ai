@echo off
title Smart Grid Intelligence Platform (AI-238)
cd /d "%~dp0"

echo ========================================================
echo Starting Enterprise Smart Grid Intelligence Platform...
echo ========================================================

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv .venv
    echo Installing requirements...
    .venv\Scripts\python.exe -m pip install -r requirements.txt
)

echo Opening browser in 5 seconds (Wait ~20s for models to load if page refreshes)...
start "" cmd /c "timeout /t 5 /nobreak >nul && start http://127.0.0.1:8000"

echo Starting FastAPI Server on http://127.0.0.1:8000 ...
.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000

pause