@echo off
REM Media Authenticity Verifier - Windows Start Script
REM Starts both backend and frontend servers

echo ============================================================
echo   Media Authenticity Verifier for Digital Evidence
echo   Forensic Analysis Pipeline v1.0
echo ============================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found! Install from https://python.org/
    pause
    exit /b 1
)

REM Check Node.js
node --version >nul 2>&1
if errorlevel 1 (
    echo WARNING: Node.js not found! Frontend will not start.
    echo The backend API will still run at http://localhost:8000
    echo Install Node.js from https://nodejs.org/
    echo.
    echo Starting backend only...
    python -m uvicorn api:app --host 0.0.0.0 --port 8000
    pause
    exit /b 0
)

REM Install frontend deps if needed
if not exist "frontend\node_modules" (
    echo Installing frontend dependencies...
    cd frontend
    call npm install
    cd ..
    echo Frontend dependencies installed.
    echo.
)

echo Starting backend and frontend servers...
echo   Frontend: http://localhost:3000
echo   Backend:  http://localhost:8000
echo.
echo Press Ctrl+C to stop both servers.
echo.

REM Start backend in a new window
start "Media Verifier - Backend" cmd /c "python -m uvicorn api:app --host 0.0.0.0 --port 8000"

REM Wait a moment for backend to initialize
timeout /t 2 /nobreak >nul

REM Start frontend in a new window
start "Media Verifier - Frontend" cmd /c "cd frontend && npm run dev"

REM Open browser after a delay
timeout /t 5 /nobreak >nul
start http://localhost:3000

echo.
echo Both servers are running. Close the terminal windows to stop.
echo Backend API docs: http://localhost:8000/docs
pause
