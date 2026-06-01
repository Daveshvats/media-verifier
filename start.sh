#!/bin/bash
# Media Authenticity Verifier - Linux/Mac Start Script
# Starts both backend and frontend servers

set -e

echo "============================================================"
echo "  Media Authenticity Verifier for Digital Evidence"
echo "  Forensic Analysis Pipeline v1.0"
echo "============================================================"
echo

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 not found!"
    exit 1
fi
echo "Python: $(python3 --version)"

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "WARNING: Node.js not found! Frontend will not start."
    echo "The backend API will still run at http://localhost:8000"
    echo "Install Node.js from https://nodejs.org/"
    echo
    echo "Starting backend only..."
    python3 -m uvicorn api:app --host 0.0.0.0 --port 8000
    exit 0
fi
echo "Node.js: $(node --version)"

# Install frontend deps if needed
if [ ! -d "frontend/node_modules" ]; then
    echo "Installing frontend dependencies..."
    cd frontend && npm install && cd ..
    echo "Frontend dependencies installed."
    echo
fi

echo
echo "Starting servers..."
echo "  Frontend: http://localhost:3000"
echo "  Backend:  http://localhost:8000"
echo
echo "Press Ctrl+C to stop both servers."
echo

# Cleanup function
cleanup() {
    echo
    echo "Shutting down servers..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    wait $BACKEND_PID $FRONTEND_PID 2>/dev/null
    echo "All servers stopped."
    exit 0
}
trap cleanup SIGINT SIGTERM

# Start backend
python3 -m uvicorn api:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
sleep 2

# Start frontend
cd frontend && npm run dev &
FRONTEND_PID=$!
cd ..

# Wait
wait
