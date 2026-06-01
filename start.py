#!/usr/bin/env python3
"""
Media Authenticity Verifier — Single Entry Point
Starts both the FastAPI backend and the Vite frontend dev server.

Usage:
  python start.py            # Start both servers
  python start.py --build    # Build frontend for production, then start backend only
"""

import os
import sys
import subprocess
import signal
import time
import webbrowser
import threading
import platform
import urllib.request
import urllib.error

# Resolve project directory (where this script lives)
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(PROJECT_DIR, "frontend")
BACKEND_PORT = 8000
FRONTEND_PORT = 3000


# On Windows, npm/bun/yarn are .cmd batch scripts — subprocess needs shell=True
IS_WINDOWS = platform.system() == 'Windows'


def check_node():
    """Check if Node.js/npm is available."""
    try:
        result = subprocess.run(
            ["node", "--version"],
            capture_output=True, text=True, timeout=10,
            shell=IS_WINDOWS,
        )
        print(f"  Node.js version: {result.stdout.strip()}")
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_python():
    """Check if Python has the required packages."""
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
        return True
    except ImportError:
        return False


def install_frontend_deps():
    """Install frontend npm dependencies if node_modules doesn't exist."""
    node_modules = os.path.join(FRONTEND_DIR, "node_modules")
    if not os.path.exists(node_modules):
        print("\n  Installing frontend dependencies (first time only)...")
        npm_cmd = "npm"
        # Try npm first, then yarn, then bun
        for cmd in ["npm", "yarn", "bun"]:
            try:
                subprocess.run(
                    [cmd, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    shell=IS_WINDOWS,
                )
                npm_cmd = cmd
                break
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue

        if npm_cmd == "yarn":
            install_args = [npm_cmd, "install", "--cwd", FRONTEND_DIR]
        elif npm_cmd == "bun":
            install_args = [npm_cmd, "install", FRONTEND_DIR]
        else:
            install_args = [npm_cmd, "install"]

        print(f"  Running: {npm_cmd} install in {FRONTEND_DIR}")
        result = subprocess.run(
            install_args,
            cwd=FRONTEND_DIR,
            capture_output=False,
            shell=IS_WINDOWS,
        )
        if result.returncode != 0:
            print(f"\n  ERROR: {npm_cmd} install failed!")
            print("  Try running manually: cd frontend && npm install")
            return False
        print("  Frontend dependencies installed successfully.")
    else:
        print("  Frontend dependencies already installed.")
    return True


def build_frontend():
    """Build the frontend for production (output to static/)."""
    print("\n  Building frontend for production...")
    result = subprocess.run(
        "npm run build",
        cwd=FRONTEND_DIR,
        capture_output=False,
        shell=True,  # Always use shell for npm commands
    )
    if result.returncode != 0:
        print("  ERROR: Frontend build failed!")
        return False
    print("  Frontend built successfully.")
    return True


def start_backend(serve_static=False):
    """Start the FastAPI backend server."""
    env = os.environ.copy()
    if serve_static:
        env["SERVE_STATIC"] = "1"

    cmd = [sys.executable, "-m", "uvicorn", "api:app",
           "--host", "0.0.0.0", "--port", str(BACKEND_PORT)]
    print(f"\n  Starting backend: {' '.join(cmd)}")
    return subprocess.Popen(cmd, cwd=PROJECT_DIR, env=env)


def start_frontend_dev():
    """Start the Vite frontend dev server."""
    # On Windows, npm is a .cmd file so we MUST use shell=True
    # On Unix, shell=True also works fine with "npm run dev"
    cmd = "npm run dev"
    print(f"  Starting frontend dev server: {cmd}")
    return subprocess.Popen(
        cmd,
        cwd=FRONTEND_DIR,
        shell=True,
        # On Windows, create a new process group so Ctrl+C doesn't cascade
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if IS_WINDOWS else 0,
    )


processes = []


def signal_handler(sig, frame):
    """Handle Ctrl+C to shut down all processes."""
    print("\n\n  Shutting down all servers...")
    for p in processes:
        try:
            if IS_WINDOWS:
                # On Windows with shell=True, we need to kill the whole process tree
                subprocess.run(
                    f"taskkill /F /T /PID {p.pid}",
                    capture_output=True, shell=True,
                )
            else:
                p.terminate()
                p.wait(timeout=5)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
    print("  All servers stopped.")
    sys.exit(0)


def open_browser():
    """Open browser after a short delay."""
    time.sleep(3)
    url = f"http://localhost:{FRONTEND_PORT}"
    print(f"\n  Opening browser: {url}")
    try:
        webbrowser.open(url)
    except Exception:
        pass


def main():
    build_mode = "--build" in sys.argv

    print("=" * 60)
    print("  Media Authenticity Verifier for Digital Evidence")
    print("  Forensic Analysis Pipeline v1.0")
    print("=" * 60)

    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Check Python deps
    print("\n  Checking Python dependencies...")
    if not check_python():
        print("  ERROR: Missing Python packages!")
        print("  Run: pip install -r requirements.txt")
        sys.exit(1)
    print("  Python dependencies OK.")

    if build_mode:
        # ---- PRODUCTION MODE: Build frontend, serve from backend ----
        print("\n  [Production Mode]")
        if not check_node():
            print("  ERROR: Node.js not found! Required to build frontend.")
            print("  Install from: https://nodejs.org/")
            sys.exit(1)

        if not install_frontend_deps():
            sys.exit(1)

        if not build_frontend():
            sys.exit(1)

        backend = start_backend(serve_static=True)
        processes.append(backend)

        # Wait for backend to be ready
        print("  Waiting for backend to start...")
        backend_ready = False
        max_wait = 120
        start_time = time.time()
        while time.time() - start_time < max_wait:
            if backend.poll() is not None:
                print("\n  ERROR: Backend process crashed!")
                sys.exit(1)
            try:
                resp = urllib.request.urlopen(
                    f"http://localhost:{BACKEND_PORT}/api/health", timeout=2
                )
                if resp.status == 200:
                    backend_ready = True
                    break
            except (urllib.error.URLError, ConnectionRefusedError, OSError):
                pass
            time.sleep(1)

        if not backend_ready:
            print("\n  ERROR: Backend did not start within {} seconds!".format(max_wait))
            sys.exit(1)

        url = f"http://localhost:{BACKEND_PORT}"
        print(f"\n  Production server running at: {url}")
        print("  Press Ctrl+C to stop.")

        threading.Thread(target=lambda: (time.sleep(1), webbrowser.open(url)), daemon=True).start()

        backend.wait()

    else:
        # ---- DEV MODE: Both servers with live reload ----
        print("\n  [Development Mode]")
        print("  Frontend: http://localhost:{} (Vite dev server)".format(FRONTEND_PORT))
        print("  Backend:  http://localhost:{} (FastAPI + API docs)".format(BACKEND_PORT))
        print()

        # Check Node.js
        has_node = check_node()
        if has_node:
            if not install_frontend_deps():
                sys.exit(1)
        else:
            print("  WARNING: Node.js not found. Frontend dev server will NOT start.")
            print("  The backend API will still run at http://localhost:{}".format(BACKEND_PORT))
            print("  Install Node.js from: https://nodejs.org/")

        # Start backend
        backend = start_backend()
        processes.append(backend)

        # Wait for backend to actually be ready (health check polling)
        print(f"  Waiting for backend to start on port {BACKEND_PORT}...")
        backend_ready = False
        max_wait = 120  # seconds — model loading can take a while on first run
        start_time = time.time()
        while time.time() - start_time < max_wait:
            elapsed = int(time.time() - start_time)
            # Check if process crashed
            if backend.poll() is not None:
                print("\n  ERROR: Backend process crashed!")
                print(f"  Exit code: {backend.returncode}")
                sys.exit(1)
            # Try the health endpoint
            try:
                resp = urllib.request.urlopen(
                    f"http://localhost:{BACKEND_PORT}/api/health", timeout=2
                )
                if resp.status == 200:
                    import json
                    data = json.loads(resp.read().decode())
                    if data.get("pipeline_loaded"):
                        backend_ready = True
                        break
                    else:
                        # Server is up but models still loading
                        if elapsed > 0 and elapsed % 10 == 0:
                            print(f"  Server up, waiting for models to load... ({elapsed}s)")
            except (urllib.error.URLError, ConnectionRefusedError, OSError):
                pass
            time.sleep(1)
            if elapsed > 0 and elapsed % 10 == 0:
                print(f"  Still waiting for backend... ({elapsed}s)")

        if not backend_ready:
            print("\n  ERROR: Backend did not start within {} seconds!".format(max_wait))
            sys.exit(1)

        print(f"  Backend is ready! (took {int(time.time() - start_time)}s)")

        # Start frontend
        if has_node:
            frontend = start_frontend_dev()
            processes.append(frontend)
            time.sleep(3)
            if frontend.poll() is not None:
                print("  WARNING: Frontend dev server failed to start.")
                print(f"  You can still access the API at http://localhost:{BACKEND_PORT}/docs")
            else:
                print(f"  Frontend running on port {FRONTEND_PORT}")

        # Open browser
        if has_node:
            threading.Thread(target=open_browser, daemon=True).start()

        print("\n  Press Ctrl+C to stop both servers.")
        print("-" * 60)

        # Wait for either process to exit
        while True:
            for p in processes:
                if p.poll() is not None:
                    print(f"\n  Process exited with code {p.returncode}")
                    signal_handler(None, None)
            time.sleep(1)


if __name__ == "__main__":
    main()
