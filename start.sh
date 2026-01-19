#!/bin/bash

# Port cleanup logic
function cleanup() {
    echo "Stopping existing services..."
    lsof -ti:8000 | xargs kill -9 2>/dev/null
}

# Start Unified Service (FastAPI + React SPA)
cleanup
echo "Starting Intelligent Research Assistant on http://0.0.0.0:8000"
source venv/bin/activate
cd backend && uvicorn main:app --host 0.0.0.0 --port 8000
