#!/bin/bash
# Helper script to run the backend server

echo "🚀 Starting Quantum Energy Backend..."
cd backend

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Run the server
echo "🌟 Starting FastAPI server..."
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000