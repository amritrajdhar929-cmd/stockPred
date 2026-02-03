#!/bin/bash

echo "🚀 Starting NSE Stock Prediction AI System..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python3 first."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip3 first."
    exit 1
fi

echo "📦 Installing dependencies..."
pip3 install -r requirements.txt

echo "📊 Generating synthetic dataset (if not exists)..."
if [ ! -f "data/dataset_metadata.json" ]; then
    python3 data/generate_dataset.py
else
    echo "✅ Dataset already exists"
fi

echo "🤖 Training ML models (if not exists)..."
if [ ! -d "backend/saved_models" ] || [ -z "$(ls -A backend/saved_models)" ]; then
    python3 backend/train_models.py
else
    echo "✅ Models already trained"
fi

echo "🌐 Starting backend server..."
echo "📱 Frontend will be available at: http://localhost:8000/static/index.html"
echo "🔗 API docs at: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

cd backend
python3 main.py
