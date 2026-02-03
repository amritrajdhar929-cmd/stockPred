#!/usr/bin/env python3
"""
FastAPI Application Entrypoint
This file serves as the main entrypoint for deployment platforms
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the FastAPI app from main.py
from backend.main import app

# Export the app for deployment
__all__ = ['app']

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
