#!/usr/bin/env python3
"""
Minimal FastAPI serverless function for Vercel deployment
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="StockPred API")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend HTML file"""
    try:
        # Try to find the frontend file
        possible_paths = [
            "../frontend/index.html",
            "frontend/index.html",
            os.path.join(current_dir.parent, "frontend/index.html"),
        ]
        
        content = None
        for path in possible_paths:
            try:
                with open(path, "r") as f:
                    content = f.read()
                    break
            except FileNotFoundError:
                continue
        
        if content:
            return HTMLResponse(content=content)
        else:
            # Fallback HTML
            fallback_html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>StockPred - Stock Prediction API</title>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <script src="https://cdn.tailwindcss.com"></script>
                <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
            </head>
            <body class="bg-gray-100">
                <div class="container mx-auto px-4 py-8">
                    <div class="bg-white rounded-lg shadow-lg p-8">
                        <h1 class="text-3xl font-bold text-center mb-4">StockPred API</h1>
                        <div class="text-center">
                            <p class="text-green-600 mb-4">✅ API is running successfully!</p>
                            <p class="mb-6">The serverless function is deployed and working.</p>
                            <div class="bg-gray-50 rounded p-4">
                                <h3 class="font-semibold mb-2">Available Endpoints:</h3>
                                <ul class="text-left">
                                    <li><a href="/api" class="text-blue-600 hover:underline">/api</a> - API Information</li>
                                    <li><a href="/health" class="text-blue-600 hover:underline">/health</a> - Health Check</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
            </body>
            </html>
            """
            return HTMLResponse(content=fallback_html)
            
    except Exception as e:
        return JSONResponse(
            content={"error": f"Failed to serve frontend: {str(e)}"},
            status_code=500
        )

@app.get("/api")
async def api_info():
    """API information endpoint"""
    return {
        "message": "StockPred API - Serverless Version",
        "status": "running",
        "platform": "Vercel"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": "2024-01-01T00:00:00Z"}

# For Vercel serverless functions
handler = app
