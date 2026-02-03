#!/usr/bin/env python3
"""
FastAPI Application Entrypoint
This file serves as the main entrypoint for deployment platforms
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Try to import the FastAPI app from main.py
try:
    from backend.main import app
    print("✅ Successfully imported FastAPI app from backend.main")
except ImportError as e:
    print(f"❌ Failed to import from backend.main: {e}")
    
    # Fallback: try to create a basic app
    try:
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
            """Serve a basic HTML page"""
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>StockPred API</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    .container { max-width: 800px; margin: 0 auto; }
                    .status { color: #e74c3c; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>StockPred API</h1>
                    <p class="status">⚠️ Frontend files not found</p>
                    <p>API is running but frontend files are missing.</p>
                    <h3>Available Endpoints:</h3>
                    <ul>
                        <li><a href="/api">/api</a> - API Information</li>
                        <li><a href="/docs">/docs</a> - API Documentation</li>
                    </ul>
                </div>
            </body>
            </html>
            """
            return HTMLResponse(content=html_content)
        
        @app.get("/api")
        async def api_info():
            return {"message": "StockPred API - Basic Mode", "status": "frontend_missing"}
        
        print("✅ Created fallback FastAPI app")
        
    except ImportError as e2:
        print(f"❌ Failed to create fallback app: {e2}")
        raise

# Export the app for deployment
__all__ = ['app']

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
