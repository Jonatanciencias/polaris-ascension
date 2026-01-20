#!/usr/bin/env python3
"""
Minimal Security Test Server
Tests security without complex dependencies.
"""

import sys
import json
from pathlib import Path
from datetime import datetime
import logging

# Setup
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from fastapi import FastAPI, Request, HTTPException, status, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Load API keys
keys_file = project_root / "config" / "api_keys.json"
API_KEYS = {}

if keys_file.exists():
    with open(keys_file) as f:
        data = json.load(f)
        API_KEYS = data.get("keys", {})
    logger.info(f"✅ Loaded {len(API_KEYS)} API keys")
else:
    logger.warning("⚠️ No API keys file found")


# Simple auth function
def validate_api_key(key: str) -> dict:
    """Validate API key and return key info."""
    if key in API_KEYS:
        return API_KEYS[key]
    return None


# FastAPI app
app = FastAPI(title="Security Test API", version="1.0")


# Middleware for rate limiting headers
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)
    
    #Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    
    return response


# Endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Security Test API",
        "version": "1.0",
        "security": "enabled",
        "keys_loaded": len(API_KEYS),
        "endpoints": {
            "health": "/health (public)",
            "models": "/models (user+)",
            "admin": "/admin/test (admin only)"
        }
    }


@app.get("/health")
async def health():
    """Public health endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/models")
async def list_models(x_api_key: str = Header(None, alias="X-API-Key")):
    """List models (requires user+ role)."""
    
    # Check auth
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API key required")
    
    key_info = validate_api_key(x_api_key)
    if not key_info:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # Check role
    role = key_info.get("role")
    if role not in ["user", "admin"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    return {
        "models": [
            {"name": "resnet50", "status": "loaded"},
            {"name": "bert-base", "status": "loaded"}
        ],
        "user": key_info.get("name"),
        "role": role
    }


@app.post("/models/load")
async def load_model(
    data: dict,
    x_api_key: str = Header(None, alias="X-API-Key")
):
    """Load model (requires admin role)."""
    
    # Check auth
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API key required")
    
    key_info = validate_api_key(x_api_key)
    if not key_info:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # Check role
    role = key_info.get("role")
    if role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    return {
        "success": True,
        "message": "Model load requested (test mode)",
        "data": data,
        "admin": key_info.get("name")
    }


@app.get("/admin/test")
async def admin_test(x_api_key: str = Header(None, alias="X-API-Key")):
    """Admin test endpoint (requires admin role)."""
    
    # Check auth
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API key required")
    
    key_info = validate_api_key(x_api_key)
    if not key_info:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # Check role
    role = key_info.get("role")
    if role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    return {
        "message": "Admin access granted",
        "admin": key_info.get("name"),
        "permissions": ["full_access", "model_management"]
    }


if __name__ == "__main__":
    logger.info("🚀 Starting minimal security test server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
