#!/usr/bin/env python3
"""
Simple Test Server - Minimal API for Testing Security
Usa solo los módulos de seguridad sin dependencias del inference engine.
"""

import sys
import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Importar security modules
SECURITY_AVAILABLE = False
require_api_key = None
require_admin = None
require_user = None
require_readonly = None

try:
    from src.api.security import require_api_key, require_admin, require_user, require_readonly, security_config
    from src.api.rate_limit import create_limiter, add_rate_limiting_middleware
    from src.api.security_headers import add_security_middleware
    SECURITY_AVAILABLE = True
    logger.info("✅ Security modules loaded successfully")
except ImportError as e:
    SECURITY_AVAILABLE = False
    logger.error(f"❌ Security modules not available: {e}")
    
    # Dummy dependencies
    def dummy_dependency():
        return {"name": "test", "role": "admin"}
    require_api_key = dummy_dependency
    require_admin = lambda: dummy_dependency
    require_user = lambda: dummy_dependency
    require_readonly = lambda: dummy_dependency


# ============================================================================
# LIFECYCLE
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Maneja el ciclo de vida del servidor."""
    logger.info("=" * 80)
    logger.info("🚀 Starting Test API Server")
    logger.info("Session 18 - Phase 4: Security Testing")
    logger.info("=" * 80)
    
    if SECURITY_AVAILABLE:
        logger.info("🔐 Initializing security middleware...")
        add_security_middleware(app)
        add_rate_limiting_middleware(app)
        logger.info("✅ Security enabled: Authentication, Rate Limiting, Headers")
    else:
        logger.warning("⚠️  Security modules not available")
    
    logger.info("✅ Server ready to accept requests")
    logger.info("=" * 80)
    
    yield
    
    logger.info("Shutting down server...")


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Radeon RX 580 Test API",
    description="Simplified API for security testing",
    version="0.6.0-test",
    lifespan=lifespan
)


# ============================================================================
# SCHEMAS
# ============================================================================

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    security_enabled: bool


class ModelInfo(BaseModel):
    name: str
    status: str


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint con información del servicio."""
    return {
        "service": "Radeon RX 580 Test API",
        "version": "0.6.0-test",
        "session": "18 - Security Testing",
        "status": "running",
        "security": {
            "enabled": SECURITY_AVAILABLE,
            "features": [
                "API Key Authentication (RBAC)",
                "Rate Limiting (Adaptive)",
                "Security Headers (CSP, HSTS, etc.)",
                "Input Validation"
            ] if SECURITY_AVAILABLE else ["None"],
            "auth_methods": ["Header", "Query", "Bearer"] if SECURITY_AVAILABLE else []
        },
        "endpoints": {
            "docs": "/docs",
            "health": "/health (public)",
            "metrics": "/metrics (readonly+)",
            "models": "/models (user+)",
            "admin_test": "/admin/test (admin only)"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Health check endpoint (público con rate limiting)."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        security_enabled=SECURITY_AVAILABLE
    )


@app.get("/metrics", tags=["Monitoring"])
async def metrics(api_key: dict = Depends(require_readonly())):
    """Metrics endpoint (readonly+)."""
    return {
        "metrics": {
            "requests_total": 1000,
            "requests_per_second": 10.5,
            "memory_usage_mb": 512,
            "cpu_usage_percent": 25.3
        },
        "user": api_key.get("name", "unknown") if api_key else "anonymous"
    }


@app.get("/models", tags=["Models"])
async def list_models(api_key: dict = Depends(require_user())):
    """List models endpoint (user+)."""
    return {
        "models": [
            {"name": "resnet50", "status": "loaded", "framework": "onnx"},
            {"name": "bert-base", "status": "loaded", "framework": "pytorch"}
        ],
        "count": 2,
        "user": api_key.get("name", "unknown") if api_key else "anonymous"
    }


@app.post("/models/load", tags=["Models"])
async def load_model(
    model_data: dict,
    api_key: dict = Depends(require_admin())
):
    """Load model endpoint (admin only)."""
    return {
        "success": True,
        "message": f"Model load requested (test mode)",
        "data": model_data,
        "admin": api_key.get("name", "unknown") if api_key else "anonymous"
    }


@app.delete("/models/{model_name}", tags=["Models"])
async def unload_model(
    model_name: str,
    api_key: dict = Depends(require_admin())
):
    """Unload model endpoint (admin only)."""
    return {
        "success": True,
        "message": f"Model '{model_name}' unload requested (test mode)",
        "admin": api_key.get("name", "unknown") if api_key else "anonymous"
    }


@app.post("/predict", tags=["Inference"])
async def predict(
    request_data: dict,
    api_key: dict = Depends(require_user())
):
    """Inference endpoint (user+)."""
    return {
        "prediction": [0.1, 0.2, 0.7],
        "model": "test-model",
        "latency_ms": 42.5,
        "user": api_key.get("name", "unknown") if api_key else "anonymous"
    }


@app.get("/admin/test", tags=["Admin"])
async def admin_test(api_key: dict = Depends(require_admin())):
    """Admin test endpoint (admin only)."""
    return {
        "message": "Admin access granted",
        "admin": api_key.get("name", "unknown") if api_key else "anonymous",
        "permissions": ["full_access", "model_management", "user_management"]
    }


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Maneja excepciones globales."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "error": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
