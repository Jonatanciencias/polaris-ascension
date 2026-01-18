"""
API Module for Radeon RX 580 AI Framework

Session 17 Enhancement: REST API + Docker Deployment

Este módulo proporciona una interfaz REST API para el framework,
permitiendo la inferencia remota, gestión de modelos y monitoreo.

Components:
- FastAPI server: HTTP interface para inferencia
- Pydantic schemas: Validación de request/response
- Prometheus monitoring: Métricas de producción
- Docker deployment: Containerización

Author: Radeon RX 580 AI Framework Team
Date: Enero 2026
"""

from .server import app
from .schemas import (
    PredictRequest,
    PredictResponse,
    LoadModelRequest,
    LoadModelResponse,
    ModelInfo,
    HealthResponse,
    MetricsResponse,
)
from .monitoring import (
    track_inference,
    track_model_load,
    track_model_unload,
    get_metrics,
)

__all__ = [
    # FastAPI app
    "app",
    # Request/Response schemas
    "PredictRequest",
    "PredictResponse",
    "LoadModelRequest",
    "LoadModelResponse",
    "ModelInfo",
    "HealthResponse",
    "MetricsResponse",
    # Monitoring
    "track_inference",
    "track_model_load",
    "track_model_unload",
    "get_metrics",
]

__version__ = "0.6.0-dev"
