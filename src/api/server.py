"""
FastAPI Server - REST API Principal

Session 17 Enhancement: Production REST API

Servidor FastAPI con endpoints para:
- Inferencia de modelos (/predict)
- Gestión de modelos (/models/*)
- Health checks (/health)
- Métricas (/metrics)

Características:
- Auto-documentation (OpenAPI/Swagger)
- Request/response validation (Pydantic)
- Error handling robusto
- CORS support
- Rate limiting ready
- Production logging

Author: Radeon RX 580 AI Framework Team
Date: Enero 2026
"""

import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Importar security modules (Session 18 - Phase 4)
SECURITY_AVAILABLE = False
require_api_key = None
require_admin = None
require_user = None
require_readonly = None

try:
    from .security import require_api_key, require_admin, require_user, require_readonly, security_config
    from .rate_limit import create_limiter, add_rate_limiting_middleware
    from .security_headers import add_security_middleware
    SECURITY_AVAILABLE = True
    print("✅ Security modules loaded successfully")
except ImportError as e:
    SECURITY_AVAILABLE = False
    print(f"⚠️ Security modules not available: {e}")
    
    # Define dummy dependencies para cuando security no está disponible
    def dummy_dependency():
        return None
    
    require_api_key = dummy_dependency
    require_admin = lambda: dummy_dependency
    require_user = lambda: dummy_dependency
    require_readonly = lambda: dummy_dependency

# Importar inference engine (Session 15 & 16)
try:
    from src.inference import (
        EnhancedInferenceEngine,
        create_loader,
        ModelMetadata
    )
    INFERENCE_AVAILABLE = True
except ImportError:
    INFERENCE_AVAILABLE = False
    print("Warning: Inference engine not available")

# Importar schemas y monitoring
from .schemas import (
    PredictRequest,
    PredictResponse,
    LoadModelRequest,
    LoadModelResponse,
    ModelInfo,
    HealthResponse,
    ModelsListResponse,
    ErrorResponse,
)
from .monitoring import (
    track_inference,
    track_model_load,
    track_model_unload,
    update_models_count,
    health_checker,
    get_metrics,
    record_error,
)


# ============================================================================
# CONFIGURACIÓN DE LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# ESTADO GLOBAL DEL SERVIDOR
# ============================================================================

class ServerState:
    """
    Estado global del servidor.
    
    Mantiene:
    - Inference engine
    - Información de modelos cargados
    - Contadores y estadísticas
    """
    
    def __init__(self):
        """Inicializa el estado del servidor"""
        self.engine: Optional[EnhancedInferenceEngine] = None
        self.models_info: Dict[str, Dict[str, Any]] = {}
        self.startup_time: datetime = datetime.now()
        logger.info("ServerState initialized")
    
    def initialize_engine(self):
        """
        Inicializa el inference engine.
        
        Se llama durante startup del servidor.
        """
        if not INFERENCE_AVAILABLE:
            logger.warning("Inference engine not available")
            return
        
        try:
            # Crear engine con configuración por defecto
            self.engine = EnhancedInferenceEngine(
                max_memory_mb=7000,  # 7GB para RX 580 (dejar 1GB para sistema)
                enable_compression=True,
                enable_batching=True
            )
            logger.info("EnhancedInferenceEngine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize engine: {e}")
            self.engine = None
    
    def add_model_info(self, name: str, metadata: Dict[str, Any]):
        """
        Agrega información de un modelo cargado.
        
        Args:
            name: Nombre del modelo
            metadata: Metadata del modelo
        """
        self.models_info[name] = {
            **metadata,
            "loaded_at": datetime.now(),
            "inference_count": 0
        }
        update_models_count(len(self.models_info))
        logger.info(f"Model '{name}' info added")
    
    def remove_model_info(self, name: str):
        """
        Elimina información de un modelo.
        
        Args:
            name: Nombre del modelo
        """
        if name in self.models_info:
            del self.models_info[name]
            update_models_count(len(self.models_info))
            logger.info(f"Model '{name}' info removed")
    
    def increment_inference_count(self, name: str):
        """
        Incrementa contador de inferencias de un modelo.
        
        Args:
            name: Nombre del modelo
        """
        if name in self.models_info:
            self.models_info[name]["inference_count"] += 1


# Instancia global del estado
server_state = ServerState()


# ============================================================================
# LIFECYCLE MANAGEMENT - Startup/Shutdown
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Maneja el ciclo de vida del servidor (startup/shutdown).
    
    Startup:
    - Inicializa inference engine
    - Carga configuración
    - Prepara recursos
    
    Shutdown:
    - Descarga modelos
    - Libera recursos
    - Guarda estado
    """
    # STARTUP
    logger.info("=" * 80)
    logger.info("Starting Radeon RX 580 AI API Server")
    logger.info("Session 17: REST API + Docker Deployment")
    logger.info("Session 18: Security Hardening")
    logger.info("=" * 80)
    
    # Inicializar security middleware
    if SECURITY_AVAILABLE:
        logger.info("Initializing security middleware...")
        add_security_middleware(app)
        add_rate_limiting_middleware(app)
        logger.info("✅ Security enabled: Authentication, Rate Limiting, Headers")
    else:
        logger.warning("⚠️  Security modules not available - running without protection")
    
    # Inicializar engine
    server_state.initialize_engine()
    
    logger.info("Server ready to accept requests")
    
    yield  # Servidor corriendo
    
    # SHUTDOWN
    logger.info("Shutting down server...")
    
    # Descargar modelos si engine disponible
    if server_state.engine:
        try:
            # Descargar todos los modelos
            for model_name in list(server_state.models_info.keys()):
                try:
                    server_state.engine.server.unload_model(model_name)
                    logger.info(f"Unloaded model: {model_name}")
                except Exception as e:
                    logger.error(f"Error unloading {model_name}: {e}")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    logger.info("Server shutdown complete")


# ============================================================================
# FASTAPI APP INITIALIZATION
# ============================================================================

app = FastAPI(
    title="Radeon RX 580 AI API",
    description="""
    REST API para inferencia de modelos de IA en AMD Radeon RX 580.
    
    ## Características
    
    * **Multi-framework**: ONNX y PyTorch (TorchScript)
    * **Compression**: Quantización, pruning, sparse matrices
    * **Dynamic batching**: Batching adaptativo automático
    * **Multi-model serving**: Múltiples modelos simultáneamente
    * **Production monitoring**: Métricas Prometheus
    * **Hardware-aware**: Selección automática de provider (ROCm/CUDA/OpenCL/CPU)
    
    ## Session 17 Implementation
    
    Completado: Enero 2026
    Components: FastAPI + Docker + Prometheus
    """,
    version="0.6.0-dev",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)


# ============================================================================
# MIDDLEWARE CONFIGURATION
# ============================================================================

# CORS middleware (configurar según necesidades)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción: especificar dominios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# EXCEPTION HANDLERS - Manejo global de errores
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """
    Handler para HTTPException.
    
    Registra el error y retorna response estándar.
    """
    record_error(
        error_type="HTTPException",
        endpoint=request.url.path
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "error_code": f"HTTP_{exc.status_code}",
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """
    Handler para excepciones no capturadas.
    
    Previene que el servidor crashee y retorna error 500.
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    record_error(
        error_type=type(exc).__name__,
        endpoint=request.url.path
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "error_code": "INTERNAL_ERROR",
            "timestamp": datetime.now().isoformat()
        }
    )


# ============================================================================
# ENDPOINTS - Root y Health
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """
    Endpoint raíz con información del servicio.
    
    Returns:
        Información básica del servidor
    """
    return {
        "service": "Radeon RX 580 AI API",
        "version": "0.6.0-dev",
        "session": "18 - Production Hardening",
        "status": "running",
        "security": {
            "enabled": SECURITY_AVAILABLE,
            "features": [
                "API Key Authentication (RBAC)",
                "Rate Limiting (Adaptive)",
                "Security Headers (CSP, HSTS, etc.)",
                "Input Validation"
            ] if SECURITY_AVAILABLE else ["None - Running without security"],
            "auth_methods": ["Header (X-API-Key)", "Query (?api_key=)", "Bearer Token"] if SECURITY_AVAILABLE else []
        },
        "endpoints": {
            "docs": "/docs",
            "health": "/health (public)",
            "metrics": "/metrics (readonly+)",
            "models": "/models (user+)",
            "inference": "/predict (user+)",
            "admin": "/models/load, /models/{id} (admin only)"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Retorna el estado de salud del servicio:
    - healthy: Todo funcionando correctamente
    - degraded: Funcionando pero con alta carga
    - unhealthy: Problemas críticos
    
    Returns:
        HealthResponse con métricas del sistema
    """
    try:
        health_data = health_checker.check_health(
            models_count=len(server_state.models_info)
        )
        return HealthResponse(**health_data)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Health check failed"
        )


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """
    Endpoint de métricas Prometheus.
    
    Retorna métricas en formato Prometheus para scraping:
    - Contadores de requests
    - Histogramas de latencias
    - Gauges de recursos
    
    Returns:
        Métricas en formato Prometheus
    """
    try:
        metrics_data, content_type = get_metrics()
        return Response(content=metrics_data, media_type=content_type)
    except Exception as e:
        logger.error(f"Metrics export failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Metrics export failed"
        )


# ============================================================================
# ENDPOINTS - Model Management
# ============================================================================

@app.post(
    "/models/load",
    response_model=LoadModelResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Models"]
)
async def load_model(
    request: LoadModelRequest,
    api_key: dict = Depends(require_admin)
):
    """
    Carga un modelo en el servidor.
    
    Soporta:
    - ONNX models (.onnx)
    - PyTorch TorchScript (.pt, .pth)
    
    Features:
    - Auto-detection de framework
    - Hardware-aware provider selection
    - Optional compression
    - Memory estimation
    
    Args:
        request: LoadModelRequest con configuración
        
    Returns:
        LoadModelResponse con información del modelo cargado
        
    Raises:
        404: Archivo no encontrado
        500: Error durante carga
    """
    if not server_state.engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine not available"
        )
    
    try:
        # Validar que el archivo existe
        model_path = Path(request.path)
        if not model_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model file not found: {request.path}"
            )
        
        # Determinar nombre del modelo
        model_name = request.model_name or model_path.stem
        
        # Detectar framework
        framework = "onnx" if request.path.endswith(".onnx") else "pytorch"
        
        # Track la operación
        with track_model_load(framework):
            # Cargar modelo
            server_state.engine.server.load_model(
                name=model_name,
                path=request.path,
                compression_config=request.compression,
                device=request.device
            )
        
        # Obtener metadata del modelo
        if model_name in server_state.engine.server.model_metadata:
            metadata = server_state.engine.server.model_metadata[model_name]
            
            # Guardar información
            server_state.add_model_info(model_name, {
                "framework": metadata.framework,
                "path": request.path,
                "device": request.device,
                "memory_mb": metadata.memory_mb,
                "input_shapes": metadata.input_shapes,
                "output_shapes": metadata.output_shapes,
            })
            
            logger.info(f"Model '{model_name}' loaded successfully")
            
            return LoadModelResponse(
                success=True,
                model_name=model_name,
                metadata={
                    "framework": metadata.framework,
                    "provider": metadata.provider,
                    "input_names": metadata.input_names,
                    "output_names": metadata.output_names,
                    "input_shapes": metadata.input_shapes,
                    "output_shapes": metadata.output_shapes,
                },
                memory_mb=metadata.memory_mb
            )
        else:
            # Fallback si no hay metadata
            logger.warning(f"No metadata found for model '{model_name}'")
            return LoadModelResponse(
                success=True,
                model_name=model_name,
                metadata={"framework": framework},
                memory_mb=None
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        return LoadModelResponse(
            success=False,
            error=str(e)
        )


@app.delete(
    "/models/{model_name}",
    tags=["Models"]
)
async def unload_model(
    model_name: str,
    api_key: dict = Depends(require_admin)
):
    """
    Descarga un modelo del servidor.
    
    Libera la memoria ocupada por el modelo.
    
    Args:
        model_name: Nombre del modelo a descargar
        
    Returns:
        Confirmación de descarga
        
    Raises:
        404: Modelo no encontrado
        500: Error durante descarga
    """
    if not server_state.engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine not available"
        )
    
    try:
        # Verificar que el modelo existe
        if model_name not in server_state.models_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_name}' not found"
            )
        
        # Descargar modelo
        server_state.engine.server.unload_model(model_name)
        
        # Actualizar estado
        server_state.remove_model_info(model_name)
        track_model_unload(status="success")
        
        logger.info(f"Model '{model_name}' unloaded successfully")
        
        return {
            "success": True,
            "message": f"Model '{model_name}' unloaded successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error unloading model: {e}", exc_info=True)
        track_model_unload(status="error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error unloading model: {str(e)}"
        )


@app.get(
    "/models",
    response_model=ModelsListResponse,
    tags=["Models"]
)
async def list_models(
    api_key: dict = Depends(require_user)
):
    """
    Lista todos los modelos cargados.
    
    Returns:
        Lista de ModelInfo con información de cada modelo
    """
    try:
        models_list = []
        
        for name, info in server_state.models_info.items():
            model_info = ModelInfo(
                name=name,
                framework=info.get("framework", "unknown"),
                memory_mb=info.get("memory_mb", 0.0),
                device=info.get("device", "unknown"),
                loaded_at=info["loaded_at"],
                input_shapes=info.get("input_shapes", {}),
                output_shapes=info.get("output_shapes", {}),
                inference_count=info.get("inference_count", 0)
            )
            models_list.append(model_info)
        
        return ModelsListResponse(
            models=models_list,
            total=len(models_list)
        )
        
    except Exception as e:
        logger.error(f"Error listing models: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error listing models"
        )


@app.get("/models/{model_name}", response_model=ModelInfo, tags=["Models"])
async def get_model_info(model_name: str):
    """
    Obtiene información de un modelo específico.
    
    Args:
        model_name: Nombre del modelo
        
    Returns:
        ModelInfo con detalles del modelo
        
    Raises:
        404: Modelo no encontrado
    """
    if model_name not in server_state.models_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' not found"
        )
    
    info = server_state.models_info[model_name]
    
    return ModelInfo(
        name=model_name,
        framework=info.get("framework", "unknown"),
        memory_mb=info.get("memory_mb", 0.0),
        device=info.get("device", "unknown"),
        loaded_at=info["loaded_at"],
        input_shapes=info.get("input_shapes", {}),
        output_shapes=info.get("output_shapes", {}),
        inference_count=info.get("inference_count", 0)
    )


# ============================================================================
# ENDPOINTS - Inference
# ============================================================================

@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Inference"]
)
async def predict(
    request: PredictRequest,
    api_key: dict = Depends(require_user)
):
    """
    Ejecuta inferencia en un modelo.
    
    Soporta:
    - Single input o multiple inputs (dict)
    - Batching automático (si habilitado)
    - Tracking de latencia
    
    Args:
        request: PredictRequest con modelo e inputs
        
    Returns:
        PredictResponse con outputs y metadata
        
    Raises:
        404: Modelo no encontrado
        400: Inputs inválidos
        500: Error durante inferencia
    """
    if not server_state.engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine not available"
        )
    
    # Verificar que el modelo existe
    if request.model_name not in server_state.models_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{request.model_name}' not found. Load it first with POST /models/load"
        )
    
    try:
        # Track la inferencia
        with track_inference(request.model_name):
            start_time = time.time()
            
            # Ejecutar inferencia
            # Nota: EnhancedInferenceEngine.server es MultiModelServer
            outputs = server_state.engine.server._run_inference(
                model_name=request.model_name,
                inputs=request.inputs
            )
            
            latency_ms = (time.time() - start_time) * 1000
        
        # Incrementar contador
        server_state.increment_inference_count(request.model_name)
        
        # Preparar response
        response_data = {
            "success": True,
            "outputs": outputs,
            "latency_ms": round(latency_ms, 2)
        }
        
        # Agregar metadata si solicitada
        if request.return_metadata:
            response_data["metadata"] = {
                "model": request.model_name,
                "framework": server_state.models_info[request.model_name].get("framework"),
                "device": server_state.models_info[request.model_name].get("device"),
                "inference_count": server_state.models_info[request.model_name].get("inference_count")
            }
        
        logger.info(
            f"Inference on '{request.model_name}' completed in {latency_ms:.2f}ms"
        )
        
        return PredictResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        return PredictResponse(
            success=False,
            error=f"Inference failed: {str(e)}"
        )


# ============================================================================
# MAIN - Para ejecutar con uvicorn
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload en desarrollo
        log_level="info"
    )
