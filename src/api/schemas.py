"""
Pydantic Schemas para REST API

Session 17 Enhancement: Request/Response validation

Define todos los modelos Pydantic para validación automática
de requests y responses en la API REST.

Características:
- Type validation automática
- Documentación OpenAPI generada automáticamente
- Error messages claros y útiles
- Default values sensibles

Author: Radeon RX 580 AI Framework Team
Date: Enero 2026
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator


# ============================================================================
# REQUEST SCHEMAS - Validación de entrada
# ============================================================================

class PredictRequest(BaseModel):
    """
    Request para ejecutar inferencia en un modelo.
    
    Attributes:
        model_name: Nombre del modelo a usar (debe estar cargado)
        inputs: Datos de entrada como dict {nombre_input: valores}
                o lista de valores para input único
        batch_size: Tamaño de batch (opcional, default=1)
        return_metadata: Si debe retornar metadata adicional
    
    Example:
        ```python
        request = PredictRequest(
            model_name="resnet50",
            inputs={"input": [[1.0, 2.0, ...]]},
            batch_size=1
        )
        ```
    """
    model_name: str = Field(
        ...,
        description="Nombre del modelo cargado",
        min_length=1,
        max_length=100,
        example="resnet50"
    )
    inputs: Union[Dict[str, Any], List[Any]] = Field(
        ...,
        description="Datos de entrada (dict o lista)",
        example={"input": [[1.0, 2.0, 3.0]]}
    )
    batch_size: Optional[int] = Field(
        default=1,
        description="Tamaño del batch",
        ge=1,
        le=128
    )
    return_metadata: bool = Field(
        default=False,
        description="Incluir metadata en respuesta"
    )
    
    @validator('model_name')
    def validate_model_name(cls, v):
        """Valida que el nombre del modelo sea válido"""
        if not v.strip():
            raise ValueError("model_name no puede estar vacío")
        return v.strip()
    
    class Config:
        """Configuración del modelo Pydantic"""
        schema_extra = {
            "example": {
                "model_name": "mobilenet_v2",
                "inputs": {"input": [[1.0, 2.0, 3.0]]},
                "batch_size": 1,
                "return_metadata": False
            }
        }


class LoadModelRequest(BaseModel):
    """
    Request para cargar un modelo en el servidor.
    
    Attributes:
        path: Ruta al archivo del modelo (.onnx, .pt, .pth)
        model_name: Nombre para identificar el modelo (opcional)
        compression: Configuración de compresión (opcional)
        device: Device para inference (cpu/cuda/auto)
    
    Example:
        ```python
        request = LoadModelRequest(
            path="/models/resnet50.onnx",
            model_name="resnet50",
            compression={"quantize": "int8"}
        )
        ```
    """
    path: str = Field(
        ...,
        description="Ruta al archivo del modelo",
        min_length=1,
        example="/models/resnet50.onnx"
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Nombre para el modelo (auto-generado si None)",
        max_length=100
    )
    compression: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Configuración de compresión",
        example={"quantize": "int8", "prune": 0.3}
    )
    device: str = Field(
        default="auto",
        description="Device de inferencia (cpu/cuda/auto)",
        pattern="^(cpu|cuda|auto)$"
    )
    optimization_level: int = Field(
        default=1,
        description="Nivel de optimización ONNX (0-2)",
        ge=0,
        le=2
    )
    
    @validator('path')
    def validate_path(cls, v):
        """Valida que la ruta tenga extensión válida"""
        valid_extensions = ['.onnx', '.pt', '.pth']
        if not any(v.endswith(ext) for ext in valid_extensions):
            raise ValueError(
                f"Extensión inválida. Use: {', '.join(valid_extensions)}"
            )
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "path": "/models/resnet50.onnx",
                "model_name": "resnet50",
                "compression": {"quantize": "int8"},
                "device": "auto",
                "optimization_level": 1
            }
        }


class CompressionConfig(BaseModel):
    """
    Configuración de compresión de modelos.
    
    Attributes:
        quantize: Tipo de quantización (int4/int8/fp16/None)
        prune_ratio: Ratio de pruning (0.0-0.9)
        sparse_format: Formato sparse (csr/csc/block/None)
    """
    quantize: Optional[str] = Field(
        default=None,
        description="Tipo de quantización",
        pattern="^(int4|int8|fp16)?$"
    )
    prune_ratio: Optional[float] = Field(
        default=None,
        description="Ratio de pruning",
        ge=0.0,
        le=0.9
    )
    sparse_format: Optional[str] = Field(
        default=None,
        description="Formato sparse",
        pattern="^(csr|csc|block)?$"
    )


# ============================================================================
# RESPONSE SCHEMAS - Formato de salida
# ============================================================================

class PredictResponse(BaseModel):
    """
    Response de una operación de inferencia.
    
    Attributes:
        success: Si la inferencia fue exitosa
        outputs: Resultados de la inferencia
        latency_ms: Latencia en milisegundos
        metadata: Metadata adicional (si solicitada)
        error: Mensaje de error (si falló)
    """
    success: bool = Field(
        ...,
        description="Estado de la operación"
    )
    outputs: Optional[Union[Dict[str, Any], List[Any]]] = Field(
        default=None,
        description="Resultados de inferencia"
    )
    latency_ms: Optional[float] = Field(
        default=None,
        description="Latencia en milisegundos",
        ge=0
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata adicional"
    )
    error: Optional[str] = Field(
        default=None,
        description="Mensaje de error"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "outputs": {"output": [[0.1, 0.9]]},
                "latency_ms": 15.3,
                "metadata": {"model": "resnet50", "device": "cpu"}
            }
        }


class LoadModelResponse(BaseModel):
    """
    Response de carga de modelo.
    
    Attributes:
        success: Si la carga fue exitosa
        model_name: Nombre asignado al modelo
        metadata: Información del modelo
        error: Mensaje de error (si falló)
    """
    success: bool = Field(..., description="Estado de la operación")
    model_name: Optional[str] = Field(
        default=None,
        description="Nombre del modelo cargado"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata del modelo"
    )
    memory_mb: Optional[float] = Field(
        default=None,
        description="Memoria utilizada en MB"
    )
    error: Optional[str] = Field(
        default=None,
        description="Mensaje de error"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "model_name": "resnet50",
                "metadata": {
                    "framework": "onnx",
                    "inputs": ["input"],
                    "outputs": ["output"]
                },
                "memory_mb": 98.5
            }
        }


class ModelInfo(BaseModel):
    """
    Información de un modelo cargado.
    
    Attributes:
        name: Nombre del modelo
        framework: Framework (onnx/pytorch)
        memory_mb: Memoria utilizada
        device: Device de inferencia
        loaded_at: Timestamp de carga
        input_shapes: Shapes de entrada
        output_shapes: Shapes de salida
    """
    name: str = Field(..., description="Nombre del modelo")
    framework: str = Field(..., description="Framework del modelo")
    memory_mb: float = Field(..., description="Memoria en MB", ge=0)
    device: str = Field(..., description="Device de inferencia")
    loaded_at: datetime = Field(..., description="Timestamp de carga")
    input_shapes: Dict[str, List[int]] = Field(
        ...,
        description="Shapes de entrada"
    )
    output_shapes: Dict[str, List[int]] = Field(
        ...,
        description="Shapes de salida"
    )
    inference_count: int = Field(
        default=0,
        description="Número de inferencias",
        ge=0
    )
    
    class Config:
        schema_extra = {
            "example": {
                "name": "resnet50",
                "framework": "onnx",
                "memory_mb": 98.5,
                "device": "cpu",
                "loaded_at": "2026-01-18T10:30:00",
                "input_shapes": {"input": [1, 3, 224, 224]},
                "output_shapes": {"output": [1, 1000]},
                "inference_count": 42
            }
        }


class HealthResponse(BaseModel):
    """
    Response del health check.
    
    Attributes:
        status: Estado del servicio (healthy/degraded/unhealthy)
        version: Versión del framework
        models_loaded: Número de modelos cargados
        memory_used_mb: Memoria GPU utilizada
        memory_available_mb: Memoria GPU disponible
        uptime_seconds: Tiempo de actividad
    """
    status: str = Field(
        ...,
        description="Estado del servicio",
        pattern="^(healthy|degraded|unhealthy)$"
    )
    version: str = Field(..., description="Versión del framework")
    models_loaded: int = Field(..., description="Modelos cargados", ge=0)
    memory_used_mb: float = Field(..., description="Memoria usada", ge=0)
    memory_available_mb: float = Field(..., description="Memoria disponible", ge=0)
    uptime_seconds: float = Field(..., description="Uptime en segundos", ge=0)
    timestamp: datetime = Field(..., description="Timestamp actual")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "version": "0.6.0-dev",
                "models_loaded": 3,
                "memory_used_mb": 350.2,
                "memory_available_mb": 7649.8,
                "uptime_seconds": 3600.5,
                "timestamp": "2026-01-18T10:30:00"
            }
        }


class MetricsResponse(BaseModel):
    """
    Response con métricas de Prometheus.
    
    Attributes:
        metrics: Métricas en formato Prometheus
        timestamp: Timestamp de generación
    """
    metrics: str = Field(..., description="Métricas formato Prometheus")
    timestamp: datetime = Field(..., description="Timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "metrics": "# HELP inference_requests_total ...\n...",
                "timestamp": "2026-01-18T10:30:00"
            }
        }


class ErrorResponse(BaseModel):
    """
    Response estándar para errores.
    
    Attributes:
        detail: Mensaje de error
        error_code: Código de error
        timestamp: Timestamp del error
    """
    detail: str = Field(..., description="Mensaje de error")
    error_code: Optional[str] = Field(
        default=None,
        description="Código de error"
    )
    timestamp: datetime = Field(..., description="Timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "detail": "Model not found",
                "error_code": "MODEL_NOT_FOUND",
                "timestamp": "2026-01-18T10:30:00"
            }
        }


# ============================================================================
# LISTA RESPONSE SCHEMAS
# ============================================================================

class ModelsListResponse(BaseModel):
    """
    Response con lista de modelos cargados.
    
    Attributes:
        models: Lista de ModelInfo
        total: Total de modelos
    """
    models: List[ModelInfo] = Field(..., description="Lista de modelos")
    total: int = Field(..., description="Total de modelos", ge=0)
    
    class Config:
        schema_extra = {
            "example": {
                "models": [
                    {
                        "name": "resnet50",
                        "framework": "onnx",
                        "memory_mb": 98.5,
                        "device": "cpu",
                        "loaded_at": "2026-01-18T10:30:00",
                        "input_shapes": {"input": [1, 3, 224, 224]},
                        "output_shapes": {"output": [1, 1000]},
                        "inference_count": 42
                    }
                ],
                "total": 1
            }
        }
