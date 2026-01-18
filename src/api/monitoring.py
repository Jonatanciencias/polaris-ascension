"""
Monitoring y Métricas con Prometheus

Session 17 Enhancement: Production monitoring

Implementa métricas de Prometheus para monitoreo en producción:
- Request/response tracking
- Latencias de inferencia
- Uso de memoria GPU
- Errores y excepciones

Características:
- Prometheus-compatible metrics
- Histograms para latencias
- Counters para requests
- Gauges para recursos
- Labels para dimensiones

Author: Radeon RX 580 AI Framework Team
Date: Enero 2026
"""

import time
import psutil
from typing import Optional, Dict, Any
from datetime import datetime
from contextlib import contextmanager

try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        Info,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("Warning: prometheus_client not installed. Monitoring disabled.")


# ============================================================================
# MÉTRICAS PROMETHEUS - Definiciones
# ============================================================================

if PROMETHEUS_AVAILABLE:
    # Información del servicio
    service_info = Info(
        'radeon_rx580_service',
        'Información del servicio de inferencia'
    )
    service_info.info({
        'version': '0.6.0-dev',
        'framework': 'Radeon RX 580 AI Framework',
        'session': '17 - REST API + Docker'
    })
    
    # Counter: Requests totales
    inference_requests_total = Counter(
        'inference_requests_total',
        'Total de requests de inferencia',
        ['model_name', 'status']  # Labels para dimensiones
    )
    
    # Counter: Modelos cargados/descargados
    model_operations_total = Counter(
        'model_operations_total',
        'Total de operaciones de modelos',
        ['operation', 'status']  # operation: load/unload, status: success/error
    )
    
    # Histogram: Latencias de inferencia
    inference_latency_seconds = Histogram(
        'inference_latency_seconds',
        'Latencia de inferencia en segundos',
        ['model_name'],
        buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
    )
    
    # Histogram: Latencias de carga de modelo
    model_load_latency_seconds = Histogram(
        'model_load_latency_seconds',
        'Latencia de carga de modelo en segundos',
        ['framework'],
        buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
    )
    
    # Gauge: Memoria GPU utilizada
    gpu_memory_used_bytes = Gauge(
        'gpu_memory_used_bytes',
        'Memoria GPU utilizada en bytes'
    )
    
    # Gauge: Memoria GPU disponible
    gpu_memory_available_bytes = Gauge(
        'gpu_memory_available_bytes',
        'Memoria GPU disponible en bytes'
    )
    
    # Gauge: Modelos actualmente cargados
    models_loaded = Gauge(
        'models_loaded',
        'Número de modelos cargados'
    )
    
    # Gauge: Uso de CPU
    cpu_usage_percent = Gauge(
        'cpu_usage_percent',
        'Porcentaje de uso de CPU'
    )
    
    # Gauge: Uso de memoria RAM
    ram_usage_bytes = Gauge(
        'ram_usage_bytes',
        'Memoria RAM utilizada en bytes'
    )
    
    # Counter: Errores totales
    errors_total = Counter(
        'errors_total',
        'Total de errores',
        ['error_type', 'endpoint']
    )

else:
    # Placeholders si Prometheus no está disponible
    inference_requests_total = None
    model_operations_total = None
    inference_latency_seconds = None
    model_load_latency_seconds = None
    gpu_memory_used_bytes = None
    gpu_memory_available_bytes = None
    models_loaded = None
    cpu_usage_percent = None
    ram_usage_bytes = None
    errors_total = None


# ============================================================================
# FUNCIONES DE TRACKING - Tracking automático
# ============================================================================

@contextmanager
def track_inference(model_name: str):
    """
    Context manager para tracking de inferencia.
    
    Registra automáticamente:
    - Latencia de inferencia
    - Success/error status
    - Counter de requests
    
    Args:
        model_name: Nombre del modelo usado
        
    Usage:
        ```python
        with track_inference("resnet50"):
            result = model.predict(data)
        ```
    """
    start_time = time.time()
    status = "success"
    
    try:
        yield
    except Exception as e:
        status = "error"
        if PROMETHEUS_AVAILABLE and errors_total:
            errors_total.labels(
                error_type=type(e).__name__,
                endpoint="/predict"
            ).inc()
        raise
    finally:
        # Registrar latencia
        latency = time.time() - start_time
        if PROMETHEUS_AVAILABLE:
            if inference_latency_seconds:
                inference_latency_seconds.labels(
                    model_name=model_name
                ).observe(latency)
            
            if inference_requests_total:
                inference_requests_total.labels(
                    model_name=model_name,
                    status=status
                ).inc()


@contextmanager
def track_model_load(framework: str):
    """
    Context manager para tracking de carga de modelo.
    
    Registra automáticamente:
    - Latencia de carga
    - Success/error status
    - Counter de operaciones
    
    Args:
        framework: Framework del modelo (onnx/pytorch)
        
    Usage:
        ```python
        with track_model_load("onnx"):
            loader.load(model_path)
        ```
    """
    start_time = time.time()
    status = "success"
    
    try:
        yield
    except Exception as e:
        status = "error"
        if PROMETHEUS_AVAILABLE and errors_total:
            errors_total.labels(
                error_type=type(e).__name__,
                endpoint="/models/load"
            ).inc()
        raise
    finally:
        # Registrar latencia
        latency = time.time() - start_time
        if PROMETHEUS_AVAILABLE:
            if model_load_latency_seconds:
                model_load_latency_seconds.labels(
                    framework=framework
                ).observe(latency)
            
            if model_operations_total:
                model_operations_total.labels(
                    operation="load",
                    status=status
                ).inc()


def track_model_unload(status: str = "success"):
    """
    Registra descarga de modelo.
    
    Args:
        status: Estado de la operación (success/error)
    """
    if PROMETHEUS_AVAILABLE and model_operations_total:
        model_operations_total.labels(
            operation="unload",
            status=status
        ).inc()


def update_system_metrics():
    """
    Actualiza métricas del sistema (CPU, RAM, GPU).
    
    Debe ser llamada periódicamente (ej: cada segundo)
    para mantener métricas actualizadas.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    
    # CPU usage
    if cpu_usage_percent:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_usage_percent.set(cpu_percent)
    
    # RAM usage
    if ram_usage_bytes:
        memory_info = psutil.virtual_memory()
        ram_usage_bytes.set(memory_info.used)
    
    # GPU memory (simulado - requeriría ROCm/CUDA API real)
    # En producción, usar rocm-smi o similar
    if gpu_memory_used_bytes and gpu_memory_available_bytes:
        # Valores de ejemplo (8GB RX 580)
        # TODO: Implementar con ROCm API real
        gpu_memory_used_bytes.set(350 * 1024 * 1024)  # 350 MB
        gpu_memory_available_bytes.set(7650 * 1024 * 1024)  # 7650 MB


def update_models_count(count: int):
    """
    Actualiza contador de modelos cargados.
    
    Args:
        count: Número de modelos actualmente cargados
    """
    if PROMETHEUS_AVAILABLE and models_loaded:
        models_loaded.set(count)


def record_error(error_type: str, endpoint: str):
    """
    Registra un error manualmente.
    
    Args:
        error_type: Tipo de error (ej: "ValueError", "ModelNotFound")
        endpoint: Endpoint donde ocurrió el error
    """
    if PROMETHEUS_AVAILABLE and errors_total:
        errors_total.labels(
            error_type=error_type,
            endpoint=endpoint
        ).inc()


# ============================================================================
# EXPORT DE MÉTRICAS - Para endpoint /metrics
# ============================================================================

def get_metrics() -> tuple[bytes, str]:
    """
    Genera métricas en formato Prometheus.
    
    Returns:
        Tuple de (metrics_data, content_type)
        
    Usage:
        ```python
        @app.get("/metrics")
        async def metrics():
            data, content_type = get_metrics()
            return Response(content=data, media_type=content_type)
        ```
    """
    if not PROMETHEUS_AVAILABLE:
        return b"# Prometheus client not installed\n", "text/plain"
    
    # Actualizar métricas del sistema antes de exportar
    update_system_metrics()
    
    # Generar métricas en formato Prometheus
    metrics_data = generate_latest()
    
    return metrics_data, CONTENT_TYPE_LATEST


def get_metrics_dict() -> Dict[str, Any]:
    """
    Obtiene métricas como diccionario (para debugging).
    
    Returns:
        Diccionario con métricas actuales
        
    Usage:
        ```python
        metrics = get_metrics_dict()
        print(f"CPU: {metrics['cpu_usage_percent']}%")
        ```
    """
    if not PROMETHEUS_AVAILABLE:
        return {"error": "Prometheus not available"}
    
    # Actualizar métricas
    update_system_metrics()
    
    # Recolectar valores actuales
    metrics_dict = {
        "timestamp": datetime.now().isoformat(),
        "cpu_usage_percent": psutil.cpu_percent(interval=0.1),
        "ram_usage_bytes": psutil.virtual_memory().used,
        "ram_usage_mb": psutil.virtual_memory().used / (1024 * 1024),
        # GPU métricas (simuladas)
        "gpu_memory_used_mb": 350.0,
        "gpu_memory_available_mb": 7650.0,
    }
    
    return metrics_dict


# ============================================================================
# HEALTH CHECK - Para endpoint /health
# ============================================================================

class HealthChecker:
    """
    Sistema de health checks para el servicio.
    
    Verifica:
    - Tiempo de actividad (uptime)
    - Uso de recursos (CPU, RAM, GPU)
    - Estado de modelos cargados
    """
    
    def __init__(self):
        """Inicializa el health checker"""
        self.start_time = time.time()
    
    def get_uptime(self) -> float:
        """
        Obtiene tiempo de actividad del servicio.
        
        Returns:
            Uptime en segundos
        """
        return time.time() - self.start_time
    
    def check_health(self, models_count: int) -> Dict[str, Any]:
        """
        Ejecuta health check completo.
        
        Args:
            models_count: Número de modelos cargados
            
        Returns:
            Dict con estado de salud del servicio
        """
        # Recolectar métricas del sistema
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_info = psutil.virtual_memory()
        
        # Determinar status (healthy/degraded/unhealthy)
        status = "healthy"
        
        # Degraded si CPU > 80% o RAM > 90%
        if cpu_percent > 80 or memory_info.percent > 90:
            status = "degraded"
        
        # Unhealthy si CPU > 95% o RAM > 95%
        if cpu_percent > 95 or memory_info.percent > 95:
            status = "unhealthy"
        
        return {
            "status": status,
            "version": "0.6.0-dev",
            "models_loaded": models_count,
            "memory_used_mb": memory_info.used / (1024 * 1024),
            "memory_available_mb": memory_info.available / (1024 * 1024),
            "memory_percent": memory_info.percent,
            "cpu_percent": cpu_percent,
            "uptime_seconds": self.get_uptime(),
            "timestamp": datetime.now()
        }


# Instancia global del health checker
health_checker = HealthChecker()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_bytes(bytes_value: float) -> str:
    """
    Formatea bytes en unidades legibles.
    
    Args:
        bytes_value: Valor en bytes
        
    Returns:
        String formateado (ej: "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def get_monitoring_summary() -> str:
    """
    Genera resumen de monitoreo para logs.
    
    Returns:
        String con resumen formateado
    """
    metrics = get_metrics_dict()
    
    return f"""
    Monitoring Summary:
    - CPU: {metrics.get('cpu_usage_percent', 0):.1f}%
    - RAM: {metrics.get('ram_usage_mb', 0):.1f} MB
    - GPU: {metrics.get('gpu_memory_used_mb', 0):.1f} MB used
    - Uptime: {health_checker.get_uptime():.0f}s
    - Timestamp: {metrics.get('timestamp', 'N/A')}
    """
