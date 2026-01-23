"""
Cluster Management Endpoints - Session 33

REST API endpoints para gestión de clúster distribuido:
- Registro y gestión de workers
- Envío y monitoreo de tareas
- Métricas y salud del clúster

Integra con src/distributed/coordinator.py

Author: Radeon RX 580 AI Framework Team
Date: Enero 22, 2026
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

from fastapi import APIRouter, HTTPException, status, Depends, BackgroundTasks
from pydantic import BaseModel, Field

# Imports de distributed layer
try:
    from ..distributed.coordinator import ClusterCoordinator, TaskPriority
    from ..distributed.load_balancing import LoadBalancingStrategy
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False
    TaskPriority = None
    LoadBalancingStrategy = None

logger = logging.getLogger(__name__)

# ============================================================================
# SCHEMAS (Pydantic Models)
# ============================================================================

class WorkerRegistration(BaseModel):
    """Request para registrar un worker"""
    worker_id: str = Field(..., description="ID único del worker")
    address: str = Field(..., description="Dirección ZMQ del worker (ej: tcp://192.168.1.10:5556)")
    capabilities: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Capacidades del worker (GPU, memoria, etc.)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "worker_id": "worker-001",
                "address": "tcp://192.168.1.10:5556",
                "capabilities": {
                    "gpu_name": "RX 580",
                    "gpu_memory_gb": 8.0,
                    "max_concurrent_tasks": 4
                }
            }
        }


class TaskSubmission(BaseModel):
    """Request para enviar una tarea distribuida"""
    model_name: str = Field(..., description="Nombre del modelo a ejecutar")
    input_data: Dict[str, Any] = Field(..., description="Datos de entrada para el modelo")
    priority: str = Field(default="NORMAL", description="Prioridad: LOW, NORMAL, HIGH, CRITICAL")
    requirements: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Requisitos del worker (GPU mínima, memoria, etc.)"
    )
    timeout: Optional[float] = Field(default=300.0, description="Timeout en segundos")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "resnet50",
                "input_data": {
                    "image_path": "/data/image.jpg",
                    "batch_size": 1
                },
                "priority": "HIGH",
                "requirements": {
                    "min_gpu_memory_gb": 4.0
                },
                "timeout": 60.0
            }
        }


class TaskStatusResponse(BaseModel):
    """Response con estado de una tarea"""
    task_id: str
    status: str  # PENDING, ASSIGNED, RUNNING, COMPLETED, FAILED
    assigned_worker: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class TaskResultResponse(BaseModel):
    """Response con resultado de una tarea"""
    task_id: str
    status: str
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time_ms: Optional[float] = None
    worker_id: Optional[str] = None


class WorkerInfo(BaseModel):
    """Información de un worker"""
    worker_id: str
    address: str
    status: str  # ACTIVE, INACTIVE, UNHEALTHY
    capabilities: Dict[str, Any]
    current_load: Dict[str, Any]
    tasks_completed: int
    tasks_failed: int
    last_heartbeat: Optional[datetime] = None


class ClusterHealth(BaseModel):
    """Estado de salud del clúster"""
    status: str  # HEALTHY, DEGRADED, CRITICAL
    total_workers: int
    healthy_workers: int
    unhealthy_workers: int
    total_tasks: int
    pending_tasks: int
    running_tasks: int
    completed_tasks: int
    failed_tasks: int
    avg_task_latency_ms: Optional[float] = None
    cluster_uptime_seconds: float


class ClusterMetrics(BaseModel):
    """Métricas agregadas del clúster"""
    timestamp: datetime
    workers_active: int
    tasks_pending: int
    tasks_running: int
    tasks_per_minute: float
    avg_latency_ms: float
    success_rate: float
    total_throughput: float


class LoadBalancingConfig(BaseModel):
    """Configuración de balanceo de carga"""
    strategy: str = Field(..., description="ROUND_ROBIN, LEAST_LOADED, GPU_MATCH, LATENCY, ADAPTIVE")
    
    class Config:
        json_schema_extra = {
            "example": {
                "strategy": "ADAPTIVE"
            }
        }


# ============================================================================
# COORDINATOR SINGLETON
# ============================================================================

_coordinator: Optional['ClusterCoordinator'] = None
_coordinator_config: Dict[str, Any] = {}


def get_coordinator() -> 'ClusterCoordinator':
    """
    Obtiene el coordinator singleton.
    Si no existe, lo crea con la configuración por defecto.
    """
    global _coordinator
    
    if not DISTRIBUTED_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Distributed computing layer not available"
        )
    
    if _coordinator is None:
        # Configuración por defecto
        bind_address = _coordinator_config.get('bind_address', 'tcp://0.0.0.0:5555')
        balancing_strategy = _coordinator_config.get('balancing_strategy', 'ADAPTIVE')
        
        logger.info(f"Creating ClusterCoordinator: bind_address={bind_address}, strategy={balancing_strategy}")
        
        _coordinator = ClusterCoordinator(
            bind_address=bind_address,
            balancing_strategy=balancing_strategy
        )
        
        # Iniciar en background (no bloqueante)
        import threading
        thread = threading.Thread(target=_coordinator.start, daemon=True)
        thread.start()
        
        # Esperar un poco para que inicie
        import time
        time.sleep(0.5)
        
        logger.info("ClusterCoordinator started successfully")
    
    return _coordinator


def configure_coordinator(config: Dict[str, Any]):
    """Configura el coordinator antes de crearlo"""
    global _coordinator_config
    _coordinator_config.update(config)


async def shutdown_coordinator():
    """Shutdown del coordinator (para cleanup)"""
    global _coordinator
    if _coordinator is not None:
        logger.info("Shutting down ClusterCoordinator...")
        _coordinator.shutdown()
        _coordinator = None


# ============================================================================
# ROUTER
# ============================================================================

router = APIRouter(
    prefix="/cluster",
    tags=["cluster"],
    responses={
        503: {"description": "Distributed layer not available"},
        500: {"description": "Internal server error"}
    }
)


# ============================================================================
# ENDPOINTS - CLUSTER INFO
# ============================================================================

@router.get(
    "/health",
    response_model=ClusterHealth,
    summary="Get cluster health",
    description="Obtiene el estado de salud del clúster y métricas básicas"
)
async def get_cluster_health(
    coordinator: ClusterCoordinator = Depends(get_coordinator)
) -> ClusterHealth:
    """Get cluster health status"""
    try:
        worker_stats = coordinator.get_worker_stats()
        task_stats = coordinator.get_task_stats()
        
        total_workers = worker_stats.get('total_workers', 0)
        healthy_workers = worker_stats.get('healthy_workers', 0)
        
        # Determinar status
        if total_workers == 0:
            health_status = "CRITICAL"
        elif healthy_workers == total_workers:
            health_status = "HEALTHY"
        elif healthy_workers > 0:
            health_status = "DEGRADED"
        else:
            health_status = "CRITICAL"
        
        return ClusterHealth(
            status=health_status,
            total_workers=total_workers,
            healthy_workers=healthy_workers,
            unhealthy_workers=total_workers - healthy_workers,
            total_tasks=task_stats.get('total_tasks', 0),
            pending_tasks=task_stats.get('pending_tasks', 0),
            running_tasks=task_stats.get('running_tasks', 0),
            completed_tasks=task_stats.get('completed_tasks', 0),
            failed_tasks=task_stats.get('failed_tasks', 0),
            avg_task_latency_ms=task_stats.get('avg_latency_ms'),
            cluster_uptime_seconds=0.0  # TODO: track uptime
        )
    
    except Exception as e:
        logger.error(f"Error getting cluster health: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cluster health: {str(e)}"
        )


@router.get(
    "/metrics",
    response_model=ClusterMetrics,
    summary="Get cluster metrics",
    description="Obtiene métricas agregadas del clúster"
)
async def get_cluster_metrics(
    coordinator: ClusterCoordinator = Depends(get_coordinator)
) -> ClusterMetrics:
    """Get aggregated cluster metrics"""
    try:
        worker_stats = coordinator.get_worker_stats()
        task_stats = coordinator.get_task_stats()
        
        # Calcular métricas
        total_tasks = task_stats.get('completed_tasks', 0) + task_stats.get('failed_tasks', 0)
        success_rate = 0.0
        if total_tasks > 0:
            success_rate = task_stats.get('completed_tasks', 0) / total_tasks
        
        return ClusterMetrics(
            timestamp=datetime.now(),
            workers_active=worker_stats.get('healthy_workers', 0),
            tasks_pending=task_stats.get('pending_tasks', 0),
            tasks_running=task_stats.get('running_tasks', 0),
            tasks_per_minute=0.0,  # TODO: track rate
            avg_latency_ms=task_stats.get('avg_latency_ms', 0.0),
            success_rate=success_rate,
            total_throughput=0.0  # TODO: track throughput
        )
    
    except Exception as e:
        logger.error(f"Error getting cluster metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cluster metrics: {str(e)}"
        )


@router.get(
    "/config",
    summary="Get cluster configuration",
    description="Obtiene la configuración actual del clúster"
)
async def get_cluster_config(
    coordinator: ClusterCoordinator = Depends(get_coordinator)
) -> Dict[str, Any]:
    """Get current cluster configuration"""
    return {
        "bind_address": coordinator.bind_address,
        "balancing_strategy": coordinator.balancing_strategy,
        "max_queue_size": 1000,  # TODO: make configurable
        "heartbeat_interval": 5.0,
        "heartbeat_timeout": 15.0
    }


@router.put(
    "/config/balancing",
    summary="Update load balancing strategy",
    description="Actualiza la estrategia de balanceo de carga"
)
async def update_balancing_strategy(
    config: LoadBalancingConfig,
    coordinator: ClusterCoordinator = Depends(get_coordinator)
) -> Dict[str, str]:
    """Update load balancing strategy"""
    try:
        # Validar estrategia
        valid_strategies = ['ROUND_ROBIN', 'LEAST_LOADED', 'GPU_MATCH', 'LATENCY', 'ADAPTIVE']
        if config.strategy not in valid_strategies:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid strategy. Must be one of: {valid_strategies}"
            )
        
        # Actualizar estrategia
        coordinator.balancing_strategy = config.strategy
        
        logger.info(f"Updated balancing strategy to: {config.strategy}")
        
        return {
            "status": "success",
            "new_strategy": config.strategy
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating balancing strategy: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update balancing strategy: {str(e)}"
        )


# ============================================================================
# ENDPOINTS - WORKER MANAGEMENT
# ============================================================================

@router.get(
    "/workers",
    response_model=List[WorkerInfo],
    summary="List all workers",
    description="Lista todos los workers registrados en el clúster"
)
async def list_workers(
    status_filter: Optional[str] = None,
    coordinator: ClusterCoordinator = Depends(get_coordinator)
) -> List[WorkerInfo]:
    """List all workers in the cluster"""
    try:
        worker_stats = coordinator.get_worker_stats()
        workers_data = worker_stats.get('workers', {})
        
        workers = []
        for worker_id, stats in workers_data.items():
            # Determinar status
            is_healthy = stats.get('status') == 'healthy'
            worker_status = 'ACTIVE' if is_healthy else 'UNHEALTHY'
            
            # Filtrar por status si se especifica
            if status_filter and worker_status != status_filter.upper():
                continue
            
            workers.append(WorkerInfo(
                worker_id=worker_id,
                address=stats.get('address', 'unknown'),
                status=worker_status,
                capabilities=stats.get('capabilities', {}),
                current_load=stats.get('load', {}),
                tasks_completed=stats.get('tasks_completed', 0),
                tasks_failed=stats.get('tasks_failed', 0),
                last_heartbeat=stats.get('last_heartbeat')
            ))
        
        return workers
    
    except Exception as e:
        logger.error(f"Error listing workers: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list workers: {str(e)}"
        )


@router.get(
    "/workers/{worker_id}",
    response_model=WorkerInfo,
    summary="Get worker details",
    description="Obtiene información detallada de un worker específico"
)
async def get_worker(
    worker_id: str,
    coordinator: ClusterCoordinator = Depends(get_coordinator)
) -> WorkerInfo:
    """Get detailed information about a specific worker"""
    try:
        worker_stats = coordinator.get_worker_stats()
        workers_data = worker_stats.get('workers', {})
        
        if worker_id not in workers_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Worker '{worker_id}' not found"
            )
        
        stats = workers_data[worker_id]
        is_healthy = stats.get('status') == 'healthy'
        
        return WorkerInfo(
            worker_id=worker_id,
            address=stats.get('address', 'unknown'),
            status='ACTIVE' if is_healthy else 'UNHEALTHY',
            capabilities=stats.get('capabilities', {}),
            current_load=stats.get('load', {}),
            tasks_completed=stats.get('tasks_completed', 0),
            tasks_failed=stats.get('tasks_failed', 0),
            last_heartbeat=stats.get('last_heartbeat')
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting worker {worker_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get worker: {str(e)}"
        )


# ============================================================================
# ENDPOINTS - TASK MANAGEMENT
# ============================================================================

@router.post(
    "/tasks",
    response_model=Dict[str, str],
    status_code=status.HTTP_201_CREATED,
    summary="Submit a task",
    description="Envía una nueva tarea al clúster para ejecución distribuida"
)
async def submit_task(
    task: TaskSubmission,
    background_tasks: BackgroundTasks,
    coordinator: ClusterCoordinator = Depends(get_coordinator)
) -> Dict[str, str]:
    """Submit a new task to the cluster"""
    try:
        # Mapear prioridad
        priority_map = {
            'LOW': TaskPriority.LOW if TaskPriority else 0,
            'NORMAL': TaskPriority.NORMAL if TaskPriority else 1,
            'HIGH': TaskPriority.HIGH if TaskPriority else 2,
            'CRITICAL': TaskPriority.CRITICAL if TaskPriority else 3
        }
        priority = priority_map.get(task.priority.upper(), TaskPriority.NORMAL if TaskPriority else 1)
        
        # Preparar payload
        payload = {
            'model_name': task.model_name,
            'input_data': task.input_data,
            'timeout': task.timeout
        }
        
        # Enviar tarea
        task_id = coordinator.submit_task(
            payload=payload,
            requirements=task.requirements or {},
            priority=priority
        )
        
        logger.info(f"Task submitted: {task_id} (priority={task.priority})")
        
        return {
            "task_id": task_id,
            "status": "PENDING",
            "message": "Task submitted successfully"
        }
    
    except Exception as e:
        logger.error(f"Error submitting task: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit task: {str(e)}"
        )


@router.get(
    "/tasks/{task_id}/status",
    response_model=TaskStatusResponse,
    summary="Get task status",
    description="Obtiene el estado actual de una tarea"
)
async def get_task_status(
    task_id: str,
    coordinator: ClusterCoordinator = Depends(get_coordinator)
) -> TaskStatusResponse:
    """Get status of a specific task"""
    try:
        # Obtener estado de la tarea
        task_stats = coordinator.get_task_stats()
        
        # TODO: Implementar tracking de tareas individuales en coordinator
        # Por ahora, retornar info básica
        
        return TaskStatusResponse(
            task_id=task_id,
            status="UNKNOWN",  # TODO: track individual task status
            created_at=datetime.now()
        )
    
    except Exception as e:
        logger.error(f"Error getting task status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get task status: {str(e)}"
        )


@router.get(
    "/tasks/{task_id}/result",
    response_model=TaskResultResponse,
    summary="Get task result",
    description="Obtiene el resultado de una tarea completada"
)
async def get_task_result(
    task_id: str,
    timeout: float = 60.0,
    coordinator: ClusterCoordinator = Depends(get_coordinator)
) -> TaskResultResponse:
    """Get result of a completed task"""
    try:
        # Intentar obtener resultado (blocking con timeout)
        result = coordinator.get_result(task_id, timeout=timeout)
        
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_408_REQUEST_TIMEOUT,
                detail=f"Task '{task_id}' result not available within {timeout}s"
            )
        
        return TaskResultResponse(
            task_id=task_id,
            status="COMPLETED",
            result=result,
            error=None,
            execution_time_ms=None,  # TODO: track execution time
            worker_id=None  # TODO: track which worker executed
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting task result: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get task result: {str(e)}"
        )


@router.get(
    "/tasks",
    summary="List tasks",
    description="Lista todas las tareas con filtros opcionales"
)
async def list_tasks(
    status_filter: Optional[str] = None,
    limit: int = 100,
    coordinator: ClusterCoordinator = Depends(get_coordinator)
) -> Dict[str, Any]:
    """List all tasks with optional filtering"""
    try:
        task_stats = coordinator.get_task_stats()
        
        return {
            "total_tasks": task_stats.get('total_tasks', 0),
            "pending": task_stats.get('pending_tasks', 0),
            "running": task_stats.get('running_tasks', 0),
            "completed": task_stats.get('completed_tasks', 0),
            "failed": task_stats.get('failed_tasks', 0),
            "tasks": []  # TODO: return actual task list
        }
    
    except Exception as e:
        logger.error(f"Error listing tasks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list tasks: {str(e)}"
        )


# ============================================================================
# ADMIN ENDPOINTS
# ============================================================================

@router.post(
    "/shutdown",
    summary="Shutdown cluster",
    description="Apaga el coordinador del clúster (admin only)"
)
async def shutdown_cluster(
    background_tasks: BackgroundTasks,
    coordinator: ClusterCoordinator = Depends(get_coordinator)
) -> Dict[str, str]:
    """Shutdown the cluster coordinator"""
    try:
        # Shutdown en background para permitir respuesta
        background_tasks.add_task(shutdown_coordinator)
        
        logger.warning("Cluster shutdown initiated")
        
        return {
            "status": "shutting_down",
            "message": "Cluster coordinator shutting down"
        }
    
    except Exception as e:
        logger.error(f"Error shutting down cluster: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to shutdown cluster: {str(e)}"
        )
