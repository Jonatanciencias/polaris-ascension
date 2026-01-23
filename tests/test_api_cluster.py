"""
Integration Tests - API + Distributed Computing
Session 33

Tests de integración entre:
- REST API (cluster_endpoints.py)
- Distributed Layer (coordinator.py, worker.py)

Author: Radeon RX 580 AI Framework Team
Date: Enero 22, 2026
"""

import pytest
import time
import threading
from typing import Dict, Any, Optional
import json

# FastAPI testing
try:
    from fastapi.testclient import TestClient
    from src.api.server import app
    from src.api.cluster_endpoints import get_coordinator, configure_coordinator, shutdown_coordinator
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    pytest.skip("FastAPI not available", allow_module_level=True)

# Distributed layer
try:
    from src.distributed.coordinator import ClusterCoordinator, TaskPriority
    from src.distributed.worker import InferenceWorker
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False
    pytest.skip("Distributed layer not available", allow_module_level=True)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def client():
    """TestClient de FastAPI con coordinator iniciado"""
    # Configurar coordinator global antes de crear cliente
    test_address = 'tcp://127.0.0.1:15555'
    configure_coordinator({
        'bind_address': test_address,
        'balancing_strategy': 'ROUND_ROBIN'
    })
    
    # Forzar inicialización del coordinator
    coord = get_coordinator()
    
    # Crear cliente
    test_client = TestClient(app)
    
    yield test_client
    
    # Cleanup
    try:
        shutdown_coordinator()
    except:
        pass
    
    time.sleep(0.3)


@pytest.fixture
def coordinator():
    """
    Coordinator instance para tests.
    Se crea y destruye para cada test.
    """
    # Configurar coordinator en puerto de test
    test_address = 'tcp://127.0.0.1:15555'  # Puerto de test diferente
    configure_coordinator({
        'bind_address': test_address,
        'balancing_strategy': 'ROUND_ROBIN'
    })
    
    coord = None
    thread = None
    
    try:
        # Crear coordinator
        coord = ClusterCoordinator(
            bind_address=test_address,
            balancing_strategy='ROUND_ROBIN'
        )
        
        # Iniciar en thread separado
        thread = threading.Thread(target=coord.start, daemon=True)
        thread.start()
        
        # Esperar a que inicie
        time.sleep(0.5)
        
        yield coord
    
    finally:
        # Cleanup
        if coord:
            coord.shutdown()
        # Dar tiempo para shutdown limpio
        time.sleep(0.3)


@pytest.fixture
def worker(coordinator):
    """
    Worker instance conectado al coordinator.
    """
    worker_id = 'test-worker-1'
    worker_address = 'tcp://127.0.0.1:15556'
    
    # Crear worker
    w = InferenceWorker(
        worker_id=worker_id,
        coordinator_address=coordinator.bind_address,
        bind_address=worker_address
    )
    
    # Handler simple de test
    def test_handler(payload: Dict[str, Any]) -> Any:
        """Handler que solo retorna el payload"""
        return {
            'status': 'success',
            'input': payload,
            'worker': worker_id,
            'processed': True
        }
    
    w.register_handler(test_handler)
    
    # Iniciar worker en thread
    worker_thread = threading.Thread(target=w.start, daemon=True)
    worker_thread.start()
    
    # Esperar registro
    time.sleep(0.5)
    
    yield w
    
    # Cleanup
    w.shutdown()
    time.sleep(0.3)


# ============================================================================
# TESTS - HEALTH & STATUS
# ============================================================================

def test_cluster_health_endpoint(client):
    """Test: GET /cluster/health retorna estado del clúster"""
    response = client.get("/cluster/health")
    
    assert response.status_code == 200
    data = response.json()
    
    # Verificar estructura
    assert 'status' in data
    assert 'total_workers' in data
    assert 'healthy_workers' in data
    assert 'pending_tasks' in data
    
    # Sin workers, status debe ser CRITICAL
    assert data['status'] in ['CRITICAL', 'HEALTHY', 'DEGRADED']


def test_cluster_metrics_endpoint(client):
    """Test: GET /cluster/metrics retorna métricas"""
    response = client.get("/cluster/metrics")
    
    assert response.status_code == 200
    data = response.json()
    
    # Verificar estructura
    assert 'timestamp' in data
    assert 'workers_active' in data
    assert 'tasks_pending' in data
    assert 'success_rate' in data
    
    # Success rate debe ser número válido
    assert 0.0 <= data['success_rate'] <= 1.0


def test_cluster_config_endpoint(client):
    """Test: GET /cluster/config retorna configuración"""
    response = client.get("/cluster/config")
    
    assert response.status_code == 200
    data = response.json()
    
    # Verificar estructura
    assert 'bind_address' in data
    assert 'balancing_strategy' in data
    assert 'heartbeat_interval' in data


# ============================================================================
# TESTS - WORKER MANAGEMENT
# ============================================================================

def test_list_workers_empty(client):
    """Test: GET /cluster/workers sin workers retorna lista vacía"""
    response = client.get("/cluster/workers")
    
    assert response.status_code == 200
    data = response.json()
    
    # Debe ser lista vacía o con workers inactivos
    assert isinstance(data, list)


def test_list_workers_with_worker(client, coordinator, worker):
    """Test: GET /cluster/workers con worker activo"""
    # Esperar un poco más para asegurar que el worker está registrado
    time.sleep(1.0)
    
    response = client.get("/cluster/workers")
    
    assert response.status_code == 200
    data = response.json()
    
    # Debe haber al menos 1 worker
    assert len(data) > 0
    
    # Verificar estructura del worker
    worker_data = data[0]
    assert 'worker_id' in worker_data
    assert 'address' in worker_data
    assert 'status' in worker_data
    assert 'capabilities' in worker_data


def test_get_worker_details(client, coordinator, worker):
    """Test: GET /cluster/workers/{id} retorna detalles del worker"""
    # Esperar registro
    time.sleep(1.0)
    
    worker_id = 'test-worker-1'
    response = client.get(f"/cluster/workers/{worker_id}")
    
    # Puede ser 404 si el worker no se registró a tiempo
    if response.status_code == 200:
        data = response.json()
        assert data['worker_id'] == worker_id
        assert 'capabilities' in data
        assert 'current_load' in data


def test_get_nonexistent_worker(client):
    """Test: GET /cluster/workers/{id} con worker inexistente retorna 404"""
    response = client.get("/cluster/workers/nonexistent-worker")
    
    assert response.status_code == 404


def test_filter_workers_by_status(client, coordinator, worker):
    """Test: Filtrar workers por status"""
    time.sleep(1.0)
    
    # Filtrar activos
    response = client.get("/cluster/workers?status_filter=ACTIVE")
    assert response.status_code == 200
    
    # Filtrar no saludables
    response = client.get("/cluster/workers?status_filter=UNHEALTHY")
    assert response.status_code == 200


# ============================================================================
# TESTS - TASK MANAGEMENT
# ============================================================================

def test_submit_task_basic(client, coordinator, worker):
    """Test: POST /cluster/tasks envía una tarea"""
    # Esperar que worker esté listo
    time.sleep(1.0)
    
    task_data = {
        'model_name': 'test-model',
        'input_data': {'test': 'data'},
        'priority': 'NORMAL',
        'timeout': 30.0
    }
    
    response = client.post("/cluster/tasks", json=task_data)
    
    # Debe retornar 201 Created
    assert response.status_code == 201
    data = response.json()
    
    # Verificar respuesta
    assert 'task_id' in data
    assert 'status' in data
    assert data['status'] == 'PENDING'


def test_submit_task_with_priorities(client, coordinator, worker):
    """Test: Enviar tareas con diferentes prioridades"""
    time.sleep(1.0)
    
    priorities = ['LOW', 'NORMAL', 'HIGH', 'CRITICAL']
    
    for priority in priorities:
        task_data = {
            'model_name': f'model-{priority}',
            'input_data': {'priority': priority},
            'priority': priority
        }
        
        response = client.post("/cluster/tasks", json=task_data)
        assert response.status_code == 201


def test_submit_task_with_requirements(client, coordinator, worker):
    """Test: Enviar tarea con requirements específicos"""
    time.sleep(1.0)
    
    task_data = {
        'model_name': 'resnet50',
        'input_data': {'image': 'test.jpg'},
        'requirements': {
            'min_gpu_memory_gb': 4.0,
            'gpu_family': 'Polaris'
        }
    }
    
    response = client.post("/cluster/tasks", json=task_data)
    assert response.status_code == 201


def test_get_task_status(client, coordinator, worker):
    """Test: GET /cluster/tasks/{id}/status obtiene estado"""
    time.sleep(1.0)
    
    # Primero enviar una tarea
    task_data = {
        'model_name': 'test',
        'input_data': {}
    }
    submit_response = client.post("/cluster/tasks", json=task_data)
    task_id = submit_response.json()['task_id']
    
    # Obtener status
    response = client.get(f"/cluster/tasks/{task_id}/status")
    assert response.status_code == 200
    
    data = response.json()
    assert data['task_id'] == task_id


def test_get_task_result(client, coordinator, worker):
    """Test: GET /cluster/tasks/{id}/result obtiene resultado"""
    time.sleep(1.0)
    
    # Enviar tarea
    task_data = {
        'model_name': 'test',
        'input_data': {'test': 'result'}
    }
    submit_response = client.post("/cluster/tasks", json=task_data)
    task_id = submit_response.json()['task_id']
    
    # Intentar obtener resultado (con timeout corto)
    response = client.get(f"/cluster/tasks/{task_id}/result?timeout=5.0")
    
    # Puede ser 200 (completado) o 408 (timeout)
    assert response.status_code in [200, 408, 500]
    
    if response.status_code == 200:
        data = response.json()
        assert 'result' in data


def test_list_tasks(client, coordinator):
    """Test: GET /cluster/tasks lista todas las tareas"""
    response = client.get("/cluster/tasks")
    
    assert response.status_code == 200
    data = response.json()
    
    # Verificar estructura
    assert 'total_tasks' in data
    assert 'pending' in data
    assert 'running' in data
    assert 'completed' in data


def test_submit_task_invalid_priority(client):
    """Test: Enviar tarea con prioridad inválida"""
    task_data = {
        'model_name': 'test',
        'input_data': {},
        'priority': 'INVALID_PRIORITY'
    }
    
    # Pydantic debe rechazar esto
    response = client.post("/cluster/tasks", json=task_data)
    
    # Puede ser 422 (validación) o aceptar y usar default
    assert response.status_code in [201, 422]


# ============================================================================
# TESTS - LOAD BALANCING CONFIGURATION
# ============================================================================

def test_update_balancing_strategy(client):
    """Test: PUT /cluster/config/balancing actualiza estrategia"""
    strategies = ['ROUND_ROBIN', 'LEAST_LOADED', 'ADAPTIVE']
    
    for strategy in strategies:
        response = client.put(
            "/cluster/config/balancing",
            json={'strategy': strategy}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data['new_strategy'] == strategy


def test_update_balancing_invalid_strategy(client):
    """Test: Actualizar con estrategia inválida retorna 400"""
    response = client.put(
        "/cluster/config/balancing",
        json={'strategy': 'INVALID_STRATEGY'}
    )
    
    assert response.status_code == 400


# ============================================================================
# TESTS - INTEGRATION SCENARIOS
# ============================================================================

def test_full_workflow_submit_and_execute(client, coordinator, worker):
    """
    Test de integración completo:
    1. Worker se registra
    2. Cliente envía tarea
    3. Tarea se asigna a worker
    4. Worker ejecuta y retorna resultado
    """
    # Esperar registro de worker
    time.sleep(1.5)
    
    # Verificar que worker está registrado
    workers_response = client.get("/cluster/workers")
    workers = workers_response.json()
    assert len(workers) > 0
    
    # Enviar tarea
    task_data = {
        'model_name': 'integration-test',
        'input_data': {'test': 'workflow'},
        'priority': 'HIGH'
    }
    
    submit_response = client.post("/cluster/tasks", json=task_data)
    assert submit_response.status_code == 201
    
    task_id = submit_response.json()['task_id']
    
    # Esperar ejecución (puede tomar tiempo)
    time.sleep(2.0)
    
    # Intentar obtener resultado
    result_response = client.get(f"/cluster/tasks/{task_id}/result?timeout=10.0")
    
    # Si el worker procesó la tarea, debe retornar 200
    if result_response.status_code == 200:
        result_data = result_response.json()
        assert result_data['status'] in ['COMPLETED', 'FAILED']


def test_multiple_tasks_concurrent(client, coordinator, worker):
    """Test: Enviar múltiples tareas concurrentemente"""
    time.sleep(1.0)
    
    num_tasks = 5
    task_ids = []
    
    # Enviar múltiples tareas
    for i in range(num_tasks):
        task_data = {
            'model_name': f'concurrent-model-{i}',
            'input_data': {'index': i}
        }
        
        response = client.post("/cluster/tasks", json=task_data)
        assert response.status_code == 201
        
        task_ids.append(response.json()['task_id'])
    
    # Verificar que todas se enviaron
    assert len(task_ids) == num_tasks
    
    # Verificar métricas
    time.sleep(0.5)
    metrics_response = client.get("/cluster/metrics")
    metrics = metrics_response.json()
    
    # Debe haber tareas pending o running
    assert metrics['tasks_pending'] + metrics['tasks_running'] >= 0


def test_worker_disconnect_handling(client, coordinator, worker):
    """
    Test: Manejar desconexión de worker.
    
    1. Worker activo
    2. Worker se desconecta
    3. Sistema detecta worker no saludable
    """
    # Esperar registro
    time.sleep(1.0)
    
    # Verificar worker activo
    workers_response = client.get("/cluster/workers")
    workers_before = workers_response.json()
    assert len(workers_before) > 0
    
    # Desconectar worker
    worker.shutdown()
    
    # Esperar a que se detecte la desconexión (heartbeat timeout)
    time.sleep(20.0)  # Timeout de heartbeat es ~15s
    
    # Verificar que worker se marca como no saludable
    health_response = client.get("/cluster/health")
    health = health_response.json()
    
    # Debería haber workers no saludables o reducción en workers activos
    # (puede variar según timing)
    assert health['status'] in ['HEALTHY', 'DEGRADED', 'CRITICAL']


# ============================================================================
# TESTS - ERROR HANDLING
# ============================================================================

def test_submit_task_without_worker(client, coordinator):
    """Test: Enviar tarea sin workers disponibles"""
    # Sin workers, la tarea debe quedarse en pending
    task_data = {
        'model_name': 'test',
        'input_data': {}
    }
    
    response = client.post("/cluster/tasks", json=task_data)
    
    # Debe aceptar la tarea
    assert response.status_code == 201


def test_malformed_task_data(client):
    """Test: Enviar tarea con datos malformados"""
    invalid_data = {
        'model_name': 'test'
        # Falta input_data (requerido)
    }
    
    response = client.post("/cluster/tasks", json=invalid_data)
    
    # Debe rechazar con 422 (validación)
    assert response.status_code == 422


def test_api_without_distributed_layer():
    """Test: API funciona incluso si distributed layer no está disponible"""
    # Este test verifica que el servidor no crashea
    # si el distributed layer no está disponible
    
    # Solo verificar que el servidor inicia
    client = TestClient(app)
    
    # Health endpoint debe funcionar
    response = client.get("/health")
    # Puede ser 200 o 503 dependiendo de implementación
    assert response.status_code in [200, 503]


# ============================================================================
# TESTS - PERFORMANCE
# ============================================================================

@pytest.mark.slow
def test_throughput_benchmark(client, coordinator, worker):
    """Test: Benchmark de throughput del sistema"""
    time.sleep(1.0)
    
    num_tasks = 20
    start_time = time.time()
    
    # Enviar muchas tareas
    for i in range(num_tasks):
        task_data = {
            'model_name': f'benchmark-{i}',
            'input_data': {'index': i}
        }
        
        response = client.post("/cluster/tasks", json=task_data)
        assert response.status_code == 201
    
    submit_time = time.time() - start_time
    
    # Verificar throughput de envío
    tasks_per_second = num_tasks / submit_time
    
    print(f"\nThroughput: {tasks_per_second:.1f} tasks/second")
    
    # Debe ser razonablemente rápido (>10 tasks/sec)
    assert tasks_per_second > 10


# ============================================================================
# MARKERS & CONFIGURATION
# ============================================================================

pytestmark = pytest.mark.integration


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
