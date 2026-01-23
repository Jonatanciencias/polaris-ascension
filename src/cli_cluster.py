"""
CLI Commands for Cluster Management - Session 33

Comandos click para gestión de clúster distribuido:
- legacygpu cluster: Gestión del coordinador
- legacygpu worker: Gestión de workers
- legacygpu task: Gestión de tareas

Author: Radeon RX 580 AI Framework Team
Date: Enero 22, 2026
"""

import click
import time
import json
import sys
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path

# Imports del distributed layer
try:
    from .distributed.coordinator import ClusterCoordinator, TaskPriority
    from .distributed.worker import InferenceWorker
    from .distributed.load_balancing import LoadBalancingStrategy
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False
    click.echo("⚠️  Warning: Distributed computing layer not available", err=True)

# Para testing con API
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# ============================================================================
# CLUSTER COMMANDS GROUP
# ============================================================================

@click.group(name='cluster')
def cluster_cli():
    """
    Gestión del clúster distribuido.
    
    Comandos para iniciar, detener y monitorear el coordinador del clúster.
    """
    pass


@cluster_cli.command(name='start')
@click.option(
    '--bind-address',
    default='tcp://0.0.0.0:5555',
    help='Dirección de enlace ZMQ (default: tcp://0.0.0.0:5555)'
)
@click.option(
    '--strategy',
    type=click.Choice(['ROUND_ROBIN', 'LEAST_LOADED', 'GPU_MATCH', 'LATENCY', 'ADAPTIVE']),
    default='ADAPTIVE',
    help='Estrategia de balanceo de carga (default: ADAPTIVE)'
)
@click.option(
    '--daemon/--no-daemon',
    default=False,
    help='Ejecutar como daemon en background'
)
def cluster_start(bind_address: str, strategy: str, daemon: bool):
    """
    Inicia el coordinador del clúster.
    
    Ejemplos:
    
        # Iniciar con configuración por defecto
        legacygpu cluster start
        
        # Iniciar con dirección personalizada
        legacygpu cluster start --bind-address tcp://192.168.1.100:5555
        
        # Iniciar con estrategia específica
        legacygpu cluster start --strategy LEAST_LOADED
        
        # Iniciar como daemon
        legacygpu cluster start --daemon
    """
    if not DISTRIBUTED_AVAILABLE:
        click.echo("❌ Error: Distributed layer not available", err=True)
        sys.exit(1)
    
    click.echo("=" * 60)
    click.echo("🚀 Starting Cluster Coordinator")
    click.echo("=" * 60)
    click.echo(f"Bind Address: {bind_address}")
    click.echo(f"Load Balancing: {strategy}")
    click.echo(f"Mode: {'daemon' if daemon else 'foreground'}")
    click.echo("")
    
    try:
        # Crear coordinador
        coordinator = ClusterCoordinator(
            bind_address=bind_address,
            balancing_strategy=strategy
        )
        
        click.echo("✅ Coordinator created successfully")
        click.echo("⏳ Starting coordinator loop...")
        
        if daemon:
            # TODO: Implementar daemonización real
            click.echo("⚠️  Daemon mode not fully implemented - running in foreground")
        
        # Iniciar coordinador (blocking)
        coordinator.start()
        
    except KeyboardInterrupt:
        click.echo("\n⚠️  Keyboard interrupt received")
        click.echo("🛑 Shutting down coordinator...")
        if 'coordinator' in locals():
            coordinator.shutdown()
        click.echo("✅ Coordinator stopped")
    except Exception as e:
        click.echo(f"❌ Error starting coordinator: {e}", err=True)
        sys.exit(1)


@cluster_cli.command(name='stop')
@click.option(
    '--api-url',
    default='http://localhost:8000',
    help='URL de la API REST (default: http://localhost:8000)'
)
def cluster_stop(api_url: str):
    """
    Detiene el coordinador del clúster.
    
    Envía una señal de shutdown al coordinador a través de la API REST.
    
    Ejemplo:
        legacygpu cluster stop
    """
    if not REQUESTS_AVAILABLE:
        click.echo("❌ Error: requests library not available", err=True)
        click.echo("Install with: pip install requests", err=True)
        sys.exit(1)
    
    click.echo("🛑 Stopping cluster coordinator...")
    
    try:
        response = requests.post(f"{api_url}/cluster/shutdown", timeout=5)
        
        if response.status_code == 200:
            click.echo("✅ Coordinator shutdown initiated")
            data = response.json()
            click.echo(f"   Status: {data.get('status', 'unknown')}")
        else:
            click.echo(f"❌ Error: {response.status_code}", err=True)
            click.echo(f"   {response.text}", err=True)
            sys.exit(1)
    
    except requests.exceptions.ConnectionError:
        click.echo("❌ Error: Cannot connect to API server", err=True)
        click.echo(f"   Make sure server is running at {api_url}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cluster_cli.command(name='status')
@click.option(
    '--api-url',
    default='http://localhost:8000',
    help='URL de la API REST (default: http://localhost:8000)'
)
@click.option(
    '--json-output/--human-readable',
    default=False,
    help='Output en formato JSON'
)
def cluster_status(api_url: str, json_output: bool):
    """
    Muestra el estado del clúster.
    
    Obtiene información de salud, workers activos y estadísticas del clúster.
    
    Ejemplos:
    
        # Estado en formato legible
        legacygpu cluster status
        
        # Estado en formato JSON
        legacygpu cluster status --json-output
    """
    if not REQUESTS_AVAILABLE:
        click.echo("❌ Error: requests library not available", err=True)
        sys.exit(1)
    
    try:
        # Obtener health
        health_response = requests.get(f"{api_url}/cluster/health", timeout=5)
        
        if health_response.status_code != 200:
            click.echo(f"❌ Error: {health_response.status_code}", err=True)
            sys.exit(1)
        
        health_data = health_response.json()
        
        # Obtener metrics
        try:
            metrics_response = requests.get(f"{api_url}/cluster/metrics", timeout=5)
            metrics_data = metrics_response.json() if metrics_response.status_code == 200 else {}
        except:
            metrics_data = {}
        
        if json_output:
            # Output JSON
            combined = {
                'health': health_data,
                'metrics': metrics_data
            }
            click.echo(json.dumps(combined, indent=2))
        else:
            # Output legible para humanos
            click.echo("=" * 60)
            click.echo("📊 CLUSTER STATUS")
            click.echo("=" * 60)
            click.echo("")
            
            # Salud general
            status_icon = {
                'HEALTHY': '✅',
                'DEGRADED': '⚠️ ',
                'CRITICAL': '❌'
            }.get(health_data.get('status', 'UNKNOWN'), '❓')
            
            click.echo(f"Status: {status_icon} {health_data.get('status', 'UNKNOWN')}")
            click.echo("")
            
            # Workers
            click.echo("Workers:")
            click.echo(f"  Total:     {health_data.get('total_workers', 0)}")
            click.echo(f"  Healthy:   {health_data.get('healthy_workers', 0)}")
            click.echo(f"  Unhealthy: {health_data.get('unhealthy_workers', 0)}")
            click.echo("")
            
            # Tareas
            click.echo("Tasks:")
            click.echo(f"  Pending:   {health_data.get('pending_tasks', 0)}")
            click.echo(f"  Running:   {health_data.get('running_tasks', 0)}")
            click.echo(f"  Completed: {health_data.get('completed_tasks', 0)}")
            click.echo(f"  Failed:    {health_data.get('failed_tasks', 0)}")
            
            if health_data.get('avg_task_latency_ms'):
                click.echo(f"  Avg Latency: {health_data['avg_task_latency_ms']:.1f}ms")
            
            click.echo("")
            
            # Métricas adicionales
            if metrics_data:
                click.echo("Performance:")
                click.echo(f"  Success Rate: {metrics_data.get('success_rate', 0)*100:.1f}%")
                if metrics_data.get('avg_latency_ms'):
                    click.echo(f"  Avg Latency:  {metrics_data['avg_latency_ms']:.1f}ms")
    
    except requests.exceptions.ConnectionError:
        click.echo("❌ Error: Cannot connect to API server", err=True)
        click.echo(f"   Make sure server is running at {api_url}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cluster_cli.command(name='workers')
@click.option(
    '--api-url',
    default='http://localhost:8000',
    help='URL de la API REST (default: http://localhost:8000)'
)
@click.option(
    '--status-filter',
    type=click.Choice(['ACTIVE', 'UNHEALTHY'], case_sensitive=False),
    help='Filtrar workers por estado'
)
def cluster_workers(api_url: str, status_filter: Optional[str]):
    """
    Lista todos los workers del clúster.
    
    Muestra información detallada de cada worker registrado.
    
    Ejemplos:
    
        # Listar todos los workers
        legacygpu cluster workers
        
        # Solo workers activos
        legacygpu cluster workers --status-filter ACTIVE
    """
    if not REQUESTS_AVAILABLE:
        click.echo("❌ Error: requests library not available", err=True)
        sys.exit(1)
    
    try:
        # Obtener lista de workers
        params = {}
        if status_filter:
            params['status_filter'] = status_filter
        
        response = requests.get(f"{api_url}/cluster/workers", params=params, timeout=5)
        
        if response.status_code != 200:
            click.echo(f"❌ Error: {response.status_code}", err=True)
            sys.exit(1)
        
        workers = response.json()
        
        if not workers:
            click.echo("No workers found")
            return
        
        click.echo("=" * 80)
        click.echo(f"📋 WORKERS ({len(workers)})")
        click.echo("=" * 80)
        click.echo("")
        
        for worker in workers:
            status_icon = '✅' if worker['status'] == 'ACTIVE' else '❌'
            
            click.echo(f"{status_icon} {worker['worker_id']}")
            click.echo(f"   Address: {worker['address']}")
            click.echo(f"   Status:  {worker['status']}")
            
            # Capabilities
            caps = worker.get('capabilities', {})
            if caps:
                click.echo(f"   GPU: {caps.get('gpu_name', 'Unknown')}")
                if 'gpu_memory_gb' in caps:
                    click.echo(f"   Memory: {caps['gpu_memory_gb']:.1f} GB")
            
            # Load
            load = worker.get('current_load', {})
            if load:
                click.echo(f"   Load: {load.get('active_tasks', 0)} tasks")
                if 'gpu_utilization' in load:
                    click.echo(f"   GPU Util: {load['gpu_utilization']*100:.0f}%")
            
            # Stats
            click.echo(f"   Completed: {worker.get('tasks_completed', 0)} tasks")
            if worker.get('tasks_failed', 0) > 0:
                click.echo(f"   Failed: {worker['tasks_failed']} tasks")
            
            click.echo("")
    
    except requests.exceptions.ConnectionError:
        click.echo("❌ Error: Cannot connect to API server", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


# ============================================================================
# WORKER COMMANDS GROUP
# ============================================================================

@click.group(name='worker')
def worker_cli():
    """
    Gestión de workers del clúster.
    
    Comandos para iniciar workers que ejecutan tareas distribuidas.
    """
    pass


@worker_cli.command(name='start')
@click.option(
    '--coordinator',
    default='tcp://localhost:5555',
    help='Dirección del coordinador (default: tcp://localhost:5555)'
)
@click.option(
    '--worker-id',
    help='ID único del worker (auto-generado si no se especifica)'
)
@click.option(
    '--bind-address',
    help='Dirección de enlace del worker (auto si no se especifica)'
)
@click.option(
    '--max-tasks',
    default=4,
    type=int,
    help='Máximo de tareas concurrentes (default: 4)'
)
def worker_start(coordinator: str, worker_id: Optional[str], bind_address: Optional[str], max_tasks: int):
    """
    Inicia un worker para ejecutar tareas.
    
    El worker se conecta al coordinador y espera tareas para ejecutar.
    
    Ejemplos:
    
        # Worker básico
        legacygpu worker start
        
        # Worker con ID personalizado
        legacygpu worker start --worker-id worker-gpu-1
        
        # Conectar a coordinador remoto
        legacygpu worker start --coordinator tcp://192.168.1.100:5555
        
        # Worker con más capacidad
        legacygpu worker start --max-tasks 8
    """
    if not DISTRIBUTED_AVAILABLE:
        click.echo("❌ Error: Distributed layer not available", err=True)
        sys.exit(1)
    
    # Auto-generar worker_id si no se especifica
    if not worker_id:
        import socket
        hostname = socket.gethostname()
        worker_id = f"worker-{hostname}-{int(time.time())}"
    
    # Auto-generar bind_address si no se especifica
    if not bind_address:
        bind_address = f"tcp://0.0.0.0:{5556 + hash(worker_id) % 1000}"
    
    click.echo("=" * 60)
    click.echo("🔧 Starting Worker")
    click.echo("=" * 60)
    click.echo(f"Worker ID: {worker_id}")
    click.echo(f"Coordinator: {coordinator}")
    click.echo(f"Bind Address: {bind_address}")
    click.echo(f"Max Tasks: {max_tasks}")
    click.echo("")
    
    try:
        # Crear worker
        worker = InferenceWorker(
            worker_id=worker_id,
            coordinator_address=coordinator,
            bind_address=bind_address
        )
        
        # Registrar handler de inferencia (simplificado)
        def inference_handler(payload: Dict[str, Any]) -> Any:
            """Handler básico de inferencia"""
            model_name = payload.get('model_name', 'unknown')
            click.echo(f"📥 Received task for model: {model_name}")
            
            # Simular procesamiento
            time.sleep(0.5)
            
            return {
                'status': 'success',
                'model': model_name,
                'worker': worker_id,
                'timestamp': datetime.now().isoformat()
            }
        
        worker.register_handler(inference_handler)
        
        click.echo("✅ Worker created successfully")
        click.echo("⏳ Starting worker loop...")
        click.echo("   Press Ctrl+C to stop")
        click.echo("")
        
        # Iniciar worker (blocking)
        worker.start()
        
    except KeyboardInterrupt:
        click.echo("\n⚠️  Keyboard interrupt received")
        click.echo("🛑 Shutting down worker...")
        if 'worker' in locals():
            worker.shutdown()
        click.echo("✅ Worker stopped")
    except Exception as e:
        click.echo(f"❌ Error starting worker: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


@worker_cli.command(name='list')
@click.option(
    '--api-url',
    default='http://localhost:8000',
    help='URL de la API REST (default: http://localhost:8000)'
)
def worker_list(api_url: str):
    """
    Lista todos los workers (alias de 'cluster workers').
    
    Ejemplo:
        legacygpu worker list
    """
    # Reutilizar comando cluster workers
    from click.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(cluster_workers, ['--api-url', api_url])
    click.echo(result.output)


# ============================================================================
# TASK COMMANDS GROUP
# ============================================================================

@click.group(name='task')
def task_cli():
    """
    Gestión de tareas distribuidas.
    
    Comandos para enviar y monitorear tareas en el clúster.
    """
    pass


@task_cli.command(name='submit')
@click.option(
    '--model',
    required=True,
    help='Nombre del modelo a ejecutar'
)
@click.option(
    '--input',
    'input_file',
    type=click.Path(exists=True),
    help='Archivo de entrada (JSON)'
)
@click.option(
    '--priority',
    type=click.Choice(['LOW', 'NORMAL', 'HIGH', 'CRITICAL']),
    default='NORMAL',
    help='Prioridad de la tarea (default: NORMAL)'
)
@click.option(
    '--timeout',
    default=300.0,
    type=float,
    help='Timeout en segundos (default: 300)'
)
@click.option(
    '--api-url',
    default='http://localhost:8000',
    help='URL de la API REST (default: http://localhost:8000)'
)
@click.option(
    '--wait/--no-wait',
    default=False,
    help='Esperar el resultado de la tarea'
)
def task_submit(model: str, input_file: Optional[str], priority: str, timeout: float, api_url: str, wait: bool):
    """
    Envía una tarea al clúster para ejecución.
    
    Ejemplos:
    
        # Enviar tarea básica
        legacygpu task submit --model resnet50 --input data.json
        
        # Tarea de alta prioridad
        legacygpu task submit --model yolo --priority HIGH
        
        # Enviar y esperar resultado
        legacygpu task submit --model bert --wait
    """
    if not REQUESTS_AVAILABLE:
        click.echo("❌ Error: requests library not available", err=True)
        sys.exit(1)
    
    # Cargar input data
    input_data = {}
    if input_file:
        try:
            with open(input_file, 'r') as f:
                input_data = json.load(f)
        except Exception as e:
            click.echo(f"❌ Error reading input file: {e}", err=True)
            sys.exit(1)
    
    # Preparar request
    task_data = {
        'model_name': model,
        'input_data': input_data,
        'priority': priority,
        'timeout': timeout
    }
    
    click.echo(f"📤 Submitting task: {model} (priority={priority})")
    
    try:
        # Enviar tarea
        response = requests.post(
            f"{api_url}/cluster/tasks",
            json=task_data,
            timeout=10
        )
        
        if response.status_code != 201:
            click.echo(f"❌ Error: {response.status_code}", err=True)
            click.echo(response.text, err=True)
            sys.exit(1)
        
        result = response.json()
        task_id = result.get('task_id')
        
        click.echo(f"✅ Task submitted successfully")
        click.echo(f"   Task ID: {task_id}")
        click.echo(f"   Status: {result.get('status', 'UNKNOWN')}")
        
        if wait:
            click.echo("")
            click.echo("⏳ Waiting for result...")
            
            # Esperar resultado
            result_response = requests.get(
                f"{api_url}/cluster/tasks/{task_id}/result",
                params={'timeout': timeout},
                timeout=timeout + 5
            )
            
            if result_response.status_code == 200:
                task_result = result_response.json()
                click.echo("")
                click.echo("✅ Task completed!")
                click.echo(json.dumps(task_result, indent=2))
            else:
                click.echo(f"❌ Error getting result: {result_response.status_code}", err=True)
    
    except requests.exceptions.ConnectionError:
        click.echo("❌ Error: Cannot connect to API server", err=True)
        sys.exit(1)
    except requests.exceptions.Timeout:
        click.echo("❌ Error: Request timeout", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@task_cli.command(name='list')
@click.option(
    '--api-url',
    default='http://localhost:8000',
    help='URL de la API REST (default: http://localhost:8000)'
)
@click.option(
    '--status',
    type=click.Choice(['PENDING', 'RUNNING', 'COMPLETED', 'FAILED']),
    help='Filtrar por estado'
)
def task_list(api_url: str, status: Optional[str]):
    """
    Lista todas las tareas.
    
    Ejemplo:
        legacygpu task list --status RUNNING
    """
    if not REQUESTS_AVAILABLE:
        click.echo("❌ Error: requests library not available", err=True)
        sys.exit(1)
    
    try:
        params = {}
        if status:
            params['status_filter'] = status
        
        response = requests.get(f"{api_url}/cluster/tasks", params=params, timeout=5)
        
        if response.status_code != 200:
            click.echo(f"❌ Error: {response.status_code}", err=True)
            sys.exit(1)
        
        data = response.json()
        
        click.echo("=" * 60)
        click.echo("📋 TASKS")
        click.echo("=" * 60)
        click.echo(f"Total:     {data.get('total_tasks', 0)}")
        click.echo(f"Pending:   {data.get('pending', 0)}")
        click.echo(f"Running:   {data.get('running', 0)}")
        click.echo(f"Completed: {data.get('completed', 0)}")
        click.echo(f"Failed:    {data.get('failed', 0)}")
        
        # TODO: Mostrar lista de tareas cuando el backend lo implemente
    
    except requests.exceptions.ConnectionError:
        click.echo("❌ Error: Cannot connect to API server", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@task_cli.command(name='status')
@click.argument('task_id')
@click.option(
    '--api-url',
    default='http://localhost:8000',
    help='URL de la API REST (default: http://localhost:8000)'
)
def task_status(task_id: str, api_url: str):
    """
    Obtiene el estado de una tarea específica.
    
    Ejemplo:
        legacygpu task status <task-id>
    """
    if not REQUESTS_AVAILABLE:
        click.echo("❌ Error: requests library not available", err=True)
        sys.exit(1)
    
    try:
        response = requests.get(f"{api_url}/cluster/tasks/{task_id}/status", timeout=5)
        
        if response.status_code != 200:
            click.echo(f"❌ Error: {response.status_code}", err=True)
            sys.exit(1)
        
        data = response.json()
        
        click.echo(f"Task ID: {data['task_id']}")
        click.echo(f"Status:  {data['status']}")
        if data.get('assigned_worker'):
            click.echo(f"Worker:  {data['assigned_worker']}")
        if data.get('error'):
            click.echo(f"Error:   {data['error']}")
    
    except requests.exceptions.ConnectionError:
        click.echo("❌ Error: Cannot connect to API server", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@task_cli.command(name='result')
@click.argument('task_id')
@click.option(
    '--api-url',
    default='http://localhost:8000',
    help='URL de la API REST (default: http://localhost:8000)'
)
@click.option(
    '--timeout',
    default=60.0,
    type=float,
    help='Timeout en segundos (default: 60)'
)
def task_result(task_id: str, api_url: str, timeout: float):
    """
    Obtiene el resultado de una tarea.
    
    Ejemplo:
        legacygpu task result <task-id>
    """
    if not REQUESTS_AVAILABLE:
        click.echo("❌ Error: requests library not available", err=True)
        sys.exit(1)
    
    click.echo(f"⏳ Getting result for task {task_id}...")
    
    try:
        response = requests.get(
            f"{api_url}/cluster/tasks/{task_id}/result",
            params={'timeout': timeout},
            timeout=timeout + 5
        )
        
        if response.status_code == 200:
            data = response.json()
            click.echo("")
            click.echo("✅ Result:")
            click.echo(json.dumps(data, indent=2))
        elif response.status_code == 408:
            click.echo("⏱️  Timeout: Result not available yet", err=True)
            sys.exit(1)
        else:
            click.echo(f"❌ Error: {response.status_code}", err=True)
            sys.exit(1)
    
    except requests.exceptions.ConnectionError:
        click.echo("❌ Error: Cannot connect to API server", err=True)
        sys.exit(1)
    except requests.exceptions.Timeout:
        click.echo("⏱️  Request timeout", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


# ============================================================================
# EXPORTS para integración con CLI principal
# ============================================================================

__all__ = ['cluster_cli', 'worker_cli', 'task_cli']
