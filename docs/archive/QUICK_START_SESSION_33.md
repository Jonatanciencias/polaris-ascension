# 🚀 QUICK START - SESSION 33
## Applications Layer Expansion

**Fecha**: 22 de Enero, 2026  
**Objetivo**: Integrar el backend distribuido con aplicaciones (API, CLI, UI)

---

## ⚡ INICIO RÁPIDO

### 1. Verificar Estado del Proyecto
```bash
cd /home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580

# Ver último commit
git log --oneline -1

# Verificar tests de Session 32
python -m pytest tests/test_distributed.py -v --tb=short
```

**Expectativa**: 22/25 tests pasando (88%)

---

### 2. Revisar Arquitectura Distribuida
```bash
# Ver estructura de distributed layer
ls -la src/distributed/

# Archivos clave:
# - communication.py (540 LOC)
# - load_balancing.py (690 LOC)
# - fault_tolerance.py (600 LOC)
# - coordinator.py (820 LOC)
# - worker.py (465 LOC)
```

---

### 3. Probar Sistema Distribuido
```bash
# Demo rápido (simulado)
python examples/distributed_comprehensive_demo.py
```

**Resultado esperado**: Demo de cluster con 3 workers

---

## 🎯 TAREAS PRINCIPALES (Por Orden)

### FASE 1: REST API Enhancement (2-3 horas) ⭐⭐⭐
**Prioridad**: CRÍTICA

#### Tarea 1.1: Crear `src/api/cluster_endpoints.py`
```python
# Endpoints a implementar:
# POST   /cluster/workers/register
# GET    /cluster/workers
# GET    /cluster/workers/{id}/stats
# POST   /cluster/tasks/submit
# GET    /cluster/tasks/{id}/status
# GET    /cluster/tasks/{id}/result
# GET    /cluster/health
# GET    /cluster/metrics
```

**Template**:
```python
from fastapi import APIRouter, Depends, HTTPException
from src.distributed.coordinator import ClusterCoordinator

router = APIRouter(prefix="/cluster", tags=["cluster"])

# Singleton coordinator
_coordinator = None

def get_coordinator():
    global _coordinator
    if _coordinator is None:
        _coordinator = ClusterCoordinator(...)
    return _coordinator

@router.post("/tasks/submit")
async def submit_task(
    payload: dict,
    coordinator: ClusterCoordinator = Depends(get_coordinator)
):
    task_id = coordinator.submit_task(payload)
    return {"task_id": task_id, "status": "submitted"}
```

**LOC Objetivo**: ~400 LOC

---

#### Tarea 1.2: Crear `src/api/websocket_handler.py`
```python
# WebSocket para updates en tiempo real
from fastapi import WebSocket

@app.websocket("/ws/cluster/events")
async def cluster_events(websocket: WebSocket):
    await websocket.accept()
    while True:
        # Stream eventos del coordinator
        event = await coordinator.wait_for_event()
        await websocket.send_json(event)
```

**LOC Objetivo**: ~250 LOC

---

#### Tarea 1.3: Modificar `src/api/server.py`
```python
# Agregar router de cluster
from src.api.cluster_endpoints import router as cluster_router

app.include_router(cluster_router)
```

**LOC Objetivo**: +150 LOC

---

### FASE 2: CLI Expansion (2-3 horas) ⭐⭐
**Prioridad**: ALTA

#### Tarea 2.1: Crear `src/cli_cluster.py`
```python
import click
from src.distributed.coordinator import ClusterCoordinator
from src.distributed.worker import InferenceWorker

@click.group()
def cluster():
    """Cluster management commands."""
    pass

@cluster.command()
@click.option('--bind-address', default='tcp://0.0.0.0:5555')
def start(bind_address):
    """Start cluster coordinator."""
    coordinator = ClusterCoordinator(bind_address=bind_address)
    coordinator.start()
    click.echo(f"Coordinator started on {bind_address}")
    coordinator.wait()

@cluster.command()
def status():
    """Show cluster status."""
    # Connect to coordinator
    # Show workers, tasks, health
    pass
```

**Comandos a implementar**:
```bash
legacygpu cluster start
legacygpu cluster stop
legacygpu cluster status
legacygpu cluster workers list
legacygpu worker start
legacygpu worker stop
legacygpu task submit --model resnet50 --input image.jpg
legacygpu task status <task-id>
legacygpu task result <task-id>
```

**LOC Objetivo**: ~400 LOC

---

#### Tarea 2.2: Crear `src/cli_monitor.py`
```python
@click.command()
@click.option('--coordinator', default='tcp://localhost:5555')
@click.option('--interval', default=2.0)
def monitor(coordinator, interval):
    """Real-time cluster monitoring."""
    while True:
        stats = get_cluster_stats(coordinator)
        
        # Clear screen
        click.clear()
        
        # Display stats
        click.echo("=== Cluster Monitor ===")
        click.echo(f"Workers: {stats['healthy']}/{stats['total']}")
        click.echo(f"Tasks: {stats['completed']} completed")
        
        time.sleep(interval)
```

**LOC Objetivo**: ~200 LOC

---

### FASE 3: Web UI Enhancement (2-3 horas) ⭐
**Prioridad**: MEDIA (si hay tiempo)

#### Tarea 3.1: Crear `src/web_ui/cluster_dashboard.py`
```python
import streamlit as st
from src.api.cluster_endpoints import get_coordinator

st.title("Cluster Dashboard")

# Worker status
coordinator = get_coordinator()
stats = coordinator.get_worker_stats()

col1, col2, col3 = st.columns(3)
col1.metric("Total Workers", stats['total_workers'])
col2.metric("Healthy Workers", stats['healthy_workers'])
col3.metric("Tasks Completed", stats['tasks_completed'])

# Worker table
st.dataframe(stats['workers'])

# Real-time updates (auto-refresh)
st.button("Refresh")
```

**LOC Objetivo**: ~200 LOC

---

### FASE 4: Testing & Documentation (1-2 horas) ⭐⭐⭐
**Prioridad**: CRÍTICA

#### Tarea 4.1: Crear `tests/test_api_cluster.py`
```python
import pytest
from fastapi.testclient import TestClient
from src.api.server import app

@pytest.fixture
def client():
    return TestClient(app)

def test_submit_task(client):
    response = client.post("/cluster/tasks/submit", json={
        "payload": {"model": "resnet50", "input": "data"}
    })
    assert response.status_code == 200
    assert "task_id" in response.json()

def test_get_cluster_health(client):
    response = client.get("/cluster/health")
    assert response.status_code == 200
```

**LOC Objetivo**: ~300 LOC

---

#### Tarea 4.2: Crear `docs/CLUSTER_DEPLOYMENT_GUIDE.md`
```markdown
# Cluster Deployment Guide

## Quick Start

### Start Coordinator
```bash
legacygpu cluster start --bind-address tcp://0.0.0.0:5555
```

### Start Workers (on each machine)
```bash
legacygpu worker start --coordinator tcp://coordinator-ip:5555
```

### Submit Task
```bash
legacygpu task submit --model resnet50 --input image.jpg
```
```

**Páginas**: ~500 líneas

---

## 📋 CHECKLIST DE IMPLEMENTACIÓN

### Must Have (Mínimo para Session 33)
- [ ] `src/api/cluster_endpoints.py` - Endpoints básicos
- [ ] `src/cli_cluster.py` - Comandos principales
- [ ] `tests/test_api_cluster.py` - Tests de integración
- [ ] `docs/CLUSTER_DEPLOYMENT_GUIDE.md` - Guía de deployment

### Should Have (Deseable)
- [ ] `src/api/websocket_handler.py` - Real-time updates
- [ ] `src/cli_monitor.py` - Monitoring interactivo
- [ ] `src/web_ui/cluster_dashboard.py` - Dashboard básico
- [ ] `tests/test_cli_cluster.py` - Tests de CLI

### Nice to Have (Si sobra tiempo)
- [ ] `src/monitoring/prometheus_exporter.py` - Métricas
- [ ] `grafana/dashboards/cluster_overview.json` - Dashboard
- [ ] `src/web_ui/worker_management.py` - Gestión avanzada
- [ ] `examples/cluster_deployment_demo.py` - Demo completo

---

## 🔧 CONFIGURACIÓN INICIAL

### Dependencias Necesarias
```bash
# Ya instaladas en el proyecto
pip install fastapi uvicorn websockets click streamlit

# Opcionales (monitoring)
pip install prometheus-client
```

---

## 🧪 TESTING DURANTE DESARROLLO

### Test Manual Rápido
```bash
# Terminal 1: Start coordinator (simulado)
python -c "
from src.distributed.coordinator import ClusterCoordinator
c = ClusterCoordinator('tcp://127.0.0.1:5555')
print('Coordinator ready')
input('Press Enter to stop...')
"

# Terminal 2: Test API
curl -X POST http://localhost:8000/cluster/tasks/submit \
  -H "Content-Type: application/json" \
  -d '{"payload": {"model": "test"}}'
```

### Test con pytest
```bash
# Test solo la nueva funcionalidad
pytest tests/test_api_cluster.py -v

# Test integración completa
pytest tests/test_api_cluster.py tests/test_distributed.py -v
```

---

## 📊 MÉTRICAS DE ÉXITO

Al finalizar Session 33, deberías tener:

✅ **API REST funcional**
- [ ] 8+ nuevos endpoints
- [ ] WebSocket para real-time (opcional)
- [ ] Tests pasando (>80%)

✅ **CLI expandido**
- [ ] 10+ nuevos comandos
- [ ] `legacygpu cluster/worker/task` funcionando
- [ ] Help text completo

✅ **Web UI básico**
- [ ] Dashboard de cluster (deseable)
- [ ] Visualización de workers (deseable)
- [ ] Métricas en tiempo real (opcional)

✅ **Documentación**
- [ ] Deployment guide completo
- [ ] CLI reference actualizado
- [ ] API reference (docstrings)

---

## ⚠️ PROBLEMAS COMUNES Y SOLUCIONES

### Problema 1: Coordinator no inicia
**Solución**: Verificar que puerto 5555 esté libre
```bash
lsof -i :5555
```

### Problema 2: Workers no se conectan
**Solución**: Verificar firewall y dirección IP
```bash
# Test conectividad
nc -zv <coordinator-ip> 5555
```

### Problema 3: Tests fallan por timeout
**Solución**: Aumentar timeouts en tests
```python
@pytest.mark.timeout(30)  # 30 segundos
def test_long_running():
    pass
```

---

## 🎯 CRITERIOS DE FINALIZACIÓN

Session 33 está completa cuando:

1. ✅ API REST tiene al menos 8 endpoints funcionando
2. ✅ CLI tiene comandos `cluster`, `worker`, y `task`
3. ✅ Tests de integración pasan (>75%)
4. ✅ Deployment guide está completo
5. ✅ Commit realizado con mensaje descriptivo
6. ✅ Session document creado (SESSION_33_COMPLETE.md)

---

## 📖 REFERENCIAS ÚTILES

### Código Existente
- Distributed Layer: `src/distributed/`
- Current API: `src/api/server.py`
- Current CLI: `src/cli.py`
- Current Web UI: `src/web_ui.py`

### Documentación
- Session 32: `SESSION_32_COMPLETE.md`
- Session 33 Plan: `SESSION_33_PLAN.md`
- Project Status: `PROJECT_STATUS.md`

### Ejemplos
- Distributed demo: `examples/distributed_comprehensive_demo.py`
- SDK demo: `examples/sdk_comprehensive_demo.py`

---

## 🚀 COMANDO PARA EMPEZAR

```bash
# 1. Verificar estado
git status
python -m pytest tests/test_distributed.py -v

# 2. Crear primer archivo
touch src/api/cluster_endpoints.py

# 3. Empezar a codear!
# Seguir el orden: API → CLI → Tests → Docs
```

---

## ⏱️ TIEMPO ESTIMADO

| Fase | Tiempo | Prioridad |
|------|--------|-----------|
| API Endpoints | 2-3h | ⭐⭐⭐ CRÍTICA |
| CLI Commands | 2-3h | ⭐⭐ ALTA |
| Web UI | 2-3h | ⭐ MEDIA |
| Testing | 1-2h | ⭐⭐⭐ CRÍTICA |
| Documentation | 1h | ⭐⭐ ALTA |
| **Total** | **8-12h** | **Full Day** |

---

## 💪 ¡LISTO PARA COMENZAR!

Session 33 es la integración final del backend distribuido con las aplicaciones.

**Meta**: Crear una experiencia de usuario completa desde CLI/API hasta el clúster distribuido.

**Resultado**: Plataforma end-to-end lista para producción.

---

**¡Éxito en Session 33!** 🎉

---

*Legacy GPU AI Platform - Session 33*  
*Prepared: January 21, 2026*
