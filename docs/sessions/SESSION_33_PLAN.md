# SESIÓN 33 - PLAN DE TRABAJO
## Applications Layer Expansion (40% → 75%)

**Fecha Planeada**: 22 de Enero, 2026  
**Prioridad**: Alta  
**Objetivo**: Expandir la capa de aplicaciones integrando el backend distribuido

---

## 🎯 OBJETIVOS PRINCIPALES

### 1. REST API Enhancement (Prioridad: ALTA)
**LOC Objetivo**: +800 LOC  
**Tiempo Estimado**: 2-3 horas

#### Tareas:
- [ ] Integrar backend distribuido con FastAPI
- [ ] Endpoints para gestión de clúster:
  - `POST /cluster/workers/register` - Registrar worker
  - `GET /cluster/workers` - Listar workers
  - `GET /cluster/workers/{id}/stats` - Stats de worker
  - `POST /cluster/tasks/submit` - Enviar tarea distribuida
  - `GET /cluster/tasks/{id}/status` - Estado de tarea
  - `GET /cluster/tasks/{id}/result` - Obtener resultado
- [ ] WebSocket para updates en tiempo real
- [ ] Endpoints de monitoring:
  - `GET /cluster/health` - Salud del clúster
  - `GET /cluster/metrics` - Métricas agregadas
  - `GET /cluster/load-distribution` - Distribución de carga

**Archivos a Crear/Modificar**:
- `src/api/cluster_endpoints.py` (nuevo, ~400 LOC)
- `src/api/websocket_handler.py` (nuevo, ~250 LOC)
- `src/api/server.py` (modificar, +150 LOC)

---

### 2. CLI Expansion (Prioridad: ALTA)
**LOC Objetivo**: +600 LOC  
**Tiempo Estimado**: 1-2 horas

#### Comandos Nuevos:
```bash
# Gestión de clúster
legacygpu cluster start --bind-address tcp://0.0.0.0:5555
legacygpu cluster stop
legacygpu cluster status
legacygpu cluster workers list
legacygpu cluster workers add <address>

# Worker management
legacygpu worker start --coordinator tcp://localhost:5555
legacygpu worker stop
legacygpu worker status

# Task management
legacygpu task submit --model resnet50 --input image.jpg
legacygpu task list
legacygpu task status <task-id>
legacygpu task result <task-id>
legacygpu task cancel <task-id>

# Monitoring
legacygpu monitor cluster
legacygpu monitor workers
legacygpu monitor tasks
```

**Archivos a Crear/Modificar**:
- `src/cli_cluster.py` (nuevo, ~400 LOC)
- `src/cli_monitor.py` (nuevo, ~200 LOC)
- `src/cli.py` (modificar, integraciones)

---

### 3. Web UI Enhancement (Prioridad: MEDIA)
**LOC Objetivo**: +500 LOC  
**Tiempo Estimado**: 2-3 horas

#### Nuevas Páginas/Componentes:
- [ ] **Dashboard de Clúster**:
  - Mapa de workers activos
  - Gráficos de utilización en tiempo real
  - Estado de salud del clúster
  - Historial de tareas

- [ ] **Gestión de Workers**:
  - Tabla de workers con stats
  - Acciones: enable/disable/remove
  - Logs de workers en tiempo real
  - Alertas de fallos

- [ ] **Monitor de Tareas**:
  - Lista de tareas activas/completadas
  - Visualización de cola de prioridad
  - Detalle de ejecución por tarea
  - Tiempos de latencia

- [ ] **Configuración del Clúster**:
  - Selección de estrategia de balanceo
  - Configuración de timeouts
  - Parámetros de retry
  - Límites de recursos

**Archivos a Crear/Modificar**:
- `src/web_ui/cluster_dashboard.py` (nuevo, ~200 LOC)
- `src/web_ui/worker_management.py` (nuevo, ~150 LOC)
- `src/web_ui/task_monitor.py` (nuevo, ~150 LOC)
- `src/web_ui.py` (modificar, integración)

---

### 4. Monitoring & Observability (Prioridad: MEDIA)
**LOC Objetivo**: +400 LOC  
**Tiempo Estimado**: 1-2 horas

#### Componentes:
- [ ] **Prometheus Integration**:
  - Exporter de métricas del coordinador
  - Métricas de workers (CPU, GPU, memoria)
  - Métricas de tareas (latencia, throughput)
  - Métricas de red (bandwidth, latency)

- [ ] **Grafana Dashboards**:
  - Dashboard de overview del clúster
  - Dashboard de performance por worker
  - Dashboard de análisis de tareas
  - Dashboard de alertas

- [ ] **Logging Centralizado**:
  - Agregador de logs de todos los workers
  - Búsqueda y filtrado
  - Niveles de log configurables
  - Rotación automática

**Archivos a Crear**:
- `src/monitoring/prometheus_exporter.py` (nuevo, ~200 LOC)
- `src/monitoring/log_aggregator.py` (nuevo, ~200 LOC)
- `grafana/dashboards/cluster_overview.json` (nuevo)
- `grafana/dashboards/worker_performance.json` (nuevo)

---

### 5. Testing & Documentation (Prioridad: ALTA)
**LOC Objetivo**: +500 LOC  
**Tiempo Estimado**: 1-2 horas

#### Tests:
- [ ] Tests de integración API + Distributed
- [ ] Tests de CLI commands
- [ ] Tests de WebSocket
- [ ] Tests de métricas de Prometheus

#### Documentación:
- [ ] Guía de despliegue de clúster
- [ ] Tutorial de uso del CLI
- [ ] API reference para endpoints nuevos
- [ ] Troubleshooting guide

**Archivos a Crear**:
- `tests/test_api_cluster.py` (nuevo, ~300 LOC)
- `tests/test_cli_cluster.py` (nuevo, ~200 LOC)
- `docs/CLUSTER_DEPLOYMENT_GUIDE.md` (nuevo)
- `docs/CLI_REFERENCE.md` (actualizar)

---

## 📊 MÉTRICAS OBJETIVO

```
Metric                    Current    Target     Gain
─────────────────────────────────────────────────────
Applications LOC          13,214     16,014    +2,800
Completeness              40%        75%       +35 pts
Test Coverage             ~60%       ~75%      +15 pts
API Endpoints             ~20        ~35       +15
CLI Commands              ~15        ~30       +15
```

---

## 🗂️ ESTRUCTURA DE ARCHIVOS NUEVOS

```
src/
├── api/
│   ├── cluster_endpoints.py          # NEW: ~400 LOC
│   ├── websocket_handler.py          # NEW: ~250 LOC
│   └── server.py                     # MODIFY: +150 LOC
├── cli_cluster.py                    # NEW: ~400 LOC
├── cli_monitor.py                    # NEW: ~200 LOC
├── monitoring/
│   ├── __init__.py                   # NEW
│   ├── prometheus_exporter.py        # NEW: ~200 LOC
│   └── log_aggregator.py             # NEW: ~200 LOC
└── web_ui/
    ├── cluster_dashboard.py          # NEW: ~200 LOC
    ├── worker_management.py          # NEW: ~150 LOC
    └── task_monitor.py               # NEW: ~150 LOC

tests/
├── test_api_cluster.py               # NEW: ~300 LOC
└── test_cli_cluster.py               # NEW: ~200 LOC

examples/
└── cluster_deployment_demo.py        # NEW: ~300 LOC

docs/
├── CLUSTER_DEPLOYMENT_GUIDE.md       # NEW
├── CLI_REFERENCE.md                  # UPDATE
└── API_REFERENCE.md                  # UPDATE

grafana/
└── dashboards/
    ├── cluster_overview.json         # NEW
    └── worker_performance.json       # NEW
```

---

## 🔄 FLUJO DE TRABAJO SUGERIDO

### Fase 1: Backend Integration (Mañana)
**Duración**: 2-3 horas

1. **Crear endpoints de clúster** (`cluster_endpoints.py`)
   - Implementar CRUD de workers
   - Implementar submit/status/result de tareas
   - Integrar con coordinator existente

2. **WebSocket para real-time** (`websocket_handler.py`)
   - Stream de eventos del clúster
   - Updates de estado de workers
   - Notificaciones de tareas completadas

3. **Modificar server.py**
   - Registrar nuevos routers
   - Configurar WebSocket
   - Agregar middleware de métricas

### Fase 2: CLI Enhancement (Tarde)
**Duración**: 2-3 horas

1. **Comandos de clúster** (`cli_cluster.py`)
   - Comandos start/stop/status
   - Gestión de workers
   - Gestión de tareas

2. **Comandos de monitoring** (`cli_monitor.py`)
   - Monitor en tiempo real
   - Visualización de stats
   - Alertas configurables

### Fase 3: UI & Monitoring (Opcional si hay tiempo)
**Duración**: 2-3 horas

1. **Web UI dashboards**
   - Dashboard principal de clúster
   - Gestión de workers
   - Monitor de tareas

2. **Prometheus & Grafana**
   - Exporter de métricas
   - Dashboards de Grafana
   - Configuración de alertas

### Fase 4: Testing & Documentation (Final)
**Duración**: 1-2 horas

1. **Tests de integración**
   - API + Distributed
   - CLI commands
   - WebSocket

2. **Documentación**
   - Deployment guide
   - CLI reference
   - API reference

---

## 🎯 PRIORIDADES SI HAY POCO TIEMPO

### Must Have (Mínimo Viable)
1. ✅ REST API endpoints básicos (cluster, tasks)
2. ✅ CLI commands principales (start/stop/submit)
3. ✅ Tests básicos de integración
4. ✅ Documentación mínima

### Should Have (Deseable)
1. ⭐ WebSocket para real-time updates
2. ⭐ Web UI dashboard básico
3. ⭐ Monitoring con Prometheus
4. ⭐ Tests comprehensivos

### Nice to Have (Si sobra tiempo)
1. 💎 Grafana dashboards completos
2. 💎 Web UI avanzado con gráficos
3. 💎 Log aggregation centralizado
4. 💎 Performance profiling

---

## 📝 CHECKLIST DE INICIO DE SESIÓN

Antes de empezar mañana:

- [ ] Revisar código de Session 32 (distributed layer)
- [ ] Verificar que todos los tests pasen
- [ ] Revisar este plan y ajustar prioridades
- [ ] Tener ejemplos de uso del coordinator/worker
- [ ] Tener ambiente de prueba listo

---

## 🎨 EJEMPLOS DE CÓDIGO A IMPLEMENTAR

### Ejemplo 1: Cluster Endpoint
```python
@router.post("/cluster/tasks/submit")
async def submit_task(
    task: TaskSubmission,
    coordinator: ClusterCoordinator = Depends(get_coordinator)
):
    """Submit distributed inference task."""
    task_id = coordinator.submit_task(
        payload=task.payload,
        requirements=task.requirements,
        priority=task.priority
    )
    return {"task_id": task_id, "status": "submitted"}
```

### Ejemplo 2: CLI Command
```python
@click.command()
@click.option('--coordinator', default='tcp://localhost:5555')
def cluster_status(coordinator: str):
    """Show cluster status."""
    coordinator = ClusterCoordinator.connect(coordinator)
    
    stats = coordinator.get_worker_stats()
    click.echo(f"Workers: {stats['healthy_workers']}/{stats['total_workers']}")
    
    task_stats = coordinator.get_task_stats()
    click.echo(f"Tasks: {task_stats['completed']} completed, {task_stats['pending']} pending")
```

### Ejemplo 3: WebSocket Handler
```python
@app.websocket("/ws/cluster/events")
async def cluster_events(websocket: WebSocket):
    """Stream cluster events to client."""
    await websocket.accept()
    
    while True:
        event = await coordinator.get_next_event()
        await websocket.send_json({
            "type": event.type,
            "data": event.data,
            "timestamp": event.timestamp
        })
```

---

## 🚀 RESULTADO ESPERADO AL FINAL DE SESIÓN 33

Al completar esta sesión, el proyecto tendrá:

✅ **API REST completa** con endpoints de clúster  
✅ **CLI expandido** con comandos de gestión distribuida  
✅ **Web UI** con dashboard de monitoring  
✅ **Monitoring** con Prometheus/Grafana  
✅ **Tests** de integración (75%+ coverage)  
✅ **Documentación** completa de deployment  

**Applications Layer: 40% → 75% (+35 puntos)**

---

## 📅 DESPUÉS DE SESIÓN 33

### Sesión 34: Polishing & Integration
- Refinamiento de todas las capas
- Optimizaciones de performance
- Documentación user-facing completa
- Preparación para release

### Sesión 35: Release v0.7.0
- Final testing
- Release notes
- Deployment packages
- Public announcement

---

## 💡 NOTAS IMPORTANTES

1. **Integración clave**: La API debe usar el coordinator de distributed layer
2. **WebSocket opcional**: Si no hay tiempo, REST polling es suficiente
3. **Priorizar funcionalidad**: UI puede ser básico pero funcional
4. **Tests críticos**: API + Distributed integration es must-have
5. **Documentación**: Deployment guide es esencial para usuarios

---

## 🔗 REFERENCIAS ÚTILES

- Session 32 Complete: `SESSION_32_COMPLETE.md`
- Distributed Layer Code: `src/distributed/`
- Current API: `src/api/server.py`
- Current CLI: `src/cli.py`
- Current Web UI: `src/web_ui.py`

---

**Preparado para Sesión 33** ✅  
**Última actualización**: 21 de Enero, 2026  
**Estado**: LISTO PARA COMENZAR

---

*"De sistema distribuido a plataforma completa de producción en una sesión"*
