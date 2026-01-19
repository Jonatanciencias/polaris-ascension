# 🚀 START HERE - Session 18: Production Hardening

**Fecha de inicio**: 19+ de Enero de 2026  
**Prerequisito**: Session 17 completada ✅  
**Objetivo**: Completar CAPA 3 (SDK) al 100%  
**Duración estimada**: 6-8 horas  
**Prioridad**: ALTA - Production readiness

---

## 🎯 Objetivo de Session 18

**Transformar el REST API de Session 17 en un sistema production-grade con CI/CD, monitoring avanzado, load testing y security hardening.**

### Resultado Esperado
- CAPA 3 (SDK): 90% → **100%** ✅
- Overall progress: 58% → **62%**
- Production deployment completo y seguro
- Monitoring dashboards operacionales
- CI/CD pipeline automatizado
- Performance validado bajo carga

---

## 📋 Estado Actual (Post Session 17)

### ✅ Lo que YA tenemos
- REST API funcionando (FastAPI + 8 endpoints)
- Docker deployment (multi-stage, GPU support)
- Prometheus metrics (8 metrics básicas)
- Tests comprehensivos (26/26 passing)
- OpenAPI documentation
- Demo client funcional

### 🔄 Lo que nos falta para 100%
1. **CI/CD Pipeline** - Automatización de testing/deployment
2. **Grafana Dashboards** - Visualización de métricas
3. **Alert Rules** - Notificaciones automáticas de problemas
4. **Load Testing** - Validación de performance bajo carga
5. **Security Hardening** - HTTPS, autenticación, rate limiting

---

## 🗺️ Plan de Trabajo Session 18

### **Fase 1: CI/CD Pipeline** (3 horas) - PRIORIDAD ALTA

**Objetivo**: Automatizar testing, building y deployment

#### Tareas
```
[ ] 1.1 - Crear .github/workflows/ci.yml
        ├─ Trigger: push, pull_request
        ├─ Jobs: test, lint, build
        ├─ Matrix strategy: Python 3.8, 3.9, 3.10
        └─ Upload coverage reports

[ ] 1.2 - Crear .github/workflows/docker.yml
        ├─ Build Docker image en cada push
        ├─ Tag con SHA y version
        ├─ Push a Docker Hub/GitHub Registry
        └─ Multi-platform build (amd64, arm64)

[ ] 1.3 - Crear .github/workflows/deploy.yml
        ├─ Deploy to staging (auto en push a main)
        ├─ Deploy to production (manual approval)
        ├─ Rollback strategy
        └─ Health check post-deployment

[ ] 1.4 - Configurar secrets en GitHub
        ├─ DOCKER_USERNAME
        ├─ DOCKER_PASSWORD
        ├─ DEPLOY_SSH_KEY (si aplica)
        └─ SLACK_WEBHOOK (notificaciones)

[ ] 1.5 - Tests del pipeline
        ├─ Hacer commit pequeño y verificar
        ├─ Verificar que tests corren
        ├─ Verificar que Docker build funciona
        └─ Documentar en README
```

**Archivos a crear**:
- `.github/workflows/ci.yml` (~150 líneas)
- `.github/workflows/docker.yml` (~120 líneas)
- `.github/workflows/deploy.yml` (~180 líneas)
- `.github/workflows/lint.yml` (~80 líneas)

**Comandos útiles**:
```bash
# Test workflow locally (con act)
act push --secret-file .secrets

# Verificar sintaxis YAML
yamllint .github/workflows/*.yml

# Ver logs de GitHub Actions
gh run list
gh run view <run-id>
```

---

### **Fase 2: Advanced Monitoring** (2 horas) - PRIORIDAD ALTA

**Objetivo**: Dashboards de Grafana y alertas de Prometheus

#### Tareas
```
[ ] 2.1 - Crear Grafana dashboards
        ├─ Dashboard 1: API Overview
        │   ├─ Request rate (req/s)
        │   ├─ Response latency (p50, p95, p99)
        │   ├─ Error rate (%)
        │   └─ Active connections
        │
        ├─ Dashboard 2: Model Inference
        │   ├─ Inference latency por modelo
        │   ├─ Throughput (inferences/s)
        │   ├─ Model load/unload events
        │   └─ Queue size
        │
        ├─ Dashboard 3: System Resources
        │   ├─ CPU usage (%)
        │   ├─ RAM usage (MB/GB)
        │   ├─ GPU memory (MB/GB)
        │   └─ Disk I/O
        │
        ├─ Dashboard 4: Docker Health
        │   ├─ Container status
        │   ├─ Restart count
        │   ├─ Network traffic
        │   └─ Volume usage
        │
        └─ Dashboard 5: Business Metrics
            ├─ Models loaded
            ├─ Total predictions
            ├─ Success rate
            └─ Average latency trend

[ ] 2.2 - Configurar Prometheus alert rules
        ├─ High error rate (>5% in 5m)
        ├─ High latency (p95 >100ms)
        ├─ API down (no requests in 1m)
        ├─ GPU memory critical (>90%)
        ├─ Container restarting
        └─ Disk space low (<10%)

[ ] 2.3 - Configurar Alertmanager
        ├─ Slack/Discord notifications
        ├─ Email notifications
        ├─ Grouping y throttling
        └─ Runbook links

[ ] 2.4 - Añadir log aggregation (opcional)
        ├─ Loki para logs
        ├─ Integration con Grafana
        └─ Log queries y filters
```

**Archivos a crear**:
- `grafana/dashboards/api-overview.json` (~300 líneas)
- `grafana/dashboards/model-inference.json` (~250 líneas)
- `grafana/dashboards/system-resources.json` (~280 líneas)
- `grafana/dashboards/docker-health.json` (~220 líneas)
- `grafana/dashboards/business-metrics.json` (~200 líneas)
- `prometheus/alerts.yml` (~150 líneas)
- `alertmanager/config.yml` (~80 líneas)
- `docker-compose.monitoring.yml` (extended, ~100 líneas)

**Comandos útiles**:
```bash
# Iniciar stack completo con monitoring
docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d

# Acceder a Grafana
open http://localhost:3000
# Usuario: admin / Password: admin

# Acceder a Prometheus
open http://localhost:9090

# Verificar alerts activas
curl http://localhost:9090/api/v1/alerts

# Importar dashboard a Grafana
curl -X POST http://admin:admin@localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @grafana/dashboards/api-overview.json
```

---

### **Fase 3: Load Testing** (2 horas) - PRIORIDAD MEDIA

**Objetivo**: Validar performance bajo carga y encontrar límites

#### Tareas
```
[ ] 3.1 - Instalar y configurar Locust
        └─ pip install locust

[ ] 3.2 - Crear locustfile.py con scenarios
        ├─ Scenario 1: Health check (warm-up)
        ├─ Scenario 2: Model loading (setup)
        ├─ Scenario 3: Light load (10 users, 1 req/s)
        ├─ Scenario 4: Medium load (50 users, 10 req/s)
        ├─ Scenario 5: Heavy load (200 users, 50 req/s)
        └─ Scenario 6: Spike test (0→500 users)

[ ] 3.3 - Ejecutar tests y recolectar métricas
        ├─ Response times (p50, p95, p99)
        ├─ Throughput (req/s)
        ├─ Error rate
        ├─ Resource usage (CPU, RAM, GPU)
        └─ Bottleneck identification

[ ] 3.4 - Crear scripts de automation
        ├─ run_load_tests.sh
        ├─ analyze_results.py
        └─ generate_report.py

[ ] 3.5 - Optimizaciones basadas en resultados
        ├─ Ajustar worker count
        ├─ Tune batch sizes
        ├─ Connection pooling
        └─ Caching strategies
```

**Archivos a crear**:
- `tests/load/locustfile.py` (~400 líneas)
- `tests/load/scenarios/` (5 scenarios, ~100 líneas c/u)
- `scripts/run_load_tests.sh` (~80 líneas)
- `scripts/analyze_load_results.py` (~150 líneas)

**Comandos útiles**:
```bash
# Instalar Locust
pip install locust

# Ejecutar load test (CLI)
locust -f tests/load/locustfile.py \
       --host http://localhost:8000 \
       --users 100 \
       --spawn-rate 10 \
       --run-time 5m \
       --headless \
       --csv results/load_test

# Ejecutar con Web UI
locust -f tests/load/locustfile.py \
       --host http://localhost:8000
# Abrir: http://localhost:8089

# Analizar resultados
python scripts/analyze_load_results.py results/load_test_stats.csv

# Ver métricas en tiempo real
watch -n 1 'curl -s http://localhost:8000/metrics | grep inference'
```

---

### **Fase 4: Security Hardening** (1 hora) - PRIORIDAD MEDIA

**Objetivo**: Asegurar el API para producción

#### Tareas
```
[ ] 4.1 - Implementar autenticación
        ├─ JWT token-based authentication
        ├─ API keys (header: X-API-Key)
        ├─ Middleware de autenticación
        └─ Endpoints de login/logout

[ ] 4.2 - Implementar rate limiting
        ├─ slowapi library
        ├─ Límites por IP: 100 req/min
        ├─ Límites por API key: 1000 req/min
        └─ Respuesta 429 con Retry-After

[ ] 4.3 - Configurar HTTPS/TLS
        ├─ Certificados SSL (Let's Encrypt)
        ├─ Nginx reverse proxy (opcional)
        ├─ Redirect HTTP → HTTPS
        └─ HSTS headers

[ ] 4.4 - Security headers
        ├─ X-Content-Type-Options: nosniff
        ├─ X-Frame-Options: DENY
        ├─ X-XSS-Protection: 1; mode=block
        ├─ Content-Security-Policy
        └─ Strict-Transport-Security

[ ] 4.5 - Input validation hardening
        ├─ File upload limits (size, type)
        ├─ Request size limits
        ├─ Timeout configurations
        └─ SQL injection prevention (N/A)

[ ] 4.6 - Secrets management
        ├─ Environment variables (.env)
        ├─ Docker secrets
        ├─ Vault integration (avanzado)
        └─ Never commit secrets!
```

**Archivos a crear/modificar**:
- `src/api/auth.py` (~250 líneas) - JWT authentication
- `src/api/middleware.py` (~150 líneas) - Rate limiting, security headers
- `src/api/security.py` (~100 líneas) - Security utilities
- `.env.example` (~30 líneas) - Environment template
- `nginx/nginx.conf` (~100 líneas, opcional) - Reverse proxy
- `docker-compose.prod.yml` (~150 líneas) - Production config

**Comandos útiles**:
```bash
# Generar secret key para JWT
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Test autenticación
curl -X POST http://localhost:8000/auth/login \
     -H "Content-Type: application/json" \
     -d '{"username": "admin", "password": "secret"}'

# Test con token
TOKEN="eyJ0eXAi..."
curl http://localhost:8000/models \
     -H "Authorization: Bearer $TOKEN"

# Test rate limiting
for i in {1..150}; do
  curl -s http://localhost:8000/health > /dev/null
  echo "Request $i"
done

# Verificar HTTPS
curl -I https://localhost:8443/health
```

---

## 📦 Estructura de Archivos a Crear

```
Session 18 - New Files:
├── .github/
│   └── workflows/
│       ├── ci.yml                     # CI pipeline
│       ├── docker.yml                 # Docker builds
│       ├── deploy.yml                 # Deployment automation
│       └── lint.yml                   # Code quality
│
├── grafana/
│   ├── dashboards/
│   │   ├── api-overview.json
│   │   ├── model-inference.json
│   │   ├── system-resources.json
│   │   ├── docker-health.json
│   │   └── business-metrics.json
│   └── provisioning/
│       ├── dashboards.yml
│       └── datasources.yml
│
├── prometheus/
│   ├── alerts.yml                     # Alert rules
│   └── rules.yml                      # Recording rules
│
├── alertmanager/
│   └── config.yml                     # Alertmanager config
│
├── tests/
│   └── load/
│       ├── locustfile.py             # Main locust file
│       ├── scenarios/
│       │   ├── health_check.py
│       │   ├── model_loading.py
│       │   ├── inference_light.py
│       │   ├── inference_heavy.py
│       │   └── spike_test.py
│       └── __init__.py
│
├── scripts/
│   ├── run_load_tests.sh
│   ├── analyze_load_results.py
│   └── generate_report.py
│
├── src/api/
│   ├── auth.py                        # JWT authentication
│   ├── middleware.py                  # Rate limiting, security
│   └── security.py                    # Security utilities
│
├── nginx/                             # Opcional
│   └── nginx.conf                     # Reverse proxy config
│
├── .env.example                       # Environment template
├── docker-compose.monitoring.yml      # Extended monitoring
├── docker-compose.prod.yml            # Production config
└── SESSION_18_PRODUCTION_HARDENING_COMPLETE.md
```

---

## 🔧 Comandos Pre-Session

### Verificar Estado Actual
```bash
# Ver último commit
git log --oneline -1

# Verificar tests passing
pytest tests/test_api.py -v

# Verificar API funcionando
uvicorn src.api.server:app --reload
# En otra terminal:
curl http://localhost:8000/health

# Verificar Docker
docker-compose up -d
curl http://localhost:8000/health
docker-compose down
```

### Preparar Entorno
```bash
# Actualizar dependencias
pip install locust slowapi python-jose[cryptography] passlib[bcrypt]

# Crear directorios
mkdir -p .github/workflows
mkdir -p grafana/dashboards grafana/provisioning
mkdir -p prometheus
mkdir -p alertmanager
mkdir -p tests/load/scenarios
mkdir -p nginx

# Verificar que Docker está corriendo
docker ps

# Verificar espacio en disco
df -h
```

---

## 📊 Checklist de Session 18

### Antes de Empezar
- [ ] Session 17 completa y commiteada ✅
- [ ] API funcionando en http://localhost:8000
- [ ] Tests 26/26 passing
- [ ] Docker Compose operacional
- [ ] Documentación actualizada

### Durante Session 18

**Fase 1: CI/CD** (3h)
- [ ] Crear workflows de GitHub Actions (4 files)
- [ ] Configurar secrets
- [ ] Test pipeline con commit
- [ ] Verificar builds automatizados
- [ ] Documentar en README

**Fase 2: Monitoring** (2h)
- [ ] Crear 5 Grafana dashboards
- [ ] Configurar alert rules (10+)
- [ ] Setup Alertmanager
- [ ] Test notificaciones
- [ ] Documentar acceso y uso

**Fase 3: Load Testing** (2h)
- [ ] Instalar Locust
- [ ] Crear 5+ test scenarios
- [ ] Ejecutar tests y recolectar datos
- [ ] Analizar resultados
- [ ] Optimizaciones basadas en findings

**Fase 4: Security** (1h)
- [ ] Implementar JWT authentication
- [ ] Rate limiting (slowapi)
- [ ] Security headers
- [ ] HTTPS setup (opcional)
- [ ] Secrets management

### Después de Session 18
- [ ] Todos los tests passing (incluye nuevos)
- [ ] Documentación SESSION_18 completa
- [ ] README actualizado
- [ ] NEXT_STEPS actualizado
- [ ] Commit comprehensivo
- [ ] CAPA 3 al 100% ✅

---

## 🎯 Criterios de Éxito

### Requisitos Mínimos (Must Have)
✅ CI/CD pipeline operacional (GitHub Actions)  
✅ Al menos 3 Grafana dashboards funcionales  
✅ Prometheus alerts configuradas (mínimo 5)  
✅ Load testing completado con resultados documentados  
✅ Authentication implementada (JWT o API keys)  
✅ Rate limiting funcional  
✅ Documentación completa de Session 18  

### Objetivos Deseables (Should Have)
✅ 5 Grafana dashboards completos  
✅ Alertmanager con notificaciones  
✅ Log aggregation (Loki)  
✅ HTTPS/TLS configurado  
✅ Nginx reverse proxy  
✅ Performance optimizations aplicadas  

### Extras Opcionales (Nice to Have)
⭐ Distributed tracing (Jaeger)  
⭐ Automated rollback en CD  
⭐ Multi-region deployment  
⭐ A/B testing infrastructure  
⭐ Chaos engineering tests  

---

## 📚 Referencias Útiles

### Session 17 (Completada)
- [SESSION_17_REST_API_COMPLETE.md](SESSION_17_REST_API_COMPLETE.md)
- [src/api/server.py](src/api/server.py)
- [src/api/schemas.py](src/api/schemas.py)
- [src/api/monitoring.py](src/api/monitoring.py)
- [docker-compose.yml](docker-compose.yml)
- [tests/test_api.py](tests/test_api.py)

### Documentación Proyecto
- [README.md](README.md) - Overview
- [STRATEGIC_ROADMAP.md](STRATEGIC_ROADMAP.md) - Roadmap general
- [NEXT_STEPS.md](NEXT_STEPS.md) - Próximos pasos
- [PROJECT_STATUS.md](PROJECT_STATUS.md) - Estado actual

### Recursos Externos
- **GitHub Actions**: https://docs.github.com/en/actions
- **Grafana Dashboards**: https://grafana.com/docs/grafana/latest/dashboards/
- **Prometheus Alerts**: https://prometheus.io/docs/prometheus/latest/configuration/alerting_rules/
- **Locust Load Testing**: https://docs.locust.io/
- **FastAPI Security**: https://fastapi.tiangolo.com/tutorial/security/
- **slowapi Rate Limiting**: https://github.com/laurents/slowapi

---

## 🚨 Notas Importantes

### ⚠️ Antes de Empezar
1. **Backup del código actual**: `git tag session-17-complete`
2. **Verificar que Session 17 funciona**: Tests passing, API running
3. **Leer este documento COMPLETO** antes de codear
4. **Tener GitHub repo configurado** si vas a usar GitHub Actions

### 💡 Tips para Session 18
1. **Empezar por CI/CD**: Es la base para todo lo demás
2. **Testing incremental**: Verificar cada componente antes de seguir
3. **No commitear secrets**: Usar .env y .env.example
4. **Documentar mientras trabajas**: No dejar para el final
5. **Mantener Session 17 funcionando**: No romper lo que ya funciona

### 🐛 Troubleshooting Común
- **GitHub Actions no corre**: Verificar permisos del repo
- **Grafana no muestra datos**: Verificar datasource Prometheus
- **Locust errores de conexión**: Verificar que API está corriendo
- **Rate limiting demasiado estricto**: Ajustar límites en código
- **Docker build falla**: Verificar espacio en disco

---

## ✅ Comando de Inicio

**Cuando estés listo para empezar Session 18:**

```bash
# 1. Crear branch (opcional pero recomendado)
git checkout -b session-18-production-hardening

# 2. Verificar que todo funciona
pytest tests/test_api.py -v
docker-compose up -d
curl http://localhost:8000/health

# 3. Crear directorios
mkdir -p .github/workflows grafana/dashboards prometheus alertmanager tests/load/scenarios

# 4. Instalar nuevas dependencias
pip install locust slowapi python-jose[cryptography] passlib[bcrypt] python-multipart

# 5. Empezar por CI/CD (Fase 1)
# Crear .github/workflows/ci.yml

# 6. ¡A codear! 🚀
```

---

## 📞 Resumen Ejecutivo

**Session 18 en una frase:**  
*"Transformar el REST API funcional de Session 17 en un sistema production-grade con CI/CD, monitoring avanzado, load testing y security."*

**Tiempo estimado**: 6-8 horas  
**Archivos nuevos**: ~20 archivos  
**Líneas de código**: ~3,500 líneas  
**Tests nuevos**: ~10 tests (load testing)  
**Resultado**: CAPA 3 al 100%, proyecto production-ready  

**Orden recomendado**:
1. CI/CD (3h) → Automatización primero
2. Monitoring (2h) → Visibilidad de lo que pasa
3. Load Testing (2h) → Validar que aguanta carga
4. Security (1h) → Asegurar para producción

---

**¡TODO ESTÁ LISTO PARA SESSION 18! 🎉**

**Última actualización**: 18 de Enero de 2026  
**Autor**: @jonatanciencias  
**Status**: Ready to Start ✅
