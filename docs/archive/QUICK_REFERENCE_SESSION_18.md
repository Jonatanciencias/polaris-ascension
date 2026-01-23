# 🚀 Quick Reference - Session 18

**Quick commands and references for Session 18**

---

## ⚡ Pre-Session Commands

```bash
# Verificar estado
git log --oneline -3
pytest tests/test_api.py -v
docker-compose up -d && curl http://localhost:8000/health && docker-compose down

# Preparar directorios
mkdir -p .github/workflows grafana/{dashboards,provisioning} prometheus alertmanager tests/load/scenarios nginx

# Instalar dependencias
pip install locust slowapi python-jose[cryptography] passlib[bcrypt]
```

---

## 🔧 Durante Session 18

### Fase 1: CI/CD (3h)
```bash
# Crear workflows
touch .github/workflows/{ci,docker,deploy,lint}.yml

# Test GitHub Actions locally (con act)
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash
act push

# Ver logs de Actions
gh run list
gh run view <run-id> --log
```

### Fase 2: Monitoring (2h)
```bash
# Iniciar stack con monitoring
docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d

# Accesos
open http://localhost:3000  # Grafana (admin/admin)
open http://localhost:9090  # Prometheus
open http://localhost:9093  # Alertmanager

# Importar dashboard
curl -X POST http://admin:admin@localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @grafana/dashboards/api-overview.json

# Test alerts
curl http://localhost:9090/api/v1/alerts
```

### Fase 3: Load Testing (2h)
```bash
# Instalar Locust
pip install locust

# Run con Web UI
locust -f tests/load/locustfile.py --host http://localhost:8000
# Abrir: http://localhost:8089

# Run headless (5 min, 100 users)
locust -f tests/load/locustfile.py \
       --host http://localhost:8000 \
       --users 100 --spawn-rate 10 \
       --run-time 5m --headless \
       --csv results/load_test

# Analizar resultados
python scripts/analyze_load_results.py results/load_test_stats.csv
```

### Fase 4: Security (1h)
```bash
# Generar secret key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Test JWT auth
curl -X POST http://localhost:8000/auth/login \
     -H "Content-Type: application/json" \
     -d '{"username": "admin", "password": "secret"}'

# Request con token
TOKEN="eyJ0eXAi..."
curl http://localhost:8000/models \
     -H "Authorization: Bearer $TOKEN"

# Test rate limiting (150 requests)
for i in {1..150}; do curl -s http://localhost:8000/health > /dev/null; echo $i; done
```

---

## 📊 Testing Commands

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_api.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run load tests
locust -f tests/load/locustfile.py --host http://localhost:8000 --headless --users 50 --spawn-rate 5 --run-time 2m
```

---

## 🐳 Docker Commands

```bash
# Build image
docker build -t rx580-api:session18 .

# Run container
docker run -d -p 8000:8000 --name rx580-api rx580-api:session18

# Full stack
docker-compose up -d

# Full stack con monitoring
docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d

# Ver logs
docker-compose logs -f api

# Stop todo
docker-compose down

# Clean rebuild
docker-compose down -v
docker-compose build --no-cache
docker-compose up -d
```

---

## 📈 Monitoring Commands

```bash
# Prometheus queries (CLI)
curl 'http://localhost:9090/api/v1/query?query=inference_requests_total'

# Ver métricas del API
curl http://localhost:8000/metrics

# Watch en tiempo real
watch -n 1 'curl -s http://localhost:8000/metrics | grep inference_requests_total'

# Grafana API - List dashboards
curl http://admin:admin@localhost:3000/api/search?query=

# Export dashboard
curl http://admin:admin@localhost:3000/api/dashboards/uid/<dashboard-uid> > backup.json
```

---

## 🔍 Debugging Commands

```bash
# Ver procesos Python
ps aux | grep python

# Ver puertos en uso
lsof -i :8000
netstat -tulpn | grep 8000

# Logs del servidor
tail -f logs/api.log

# Verificar GPU
rocm-smi
watch -n 1 rocm-smi

# Verificar RAM/CPU
htop
free -h
```

---

## 📝 Git Commands

```bash
# Crear branch
git checkout -b session-18-production-hardening

# Ver cambios
git status
git diff

# Add y commit
git add .
git commit -m "Session 18: <descripción>"

# Tag
git tag session-18-complete
git push origin session-18-complete

# Ver historial
git log --oneline -10
```

---

## 🎯 Quick Checks

```bash
# Health check
curl http://localhost:8000/health | jq

# Metrics check
curl http://localhost:8000/metrics | head -20

# Load model
curl -X POST http://localhost:8000/models/load \
     -H "Content-Type: application/json" \
     -d '{"path": "/models/test.onnx", "model_name": "test"}'

# List models
curl http://localhost:8000/models | jq

# Run inference
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"model_name": "test", "inputs": {"input": [[1.0, 2.0, 3.0]]}}'

# OpenAPI docs
open http://localhost:8000/docs
open http://localhost:8000/redoc
```

---

## 📦 Dependencies Check

```bash
# Verificar instaladas
pip list | grep -E "fastapi|uvicorn|locust|slowapi|jose"

# Instalar faltantes
pip install -r requirements.txt

# Freeze actual
pip freeze > requirements.txt

# Check versions
python --version
docker --version
docker-compose --version
```

---

## 🚨 Emergency Commands

```bash
# Stop everything
docker-compose down
pkill -f uvicorn

# Clean Docker
docker system prune -a --volumes

# Reset database (si hay)
rm -f data/*.db

# Restore from backup
git reset --hard HEAD
git clean -fd

# Ver logs de error
grep -i error logs/api.log
journalctl -xe | grep -i error
```

---

## 📊 Performance Check

```bash
# API latency
time curl http://localhost:8000/health

# Multiple requests
ab -n 100 -c 10 http://localhost:8000/health

# Memory usage
docker stats rx580-api --no-stream

# Disk usage
du -sh *
df -h
```

---

## ✅ Session 18 Checklist

```bash
# Pre-session
[ ] git log --oneline -1
[ ] pytest tests/test_api.py -v
[ ] docker-compose up -d && curl http://localhost:8000/health

# During session
[ ] mkdir -p .github/workflows grafana/dashboards prometheus
[ ] pip install locust slowapi python-jose passlib

# Post-session
[ ] pytest tests/ -v
[ ] locust -f tests/load/locustfile.py --headless --users 10 --run-time 1m
[ ] git add . && git commit -m "Session 18: Production Hardening - Complete"
[ ] git tag session-18-complete
```

---

## 📚 Files to Create

```
Session 18 Files:
├── .github/workflows/ci.yml
├── .github/workflows/docker.yml
├── .github/workflows/deploy.yml
├── grafana/dashboards/api-overview.json
├── grafana/dashboards/model-inference.json
├── grafana/dashboards/system-resources.json
├── prometheus/alerts.yml
├── tests/load/locustfile.py
├── src/api/auth.py
├── src/api/middleware.py
└── SESSION_18_PRODUCTION_HARDENING_COMPLETE.md
```

---

## 🎯 Success Criteria

```bash
# CI/CD: GitHub Actions running
gh run list | head -5

# Monitoring: Grafana dashboards
curl -s http://admin:admin@localhost:3000/api/search | jq '. | length'

# Load Testing: Results exist
ls -lh results/load_test_*.csv

# Security: Auth working
curl -X POST http://localhost:8000/auth/login -d '{"username":"test","password":"test"}'

# All tests passing
pytest tests/ -v --tb=short
```

---

**Ver [START_HERE_SESSION_18.md](START_HERE_SESSION_18.md) para guía completa**

**Status**: Ready to Start ✅  
**Fecha**: 18 Enero 2026
