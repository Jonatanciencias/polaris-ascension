# ==============================================================================
# GitHub CI/CD Workflows Documentation
# ==============================================================================

# GitHub Actions Workflows - Radeon RX 580 Project

Este directorio contiene los workflows de GitHub Actions para automatización de CI/CD.

## 📋 Workflows Disponibles

### 1. **ci.yml** - Continuous Integration
Workflow principal para testing y code quality.

**Triggers:**
- Push a `main`, `master`, `develop`
- Pull requests
- Manual dispatch

**Jobs:**
- **test-python-38/39/310/311**: Testing en múltiples versiones de Python
- **code-quality**: Linting (Black, isort, flake8, mypy, pylint)
- **security**: Security scanning (Safety, Bandit)
- **build**: Build verification del paquete
- **ci-success**: Gate para branch protection

**Características:**
- ✅ Multi-version Python testing (3.8-3.11)
- ✅ Parallel test execution con pytest-xdist
- ✅ Coverage reporting con comentarios en PRs
- ✅ Timeouts para evitar jobs colgados
- ✅ Artifact uploading para debugging
- ✅ Dependency caching para velocidad

**Duración estimada:** 15-20 minutos

---

### 2. **docker.yml** - Docker Build & Push
Workflow para construir y publicar imágenes Docker.

**Triggers:**
- Push a `main`/`master` (cambios en src/, Dockerfile, etc.)
- Tags de versión (`v*.*.*`)
- Manual dispatch

**Jobs:**
- **docker-build**: Build y push de imagen Docker
- **docker-compose-test**: Testing del stack completo

**Características:**
- ✅ Multi-platform support (amd64, arm64)
- ✅ Automatic tagging (latest, version, SHA)
- ✅ GitHub Container Registry integration
- ✅ Security scanning con Trivy
- ✅ Build cache optimization
- ✅ Health checks post-deployment

**Registry:** `ghcr.io/<tu-username>/radeon_rx_580`

**Duración estimada:** 10-15 minutos

---

### 3. **deploy.yml** - Deployment Automation
Workflow para deployment a staging y production.

**Triggers:**
- Push a `develop` (auto-deploy to staging)
- Manual dispatch para production

**Environments:**
- **Staging**: Auto-deploy desde `develop`
- **Production**: Manual approval requerido

**Jobs:**
- **deploy-staging**: Deployment automático a staging
- **deploy-production**: Deployment manual a production
- **post-deployment-monitoring**: Monitoreo post-deployment

**Características:**
- ✅ Automatic staging deployments
- ✅ Manual production approvals
- ✅ Pre-deployment checks
- ✅ Backup antes de deployment
- ✅ Health checks comprehensivos
- ✅ Automatic rollback en fallos
- ✅ Post-deployment monitoring

**Duración estimada:** 10-20 minutos

---

### 4. **release.yml** - Release Automation
Workflow para automatizar el proceso de release.

**Triggers:**
- Push de tags de versión (`v*.*.*`)
- Manual dispatch

**Jobs:**
- **validate**: Validación de versión semántica
- **build**: Build de distribución (wheel, sdist)
- **changelog**: Generación automática de changelog
- **release**: Creación de GitHub Release
- **publish-pypi**: Publicación a PyPI (opcional)

**Características:**
- ✅ Semantic versioning validation
- ✅ Automatic changelog generation desde commits
- ✅ GitHub Release creation con artifacts
- ✅ Pre-release detection
- ✅ PyPI publishing (comentado por defecto)
- ✅ Build artifact verification

**Duración estimada:** 10-15 minutos

---

### 5. **test-tiers.yml** - CPU/GPU Test Split (Phase 4)
Workflow con separación explícita entre pruebas rápidas CPU y validación GPU/OpenCL.

**Triggers:**
- Push / Pull request: ejecuta solo tier rápido CPU
- Manual dispatch: permite activar tier GPU/OpenCL

**Jobs:**
- **cpu-fast**: `pytest -m "not slow and not gpu and not opencl"`
- **gpu-opencl** (manual): `pytest -m "gpu or opencl"` + bucle anti-flakiness

**Características:**
- ✅ Feedback rápido en CI estándar
- ✅ Validación de hardware en runner dedicado
- ✅ Repetición de pruebas críticas para detectar flakiness

**Duración estimada:** 5-10 min (CPU), 15-45 min (GPU)

---

## 🔧 Configuración Requerida

### GitHub Secrets
Para usar todos los workflows, configura estos secrets en GitHub:

```bash
# Para Docker Hub (opcional)
DOCKERHUB_USERNAME=<tu-username>
DOCKERHUB_TOKEN=<tu-token>

# Para PyPI (opcional, para releases)
PYPI_API_TOKEN=<tu-token>

# GitHub Token (automático)
GITHUB_TOKEN=<auto-generado>
```

### Environment Variables
Actualiza estos valores en los workflows:

**deploy.yml:**
- `url: https://staging.example.com` → Tu URL de staging
- `url: https://production.example.com` → Tu URL de producción

**dependabot.yml:**
- `reviewers: ["jonatanciencias"]` → Tu username

---

## 🚀 Uso

### Ejecutar CI manualmente
```bash
gh workflow run ci.yml
```

### Ejecutar validación GPU/OpenCL manual
```bash
gh workflow run test-tiers.yml -f run_gpu=true
```

### Ejecutar build de Docker
```bash
gh workflow run docker.yml
```

### Deployment a staging (automático en push a develop)
```bash
git push origin develop
```

### Deployment a production (manual)
```bash
gh workflow run deploy.yml -f environment=production -f version=v1.0.0
```

### Crear release
```bash
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0
```

---

## 📊 Status Badges

Agrega estos badges a tu README.md:

```markdown
![CI](https://github.com/<username>/Radeon_RX_580/workflows/CI%20-%20Continuous%20Integration/badge.svg)
![Docker](https://github.com/<username>/Radeon_RX_580/workflows/Docker%20Build%20%26%20Push/badge.svg)
![Deploy](https://github.com/<username>/Radeon_RX_580/workflows/Deploy/badge.svg)
```

---

## 🔒 Security

### Dependabot
Configurado para actualizar automáticamente:
- Python dependencies (weekly)
- GitHub Actions (weekly)
- Docker base images (weekly)

### Security Scanning
- **Safety**: Escaneo de vulnerabilidades en dependencies
- **Bandit**: Escaneo de issues de seguridad en código
- **Trivy**: Escaneo de vulnerabilidades en imágenes Docker

---

## 📈 Optimizaciones

### Caching
- Python dependencies: `cache: 'pip'`
- Docker layers: `cache-from: type=gha`

### Parallel Execution
- Tests: `pytest -n auto` (pytest-xdist)
- Jobs: Múltiples jobs en paralelo

### Timeouts
- CI jobs: 15 minutos
- Deployment: 20 minutos
- Previene jobs colgados

---

## 🐛 Troubleshooting

### CI failing
```bash
# Ver logs
gh run view <run-id>

# Re-ejecutar failed jobs
gh run rerun <run-id> --failed
```

### Docker build failing
```bash
# Build localmente para debugging
docker build -t radeon-rx-580:test .

# Ver logs del workflow
gh run view <run-id> --log
```

### Deployment failing
```bash
# Check health endpoint
curl -f https://staging.example.com/health

# Ver logs del servicio
docker-compose logs api
```

---

## 📚 Recursos

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Build Push Action](https://github.com/docker/build-push-action)
- [Dependabot Configuration](https://docs.github.com/en/code-security/dependabot)

---

**Última actualización:** Enero 2026  
**Versión:** 1.0.0  
**Mantenedor:** Jonathan (@jonatanciencias)
