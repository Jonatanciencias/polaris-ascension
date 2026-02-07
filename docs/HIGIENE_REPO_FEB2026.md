# Higiene de Repositorio - Febrero 2026

## 1) Clasificacion de archivos locales

### Core (deben versionarse si cambian)
- `src/**` (codigo fuente)
- `tests/test_*.py` fuera de `tests/legacy/`
- `test_production_system.py`
- `src/ml_models/kernel_selector_dataset.json`
- `src/ml_models/kernel_selector_model.pkl`
- `infrastructure/.dockerignore`

### Local/artefactos (no deben versionarse)
- caches: `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`
- entorno: `venv/`, `.venv/`
- resultados: `results/`, `benchmark_data/`, `scripts/results/`
- experimentacion legacy: `tests/legacy/test_*.py`, `scripts/test_*.py`
- artefactos de investigacion: `research/tile_20_investigation/*.json|*.csv|*.pkl`
- dashboards locales: `infrastructure/grafana/dashboards/`
- secretos: `.env*`, `configs/api_keys.json`, `*.pem`, `*.key`

## 2) Git add recomendado

### Commit de higiene (este cambio)
```bash
git add .gitignore docs/HIGIENE_REPO_FEB2026.md
```

### Commit funcional (si aplica en la siguiente fase)
```bash
git add README.md \
  scripts/benchmark_gcn4_optimized.py \
  scripts/benchmark_phase3_reproducible.py \
  scripts/quick_validation.py \
  src/benchmarking/__init__.py \
  src/benchmarking/production_kernel_benchmark.py \
  src/benchmarking/reporting.py \
  src/cli.py \
  test_production_system.py \
  src/ml_models/kernel_selector_dataset.json \
  src/ml_models/kernel_selector_model.pkl \
  tests/test_advanced_memory_manager.py \
  tests/test_calibrated_selector.py \
  tests/test_intelligent_selector.py \
  tests/test_optimized_kernel_engine.py \
  tests/test_system_integration.py \
  infrastructure/.dockerignore
```

## 3) Estado estructural

- Estructura principal correcta: `src/`, `scripts/`, `tests/`, `docs/`, `research/`, `infrastructure/`.
- `tests/unit`, `tests/integration`, `tests/benchmark` estan vacios actualmente (pendiente de consolidacion futura).
- `docs/api_reference`, `docs/architecture`, `docs/benchmarks`, `docs/development`, `docs/techniques` estan creados pero vacios.
