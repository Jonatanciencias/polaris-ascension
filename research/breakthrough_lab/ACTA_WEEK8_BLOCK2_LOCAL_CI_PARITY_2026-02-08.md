# Acta Week 8 - Block 2 (Local/CI Parity Hardening)

- Date: 2026-02-08
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope: endurecer paridad local/CI integrando runner unificado en `ci.yml`, añadir pruebas unitarias del runner y registrar evidencia formal del bloque.

## Objetivo

1. Integrar `scripts/run_validation_suite.py` en `ci.yml` dentro de un job principal.
2. Cubrir unitariamente el parseo de JSON smoke y manejo de `pytest` exit code `5`.
3. Ejecutar validación formal del bloque y registrar decisión governance.

## Implementacion

Cambios aplicados:

- `.github/workflows/ci.yml`
  - job principal `test-python-310`: nuevo gate:
    - `python scripts/run_validation_suite.py --tier canonical --driver-smoke`

- `tests/test_run_validation_suite.py`
  - tests de `_extract_json_payload` (JSON plano y con prefijo de texto)
  - tests de `_evaluate` para `pytest rc=5` con `allow_no_tests=True/False`
  - test de contrato mínimo para `verify_drivers_json_smoke`

- `.github/workflows/README.md`
  - documentación actualizada para reflejar el gate unificado en CI principal

## Ejecucion Formal

Commands:

- `./venv/bin/python -m pytest -q tests/test_run_validation_suite.py`
- `./venv/bin/python scripts/run_validation_suite.py --tier cpu-fast --allow-no-tests --driver-smoke --report-dir research/breakthrough_lab/week8_validation_discipline`
- `./venv/bin/python scripts/run_validation_suite.py --tier canonical --driver-smoke --report-dir research/breakthrough_lab/week8_validation_discipline`

Artifacts:

- `research/breakthrough_lab/week8_validation_discipline/validation_suite_cpu-fast_20260208_012828.json`
- `research/breakthrough_lab/week8_validation_discipline/validation_suite_cpu-fast_20260208_012828.md`
- `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_012835.json`
- `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_012835.md`

## Resultados

- Unit tests del runner: `5 passed`
- Tier `cpu-fast`: **promote**
  - `pytest`: `22 passed, 61 deselected`
  - schema: `rc=0`
  - driver smoke JSON: parse `ok`, claves requeridas `ok`
- Tier `canonical`: **promote**
  - `pytest`: `83 passed`
  - schema: `rc=0`
  - driver smoke JSON: parse `ok`, claves requeridas `ok`

## Decision Formal

Track `validation_discipline`: **promote**.

Razonamiento:

- La paridad local/CI queda reforzada al ejecutar el mismo runner unificado en el job principal de `ci.yml`.
- El comportamiento crítico del runner (parseo smoke y `rc=5`) quedó cubierto por pruebas unitarias.
- La evidencia formal del bloque cierra en verde en ambos tiers (`cpu-fast`, `canonical`).

## Estado del Bloque

`Week 8 - Block 2` queda ejecutado con evidencia reproducible y decision formal registrada.
