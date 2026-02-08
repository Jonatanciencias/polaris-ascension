# Acta Week 8 - Block 1 (Validation Discipline Hardening)

- Date: 2026-02-08
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope: unificar el runner de validacion local/CI, agregar smoke de diagnostico de drivers y registrar evidencia formal de arranque 2026Q2.

## Objetivo

1. Crear scaffold operativo de Week 8 con plantillas de acta/decision.
2. Introducir smoke test CI para `verify_drivers.py --json` sin dependencia de hardware dedicado.
3. Unificar validaciones de contrato + pytest bajo un runner unico reutilizable.
4. Ejecutar corrida formal de arranque y registrar decision del bloque.

## Implementacion

Cambios aplicados:

- `scripts/run_validation_suite.py`
  - runner unificado por tier (`cpu-fast`, `canonical`, `full`)
  - incluye validacion de contrato (`validate_breakthrough_results.py`)
  - incluye smoke opcional de `verify_drivers.py --json` con parseo y validacion de claves
  - genera artifacts JSON/Markdown para trazabilidad y governance

- `.github/workflows/test-tiers.yml`
  - job `cpu-fast` migra a `scripts/run_validation_suite.py --tier cpu-fast --allow-no-tests --driver-smoke`

- `scripts/verify_drivers.py`
  - ajuste para salida JSON limpia (sin logs de progreso en modo `--json`)

- `research/breakthrough_lab/week8_validation_discipline/`
  - `README.md`
  - `acta_template.md`
  - `decision_template.json`

## Ejecucion Formal

Commands:

- `./venv/bin/python scripts/run_validation_suite.py --tier cpu-fast --allow-no-tests --driver-smoke --report-dir research/breakthrough_lab/week8_validation_discipline`
- `./venv/bin/python scripts/run_validation_suite.py --tier canonical --driver-smoke --report-dir research/breakthrough_lab/week8_validation_discipline`

Artifacts:

- `research/breakthrough_lab/week8_validation_discipline/validation_suite_cpu-fast_20260208_012317.json`
- `research/breakthrough_lab/week8_validation_discipline/validation_suite_cpu-fast_20260208_012317.md`
- `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_012325.json`
- `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_012325.md`

## Resultados

- Tier `cpu-fast`: **promote**
  - schema validation: `rc=0`
  - pytest tier: `17 passed, 61 deselected`, `rc=0`
  - driver smoke JSON: parse `ok`, claves requeridas `ok`, `overall_status=good`

- Tier `canonical`: **promote**
  - schema validation: `rc=0`
  - pytest tier: `78 passed`, `rc=0`
  - driver smoke JSON: parse `ok`, claves requeridas `ok`, `overall_status=good`

## Decision Formal

Track `validation_discipline`: **promote**.

Razonamiento:

- El pipeline de validacion ya tiene un entrypoint unico reutilizable en local y CI.
- El smoke de diagnostico de drivers queda integrado en el tier rapido sin requerir GPU dedicada para evaluar parseo/contrato JSON.
- La corrida formal inicial de Week 8 cierra en verde en tiers `cpu-fast` y `canonical`.

## Estado del Bloque

`Week 8 - Block 1` queda ejecutado con evidencia reproducible y decision formal registrada.
