# Acta Week 3 - Block 2 (T4 Approximate GEMM Contract + Fallback)

- Date: 2026-02-07
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope: primer bloque ejecutable de T4 con contrato explícito de error y fallback instrumentado.

## Objetivo

Implementar y validar:
1. contrato de error para ruta aproximada,
2. fallback automático,
3. telemetría formal de cumplimiento y activación de fallback.

## Implementación

Nuevo runner:
- `research/breakthrough_lab/t4_approximate_gemm/run_t4_error_contract.py`

Características:
- política contract-aware con `error_budget`,
- precheck de compresibilidad (`energy threshold`) para fallback preventivo,
- postcheck de error para fallback correctivo,
- registro de razones de fallback (`precheck_low_energy`, `postcheck_contract_violation`),
- artefactos JSON/Markdown reproducibles.

## Ejecución Formal

Command:
- `./venv/bin/python research/breakthrough_lab/t4_approximate_gemm/run_t4_error_contract.py --sessions 3 --sizes 512 1024 1400 --families dense_random compressible_lowrank --target-rank 16 --error-budget 0.005 --precheck-energy-threshold 0.95 --sample-size 64 --seed 42`

Artifacts:
- `research/breakthrough_lab/t4_approximate_gemm/week3_t4_contract_run_20260207_200118.json`
- `research/breakthrough_lab/t4_approximate_gemm/week3_t4_contract_run_20260207_200118.md`

## Resultados

- Contract compliance: `1.000`
- Post-fallback violations: `0.000`
- Fallback rate total: `0.500`

Por familia:
- `compressible_lowrank`:
  - speedup vs exact: `2.972x`
  - fallback rate: `0.000`
  - raw error mean: `0.002500` (dentro de presupuesto `0.005`)
- `dense_random`:
  - speedup vs exact: `0.856x`
  - fallback rate: `1.000`
  - contract escapes: `0`

## Stop Rule

Regla T4:
- detener si violación de contrato > 5% de runs.

Resultado:
- `post_fallback_violation_rate = 0.000`
- stop rule: **not triggered**.

## Decisión Formal

Track `t4_approximate_gemm`: **iterate**.

Razonamiento:
- La seguridad del contrato/fallback queda validada.
- El beneficio de performance es fuerte en cargas compresibles.
- La ruta aún es dependiente de supuestos de compresibilidad/factorización y no está lista para promoción general.

## Estado de Bloque

`Week 3 - Block 2 (T4)` queda ejecutado con evidencia y decisión formal registradas.
