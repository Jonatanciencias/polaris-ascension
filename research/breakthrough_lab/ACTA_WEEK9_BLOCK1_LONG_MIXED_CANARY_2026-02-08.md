# Acta Week 9 - Block 1 (Long Mixed Canary)

- Date: 2026-02-08
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope: 24h-equivalent long mixed canary for `auto_t3_controlled` + `auto_t5_guarded` under queue pressure.

## Objetivo

1. Validar estabilidad sostenida de T3 y T5 en tamanos criticos (`1400`, `2048`).
2. Verificar guardrails operativos bajo presion de cola reproducible.
3. Cerrar bloque con evidencia machine-readable y gate canonico obligatorio.

## Ejecucion Formal

Commands:

- `./venv/bin/python research/breakthrough_lab/run_week9_block1_long_canary.py --batches 24 --sessions-per-batch 1 --iterations-per-session 8 --sizes 1400 2048 --pressure-size 896 --pressure-iterations 3 --pressure-pulses-per-batch 2 --seed 42`
- `./venv/bin/python scripts/run_validation_suite.py --tier canonical --driver-smoke --report-dir research/breakthrough_lab/week8_validation_discipline`

Artifacts:

- `research/breakthrough_lab/run_week9_block1_long_canary.py`
- `research/breakthrough_lab/week9_block1_long_canary_20260208_030816.json`
- `research/breakthrough_lab/week9_block1_long_canary_20260208_030816.md`
- `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_030950.json`
- `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_030950.md`

## Resultados

Queue pressure:
- Pulses requested/completed/failures: `48/48/0`

T3 (`auto_t3_controlled`):
- 1400: avg `876.480` GFLOPS, p95 `6.137` ms, max error `0.0003891`
- 2048: avg `773.579` GFLOPS, p95 `22.147` ms, max error `0.0006104`
- fallback mean/max: `0.0/0.0`
- policy disabled count: `0`

T5 (`auto_t5_guarded`):
- 1400: avg `898.196` GFLOPS, p95 `5.985` ms, max error `0.0003815`
- 2048: avg `778.710` GFLOPS, p95 `21.966` ms, max error `0.0005798`
- overhead mean: `1.339%` (guardrail max `3.0%`)
- false positive mean: `0.0`
- disable events total: `1`

Guardrails:
- Pass: `pressure_failures_zero`, `t3_correctness_bound`, `t3_fallback_rate_mean`, `t3_policy_not_disabled`, `t3_drift_abs_percent`, `t5_correctness_bound`, `t5_overhead_mean_percent`, `t5_false_positive_rate_mean`, `t5_drift_abs_percent`
- Fail: `t5_disable_events_zero` (`observed=1`, `required=0`)

Canonical gate:
- `validation_suite canonical + driver_smoke`: **promote**

## Decision Formal

Tracks:
- `t3_long_canary`: **promote**
- `t5_long_canary`: **iterate**

Block decision:
- **iterate**

Razonamiento:
- El bloque mantiene correctness y overhead en rango, pero no puede promover completo porque T5 reporta un evento de auto-disable en el horizonte largo.
- Se requiere Block 2 para hardening de T5 (histeresis/threshold tuning) y rerun estricto del mismo escenario.

## Estado del Bloque

`Week 9 - Block 1` queda cerrado con decision `iterate`, evidencia reproducible y gate canonico en verde.
