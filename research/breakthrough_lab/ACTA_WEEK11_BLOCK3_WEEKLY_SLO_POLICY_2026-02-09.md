# Acta Week 11 - Block 3 (Versionado de SLO Semanales)

- Date: 2026-02-09
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope:
  - formalizar policy semanal de monitoreo (GFLOPS/p95/overhead/fallback/disable_events),
  - anclar thresholds al baseline de Week 11 Block 2,
  - cerrar con gate canónico obligatorio.

## Objetivo

1. Definir contrato SLO semanal reproducible y versionado.
2. Mantener defaults rollback-safe para operación continua.
3. Dejar base formal para evaluación del replay semanal (Block 4).

## Implementación

Policy formal creada:

- `research/breakthrough_lab/week11_controlled_rollout/policy_week11_block3_weekly_slo_v1.json`

Contenido clave:

- `minimum_snapshots = 6`
- `max_correctness_error = 0.001`
- `max_t3_fallback_rate = 0.08`
- `max_t5_disable_events_total = 0`
- `max_t5_overhead_percent = 3.0`
- `max_abs_throughput_drift_percent = 3.0`
- `max_p95_drift_percent = 8.0`

SLO por `kernel:size`:

- `min_avg_gflops`: floor de `95%` sobre media observada en Block 2.
- `max_p95_time_ms`: headroom de `+8%` sobre media observada en Block 2.

Rollback defaults incluidos:

- `auto_rollback_enabled = true`
- `rollback_after_consecutive_soft_overhead_violations = 2`
- `rollback_script_path = research/breakthrough_lab/platform_compatibility/week9_block5_rusticl_rollback.sh`

## Gate Canónico (obligatorio)

- `./venv/bin/python scripts/run_validation_suite.py --tier canonical --driver-smoke --report-dir research/breakthrough_lab/week8_validation_discipline`
  - Artifact JSON: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260209_010222.json`
  - Artifact MD: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260209_010222.md`
  - Decision: `promote`

## Decision Formal

Tracks:

- `week11_block3_policy_formalization`: **promote**
- `week11_block3_rollback_safe_defaults`: **promote**
- `week11_block3_mandatory_canonical_gate`: **promote**

Block decision:

- **promote**

Razonamiento:

- El policy semanal queda formalizado con umbrales explícitos, trazables y derivados de baseline real.
- Se mantiene disciplina de validación canónica en `promote`.

## Estado del Bloque

`Week 11 - Block 3` cerrado en `promote`.
