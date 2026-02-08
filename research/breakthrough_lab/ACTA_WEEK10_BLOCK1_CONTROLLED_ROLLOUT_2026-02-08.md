# Acta Week 10 - Block 1 (Controlled Rollout + Auto Rollback Guardrails)

- Date: 2026-02-08
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope: arranque de rollout controlado de bajo alcance con snapshots horarios logicos, rollback automatico por guardrails y extension del dashboard con Block 6 explicito + drift semanal.

## Objetivo

1. Ejecutar primer rollout controlado en Clover con scope reducido y guardrails T3/T5 activos.
2. Verificar operatividad de rollback automatico ante breach de guardrails.
3. Extender el dashboard comparativo para incluir `block6` y seguimiento semanal de drift (cadena activa).

## Implementacion

Cambios del bloque:

- Nuevo runner:
  - `research/breakthrough_lab/platform_compatibility/run_week10_block1_controlled_rollout.py`
- Dashboard extendido:
  - `research/breakthrough_lab/build_week9_comparative_dashboard.py`
  - Soporte de `--block6-path` y `--block10-path`
  - Seccion nueva de `weekly_drift_tracking` por transicion de bloque.

## Ejecucion Formal

Commands:

- `./venv/bin/python research/breakthrough_lab/platform_compatibility/run_week10_block1_controlled_rollout.py --snapshots 3 --snapshot-interval-minutes 60 --sleep-between-snapshots-seconds 0 --sizes 1400 --kernels auto_t3_controlled auto_t5_guarded --sessions 1 --iterations 4 --pressure-size 896 --pressure-iterations 2 --pressure-pulses-per-snapshot 2 --rollback-after-consecutive-soft-overhead-violations 2`
- `./venv/bin/python research/breakthrough_lab/build_week9_comparative_dashboard.py --block4-path research/breakthrough_lab/platform_compatibility/week9_block4_stress_split_20260208_033946.json --block5-path research/breakthrough_lab/platform_compatibility/week9_block5_preprod_pilot_20260208_035240.json --block6-path research/breakthrough_lab/platform_compatibility/week9_block6_wallclock_canary_20260208_043949.json --block10-path research/breakthrough_lab/platform_compatibility/week10_block1_controlled_rollout_20260208_160122.json`

Artifacts:

- `research/breakthrough_lab/platform_compatibility/week10_block1_controlled_rollout_20260208_160122.json`
- `research/breakthrough_lab/platform_compatibility/week10_block1_controlled_rollout_20260208_160122.md`
- `research/breakthrough_lab/platform_compatibility/week9_block5_rollback_20260208_160103.md`
- `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_160122.json`
- `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_160122.md`
- `research/breakthrough_lab/week9_comparative_dashboard_20260208_160146.json`
- `research/breakthrough_lab/week9_comparative_dashboard_20260208_160146.md`

## Resultados

Rollout controlado Block 1:

- Decision: `iterate`
- Snapshots ejecutados: `2/3` (detenido por rollback automatico)
- Snapshot 1:
  - estado: `warn`
  - T5 overhead max: `3.3409%` (soft breach > `3.0%`, sin breach duro)
- Snapshot 2:
  - estado: `rollback`
  - trigger duro: `t5_guardrails_hard` (disable events `1`)
  - T5 overhead max: `3.2207%`
- Guardrails T3: pass en snapshots ejecutados (fallback `0.0`, policy disabled `0`)

Rollback automatico:

- Activado automaticamente en snapshot `2`
- Trigger type: `hard_guardrail`
- Script rollback (`week9_block5_rusticl_rollback.sh apply`): **success**
- Gate canonico post-rollback: **promote**

Dashboard extendido (Block 6 + drift semanal):

- Artifact decision: `iterate` (por `block10=iterate`)
- Cadena activa cargada: `block2 -> block3 -> block4 -> block5 -> block6 -> block10`
- Track T3 (`block6 -> block10`):
  - avg GFLOPS delta: `+4.598%`
  - p95 delta: `-56.074%`
- Track T5 (`block6 -> block10`):
  - avg GFLOPS delta: `+7.372%`
  - p95 delta: `-56.969%`
  - overhead delta: `+1.673%`
  - disable events delta: `+1`

## Decision Formal

Tracks:

- `week10_block1_controlled_rollout_scope_low`: **iterate**
- `week10_block1_auto_rollback_operability`: **promote**
- `week10_block2_dashboard_extension_block6_weekly_drift`: **promote**

Block decision:

- **iterate**

Razonamiento:

- El objetivo de seguridad operacional se cumple: rollback automatico funciono correctamente y el gate post-rollback se mantiene `promote`.
- El rollout de bajo alcance no puede promoverse todavia por evento de disable T5 en snapshot 2; requiere hardening adicional antes de aumentar alcance.

## Estado del Bloque

`Week 10 - Block 1` iniciado y formalizado con `iterate`, evidencia reproducible y rollback automatico validado en ejecucion real.

