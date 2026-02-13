# Acta Week 14 - Block 2 (Extended horizon + verificación policy v2)

- Date: 2026-02-09
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope:
  - validar `policy v2` en horizonte extendido,
  - ejecutar split Clover/rusticl con presión de cola,
  - cerrar con gate canónico obligatorio.

## Objetivo

1. Confirmar que el hardening T5 de Week 14 Block 1 se mantiene estable en ventana más larga.
2. Verificar cumplimiento de guardrails y SLO por `kernel:size` sin relajar correctness/fallback/disable-events.
3. Cerrar formalmente con evidencia reproducible y decisión `promote|iterate|refine|stop`.

## Ejecución Formal

Replay extendido contra `policy v2`:

- `./venv/bin/python research/breakthrough_lab/week12_controlled_rollout/run_week12_weekly_replay_automation.py --mode local --policy-path research/breakthrough_lab/week13_controlled_rollout/policy_week13_block3_weekly_slo_v2.json --t5-policy-path research/breakthrough_lab/t5_reliability_abft/policy_hardening_week14_block1_low_overhead.json --baseline-path research/breakthrough_lab/week14_controlled_rollout/week14_block1_policy_v2_cycle_rerun_canary_20260209_024440.json --sizes 1400 2048 3072 --snapshots 10 --sessions 2 --iterations 8 --pressure-size 896 --pressure-iterations 3 --pressure-pulses-per-snapshot 4 --seed 16041 --output-dir research/breakthrough_lab/week14_controlled_rollout --output-prefix week14_block2_extended_horizon`
  - Artifact JSON: `research/breakthrough_lab/week14_controlled_rollout/week14_block2_extended_horizon_20260209_131849.json`
  - Canary JSON: `research/breakthrough_lab/week14_controlled_rollout/week14_block2_extended_horizon_canary_20260209_132519.json`
  - Eval JSON: `research/breakthrough_lab/week14_controlled_rollout/week14_block2_extended_horizon_eval_20260209_132519.json`
  - Decision: `promote`

Split Clover/rusticl con presión:

- `./venv/bin/python research/breakthrough_lab/platform_compatibility/run_week9_block4_stress_split.py --seeds 1212 1312 --sizes 1400 2048 3072 --kernels auto_t3_controlled auto_t5_guarded --sessions 1 --iterations 8 --pressure-size 896 --pressure-iterations 3 --pressure-pulses-per-seed 4 --t5-policy-path research/breakthrough_lab/t5_reliability_abft/policy_hardening_week14_block1_low_overhead.json --baseline-block3-path research/breakthrough_lab/week14_controlled_rollout/week14_block1_policy_v2_cycle_rerun_canary_20260209_024440.json --output-dir research/breakthrough_lab/week14_controlled_rollout --output-prefix week14_block2_platform_split`
  - Artifact JSON: `research/breakthrough_lab/week14_controlled_rollout/week14_block2_platform_split_20260209_132747.json`
  - Decision: `promote`

Evaluación formal del split contra `policy v2`:

- `./venv/bin/python research/breakthrough_lab/week12_controlled_rollout/evaluate_week12_platform_split_policy.py --split-artifact research/breakthrough_lab/week14_controlled_rollout/week14_block2_platform_split_20260209_132747.json --policy-path research/breakthrough_lab/week13_controlled_rollout/policy_week13_block3_weekly_slo_v2.json --min-rusticl-ratio 0.85 --required-sizes 1400 2048 3072 --output-dir research/breakthrough_lab/week14_controlled_rollout --output-prefix week14_block2_platform_split_eval`
  - Artifact JSON: `research/breakthrough_lab/week14_controlled_rollout/week14_block2_platform_split_eval_20260209_132755.json`
  - Decision: `promote`

Gate canónico obligatorio:

- Pre-gate del replay: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260209_131909.json` (`promote`)
- Post-gate del replay: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260209_132538.json` (`promote`)
- Gate final de cierre de bloque:
  - `./venv/bin/python scripts/run_validation_suite.py --tier canonical --driver-smoke --report-dir research/breakthrough_lab/week8_validation_discipline`
  - Artifact JSON: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260209_132820.json`
  - Artifact MD: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260209_132820.md`
  - Decision: `promote`

## Resultados

Replay extendido (`10 snapshots`):

- `decision = promote`
- `max_error = 0.0008544921875`
- `t3_fallback_max = 0.0`
- `t5_overhead_max = 1.4280063948041821%`
- `t5_disable_events_total = 0`
- `max_abs_drift = 1.375915143979322%`

Split Clover/rusticl:

- `decision = promote`
- `max_error = 0.0008087158203125`
- `t5_overhead_max = 1.5015676951729275%`
- `t5_disable_total = 0`
- `ratio_min = 0.9198754641296375` (`>= 0.85`)
- `required_sizes_present_on_split = true`

## Decisión Formal

Tracks:

- `week14_block2_extended_horizon_replay`: **promote**
- `week14_block2_platform_split_stress`: **promote**
- `week14_block2_platform_split_eval`: **promote**
- `week14_block2_mandatory_canonical_gate`: **promote**

Block decision:

- **promote**

Razonamiento:

- Se mantuvo estabilidad en horizonte extendido sin rollback y con `disable_events=0`.
- El split Clover/rusticl cumplió contrato y ratio mínimo con tamaños críticos presentes.
- El gate canónico cerró en `promote` antes del cierre formal del bloque.

## Estado del Bloque

`Week 14 - Block 2` cerrado en `promote`.
