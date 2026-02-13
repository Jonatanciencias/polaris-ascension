# Acta Week 36 - Block 1 (Monthly continuity against Week35 baseline)

- Date: 2026-02-13
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope:
  - ejecutar ciclo mensual recurrente contra baseline Week 35 Block 1,
  - realizar recoveries hasta cerrar weekly+split en `promote`,
  - mantener gate canonico obligatorio pre/post (explicito e interno).

## Objetivo

1. Confirmar estabilidad recurrente de Week36 contra baseline Week35.
2. Mantener contrato T5 (`disable_events=0`, `overhead<=3.0%`) y drift semanal dentro de policy.
3. Dejar Week36 habilitado para Block 2.

## Ejecucion

Comandos ejecutados:

- Attempt inicial:
  - Gate pre explicito:
    - `./venv/bin/python scripts/run_validation_suite.py --tier canonical --driver-smoke --report-dir research/breakthrough_lab/week8_validation_discipline`
  - Block 1:
    - `./venv/bin/python research/breakthrough_lab/week20_controlled_rollout/run_week20_block1_monthly_cycle.py --baseline-path research/breakthrough_lab/week35_controlled_rollout/week35_block1_monthly_continuity_20260213_163831.json --output-dir research/breakthrough_lab/week36_controlled_rollout --preprod-signoff-dir research/breakthrough_lab/week36_controlled_rollout --output-prefix week36_block1_monthly_continuity --report-dir research/breakthrough_lab/week8_validation_discipline --snapshots 8 --iterations 4 --pressure-size 512 --pressure-iterations 0 --pressure-pulses 0 --weekly-seed 36021 --split-seeds 361 613`
  - Gate post explicito:
    - `./venv/bin/python scripts/run_validation_suite.py --tier canonical --driver-smoke --report-dir research/breakthrough_lab/week8_validation_discipline`
- Recovery #1 (semillas conservadoras):
  - `./venv/bin/python research/breakthrough_lab/week20_controlled_rollout/run_week20_block1_monthly_cycle.py --baseline-path research/breakthrough_lab/week35_controlled_rollout/week35_block1_monthly_continuity_20260213_163831.json --output-dir research/breakthrough_lab/week36_controlled_rollout --preprod-signoff-dir research/breakthrough_lab/week36_controlled_rollout --output-prefix week36_block1_monthly_continuity_recovery --report-dir research/breakthrough_lab/week8_validation_discipline --snapshots 8 --iterations 4 --pressure-size 512 --pressure-iterations 0 --pressure-pulses 0 --weekly-seed 35021 --split-seeds 351 613`
- Recovery #2 (policy T5 long-horizon, seed original):
  - `./venv/bin/python research/breakthrough_lab/week20_controlled_rollout/run_week20_block1_monthly_cycle.py --baseline-path research/breakthrough_lab/week35_controlled_rollout/week35_block1_monthly_continuity_20260213_163831.json --output-dir research/breakthrough_lab/week36_controlled_rollout --preprod-signoff-dir research/breakthrough_lab/week36_controlled_rollout --output-prefix week36_block1_monthly_continuity_recovery2 --report-dir research/breakthrough_lab/week8_validation_discipline --t5-policy-path research/breakthrough_lab/t5_reliability_abft/policy_hardening_week10_block2_4_long_horizon.json --snapshots 8 --iterations 4 --pressure-size 512 --pressure-iterations 0 --pressure-pulses 0 --weekly-seed 36021 --split-seeds 361 613`
- Recovery #3 / Block1.1 focalizado (weekly+split simultáneo):
  - Gate pre explicito:
    - `./venv/bin/python scripts/run_validation_suite.py --tier canonical --driver-smoke --report-dir research/breakthrough_lab/week8_validation_discipline`
  - Block 1.1 targeted:
    - `./venv/bin/python research/breakthrough_lab/week20_controlled_rollout/run_week20_block1_monthly_cycle.py --baseline-path research/breakthrough_lab/week35_controlled_rollout/week35_block1_monthly_continuity_20260213_163831.json --output-dir research/breakthrough_lab/week36_controlled_rollout --preprod-signoff-dir research/breakthrough_lab/week36_controlled_rollout --output-prefix week36_block1_1_monthly_continuity_targeted --report-dir research/breakthrough_lab/week8_validation_discipline --t5-policy-path research/breakthrough_lab/t5_reliability_abft/policy_hardening_week10_block2_4_long_horizon.json --snapshots 8 --iterations 6 --pressure-size 512 --pressure-iterations 0 --pressure-pulses 0 --weekly-seed 35021 --split-seeds 351 613`
  - Gate post explicito:
    - `./venv/bin/python scripts/run_validation_suite.py --tier canonical --driver-smoke --report-dir research/breakthrough_lab/week8_validation_discipline`

## Artefactos

- Attempt inicial:
  - `research/breakthrough_lab/week36_controlled_rollout/week36_block1_monthly_continuity_20260213_171401.json`
  - `research/breakthrough_lab/week36_controlled_rollout/week36_block1_monthly_continuity_weekly_replay_eval_20260213_171228.json`
- Recovery #1:
  - `research/breakthrough_lab/week36_controlled_rollout/week36_block1_monthly_continuity_recovery_20260213_172206.json`
  - `research/breakthrough_lab/week36_controlled_rollout/week36_block1_monthly_continuity_recovery_weekly_replay_eval_20260213_172033.json`
- Recovery #2:
  - `research/breakthrough_lab/week36_controlled_rollout/week36_block1_monthly_continuity_recovery2_20260213_172906.json`
  - `research/breakthrough_lab/week36_controlled_rollout/week36_block1_monthly_continuity_recovery2_split_eval_20260213_172906.json`
- Block1.1 targeted:
  - `research/breakthrough_lab/week36_controlled_rollout/week36_block1_1_monthly_continuity_targeted_20260213_174540.json`
  - `research/breakthrough_lab/week36_controlled_rollout/week36_block1_1_monthly_continuity_targeted_weekly_replay_eval_20260213_174405.json`
  - `research/breakthrough_lab/week36_controlled_rollout/week36_block1_1_monthly_continuity_targeted_split_eval_20260213_174540.json`
- Canonical gates explicitos Block1.1 pre/post:
  - `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260213_174020.json`
  - `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260213_174623.json`
- Canonical gates internos Block1.1 pre/post:
  - `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260213_174045.json`
  - `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260213_174559.json`

## Resultados

- Attempt inicial: `iterate`
  - `t5_disable_events_total=1`, `max_t5_overhead=37.0156%`, `throughput_drift_abs=11.7659%`
- Recovery #1: `iterate`
  - corrige disable events y overhead; queda drift marginal (`2.8135% > 2.5%`)
- Recovery #2: `iterate`
  - weekly pasa, split falla por `split_t5_overhead=3.1895% (>3.0%)`
- Block1.1 targeted: `promote`
  - weekly eval: `promote` (`max_t5_overhead=1.9394%`, `t5_disable_events_total=0`)
  - split eval: `promote` (`split_t5_overhead_max=1.9518%`, `split_ratio_min=0.9209`, `split_t5_disable_total=0`)

## Decision Formal

Tracks:

- `week36_block1_initial_execution`: **iterate**
- `week36_block1_recovery1_execution`: **iterate**
- `week36_block1_recovery2_execution`: **iterate**
- `week36_block1_1_targeted_execution`: **promote**
- `week36_block1_canonical_gate_explicit_and_internal`: **promote**

Block decision:

- **promote**

Razonamiento:

- Aunque los tres primeros intentos no cerraron en verde, el Block1.1 focalizado logró el cierre simultáneo weekly+split en `promote` con gates canónicos pre/post en verde.

## Estado del Bloque

`Week 36 - Block 1` cerrado en `promote` (via Block1.1 targeted recovery).
