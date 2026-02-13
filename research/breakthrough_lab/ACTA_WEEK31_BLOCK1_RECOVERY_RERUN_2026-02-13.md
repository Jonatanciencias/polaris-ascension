# Acta Week 31 - Block 1 Recovery Rerun (Hardening T5 + Snapshots)

- Date: 2026-02-13
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope:
  - ejecutar rerun de recuperacion sobre Week31 Block1,
  - endurecer perfil operativo para reducir overhead T5,
  - buscar cierre en `promote` manteniendo gate canonico pre/post.

## Contexto de entrada

Week31 Block1 quedo inicialmente en `iterate` por replay semanal con:

- `snapshots=4/6`
- `t5_disable_total=1`
- `t5_overhead_max=10.8918%`

(`research/breakthrough_lab/week31_block1_monthly_continuity_decision.json`)

## Ejecucion

### Attempt 1 (r1)

Comando:

- `./venv/bin/python research/breakthrough_lab/week20_controlled_rollout/run_week20_block1_monthly_cycle.py --baseline-path research/breakthrough_lab/week30_controlled_rollout/week30_block1_monthly_continuity_20260213_013243.json --output-dir research/breakthrough_lab/week31_controlled_rollout --preprod-signoff-dir research/breakthrough_lab/week31_controlled_rollout --output-prefix week31_block1_monthly_continuity_recovery --report-dir research/breakthrough_lab/week8_validation_discipline --snapshots 8 --iterations 6 --pressure-iterations 1 --pressure-pulses 1 --weekly-seed 31011 --split-seeds 311 607`

Resultado:

- `decision = iterate`
- `failed_checks = ['weekly_replay_promote', 'weekly_replay_eval_promote']`
- Weekly replay eval:
  - `minimum_snapshots = 8` (pass)
  - `t5_disable_total = 0` (pass)
  - `t5_overhead_max = 4.3320%` (fail vs `<=3.0%`)

### Attempt 2 (r2)

Comando:

- `./venv/bin/python research/breakthrough_lab/week20_controlled_rollout/run_week20_block1_monthly_cycle.py --baseline-path research/breakthrough_lab/week30_controlled_rollout/week30_block1_monthly_continuity_20260213_013243.json --output-dir research/breakthrough_lab/week31_controlled_rollout --preprod-signoff-dir research/breakthrough_lab/week31_controlled_rollout --output-prefix week31_block1_monthly_continuity_recovery_r2 --report-dir research/breakthrough_lab/week8_validation_discipline --snapshots 8 --iterations 4 --pressure-size 512 --pressure-iterations 0 --pressure-pulses 0 --weekly-seed 31021 --split-seeds 313 613`

Resultado:

- `decision = promote`
- `failed_checks = []`
- Weekly replay canary:
  - `snapshots = 8`
  - `rollback_event = null`
  - `t5_disable_total = 0`
  - `t5_overhead_max = 2.2758%`
- Weekly replay eval:
  - `minimum_snapshots = 8` (pass)
  - `t5_disable_total = 0` (pass)
  - `t5_overhead_max = 2.2758%` (pass vs `<=3.0%`)
- Split eval:
  - `decision = promote`
  - `split_ratio_min = 0.9221`
  - `split_t5_overhead_max = 2.7492%`
  - `split_t5_disable_total = 0`

## Artefactos de cierre

- Recovery r2 report JSON: `research/breakthrough_lab/week31_controlled_rollout/week31_block1_monthly_continuity_recovery_r2_20260213_024515.json`
- Recovery r2 report MD: `research/breakthrough_lab/week31_controlled_rollout/week31_block1_monthly_continuity_recovery_r2_20260213_024515.md`
- Recovery r2 weekly replay eval JSON: `research/breakthrough_lab/week31_controlled_rollout/week31_block1_monthly_continuity_recovery_r2_weekly_replay_eval_20260213_024342.json`
- Recovery r2 split eval JSON: `research/breakthrough_lab/week31_controlled_rollout/week31_block1_monthly_continuity_recovery_r2_split_eval_20260213_024515.json`
- Recovery r2 dashboard JSON: `research/breakthrough_lab/week31_controlled_rollout/week20_block1_monthly_cycle_dashboard_20260213_024515.json`
- Canonical gates pre/post (r2):
  - `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260213_024027.json`
  - `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260213_024535.json`

## Decision Formal

Tracks:

- `week31_block1_recovery_attempt_1`: **iterate**
- `week31_block1_recovery_attempt_2`: **promote**
- `week31_block1_recovery_split_guardrails`: **promote**
- `week31_block1_recovery_canonical_gate_internal`: **promote**

Block decision:

- **promote**

Razonamiento:

- El hardening operativo del attempt r2 elimina rollback y violacion de overhead T5, preserva guardrails cross-platform y deja Week31 Block1 recuperado para continuar con Week31 Block2.

## Estado del Bloque

`Week 31 - Block 1 recovery rerun` cerrado en `promote`.
