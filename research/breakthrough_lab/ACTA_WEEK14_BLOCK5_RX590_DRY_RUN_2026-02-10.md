# Acta Week 14 - Block 5 (Dry-run RX590 low scope + GO/NO-GO formal)

- Date: 2026-02-10
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope:
  - ejecutar dry-run RX590 de bajo alcance,
  - cerrar go/no-go formal con rollback SLA,
  - mantener gate canónico obligatorio pre/post.

## Objetivo

1. Verificar comportamiento estable en alcance bajo antes de ampliar scope.
2. Validar readiness operativa (rollback path + SLA + gate canónico).
3. Cerrar con decisión formal `go|no-go`.

## Ejecución Formal

Intento inicial:

- `./venv/bin/python research/breakthrough_lab/week14_controlled_rollout/run_week14_block5_rx590_dry_run.py --duration-minutes 3 --snapshot-interval-minutes 1 --sizes 1400 2048 --sessions 1 --iterations 4 --pressure-size 896 --pressure-iterations 2 --pressure-pulses-per-snapshot 1 --seed 17051 --output-dir research/breakthrough_lab/week14_controlled_rollout --output-prefix week14_block5_rx590_dry_run`
  - Artifact JSON: `research/breakthrough_lab/week14_controlled_rollout/week14_block5_rx590_dry_run_20260210_003416.json`
  - Decision: `no-go`
  - Hallazgo: `canary_promote=false`, `canary_t5_disable_zero=false` (`disable_total=1`).

Rerun conservador:

- `./venv/bin/python research/breakthrough_lab/week14_controlled_rollout/run_week14_block5_rx590_dry_run.py --duration-minutes 3 --snapshot-interval-minutes 1 --sizes 1400 2048 --sessions 1 --iterations 4 --pressure-size 896 --pressure-iterations 2 --pressure-pulses-per-snapshot 0 --seed 17071 --t5-policy-path research/breakthrough_lab/t5_reliability_abft/policy_hardening_week10_block2_4_long_horizon.json --output-dir research/breakthrough_lab/week14_controlled_rollout --output-prefix week14_block5_rx590_dry_run_rerun`
  - Artifact JSON: `research/breakthrough_lab/week14_controlled_rollout/week14_block5_rx590_dry_run_rerun_20260210_003819.json`
  - Decision: `no-go`
  - Hallazgo: `t5_disable_total=0`, pero `canary_promote=false` por overhead máximo `3.003669%` (límite `3.0%`).

Rerun endurecido v2 (cierre):

- `./venv/bin/python research/breakthrough_lab/week14_controlled_rollout/run_week14_block5_rx590_dry_run.py --duration-minutes 3 --snapshot-interval-minutes 1 --sizes 1400 2048 --sessions 1 --iterations 8 --pressure-size 896 --pressure-iterations 2 --pressure-pulses-per-snapshot 0 --seed 17111 --t5-policy-path research/breakthrough_lab/t5_reliability_abft/policy_hardening_week14_block1_low_overhead.json --output-dir research/breakthrough_lab/week14_controlled_rollout --output-prefix week14_block5_rx590_dry_run_hardened_v2`
  - Artifact JSON: `research/breakthrough_lab/week14_controlled_rollout/week14_block5_rx590_dry_run_hardened_v2_20260210_004609.json`
  - Canary JSON: `research/breakthrough_lab/week14_controlled_rollout/week14_block5_rx590_dry_run_hardened_v2_canary_20260210_004549.json`
  - Decision: `go`
  - Checklist: `research/breakthrough_lab/preprod_signoff/WEEK14_BLOCK5_GO_NO_GO_CHECKLIST.md`

## Resultados Finales (v2)

- `pre_gate_decision = promote`
- `canary_decision = promote`
- `canary_t5_disable_total = 0`
- `canary_correctness_max = 0.0005645751953125`
- `post_gate_decision = promote`
- `rollback_dry_run_ok = true`
- `rollback_sla_exists = true`

## Decisión Formal

Tracks:

- `week14_block5_initial_dry_run`: **iterate / no-go**
- `week14_block5_conservative_rerun`: **iterate / no-go**
- `week14_block5_hardened_v2_rerun`: **promote / go**
- `week14_block5_rollback_readiness`: **promote**
- `week14_block5_mandatory_canonical_gate`: **promote**

Block decision:

- **promote**
- Operational decision: **go**

Razonamiento:

- Se aplicó endurecimiento incremental sin relajar contratos.
- El rerun v2 cerró en `go` con guardrails y gates canónicos en verde.

## Estado del Bloque

`Week 14 - Block 5` cerrado en `promote` con `GO` operativo.
