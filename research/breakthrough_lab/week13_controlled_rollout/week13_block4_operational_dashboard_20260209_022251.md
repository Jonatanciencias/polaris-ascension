# Week 13 Block 4 - Operational Consolidation Dashboard

- Date: 2026-02-09T02:22:51.407239+00:00
- Baseline canary: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week13_controlled_rollout/week13_block1_extended_controlled_canary_20260209_014522.json`
- Current canary: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week13_controlled_rollout/week13_block2_weekly_replay_canary_20260209_020049.json`
- Policy eval v2: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week13_controlled_rollout/week13_block3_recalibrated_policy_eval_20260209_021320.json`
- Split eval: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week13_controlled_rollout/week13_block2_platform_split_eval_20260209_020309.json`

## Block Summaries

| Window | Decision | Snapshots | Rollback | Correctness max | T3 fallback max | T5 overhead max % | T5 disable total |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| baseline | promote | 8 | False | 0.000885010 | 0.0000 | 1.4782 | 0 |
| current | promote | 6 | False | 0.000869751 | 0.0000 | 1.3788 | 0 |

## Comparative Deltas (current vs baseline)

| Key | Baseline avg GFLOPS | Current avg GFLOPS | Delta % | Baseline p95 ms | Current p95 ms | Delta % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| auto_t3_controlled:1400 | 880.860 | 881.746 | 0.101 | 6.140 | 6.140 | -0.006 |
| auto_t3_controlled:2048 | 773.110 | 773.246 | 0.018 | 22.196 | 22.178 | -0.081 |
| auto_t3_controlled:3072 | 704.725 | 706.514 | 0.254 | 72.111 | 72.072 | -0.053 |
| auto_t5_guarded:1400 | 907.427 | 901.821 | -0.618 | 6.000 | 6.003 | 0.056 |
| auto_t5_guarded:2048 | 777.284 | 777.996 | 0.092 | 21.994 | 21.987 | -0.035 |
| auto_t5_guarded:3072 | 803.754 | 803.699 | -0.007 | 71.974 | 71.974 | -0.000 |

## Drift v2 and Split Status

- Drift v2 decision: `promote`
- Drift v2 failed checks: []
- Split eval decision: `promote`
- Split ratio rusticl/clover min: `0.922765`

## Package Decision

- Decision: `promote`
- Failed checks: []
- Rationale: Quincenal operational package is stable under policy v2 with healthy split compatibility.

