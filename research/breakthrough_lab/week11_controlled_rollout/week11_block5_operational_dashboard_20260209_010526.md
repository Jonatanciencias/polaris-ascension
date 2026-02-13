# Week 11 Operational Dashboard

- Date: 2026-02-09T01:05:26.488222+00:00
- Block 2 canary: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week11_controlled_rollout/week11_block2_continuous_canary_20260209_005442.json`
- Block 4 canary: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week11_controlled_rollout/week11_block4_weekly_replay_canary_20260209_010447.json`
- Block 4 eval: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week11_controlled_rollout/week11_block4_weekly_replay_eval_20260209_010454.json`

## Block Summaries

| Block | Decision | Snapshots | Rollback | Correctness max | T3 fallback max | T5 overhead max % | T5 disable total |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| block2 | promote | 6 | False | 0.000579834 | 0.0000 | 2.8982 | 0 |
| block4 | promote | 6 | False | 0.000549316 | 0.0000 | 1.8522 | 0 |

## Comparative Deltas (Block4 vs Block2)

| Key | Block2 avg GFLOPS | Block4 avg GFLOPS | Delta % | Block2 p95 ms | Block4 p95 ms | Delta % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| auto_t3_controlled:1400 | 882.667 | 883.185 | 0.059 | 6.118 | 6.150 | 0.517 |
| auto_t3_controlled:2048 | 773.013 | 773.046 | 0.004 | 22.180 | 22.188 | 0.035 |
| auto_t5_guarded:1400 | 906.793 | 905.881 | -0.101 | 5.999 | 5.999 | -0.003 |
| auto_t5_guarded:2048 | 777.524 | 777.040 | -0.062 | 22.004 | 21.995 | -0.041 |

## Drift Status

- Decision: `promote`
- Failed checks: []
- Rationale: Weekly replay satisfies all global guardrails, per-kernel SLO rows, and drift bounds.

## Package Decision

- Decision: `promote`
- Rationale: Week 11 package keeps operational canary and policy replay in promote with healthy drift status.

