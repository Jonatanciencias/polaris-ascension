# Week 13 Block 1 - Biweekly Comparative Report

- Date: 2026-02-09T01:47:33.303225+00:00
- Baseline canary: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week12_controlled_rollout/week12_block3_size3072_pilot_canary_20260209_012745.json`
- Current canary: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week13_controlled_rollout/week13_block1_extended_controlled_canary_20260209_014522.json`

## Guardrail Summary

| Window | Decision | Snapshots | Rollback | Max error | T3 fallback max | T5 overhead max % | T5 disable total |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| baseline | promote | 6 | False | 0.000976562 | 0.0000 | 1.8956 | 0 |
| current | promote | 8 | False | 0.000885010 | 0.0000 | 1.4782 | 0 |

## Per-Key Deltas (current vs baseline)

| Key | Baseline avg GFLOPS | Current avg GFLOPS | Delta % | Baseline p95 ms | Current p95 ms | Delta % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| auto_t3_controlled:1400 | 881.899 | 880.860 | -0.118 | 6.148 | 6.140 | -0.131 |
| auto_t3_controlled:2048 | 773.432 | 773.110 | -0.042 | 22.188 | 22.196 | 0.034 |
| auto_t3_controlled:3072 | 706.618 | 704.725 | -0.268 | 72.092 | 72.111 | 0.026 |
| auto_t5_guarded:1400 | 903.671 | 907.427 | 0.416 | 6.005 | 6.000 | -0.091 |
| auto_t5_guarded:2048 | 779.591 | 777.284 | -0.296 | 21.978 | 21.994 | 0.072 |
| auto_t5_guarded:3072 | 803.701 | 803.754 | 0.007 | 71.981 | 71.974 | -0.010 |

## Decision

- Decision: `promote`
- Failed checks: []
- Rationale: Extended controlled production remains stable with bounded deltas and healthy guardrails.

