# Week 9 Comparative Dashboard - T3/T4/T5

- Date: 2026-02-08T03:40:22.359455+00:00
- Branch: feat/breakthrough-roadmap-2026q1

## Stage Decisions

| Stage | Decision |
| --- | --- |
| block1 | iterate |
| block2 | promote |
| block3 | promote |
| block4 | promote |

## T3/T5 Aggregates (Clover-normalized for block3/4)

| Track | Stage | Avg GFLOPS | P95 ms | Max error | Extra |
| --- | --- | ---: | ---: | ---: | --- |
| t3 | block1 | 825.029 | 14.142 | 0.0006104 | fallback=0.0000, disabled=0 |
| t3 | block2 | 824.660 | 14.152 | 0.0006104 | fallback=0.0000, disabled=0 |
| t3 | block3 | 823.805 | 14.145 | 0.0005188 | fallback=0.0000, disabled=0 |
| t3 | block4 | 823.677 | 14.175 | 0.0005646 | fallback=0.0000, disabled=0 |
| t5 | block1 | 838.453 | 13.975 | 0.0005798 | overhead=1.339%, disable=1 |
| t5 | block2 | 840.502 | 13.973 | 0.0005798 | overhead=1.313%, disable=0 |
| t5 | block3 | 843.745 | 13.993 | 0.0006409 | overhead=1.559%, disable=0 |
| t5 | block4 | 840.002 | 14.002 | 0.0005493 | overhead=1.691%, disable=0 |

## Week9 Deltas (Block1/2/3)

| Track | Delta | Avg GFLOPS % | P95 % |
| --- | --- | ---: | ---: |
| t3 | block2_vs_block1 | -0.045 | +0.070 |
| t3 | block3_vs_block2 | -0.104 | -0.052 |
| t3 | block3_vs_block1 | -0.148 | +0.018 |
| t5 | block2_vs_block1 | +0.244 | -0.019 |
| t5 | block3_vs_block2 | +0.386 | +0.150 |
| t5 | block3_vs_block1 | +0.631 | +0.131 |

## T4 Reference

- Source: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week8_block6_t4_t5_interaction_20260208_024510.json` (latest validated T4 evidence; Week9 blocks 1/2/3 did not modify T4 policy)
- Contract compliance: 1.000 | Fallback: 0.000 | Post-fallback violation: 0.000
- Compressible speedup vs exact: 5.104 | Delta vs exact: +227.658%

## Decision

- Dashboard status: `promote`
- Rationale: Block1 iterate was superseded by Block2 hardening; active Week9 chain (Block2/3/4) is fully promote.

