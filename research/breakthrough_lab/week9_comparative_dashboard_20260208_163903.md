# Comparative Dashboard - T3/T4/T5

- Date: 2026-02-08T16:39:03.679971+00:00
- Branch: feat/breakthrough-roadmap-2026q1

## Stage Decisions

| Stage | Decision |
| --- | --- |
| block1 | iterate |
| block2 | promote |
| block3 | promote |
| block4 | promote |
| block5 | promote |
| block6 | promote |
| block10 | promote |

## T3/T5 Aggregates (Clover-normalized for platform campaigns)

| Track | Stage | Avg GFLOPS | P95 ms | Max error | Extra |
| --- | --- | ---: | ---: | ---: | --- |
| t3 | block1 | 825.029 | 14.142 | 0.0006104 | fallback=0.0000, disabled=0 |
| t3 | block2 | 824.660 | 14.152 | 0.0006104 | fallback=0.0000, disabled=0 |
| t3 | block3 | 823.805 | 14.145 | 0.0005188 | fallback=0.0000, disabled=0 |
| t3 | block4 | 823.677 | 14.175 | 0.0005646 | fallback=0.0000, disabled=0 |
| t3 | block5 | 824.237 | 14.146 | 0.0005493 | fallback=0.0000, disabled=0 |
| t3 | block6 | 823.669 | 14.150 | 0.0005493 | fallback=0.0000, disabled=0 |
| t3 | block10 | 825.374 | 14.145 | 0.0005798 | fallback=0.0000, disabled=0 |
| t5 | block1 | 838.453 | 13.975 | 0.0005798 | overhead=1.339%, disable=1 |
| t5 | block2 | 840.502 | 13.973 | 0.0005798 | overhead=1.313%, disable=0 |
| t5 | block3 | 843.745 | 13.993 | 0.0006409 | overhead=1.559%, disable=0 |
| t5 | block4 | 840.002 | 14.002 | 0.0005493 | overhead=1.691%, disable=0 |
| t5 | block5 | 843.868 | 13.985 | 0.0007019 | overhead=1.586%, disable=0 |
| t5 | block6 | 842.743 | 13.987 | 0.0005646 | overhead=1.608%, disable=0 |
| t5 | block10 | 841.886 | 13.986 | 0.0005035 | overhead=1.068%, disable=0 |

## Week9 Deltas (Block1/2/3)

| Track | Delta | Avg GFLOPS % | P95 % |
| --- | --- | ---: | ---: |
| t3 | block2_vs_block1 | -0.045 | +0.070 |
| t3 | block3_vs_block2 | -0.104 | -0.052 |
| t3 | block3_vs_block1 | -0.148 | +0.018 |
| t5 | block2_vs_block1 | +0.244 | -0.019 |
| t5 | block3_vs_block2 | +0.386 | +0.150 |
| t5 | block3_vs_block1 | +0.631 | +0.131 |

## Weekly Drift Tracking (Active Chain)

| Track | From -> To | Avg GFLOPS % | P95 % | Extra |
| --- | --- | ---: | ---: | --- |
| t3 | block2 -> block3 | -0.104 | -0.052 | fallback_delta=+0.0000 |
| t3 | block3 -> block4 | -0.015 | +0.218 | fallback_delta=+0.0000 |
| t3 | block4 -> block5 | +0.068 | -0.205 | fallback_delta=+0.0000 |
| t3 | block5 -> block6 | -0.069 | +0.025 | fallback_delta=+0.0000 |
| t3 | block6 -> block10 | +0.207 | -0.034 | fallback_delta=+0.0000 |
| t5 | block2 -> block3 | +0.386 | +0.150 | overhead_delta=+0.246%, disable_delta=0 |
| t5 | block3 -> block4 | -0.444 | +0.061 | overhead_delta=+0.132%, disable_delta=0 |
| t5 | block4 -> block5 | +0.460 | -0.118 | overhead_delta=-0.106%, disable_delta=0 |
| t5 | block5 -> block6 | -0.133 | +0.009 | overhead_delta=+0.022%, disable_delta=0 |
| t5 | block6 -> block10 | -0.102 | -0.008 | overhead_delta=-0.540%, disable_delta=0 |

## T4 Reference

- Source: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week8_block6_t4_t5_interaction_20260208_024510.json` (latest validated T4 evidence; dashboard compares operational chain without mutating this baseline)
- Contract compliance: 1.000 | Fallback: 0.000 | Post-fallback violation: 0.000
- Compressible speedup vs exact: 5.104 | Delta vs exact: +227.658%

## Decision

- Dashboard status: `promote`
- Rationale: Block1 iterate is superseded; active chain from Block2 onward remains promote including explicit Block6 tracking.

