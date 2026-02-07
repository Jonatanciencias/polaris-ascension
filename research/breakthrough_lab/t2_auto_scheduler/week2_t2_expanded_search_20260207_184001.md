# T2 Week 2 Search (Deterministic + Strict)

- Date: 2026-02-07T18:40:01.563012+00:00
- Search space: expanded
- Budget: 6 configs x [1400, 2048, 3072] sizes x 12 runs
- Input distribution: standard_normal
- Correctness threshold: 0.001

## Search Results

| Rank | Config | Family | Vec | Unroll | Local | Size | GFLOPS | Max Error | Delta vs Baseline | Status |
| ---: | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | t20_v3vec_v4_u0_l10 | tile20 | 4 | 0 | 10x10 | 1400 | 926.639 | 0.000397 | +15.880% | valid |
| 2 | t20_prefetch_v4_u4_l10 | tile20 | 4 | 4 | 10x10 | 1400 | 837.811 | 0.000336 | +4.771% | valid |
| 3 | t20_prod_v4_u10_l10 | tile20 | 4 | 10 | 10x10 | 1400 | 799.657 | 0.000412 | +0.000% | valid |
| 4 | t24_prod_v4_u0_l12 | tile24 | 4 | 0 | 12x12 | 2048 | 739.847 | 0.000496 | +0.000% | valid |
| 5 | t24_prod_v4_u0_l12 | tile24 | 4 | 0 | 12x12 | 3072 | 710.442 | 0.000824 | +0.000% | valid |
| 6 | t24_prod_v4_u0_l12 | tile24 | 4 | 0 | 12x12 | 1400 | 768.543 | 0.000305 | -3.891% | valid |
| 7 | t20_regblock_v4_u0_l5 | tile20 | 4 | 0 | 5x5 | 1400 | 631.964 | 0.000320 | -20.971% | valid |
| 8 | t20_regblock_v4_u0_l5 | tile20 | 4 | 0 | 5x5 | 2048 | 435.359 | 0.000488 | -41.156% | valid |
| 9 | t20_v3vec_v4_u0_l10 | tile20 | 4 | 0 | 10x10 | 2048 | 336.061 | 0.000519 | -54.577% | valid |
| 10 | t20_regblock_v4_u0_l5 | tile20 | 4 | 0 | 5x5 | 3072 | 302.741 | 0.000732 | -57.387% | valid |
| 11 | t20_prod_v4_u10_l10 | tile20 | 4 | 10 | 10x10 | 2048 | 291.370 | 0.000473 | -60.618% | valid |
| 12 | t20_float8_v8_u8_l10 | tile20 | 8 | 8 | 10x10 | 1400 | 308.986 | 0.000336 | -61.360% | valid |
| 13 | t20_prefetch_v4_u4_l10 | tile20 | 4 | 4 | 10x10 | 2048 | 234.310 | 0.000534 | -68.330% | valid |
| 14 | t20_v3vec_v4_u0_l10 | tile20 | 4 | 0 | 10x10 | 3072 | 222.687 | 0.000809 | -68.655% | valid |
| 15 | t20_prefetch_v4_u4_l10 | tile20 | 4 | 4 | 10x10 | 3072 | 193.453 | 0.000732 | -72.770% | valid |
| 16 | t20_prod_v4_u10_l10 | tile20 | 4 | 10 | 10x10 | 3072 | 156.298 | 0.000717 | -78.000% | valid |

## Replay Summary

| Candidate | Vec | Unroll | Local | Mean GFLOPS | CV | Max Error (max) | Delta vs Baseline | Correctness | Stability | Promotion |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| t20_v3vec_v4_u0_l10@1400 | 4 | 0 | 10x10 | 926.303 | 0.00421 | 0.000336 | +15.838% | True | True | True |
| t20_prefetch_v4_u4_l10@1400 | 4 | 4 | 10x10 | 839.477 | 0.00068 | 0.000443 | +4.980% | True | True | False |
| t20_prod_v4_u10_l10@1400 | 4 | 10 | 10x10 | 803.049 | 0.00090 | 0.000320 | +0.424% | True | True | False |
| t24_prod_v4_u0_l12@2048 | 4 | 0 | 12x12 | 782.770 | 0.00078 | 0.000549 | +5.802% | True | True | False |
| t24_prod_v4_u0_l12@3072 | 4 | 0 | 12x12 | 708.494 | 0.00117 | 0.000916 | -0.274% | True | True | False |

## Decision Hint

- Suggested decision: `promote`
- Decision rationale: At least one replayed candidate passed performance, correctness and stability gates.
