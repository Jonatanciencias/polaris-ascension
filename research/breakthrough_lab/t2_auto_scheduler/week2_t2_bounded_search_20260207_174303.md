# T2 Week 2 Bounded Search

- Date: 2026-02-07T17:43:03.620761+00:00
- Budget: ['tile20', 'tile24'] kernels x [1400, 2048, 3072] sizes x 12 runs
- Correctness threshold: 0.001

## Search Results

| Rank | Kernel | Size | GFLOPS | Max Error | Delta vs Baseline | Status |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | tile20 | 1400 | 801.716 | 0.001068 | +0.000% | filtered |
| 2 | tile24 | 1400 | 768.470 | 0.001068 | -4.147% | filtered |
| 3 | tile24 | 2048 | 734.452 | 0.001556 | +0.000% | filtered |
| 4 | tile24 | 3072 | 693.426 | 0.003601 | +0.000% | filtered |
| 5 | tile20 | 2048 | 291.445 | 0.001526 | -60.318% | filtered |
| 6 | tile20 | 3072 | 155.719 | 0.003967 | -77.544% | filtered |

## Replay Summary

| Candidate | Mean GFLOPS | CV | Max Error (max) | Delta vs Baseline | Correctness | Stability | Promotion |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| tile20@1400 | 802.609 | 0.00190 | 0.001099 | +0.111% | False | True | False |
| tile24@1400 | 769.745 | 0.00042 | 0.001099 | -3.988% | False | True | False |
| tile24@2048 | 731.248 | 0.00188 | 0.001831 | -0.436% | False | True | False |

## Decision Hint

- Suggested decision: `refine`
- Decision rationale: No candidate passed strict correctness filter in search phase.
