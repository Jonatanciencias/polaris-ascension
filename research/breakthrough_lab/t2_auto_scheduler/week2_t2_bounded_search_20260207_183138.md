# T2 Week 2 Bounded Search

- Date: 2026-02-07T18:31:38.181820+00:00
- Budget: ['tile20', 'tile24'] kernels x [1400, 2048, 3072] sizes x 12 runs
- Input distribution: standard_normal
- Correctness threshold: 0.001

## Search Results

| Rank | Kernel | Size | GFLOPS | Max Error | Delta vs Baseline | Status |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | tile20 | 1400 | 799.524 | 0.000298 | +0.000% | valid |
| 2 | tile24 | 1400 | 768.223 | 0.000351 | -3.915% | valid |
| 3 | tile24 | 2048 | 739.372 | 0.000488 | +0.000% | valid |
| 4 | tile24 | 3072 | 712.511 | 0.000809 | +0.000% | valid |
| 5 | tile20 | 2048 | 286.983 | 0.000519 | -61.186% | valid |
| 6 | tile20 | 3072 | 152.550 | 0.000717 | -78.590% | valid |

## Replay Summary

| Candidate | Mean GFLOPS | CV | Max Error (max) | Delta vs Baseline | Correctness | Stability | Promotion |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| tile20@1400 | 802.622 | 0.00234 | 0.000336 | +0.387% | True | True | False |
| tile24@1400 | 769.843 | 0.00023 | 0.000443 | -3.712% | True | True | False |
| tile24@2048 | 783.923 | 0.00039 | 0.000610 | +6.026% | True | True | False |

## Decision Hint

- Suggested decision: `iterate`
- Decision rationale: Correct/stable candidates exist but promotion uplift threshold is not met.
