# Week 11 T5 Effect Probe (rusticl subprocess)

- Date: 2026-02-09T00:36:19.909922+00:00

- Errors: 0

## Summary

| Profile | Size | Avg GFLOPS | Peak GFLOPS | P95 ms | Max error | T5 over mean % | T5 over max % | T5 disable total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| t5_old_policy | 1400 | 916.018 | 924.581 | 5.940 | 0.0003204 | 1.413 | 1.415 | 0 |
| t5_old_policy | 2048 | 717.547 | 721.975 | 23.799 | 0.0005798 | 0.511 | 0.523 | 0 |
| t5_new_policy | 1400 | 921.047 | 924.347 | 5.939 | 0.0003204 | 1.253 | 1.271 | 0 |
| t5_new_policy | 2048 | 716.664 | 722.315 | 23.803 | 0.0005798 | 0.461 | 0.464 | 0 |

## Delta new vs old

| Size | Old Avg | New Avg | Delta % | Old Over Mean | New Over Mean | New Over Max | Old Disable | New Disable |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1400 | 916.018 | 921.047 | +0.549% | 1.413 | 1.253 | 1.271 | 0 | 0 |
| 2048 | 717.547 | 716.664 | -0.123% | 0.511 | 0.461 | 0.464 | 0 | 0 |
