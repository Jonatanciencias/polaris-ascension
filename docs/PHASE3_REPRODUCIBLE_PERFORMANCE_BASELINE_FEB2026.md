# Phase 3 Reproducible Performance Baseline (Feb 7, 2026)

## Objective
Define a reproducible, host-verified performance baseline for production GEMM kernels on AMD Radeon RX 590 GME.

This baseline is the reference used by:
- `test_production_system.py` (Phase 3 validation section)
- `README.md` (public performance claims)

## Protocol
- Date: February 7, 2026
- Hardware: AMD Radeon RX 590 GME
- Runtime: PyOpenCL + Mesa Clover
- Matrices: `1400x1400`, `2048x2048`, `512x512`
- Sessions: 10
- Iterations per session: 20
- Seed: `42`
- Metric:
  - Peak GFLOPS: best iteration in each session
  - Avg GFLOPS: mean iteration performance in each session

Command used:

```bash
./venv/bin/python scripts/benchmark_phase3_reproducible.py \
  --sessions 10 \
  --iterations 20 \
  --output-json /tmp/phase3_reproducible_baseline_feb2026.json
```

Note: a known PyOpenCL compiler-cache warning may appear (`%b requires bytes-like object`); benchmark execution is not blocked.

## Results

| Size | Kernel | Peak GFLOPS mean | Peak range [min,max] | Avg GFLOPS mean |
|------|--------|------------------|----------------------|-----------------|
| 1400x1400 | tile20 | 776.1 | [772.6, 781.6] | 765.5 |
| 2048x2048 | tile24 | 774.3 | [772.8, 777.2] | 712.8 |
| 512x512 | tile24 | 455.6 | [436.3, 473.0] | 409.5 |

## Baseline Constants in Code
`test_production_system.py` uses this baseline for verification messages:

- `PHASE3_BASELINE["1400x1400"] = mean 776.1, min 772.6, max 781.6`
- `PHASE3_BASELINE["2048x2048"] = mean 774.3, min 772.8, max 777.2`
- `PHASE3_BASELINE["512x512"] = mean 455.6, min 436.3, max 473.0`

## Claim Policy
- Public claims should use reproducible baseline values by default.
- Historical peaks (for example, 831.2 GFLOPS @ 1300x1300) must be labeled as archived/historical discovery runs.
- When updating claims, re-run the same protocol and update all three locations together:
  1. `scripts/benchmark_phase3_reproducible.py` output (new report)
  2. `test_production_system.py` baseline constants
  3. `README.md` performance section
