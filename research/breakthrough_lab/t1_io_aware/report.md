# T1 Report - Week 2 Block 1 (IO-Aware Variants)

- Status: completed
- Decision: iterate
- Promotion gate: FAILED (performance threshold not met)

## Summary
- Best candidate (`io_prefetch_v1` @ 1400): 820.678 GFLOPS (`+4.708%` vs baseline tile20)
- Correctness/stability on best candidate: pass (`max_error=0.000323`, `cv_peak=0.00566`)
- `io_regblock_v1` overall mean delta across 1400/2048/3072: `-42.253%`
- Large-size regressions remain severe (2048/3072), so promotion is blocked.

## Evidence
- research/breakthrough_lab/t1_io_aware/week2_t1_io_campaign_20260207_174226.json
- research/breakthrough_lab/t1_io_aware/week2_t1_io_campaign_20260207_174226.md
- research/breakthrough_lab/t1_io_aware/kernels/gemm_tile20_io_prefetch.cl
- research/breakthrough_lab/t1_io_aware/kernels/gemm_tile20_io_regblock.cl

## Next
- Add one more IO-aware variant explicitly tuned for 2048/3072 behavior.
- Introduce size-weighted objective so 1400 uplift cannot hide large-size regressions.
- Keep strict correctness target (`max_error <= 1e-3`) and re-run full 10x20 protocol.
