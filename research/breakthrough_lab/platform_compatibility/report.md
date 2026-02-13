# Platform Compatibility Report - Week 5 Block 4 (Rusticl/ROCm Feasibility)

- Status: completed
- Decision: refine
- Promotion gate: PARTIAL (Rusticl viable in shadow, production selection hardening pending)

## Summary
- Baseline production platform remains `Clover` with AMD RX 590 detected and stable.
- `rusticl` appears by default but without active GPU device in this host configuration.
- With `RUSTICL_ENABLE=radeonsi`, `rusticl` exposes the RX 590 GPU and runs production GEMM kernel microbench successfully.
- Microbench signal (`tile24`, 1024):
  - Clover avg: `641.616` GFLOPS
  - Rusticl avg: `658.612` GFLOPS
  - Ratio rusticl/clover: `1.026`
- ROCm userspace tools (`rocminfo`, `rocm-smi`) are not installed in this environment.

## Guardrail Outcome
- Pass: Clover default GPU availability.
- Pass: Rusticl can be activated in shadow mode via env.
- Pass: Rusticl microbench performance ratio >= 0.9.
- Pending: production path still hardcodes `cl.get_platforms()[0]`, so canary migration by explicit platform is not yet wired.
- Informational: `verify_drivers.py` currently reports OpenCL false negatives in this host despite positive pyopencl/clinfo evidence.

## Interpretation
- Platform migration is technically feasible for Rusticl in controlled/shadow mode.
- Promotion to production platform policy requires platform-selection hardening first.
- ROCm should remain non-blocking/optional for this Polaris setup until explicit package-level validation exists.

## Evidence
- research/breakthrough_lab/platform_compatibility/run_week5_platform_compatibility.py
- research/breakthrough_lab/platform_compatibility/week5_platform_compatibility_20260207_234905.json
- research/breakthrough_lab/platform_compatibility/week5_platform_compatibility_20260207_234905.md
- research/breakthrough_lab/ACTA_WEEK5_BLOCK4_PLATFORM_COMPATIBILITY_2026-02-07.md

## Week 7 Block 1 - Explicit Selector Hardening

Status: completed  
Decision: promote

- Added explicit OpenCL platform/device selectors to production benchmark path and CLI.
- Verified canary path on Rusticl with `RUSTICL_ENABLE=radeonsi` and explicit `opencl_platform='rusticl'`.
- Evidence:
  - `research/breakthrough_lab/platform_compatibility/week7_platform_selector_hardening_20260208_000425.json`
  - `research/breakthrough_lab/platform_compatibility/week7_platform_selector_hardening_20260208_000425.md`
  - `research/breakthrough_lab/ACTA_WEEK7_BLOCK1_PLATFORM_SELECTOR_HARDENING_2026-02-08.md`
