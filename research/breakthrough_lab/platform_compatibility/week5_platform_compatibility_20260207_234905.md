# Week 5 Block 4 - Rusticl/ROCm Compatibility Report

- Date: 2026-02-07T23:49:02.381681+00:00
- Host: `XeonStation`
- Python: `3.12.3`

## Compatibility Summary

- Clover GPU available (default): True
- Rusticl GPU visible (default env): False
- Rusticl GPU activatable (`RUSTICL_ENABLE=radeonsi`): True
- Clover avg GFLOPS (microbench): 641.616
- Rusticl avg GFLOPS (microbench): 658.612
- Rusticl/Clover ratio: 1.026
- ROCm tools present: False
- Production platform selection hardcoded index-0: True

## Guardrail Checks

| Check | Observed | Requirement | Pass |
| --- | --- | --- | --- |
| clover_default_gpu_available | True | True | True |
| rusticl_gpu_visible_without_env | False | True | False |
| rusticl_gpu_activatable_with_env | True | True | True |
| rusticl_perf_ratio_vs_clover | 1.0264887473478093 | >= 0.9 | True |
| rocm_tools_present | False | False | True |
| production_platform_selection_explicit | False | True | False |

## Formal Decision

- Decision: `refine`
- Rationale: Rusticl is technically viable in shadow mode, but production path is pinned to platform index 0; selector hardening is required before any canary promotion.

## Raw Command Exit Codes

- `verify_hardware.py`: 0
- `verify_drivers.py --json`: 1
- `clinfo --list`: 0
- `rocminfo`: 127
- `rocm-smi`: 127

