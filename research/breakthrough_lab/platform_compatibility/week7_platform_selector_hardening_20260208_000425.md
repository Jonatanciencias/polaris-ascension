# Week 7 Block 1 - Platform Selector Hardening Report

- Date: 2026-02-08T00:04:25.876256+00:00
- Size: 1024
- Sessions/Iterations: 3/5

## Run Summary

| Route | Status | Platform | Peak mean GFLOPS | Max error |
| --- | --- | --- | ---: | ---: |
| Clover explicit | ok | Clover | 432.698 | 0.0001907 |
| Rusticl canary | ok | rusticl | 396.921 | 0.0001907 |

## Guardrail Checks

| Check | Pass |
| --- | --- |
| clover_explicit_selection | True |
| rusticl_canary_selection | True |
| correctness_bound_clover | True |
| correctness_bound_rusticl | True |
| rusticl_peak_ratio_vs_clover | True |

## Decision

- Decision: `promote`
- Rationale: Explicit platform selection works for Clover and Rusticl canary with bounded correctness.

