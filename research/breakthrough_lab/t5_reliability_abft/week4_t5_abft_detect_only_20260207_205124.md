# T5 Week 4 Block 2 - ABFT-lite Coverage Refinement Report

- Date: 2026-02-07T20:51:24.779674+00:00
- Sizes: [1400, 2048] | Sessions=4 | Iterations=8
- Sampling periods: [4, 8] | Row samples=16 | Col samples=16
- Projection checks: count=4
- Fault injection: faults_per_matrix=2, models=['critical_monitored', 'uniform_random']

## Summary

- Recommended mode: `periodic_8`
- Decision hint: `iterate` (ABFT-lite detect-only achieves critical and uniform recall targets with low overhead in validated periodic mode; continue toward integration hardening.)
- Stop rule triggered: False (not_triggered)

## Mode Comparison

| Mode | Coverage | Overhead % | Critical Recall | Uniform Recall | Critical Misses | False Pos Rate | Correctness | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| periodic_4 | 0.250 | 2.335 | 1.000 | 1.000 | 0 | 0.000 | True | True |
| periodic_8 | 0.125 | 1.206 | 1.000 | 1.000 | 0 | 0.000 | True | True |

## Recommended Mode Details

- Kernel GFLOPS mean: 858.164
- Effective GFLOPS mean (with ABFT): 845.935
- Effective overhead: 1.206%
- Critical recall: 1.000
- Uniform-random recall: 1.000

## Per-Size (Recommended Mode)

| Size | Kernel | Coverage | Overhead % | Critical Recall | Uniform Recall |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1400 | tile20_v3_1400 | 0.125 | 2.256 | 1.000 | 1.000 |
| 2048 | tile24 | 0.125 | 0.922 | 1.000 | 1.000 |
