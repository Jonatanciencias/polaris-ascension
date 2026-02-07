# T5 Week 4 Block 1 - ABFT-lite Detect-only Report

- Date: 2026-02-07T20:39:36.989171+00:00
- Sizes: [1400, 2048] | Sessions=4 | Iterations=8
- Sampling periods: [1, 4] | Row samples=16 | Col samples=16
- Fault injection: faults_per_matrix=2, models=['critical_monitored', 'uniform_random']

## Summary

- Recommended mode: `periodic_4`
- Decision hint: `iterate` (ABFT-lite detect-only achieves fault-detection and overhead targets in at least one validated mode; continue toward integration hardening.)
- Stop rule triggered: False (not_triggered)

## Mode Comparison

| Mode | Coverage | Overhead % | Critical Recall | Critical Misses | False Pos Rate | Correctness | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| always | 1.000 | 3.639 | 1.000 | 0 | 0.000 | True | True |
| periodic_4 | 0.250 | 0.973 | 1.000 | 0 | 0.000 | True | True |

## Recommended Mode Details

- Kernel GFLOPS mean: 858.636
- Effective GFLOPS mean (with ABFT): 847.600
- Effective overhead: 0.973%
- Critical recall: 1.000
- Uniform-random recall: 0.000

## Per-Size (Recommended Mode)

| Size | Kernel | Coverage | Overhead % | Critical Recall | Uniform Recall |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1400 | tile20_v3_1400 | 0.250 | 1.919 | 1.000 | 0.000 |
| 2048 | tile24 | 0.250 | 0.718 | 1.000 | 0.000 |
