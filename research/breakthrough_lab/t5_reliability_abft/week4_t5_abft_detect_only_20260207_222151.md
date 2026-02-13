# T5 Week 4 Block 2 - ABFT-lite Coverage Refinement Report

- Date: 2026-02-07T22:21:51.202066+00:00
- Sizes: [1400, 2048] | Sessions=10 | Iterations=24
- Sampling periods: [8, 16] | Row samples=16 | Col samples=16
- Projection checks: count=4
- Fault injection: faults_per_matrix=2, models=['critical_monitored', 'uniform_random']

## Summary

- Recommended mode: `periodic_8`
- Decision hint: `iterate` (ABFT-lite detect-only achieves critical and uniform recall targets with low overhead in validated periodic mode; continue toward integration hardening.)
- Stop rule triggered: False (not_triggered)

## Mode Comparison

| Mode | Coverage | Overhead % | Critical Recall | Uniform Recall | Critical Misses | False Pos Rate | Correctness | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| periodic_8 | 0.125 | 1.209 | 1.000 | 0.967 | 0 | 0.000 | True | True |
| periodic_16 | 0.083 | 0.922 | 1.000 | 0.950 | 0 | 0.000 | True | True |

## Recommended Mode Details

- Kernel GFLOPS mean: 857.840
- Effective GFLOPS mean (with ABFT): 846.316
- Effective overhead: 1.209%
- Critical recall: 1.000
- Uniform-random recall: 0.967

## Per-Size (Recommended Mode)

| Size | Kernel | Coverage | Overhead % | Critical Recall | Uniform Recall |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1400 | tile20_v3_1400 | 0.125 | 2.043 | 1.000 | 0.967 |
| 2048 | tile24 | 0.125 | 0.984 | 1.000 | 0.967 |
