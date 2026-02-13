# Baseline Runbook (Source of Truth)

## Objective

Provide one canonical baseline procedure used by all breakthrough experiments.

## Preconditions

- run from repository root
- active Python venv at `./venv`
- OpenCL device available

## Canonical Commands

1. Functional validation:
```bash
./venv/bin/python test_production_system.py
```

2. Test suite:
```bash
./venv/bin/pytest tests/ -q
```

3. Reproducible production baseline:
```bash
./venv/bin/python scripts/benchmark_phase3_reproducible.py \
  --sessions 10 \
  --iterations 20 \
  --output-dir results/benchmark_reports \
  --prefix phase3_repro_baseline
```

4. CLI production benchmark:
```bash
./venv/bin/python -m src.cli benchmark \
  --mode production \
  --kernel auto \
  --size 1400 \
  --sessions 10 \
  --iterations 20 \
  --report-dir results/benchmark_reports
```

## Required Artifacts

- `results/benchmark_reports/phase3_repro_baseline_*.json`
- `results/benchmark_reports/phase3_repro_baseline_*.md`
- `results/benchmark_reports/cli_production_benchmark_*.json`
- `results/benchmark_reports/cli_production_benchmark_*.md`

## Baseline Reference

- protocol document:
  - `docs/PHASE3_REPRODUCIBLE_PERFORMANCE_BASELINE_FEB2026.md`
- baseline commit:
  - `57db7b4`
