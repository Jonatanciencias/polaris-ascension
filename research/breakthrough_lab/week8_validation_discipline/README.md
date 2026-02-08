# Week 8 - Validation Discipline

This folder contains the Week 8 Block 1 assets for the 2026Q2 cycle:

- unified local/CI validation execution
- driver diagnostics smoke checks
- formal evidence artifacts for promote/iterate decisions

## Standard Execution

CPU fast tier (CI-compatible):

`python scripts/run_validation_suite.py --tier cpu-fast --allow-no-tests --driver-smoke`

Canonical local tier:

`python scripts/run_validation_suite.py --tier canonical --driver-smoke --report-dir research/breakthrough_lab/week8_validation_discipline`

## Templates

- `acta_template.md`
- `decision_template.json`

Use these templates for every subsequent Week 8+ block to keep governance consistent.
