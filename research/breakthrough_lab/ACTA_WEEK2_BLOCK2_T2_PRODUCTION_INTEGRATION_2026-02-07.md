# Acta Week 2 - Block 2 (Integración Producción T2)

- Date: 2026-02-07
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope: integración en producción del candidato T2 promovido con alcance 1400 y fallback seguro.

## Cambios Integrados

1. Selector de producción:
- `src/optimization_engines/adaptive_kernel_selector.py`
- Nuevo target: `tile20_v3_1400` (`src/kernels/gemm_tile20_v3_vectorized.cl`)
- Política:
  - `M=N=K=1400` -> `tile20_v3_1400`
  - resto de tamaños -> fallback a política previa (`tile20`/`tile24`)

2. Benchmark de producción en modo `auto`:
- `src/benchmarking/production_kernel_benchmark.py`
- `auto` ahora respeta el scope promovido para `size == 1400`.

3. CLI:
- `src/cli.py`
- soporte explícito para `--kernel tile20_v3_1400`.

4. Validación de integración:
- `test_production_system.py`
- chequeo de archivo nuevo + asserts de scope/fallback.

## Evidencia Ejecutada

1. Suite funcional de producción:
- Command: `./venv/bin/python test_production_system.py`
- Result: `4/4 tests passed`

2. Test suite del repo:
- Command: `./venv/bin/pytest tests/ -q`
- Result: `69 passed in 11.74s`

3. Baseline reproducible canónico (referencia):
- Command: `./venv/bin/python scripts/benchmark_phase3_reproducible.py --sessions 10 --iterations 20 --output-dir results/benchmark_reports --prefix week2_block2_t2_integration_phase3_repro`
- Artifacts:
  - `results/benchmark_reports/week2_block2_t2_integration_phase3_repro_20260207_135909.json`
  - `results/benchmark_reports/week2_block2_t2_integration_phase3_repro_20260207_135909.md`

4. Benchmark producción `auto` post-integración (10x20):
- 1400:
  - `results/benchmark_reports/cli_production_benchmark_20260207_135929.json`
  - `results/benchmark_reports/cli_production_benchmark_20260207_135929.md`
- 2048:
  - `results/benchmark_reports/cli_production_benchmark_20260207_135945.json`
  - `results/benchmark_reports/cli_production_benchmark_20260207_135945.md`
- 3072:
  - `results/benchmark_reports/cli_production_benchmark_20260207_140016.json`
  - `results/benchmark_reports/cli_production_benchmark_20260207_140016.md`

## Resumen de Métricas (Post-Integración)

- `1400 auto` (scope promovido):
  - peak mean: `909.7 GFLOPS`
  - avg mean: `889.5 GFLOPS`
  - max_error mean: `0.000298`
  - vs baseline reproducible 1400 (`776.1`): `+17.2%`

- `2048 auto` (fallback):
  - peak mean: `775.5 GFLOPS`
  - avg mean: `712.9 GFLOPS`
  - max_error mean: `0.000641`
  - comportamiento en línea con baseline de fallback.

- `3072 auto` (fallback):
  - peak mean: `804.9 GFLOPS`
  - avg mean: `694.0 GFLOPS`
  - max_error mean: `0.000763`
  - sin evidencia de regresión de correctitud.

## Checklist de Integración

- [x] integración del kernel promovido en ruta de producción
- [x] scope acotado + fallback documentado y validado
- [x] tests funcionales y suite de pruebas en verde
- [x] benchmark reproducible ejecutado con artefactos
- [x] benchmark `auto` multi-size ejecutado con artefactos
- [x] correctitud dentro de umbral estricto observado

## Decisión

`T2` integración de producción: **approved** (scope `1400` + fallback).

La integración queda formalmente validada para continuar con el cierre de `T1` en el siguiente bloque.
