# Acta Week 7 - Block 1 (Hardening Selector de Plataforma OpenCL)

- Date: 2026-02-08
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope: hardening de selección explícita de plataforma/dispositivo para habilitar canary Rusticl con fallback operativo.

## Objetivo

1. Eliminar dependencia de selección por índice (`cl.get_platforms()[0]`) en benchmark productivo.
2. Exponer selector explícito de plataforma/dispositivo desde API y CLI.
3. Validar canary Rusticl en ejecución real, manteniendo correctness bajo contrato.

## Implementación

Cambios de código:
- `src/benchmarking/production_kernel_benchmark.py`
  - nuevo selector `_select_opencl_runtime(...)`
  - soporte `opencl_platform`, `opencl_device`, `rusticl_enable`
  - metadata de selección en reportes productivos
- `src/cli.py`
  - flags nuevos: `--opencl-platform`, `--opencl-device`, `--rusticl-enable`
  - inicialización lazy para permitir aplicar `RUSTICL_ENABLE` antes de descubrimiento OpenCL
- `tests/test_production_opencl_selection.py`
  - cobertura de selección explícita, env selectors y error path
- `research/breakthrough_lab/platform_compatibility/run_week7_platform_selector_hardening.py`

## Ejecución formal

Command:
- `./venv/bin/python research/breakthrough_lab/platform_compatibility/run_week7_platform_selector_hardening.py --size 1024 --sessions 3 --iterations 5 --seed 42`

Artifacts:
- `research/breakthrough_lab/platform_compatibility/week7_platform_selector_hardening_20260208_000425.json`
- `research/breakthrough_lab/platform_compatibility/week7_platform_selector_hardening_20260208_000425.md`

## Resultados

Clover explícito:
- platform: `Clover`
- peak mean: `432.698` GFLOPS
- max error: `0.0001907`

Rusticl canary explícito:
- platform: `rusticl`
- peak mean: `396.921` GFLOPS
- max error: `0.0001907`

Checks:
- selección explícita Clover: `pass`
- selección explícita Rusticl: `pass`
- correctness bound (`<=1e-3`) Clover/Rusticl: `pass`
- ratio rusticl/clover (`>=0.85`): `pass` (`0.917`)

Tests de soporte:
- `pytest -q tests/test_production_opencl_selection.py tests/test_t3_controlled_policy.py tests/test_t5_abft_guardrails.py`
- resultado: `9 passed`

## Decisión formal

Track `platform_compatibility`: **promote**.

Razonamiento:
- La selección explícita quedó productiva y verificable por metadata.
- El canary Rusticl ahora puede activarse de forma controlada desde CLI/API sin fallback implícito por índice.
- La exactitud se mantiene dentro de contrato en ambas rutas.

## Estado del bloque

`Week 7 - Block 1` queda ejecutado con evidencia reproducible y decisión formal registrada.
