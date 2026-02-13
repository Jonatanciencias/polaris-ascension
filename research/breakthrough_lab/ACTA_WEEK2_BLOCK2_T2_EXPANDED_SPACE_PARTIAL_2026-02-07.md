# Acta Week 2 - Block 2 (Parcial T2 Expansión de Espacio)

- Date: 2026-02-07
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope: expansión de espacio T2 en dimensiones `vector/unroll/local-size`, manteniendo modo estricto y determinista.

## Cambios Ejecutados

1. `research/auto_tuner/gemm_auto_tuner.py`
- soporte de benchmark para kernels externos (`benchmark_custom_kernel`).
- caché de compilación por `(kernel_file, kernel_name, build_options)`.
- ejecución determinista por `seed` + distribución de entrada canónica.

2. `research/breakthrough_lab/t2_auto_scheduler/run_week2_t2_search.py`
- nuevo modo `--search-space expanded`.
- catálogo explícito de configuraciones con metadata de dimensiones:
  - `vector_width`
  - `unroll_k`
  - `local_size`
- ranking estricto con filtro de correctitud antes de replay.

## Evidencia Revisada

- `research/breakthrough_lab/t2_auto_scheduler/week2_t2_expanded_search_20260207_184001.json`
- `research/breakthrough_lab/t2_auto_scheduler/week2_t2_expanded_search_20260207_184001.md`
- `research/breakthrough_lab/t2_auto_scheduler/results.json`
- `research/breakthrough_lab/t2_auto_scheduler/report.md`
- `research/breakthrough_lab/PROMOTION_GATE_CHECKLIST.md`

## Resultado Formal del Re-Run Expandido

- Search space ejecutado: `6 configs x 3 sizes x 12 runs`
- Dimensiones cubiertas:
  - `vector_width`: `{4, 8}`
  - `unroll_k`: `{0, 4, 8, 10}`
  - `local_size`: `{5x5, 10x10, 12x12}`
- Candidatos válidos tras filtro estricto (`max_error <= 1e-3`): `16/18`

Top candidato replay:
- Config: `t20_v3vec_v4_u0_l10 @ 1400`
- Throughput: `926.303 GFLOPS`
- Delta vs baseline: `+15.838%`
- Correctness: `max_error max = 0.000336` (pass)
- Stability: `cv_peak = 0.00421` (pass)

## Decisión Parcial

- Track `t2_auto_scheduler`: `promote` (scoped)

Rationale:
- Se cumple gate de promoción para el candidato objetivo en tamaño 1400.
- Promoción acotada por rango de tamaño; para tamaños grandes se mantiene baseline actual.

## Acciones Aprobadas

1. Integrar candidato promovido como política de scheduling por rango de tamaño.
2. Mantener fallback a baseline en 2048/3072.
3. Ejecutar validación de frontera (`1200-1600` y `1536-2048`) antes de cierre de bloque completo.
