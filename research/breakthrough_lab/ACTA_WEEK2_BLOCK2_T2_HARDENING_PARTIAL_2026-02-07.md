# Acta Week 2 - Block 2 (Parcial T2 Hardening)

- Date: 2026-02-07
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope: hardening de precisión en `t2_auto_scheduler` + rerun estricto con evidencia formal.

## Cambios de Hardening Ejecutados

1. `research/auto_tuner/gemm_auto_tuner.py`
- benchmark con `seed` determinista por configuración/sesión.
- distribución de entrada parametrizable (`standard_normal`/`uniform`).
- default actualizado a `standard_normal` para alinear con protocolo canónico.

2. `research/breakthrough_lab/t2_auto_scheduler/run_week2_t2_search.py`
- propagación de `seed` determinista en search/replay.
- trazabilidad explícita de `input_distribution` en metadata.
- ejecución en modo estricto (`max_error <= 1e-3`) mantenida.

## Evidencia Revisada

- `research/breakthrough_lab/t2_auto_scheduler/week2_t2_bounded_search_20260207_183138.json`
- `research/breakthrough_lab/t2_auto_scheduler/week2_t2_bounded_search_20260207_183138.md`
- `research/breakthrough_lab/t2_auto_scheduler/results.json`
- `research/breakthrough_lab/t2_auto_scheduler/report.md`
- `research/breakthrough_lab/PROMOTION_GATE_CHECKLIST.md`

## Resultado Formal del Re-Run Estricto

- Candidatos válidos tras filtro estricto: `6/6`
- Mejor candidato replay para uplift: `tile24@2048`
  - `783.923 GFLOPS`
  - `+6.026%` vs baseline
  - `max_error max = 0.000610` (pass)
  - `cv_peak = 0.000386` (pass)

## Decisión Parcial

- Track `t2_auto_scheduler`: `iterate`

Rationale:
- El bloqueo de correctitud estricta quedó resuelto con el hardening.
- Aún no se cumple gate de promoción (`>= +10%`), por lo que no procede promoción.

## Acciones Aprobadas para el Siguiente Paso

1. Expandir espacio de búsqueda en T2 (vector/unroll/local-size) manteniendo filtro estricto.
2. Conservar distribución canónica y seeds deterministas para comparabilidad.
3. Re-ejecutar replay top-k y reevaluar umbral de promoción.
