# Acta Week 4 - Block 2 (T5 Coverage Refinement)

- Date: 2026-02-07
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope: refinar cobertura ABFT detect-only para elevar recall en `uniform_random` manteniendo overhead bajo.

## Objetivo

Cumplir el objetivo explicitado para Block 2:
1. mejorar deteccion de fallos no monitoreados (`uniform_random`),
2. sostener overhead operativo bajo,
3. validar correctness y stop-rule,
4. dejar decision formal del track.

## Implementacion

Runner refinado:
- `research/breakthrough_lab/t5_reliability_abft/run_t5_abft_detect_only.py`

Cambios clave:
- adicion de banco de proyecciones checksum (`projection_count=4`) en `float32`,
- deteccion basada en union de:
  - checksums de filas/columnas muestreadas,
  - checksums por proyeccion,
- ajuste de modos a verificacion periodica esparsa (`periodic_4`, `periodic_8`),
- criterios de pase de modo incluyen recall critico y uniform.

## Ejecucion Formal

Command:
- `./venv/bin/python research/breakthrough_lab/t5_reliability_abft/run_t5_abft_detect_only.py --sessions 4 --iterations 8 --warmup 2 --sizes 1400 2048 --sampling-periods 4 8 --row-samples 16 --col-samples 16 --projection-count 4 --faults-per-matrix 2 --seed 42`

Artifacts:
- `research/breakthrough_lab/t5_reliability_abft/week4_t5_abft_detect_only_20260207_205124.json`
- `research/breakthrough_lab/t5_reliability_abft/week4_t5_abft_detect_only_20260207_205124.md`

## Resultados

Comparativa de modos:
- `periodic_4`:
  - overhead: `2.335%`
  - critical recall: `1.000`
  - uniform recall: `1.000`
  - false positive rate: `0.000`
- `periodic_8` (recomendado):
  - overhead: `1.206%`
  - critical recall: `1.000`
  - uniform recall: `1.000`
  - critical misses: `0`
  - false positive rate: `0.000`

Validaciones:
- correctness pass (`max_error <= 1e-3`): **True**
- stop rule (`overhead > 8%` sin recall suficiente): **not triggered**

## Decision Formal

Track `t5_reliability_abft`: **iterate**.

Razonamiento:
- El objetivo del bloque se cumple: mejora fuerte de cobertura (`uniform_random`) con overhead bajo.
- Se requiere siguiente bloque de endurecimiento (stress prolongado e integracion controlada) antes de promocion.

## Estado de Bloque

`Week 4 - Block 2 (T5)` queda ejecutado con evidencia y decision formal registradas.
