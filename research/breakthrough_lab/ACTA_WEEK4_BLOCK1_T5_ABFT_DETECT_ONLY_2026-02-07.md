# Acta Week 4 - Block 1 (T5 ABFT-lite Detect-only)

- Date: 2026-02-07
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope: primer bloque ejecutable de T5 con detector ABFT-lite, fault injection y medicion formal de overhead.

## Objetivo

Ejecutar el arranque formal de T5:
1. implementar ruta detect-only sin correccion automatica,
2. validar recall en campana de fallos inyectados,
3. cuantificar overhead operativo,
4. emitir decision formal del track.

## Implementacion

Nuevo runner:
- `research/breakthrough_lab/t5_reliability_abft/run_t5_abft_detect_only.py`

Capacidades incluidas:
- seleccion de kernel productivo por tamano (`tile20_v3_1400` / `tile24`),
- checksums ABFT-lite por filas/columnas muestreadas,
- inyeccion de fallos en dos modelos:
  - `critical_monitored`
  - `uniform_random`
- comparacion de modos de muestreo:
  - `always`
  - `periodic_4`
- metricas formales de recall, falsos positivos, overhead y correctness.

## Ejecucion Formal

Command:
- `./venv/bin/python research/breakthrough_lab/t5_reliability_abft/run_t5_abft_detect_only.py --sessions 4 --iterations 8 --warmup 2 --sizes 1400 2048 --sampling-periods 1 4 --row-samples 16 --col-samples 16 --faults-per-matrix 2 --seed 42`

Artifacts:
- `research/breakthrough_lab/t5_reliability_abft/week4_t5_abft_detect_only_20260207_203936.json`
- `research/breakthrough_lab/t5_reliability_abft/week4_t5_abft_detect_only_20260207_203936.md`

## Resultados

Comparativa de modos:
- `always`:
  - overhead efectivo: `3.639%`
  - critical recall: `1.000`
  - critical misses: `0`
  - false positive rate: `0.000`
- `periodic_4` (recomendado):
  - overhead efectivo: `0.973%`
  - critical recall: `1.000`
  - critical misses: `0`
  - false positive rate: `0.000`

Observacion de cobertura:
- En espacio de fallo `uniform_random`, el recall observado fue bajo (`0.000`) bajo muestreo esparso.

Stop rule T5:
- regla: detener si overhead > 8% sin ganancia de confiabilidad.
- resultado: **not triggered**.

## Decision Formal

Track `t5_reliability_abft`: **iterate**.

Razonamiento:
- El prototipo detect-only es funcional y cumple objetivo de overhead + deteccion critica.
- La cobertura de fallos no monitoreados aun no es suficiente para promocion.
- El siguiente bloque debe refinar policy de cobertura/sampling antes de integracion productiva.

## Estado de Bloque

`Week 4 - Block 1 (T5)` queda ejecutado con evidencia y decision formal registradas.
