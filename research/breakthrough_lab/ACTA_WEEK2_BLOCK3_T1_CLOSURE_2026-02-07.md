# Acta Week 2 - Block 3 (Cierre T1)

- Date: 2026-02-07
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope: cierre formal de `t1_io_aware` con último intento técnico y decisión `continue/refine/stop`.

## Objetivo del Bloque

Ejecutar el punto 3 del roadmap:
- último intento técnico de T1, o
- decisión formal de corte si no cumple criterios.

## Cambios Ejecutados

1. Runner T1 actualizado:
- `research/breakthrough_lab/t1_io_aware/run_week2_t1_campaign.py`
- Se añadió tercera variante: `io_hybrid_sizeaware_v1` (estrategia explícita large-size).
- Se añadió objetivo ponderado por tamaño: `1400:0.2`, `2048:0.4`, `3072:0.4`.
- Se añadió evaluación explícita de `stop_rule` del experimento.

2. Corrida formal canónica:
- Command: `./venv/bin/python research/breakthrough_lab/t1_io_aware/run_week2_t1_campaign.py --sessions 10 --iterations 20`
- Artifacts:
  - `research/breakthrough_lab/t1_io_aware/week2_t1_io_campaign_20260207_193227.json`
  - `research/breakthrough_lab/t1_io_aware/week2_t1_io_campaign_20260207_193227.md`

3. Snapshot del track actualizado:
- `research/breakthrough_lab/t1_io_aware/results.json`
- `research/breakthrough_lab/t1_io_aware/report.md`

## Resumen de Resultados

Baseline vs candidatos (peak mean GFLOPS):

- Size `1400` (baseline `784.401`):
  - `io_prefetch_v1`: `+4.945%`
  - `io_regblock_v1`: `-21.121%`
  - `io_hybrid_sizeaware_v1`: `+4.891%`

- Size `2048` (baseline `776.946`):
  - `io_prefetch_v1`: `-68.473%`
  - `io_regblock_v1`: `-43.788%`
  - `io_hybrid_sizeaware_v1`: `-0.156%`

- Size `3072` (baseline `804.174`):
  - `io_prefetch_v1`: `-75.637%`
  - `io_regblock_v1`: `-62.009%`
  - `io_hybrid_sizeaware_v1`: `-0.375%`

Mejor variante por objetivo ponderado:
- `io_hybrid_sizeaware_v1`
- weighted delta: `+0.766%`
- mean delta: `+1.454%`
- correctness/stability: pass en todos los tamaños.

## Verificación de Stop Rule

Regla (experiment card T1):
- después de 3 variantes,
- si ninguna logra `+5%` en `1400` o `2048`, cortar.

Resultado:
- variantes evaluadas: `3`
- hits >= `+5%` en `1400/2048`: `0`
- `stop_rule.triggered = true`

## Decisión Formal

Track `t1_io_aware`: **stop**.

Razonamiento:
- Se completó el último intento técnico solicitado.
- Persisten regresiones estructurales en large-size para variantes puras T1.
- La estrategia híbrida estabiliza pero no logra uplift suficiente para justificar continuidad del track en su hipótesis actual.
- No cumple objetivo de promoción ni criterio mínimo de continuidad definido por la regla de parada.

## Estado del Roadmap

- `T2` producción (scope 1400 + fallback): integrado y validado.
- `T1`: cerrado formalmente (`stop`) en este bloque.

