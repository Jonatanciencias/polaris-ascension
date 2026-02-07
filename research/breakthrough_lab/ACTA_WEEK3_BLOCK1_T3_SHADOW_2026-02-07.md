# Acta Week 3 - Block 1 (T3 Shadow Policy)

- Date: 2026-02-07
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope: primer experimento ejecutable de `t3_online_control` en modo shadow contra selector estático.

## Preparación y Correcciones Previas

Durante la ejecución se detectó y corrigió un desvío de política en producción:
- `src/optimization_engines/adaptive_kernel_selector.py`
  - `tile20_v3_1400` estaba participando fuera de su scope validado.
  - `tile16` (debug path) aparecía como elegible en selección productiva.

Corrección aplicada:
- scope estricto para `tile20_v3_1400`.
- exclusión de `tile16` del conjunto elegible de producción.

Validación posterior:
- `./venv/bin/python test_production_system.py` -> `4/4 tests passed`.

## Experimento Ejecutado

Command:
- `./venv/bin/python research/breakthrough_lab/t3_online_control/run_t3_shadow_policy.py --epochs 2 --runs-per-decision 8 --warmup 2 --epsilon 0.15 --fallback-regression-limit 0.10 --max-fallback-rate 0.20 --seed 42`

Artifacts:
- `research/breakthrough_lab/t3_online_control/week3_t3_shadow_policy_20260207_195235.json`
- `research/breakthrough_lab/t3_online_control/week3_t3_shadow_policy_20260207_195235.md`

## Resultados

- Steps executed: `3/16`
- Mean delta vs static: `+0.000%`
- Fallback rate: `0.333` (threshold `0.200`)
- Correctness failures: `0`
- Stop rule: **triggered** (`fallback rate exceeded threshold`)

## Decisión Formal

Track `t3_online_control`: **iterate** (prototype dropped, track continues).

Razonamiento:
- El primer prototipo no cumple condición mínima de seguridad operacional en shadow mode.
- La falla es de política (control de fallback temprano), no de exactitud numérica.
- Se requiere rediseño del policy bootstrap (prior conservador + warm-start por contexto) antes del siguiente intento.

## Estado de Bloque

`Week 3 - Block 1 (T3)` queda ejecutado con evidencia formal y decisión registrada.
