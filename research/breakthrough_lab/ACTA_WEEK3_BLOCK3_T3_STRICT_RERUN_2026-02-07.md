# Acta Week 3 - Block 3 (T3 Bootstrap/Guardrails Redesign + Strict Rerun)

- Date: 2026-02-07
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope: rediseno de politica T3 en shadow mode con bootstrap de evidencia T2, guardrails formales y rerun estricto.

## Objetivo

Ejecutar el cierre tecnico de T3 en Week 3 / Block 3:
1. redisenar bootstrap de politica online usando evidencia formal de T2,
2. agregar guardrails operacionales por contexto/tamano,
3. validar rerun estricto y emitir decision formal `continue/refine/stop/promote`.

## Implementacion

Runner actualizado:
- `research/breakthrough_lab/t3_online_control/run_t3_shadow_policy.py`

Cambios clave:
- bootstrap priors por tamano desde `week2_t2_expanded_search_20260207_194454.json`,
- politica `GuardedContextualBandit` con pseudo-count inicial y warmup por contexto,
- allowlist de brazos por tamano con umbral minimo de uplift esperado (`min_delta_for_nonstatic`),
- freeze explicito en tamanos criticos (`1400`, `2048`) con forzado a brazo estatico,
- contexto determinista por tamano (`size_<N>`) y carga estricta por rotacion,
- stop rules estrictas por correctness/fallback rate.

## Ejecucion Formal

Command:
- `./venv/bin/python research/breakthrough_lab/t3_online_control/run_t3_shadow_policy.py --epochs 3 --runs-per-decision 8 --warmup 2 --epsilon 0.05 --fallback-regression-limit 0.08 --max-fallback-rate 0.10 --correctness-threshold 1e-3 --seed 42 --bootstrap-weight 3.0 --warmup-steps-per-context 2 --min-delta-for-nonstatic 8.0 --freeze-sizes 1400 2048 --boundary-report-path research/breakthrough_lab/t2_auto_scheduler/week2_t2_expanded_search_20260207_194454.json`

Artifacts:
- `research/breakthrough_lab/t3_online_control/week3_t3_shadow_policy_20260207_201856.json`
- `research/breakthrough_lab/t3_online_control/week3_t3_shadow_policy_20260207_201856.md`

## Resultados

- Steps executed: `24/24`
- Mean uplift vs static: `+10.631%`
- P95 latency delta vs static: `+0.000%`
- Fallback rate: `0.000` (threshold `0.100`)
- Correctness failures: `0`
- Stop rule: **not triggered**
- Promotion gate: **passed**

Lectura por tamanos (resumen):
- mejoras fuertes en `1200`, `1280`, `1600`, `1920`,
- neutralidad controlada en `1400` y `2048` por freeze guardrail,
- leve mejora positiva en `1792` sin regresion.

## Decision Formal

Track `t3_online_control`: **promote** (para progresion controlada del roadmap).

Razonamiento:
- Se cumple el gate formal en rerun estricto: uplift, latencia, correctness y fallback.
- El rediseno elimina el riesgo observado en Block 1 (fallback temprano).
- El comportamiento es determinista y auditable bajo guardrails explicitos.

## Estado de Bloque

`Week 3 - Block 3 (T3)` queda ejecutado con evidencia y decision formal registradas.
