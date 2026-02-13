# Acta Week 9 - Block 3 (Robustness Replay Seeds + Platform Split)

- Date: 2026-02-08
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope: replay robusto post-hardening T5 con semillas alternas y split corto por plataforma (`Clover`/`rusticl`).

## Objetivo

1. Confirmar que el hardening de Week 9 Block 2 no introduce regresiones bajo semillas alternas.
2. Validar guardrails T3/T5 en split corto por plataforma.
3. Comparar Clover post-replay contra referencia de Week 9 Block 2.

## Implementacion

Nuevo runner del bloque:

- `research/breakthrough_lab/platform_compatibility/run_week9_block3_robustness_replay.py`

Capacidades clave:
- barrido por semillas (`7`, `42`, `1337`) y tamanos criticos (`1400`, `2048`),
- ejecucion dual plataforma (`Clover` + `rusticl`),
- policy T5 hardening de Block 2 aplicada explicita,
- comparativa automatica contra baseline Block 2 (delta throughput/p95 en Clover),
- decision formal automatica (`promote|iterate|drop`).

## Ejecucion Formal

Commands:

- `./venv/bin/python research/breakthrough_lab/platform_compatibility/run_week9_block3_robustness_replay.py --seeds 7 42 1337 --sizes 1400 2048 --kernels auto_t3_controlled auto_t5_guarded --sessions 1 --iterations 6`
- `./venv/bin/python scripts/run_validation_suite.py --tier canonical --driver-smoke --report-dir research/breakthrough_lab/week8_validation_discipline`

Artifacts:

- `research/breakthrough_lab/platform_compatibility/run_week9_block3_robustness_replay.py`
- `research/breakthrough_lab/platform_compatibility/week9_block3_robustness_replay_20260208_033111.json`
- `research/breakthrough_lab/platform_compatibility/week9_block3_robustness_replay_20260208_033111.md`
- `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_033147.json`
- `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_033147.md`

## Resultados

Replay robusto:
- Decision: `promote`
- Runs OK: `24/24`
- Correctness max global: `0.000640869140625` (`<= 1e-3`)
- T3 fallback max: `0.0`, policy disabled count: `0`
- T5 disable total: `0`, false-positive max: `0.0`, overhead max: `2.476%` (`<= 3.0`)

Split plataforma:
- Clover y rusticl ejecutados correctamente en todas las combinaciones.
- Ratio minimo rusticl/clover (peak): `0.9209` (`>= 0.80`)

No regresion vs Week9 Block2 (Clover):
- `auto_t5_guarded` 1400: throughput `+0.688%`, p95 `+0.355%`
- `auto_t5_guarded` 2048: throughput `+0.036%`, p95 `+0.094%`
- Ambos dentro de limites de no-regresion configurados.

Gate canonico obligatorio:
- `validation_suite canonical + driver_smoke`: **promote**

## Decision Formal

Tracks:
- `robustness_replay_alternate_seeds`: **promote**
- `platform_split_short_post_hardening`: **promote**

Block decision:
- **promote**

Razonamiento:
- El hardening T5 se mantiene estable en semillas alternas.
- No hay regresion de guardrails ni de performance relevante respecto a Block 2.
- Split Clover/rusticl queda validado para escenario corto de control.

## Estado del Bloque

`Week 9 - Block 3` cerrado con `promote`, evidencia reproducible y gate canonico en verde.
