# Acta Week 4 - Block 3 (T5 Integration Hardening + Long Stress)

- Date: 2026-02-07
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope: hardening de integracion para T5 mediante campana de stress prolongada y politica operativa candidata.

## Objetivo

Ejecutar el bloque de endurecimiento previo a canary:
1. validar estabilidad en ventana larga,
2. confirmar recall critico sin misses y recall robusto en `uniform_random`,
3. mantener overhead bajo,
4. dejar policy de integracion con guardrails explicitos.

## Implementacion

Runner reutilizado:
- `research/breakthrough_lab/t5_reliability_abft/run_t5_abft_detect_only.py`

Hardening policy generada:
- `research/breakthrough_lab/t5_reliability_abft/policy_hardening_block3.json`

Parametros de stress:
- `sessions=10`, `iterations=24`, `sizes=1400 2048`
- modos: `periodic_8`, `periodic_16`
- `projection_count=4`

## Ejecucion Formal

Command:
- `./venv/bin/python research/breakthrough_lab/t5_reliability_abft/run_t5_abft_detect_only.py --sessions 10 --iterations 24 --warmup 2 --sizes 1400 2048 --sampling-periods 8 16 --row-samples 16 --col-samples 16 --projection-count 4 --faults-per-matrix 2 --seed 42`

Artifacts:
- `research/breakthrough_lab/t5_reliability_abft/week4_t5_abft_detect_only_20260207_222151.json`
- `research/breakthrough_lab/t5_reliability_abft/week4_t5_abft_detect_only_20260207_222151.md`
- `research/breakthrough_lab/t5_reliability_abft/policy_hardening_block3.json`

## Resultados

Modo recomendado: `periodic_8`
- overhead: `1.209%`
- critical recall: `1.000` (`60/60`)
- critical misses: `0`
- uniform recall: `0.967` (`58/60`)
- false positive rate: `0.000`
- correctness: pass (`max_error=0.0005646`)

Modo alterno `periodic_16`:
- overhead: `0.922%`
- uniform recall: `0.950` (limite inferior aceptable)

Stop rule:
- **not triggered**.

## Decision Formal

Track `t5_reliability_abft`: **iterate**.

Razonamiento:
- La evidencia de stress valida hardening suficiente para pasar a canary de integracion controlada.
- Aun no se promueve a produccion hasta completar el siguiente gate operativo (shadow canary + campaña extendida).

## Estado de Bloque

`Week 4 - Block 3 (T5)` queda ejecutado con evidencia, policy candidata y decision formal registradas.
