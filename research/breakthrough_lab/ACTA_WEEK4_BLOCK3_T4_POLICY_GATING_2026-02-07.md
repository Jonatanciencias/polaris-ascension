# Acta Week 4 - Block 3 (T4 Policy Gating by Compressibility)

- Date: 2026-02-07
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope: refinar policy de activacion T4 por compresibilidad para reducir fallback total sin romper contrato de error.

## Objetivo

Ejecutar el bloque solicitado de T4:
1. introducir gating por compresibilidad,
2. reservar fallback solo para violaciones reales de contrato post-check,
3. medir impacto en fallback total, seguridad y speedup en cargas elegibles,
4. emitir decision formal de gate.

## Implementacion

Nuevo runner:
- `research/breakthrough_lab/t4_approximate_gemm/run_t4_policy_gating.py`

Policy artifact generado:
- `research/breakthrough_lab/t4_approximate_gemm/policy_activation_block3.json`

## Ejecucion Formal

Command:
- `./venv/bin/python research/breakthrough_lab/t4_approximate_gemm/run_t4_policy_gating.py --sessions 6 --sizes 512 1024 1400 --families dense_random compressible_lowrank --target-rank 16 --error-budget 0.005 --precheck-energy-threshold 0.95 --sample-size 64 --seed 42`

Artifacts:
- `research/breakthrough_lab/t4_approximate_gemm/week4_t4_policy_gating_20260207_224256.json`
- `research/breakthrough_lab/t4_approximate_gemm/week4_t4_policy_gating_20260207_224256.md`
- `research/breakthrough_lab/t4_approximate_gemm/policy_activation_block3.json`

## Resultados

Resumen global:
- contract compliance: `1.000`
- post-fallback violation rate: `0.000`
- fallback rate: `0.000`
- policy exact-route rate: `0.500`
- approximate-attempt rate: `0.500`
- stop rule: **not triggered**

Metricas clave:
- speedup compresible vs exact: `3.022x`
- dense_random: `policy_exact_route=1.000`, `fallback=0.000`
- max error ejecutado: `0.002532` (`<= 0.005`)

Interpretacion:
- La policy separa correctamente workloads elegibles/no elegibles por compresibilidad.
- Se elimina fallback operacional innecesario: exact route por policy para baja compresibilidad en lugar de fallback preventivo.
- Se mantiene seguridad contractual sin escapes.

## Decision Formal

Track `t4_approximate_gemm`: **promote** (scoped).

Razonamiento:
- Gate de seguridad pasa (compliance y post-fallback violation).
- Fallback total se reduce a cero bajo la nueva semantica operacional.
- Speedup fuerte y estable en el dominio elegible (`compressible_lowrank`).

## Estado de Bloque

`T4 - Block 3` queda ejecutado con evidencia reproducible, policy artefactada y decision formal registrada.
