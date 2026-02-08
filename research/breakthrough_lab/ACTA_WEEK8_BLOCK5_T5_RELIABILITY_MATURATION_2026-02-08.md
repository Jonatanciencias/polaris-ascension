# Acta Week 8 - Block 5 (T5 Reliability Maturation)

- Date: 2026-02-08
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope: maduracion de ABFT-lite con fault-injection y tuning recall/overhead sobre policy T5.

## Objetivo

1. Ejecutar comparativa formal baseline vs candidate en fault-injection determinista.
2. Mejorar recall en `uniform_random` sin romper correctness ni guardrails de overhead.
3. Versionar policy refinada de T5 con thresholds explícitos de maduración.
4. Cerrar bloque con evidencia + decision formal + gate canónico obligatorio.

## Implementación

Cambios aplicados:

- `research/breakthrough_lab/t5_reliability_abft/policy_hardening_block5.json`
  - policy candidata de maduración con `projection_count=6` y guardrails de delta vs baseline.
- `research/breakthrough_lab/t5_reliability_abft/run_week8_t5_maturation.py`
  - runner comparativo baseline/candidate para evaluar tradeoff recall-overhead.
- `research/breakthrough_lab/t5_reliability_abft/results.json`
  - actualizado a experimento Week 8 Block 5.
- `research/breakthrough_lab/t5_reliability_abft/report.md`
  - actualizado con resultado de promoción del bloque.

## Ejecución Formal

Commands:

- `./venv/bin/python research/breakthrough_lab/t5_reliability_abft/run_week8_t5_maturation.py`
- `./venv/bin/python scripts/run_validation_suite.py --tier canonical --driver-smoke --report-dir research/breakthrough_lab/week8_validation_discipline`

Artifacts:

- `research/breakthrough_lab/t5_reliability_abft/week8_t5_maturation_20260208_022633.json`
- `research/breakthrough_lab/t5_reliability_abft/week8_t5_maturation_20260208_022633.md`
- `research/breakthrough_lab/t5_reliability_abft/policy_hardening_block5.json`
- `research/breakthrough_lab/t5_reliability_abft/results.json`
- `research/breakthrough_lab/t5_reliability_abft/report.md`
- `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_022835.json`
- `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_022835.md`

## Resultados

- Uniform recall: `0.967 -> 0.983` (`+0.017`).
- Critical recall: `1.000 -> 1.000`.
- Effective overhead: `1.165% -> 1.230%` (`+0.065%`, dentro de guardrail).
- False positive rate: `0.000`.
- Correctness max error: `0.0005646` (`<=1e-3`).
- Gate canónico (`validation_suite canonical + driver_smoke`): **promote**.

## Decisión Formal

Track `t5_reliability_abft`: **promote**.

Razonamiento:

- La policy candidata mejora recall uniform de forma verificable sin degradar seguridad crítica.
- El costo en overhead permanece acotado y por debajo de límites de operación.
- El gate canónico de validación cerró en verde, consolidando paridad local/CI.

## Estado del Bloque

`Week 8 - Block 5` ejecutado con evidencia reproducible y decisión formal registrada.
