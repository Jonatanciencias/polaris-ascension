# Acta Week 8 - Block 4 (T4 Mixed Policy Refinement)

- Date: 2026-02-08
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope: refinar policy de activacion T4 en workload mixto para reducir fallback sin romper contrato de error.

## Objetivo

1. Ejecutar campaña mixta `baseline vs candidate` para T4 con comparación formal de fallback/contrato.
2. Versionar policy refinada con guardrails rollback-safe.
3. Cerrar bloque con evidencia formal + decisión governance.
4. Validar gate obligatorio: `run_validation_suite.py --tier canonical --driver-smoke`.

## Implementación

Cambios aplicados:

- `research/breakthrough_lab/t4_approximate_gemm/policy_activation_block4.json`
  - policy versionada Block 4 con `target_rank=18`, scope mixto y guardrails de reducción de fallback.
- `research/breakthrough_lab/t4_approximate_gemm/run_week8_t4_mixed_campaign.py`
  - runner formal para comparar baseline (`block3`) vs candidate (`block4`) en mismo scope.
- `research/breakthrough_lab/t4_approximate_gemm/results.json`
  - normalizado al experimento Week 8 Block 4.
- `research/breakthrough_lab/t4_approximate_gemm/report.md`
  - actualizado al estado de promoción del bloque.

## Ejecución Formal

Commands:

- `./venv/bin/python research/breakthrough_lab/t4_approximate_gemm/run_week8_t4_mixed_campaign.py --sessions 6 --seed 42`
- `./venv/bin/python scripts/run_validation_suite.py --tier canonical --driver-smoke --report-dir research/breakthrough_lab/week8_validation_discipline`

Artifacts:

- `research/breakthrough_lab/t4_approximate_gemm/week8_t4_mixed_campaign_20260208_021541.json`
- `research/breakthrough_lab/t4_approximate_gemm/week8_t4_mixed_campaign_20260208_021541.md`
- `research/breakthrough_lab/t4_approximate_gemm/policy_activation_block4.json`
- `research/breakthrough_lab/t4_approximate_gemm/results.json`
- `research/breakthrough_lab/t4_approximate_gemm/report.md`
- `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_021724.json`
- `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_021724.md`

## Resultados

- Candidate contract compliance: `1.000` (baseline `1.000`).
- Candidate post-fallback violation rate: `0.000`.
- Fallback rate: `0.194 -> 0.000` (reducción absoluta `0.194`).
- Compressible speedup vs exact: `1.370x -> 1.999x`.
- Gate canónico (`validation_suite`): **promote**.

## Decisión Formal

Track `t4_approximate_gemm`: **promote**.

Razonamiento:

- La policy refinada reduce fallback a cero manteniendo contrato de error y sin escapes post-fallback.
- La campaña mixta pasa todos los guardrails versionados de Block 4.
- El gate canónico de validación (local/CI parity) cerró en verde.

## Estado del Bloque

`Week 8 - Block 4` ejecutado con evidencia reproducible y decisión formal registrada.
