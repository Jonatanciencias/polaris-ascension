# Week15 Block3 Dependent Projects Onboarding

## Tier A - Core Runtime Consumers

- Inference runtimes using GEMM selector in production profile.
- Benchmark suites consuming `run_production_benchmark` contracts.

## Tier B - Extension / Plugin Projects

- Kernel-policy plugins adding strategy metadata and telemetry.
- Reliability plugins extending T5 checks without bypassing guardrails.

## Tier C - Ops / CI Projects

- CI jobs enforcing canonical gate and schema checks.
- Monitoring jobs running weekly replay + drift control.

## Adoption Sequence

1. Integrate template from `WEEK14_BLOCK6_PLUGIN_TEMPLATE.md`.
2. Run canonical gate + driver smoke.
3. Execute controlled dry-run profile and collect evidence.
4. Close local acta + decision before scope expansion.

