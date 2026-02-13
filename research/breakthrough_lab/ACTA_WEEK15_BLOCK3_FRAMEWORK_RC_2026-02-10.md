# Acta Week 15 - Block 3 (Paquete RC del framework base)

- Date: 2026-02-10
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope:
  - publicar paquete RC del framework base,
  - dejar checklist de adopción para proyectos dependientes,
  - cerrar decisión formal de publicación RC.

## Objetivo

1. Consolidar un release candidate reutilizable por proyectos dependientes.
2. Encadenar la publicación RC a evidencia real (`Block1=go`, `Block2=promote`, gate canónico verde).
3. Formalizar documentación de adopción y onboarding.

## Ejecución Formal

Gate canónico previo:

- `./venv/bin/python scripts/run_validation_suite.py --tier canonical --driver-smoke --report-dir research/breakthrough_lab/week8_validation_discipline`
  - Artifact JSON: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260210_012516.json`
  - Decision: `promote`

Construcción paquete RC:

- `./venv/bin/python research/breakthrough_lab/week15_controlled_rollout/build_week15_block3_framework_rc.py --canonical-gate-path research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260210_012516.json --block1-report-path research/breakthrough_lab/week15_controlled_rollout/week15_block1_expanded_pilot_rerun_20260210_011756.json --block2-report-path research/breakthrough_lab/week15_controlled_rollout/week15_block2_plugin_pilot_rerun_20260210_012358.json --rc-tag v0.15.0-rc1 --output-dir research/breakthrough_lab/week15_controlled_rollout --output-prefix week15_block3_framework_rc`
  - Artifact JSON: `research/breakthrough_lab/week15_controlled_rollout/week15_block3_framework_rc_20260210_012522.json`
  - Artifact MD: `research/breakthrough_lab/week15_controlled_rollout/week15_block3_framework_rc_20260210_012522.md`
  - Release notes: `research/breakthrough_lab/preprod_signoff/WEEK15_BLOCK3_FRAMEWORK_RC_RELEASE_NOTES.md`
  - Adoption checklist: `research/breakthrough_lab/preprod_signoff/WEEK15_BLOCK3_FRAMEWORK_RC_ADOPTION_CHECKLIST.md`
  - Dependent projects onboarding: `research/breakthrough_lab/preprod_signoff/WEEK15_BLOCK3_DEPENDENT_PROJECTS_ONBOARDING.md`
  - RC manifest: `research/breakthrough_lab/preprod_signoff/WEEK15_BLOCK3_FRAMEWORK_RC_MANIFEST.json`
  - Decision: `promote`

## Resultados

- `rc_tag = v0.15.0-rc1`
- `block1_decision = go`
- `block2_decision = promote`
- `canonical_gate_decision = promote`
- `failed_checks = []`

## Decisión Formal

Tracks:

- `week15_block3_block1_dependency`: **promote**
- `week15_block3_block2_dependency`: **promote**
- `week15_block3_canonical_gate`: **promote**
- `week15_block3_rc_package_generation`: **promote**

Block decision:

- **promote**

Razonamiento:

- El RC queda publicable con dependencias técnicas en verde y documentación de adopción lista.

## Estado del Bloque

`Week 15 - Block 3` cerrado en `promote`.
