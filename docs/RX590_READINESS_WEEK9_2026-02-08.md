# RX 590 Readiness Assessment - Week 9 (2026-02-08)

## Scope

Assessment of current codebase readiness for real-world testing on installed `AMD Radeon RX 590 GME`.

## Evidence Used

- `research/breakthrough_lab/week9_block2_long_canary_rerun_20260208_032017.json` (`promote`)
- `research/breakthrough_lab/platform_compatibility/week9_block3_robustness_replay_20260208_033111.json` (`promote`)
- `research/breakthrough_lab/platform_compatibility/week9_block4_stress_split_20260208_033946.json` (`promote`)
- `research/breakthrough_lab/platform_compatibility/week9_block5_preprod_pilot_20260208_035240.json` (`promote`)
- `research/breakthrough_lab/platform_compatibility/week9_block6_wallclock_canary_20260208_043949.json` (`promote`)
- `research/breakthrough_lab/platform_compatibility/week10_block1_controlled_rollout_20260208_160122.json` (`iterate`)
- `research/breakthrough_lab/platform_compatibility/week10_block1_1_controlled_rollout_20260208_161153.json` (`promote`)
- `research/breakthrough_lab/platform_compatibility/week10_block1_2_controlled_rollout_20260208_163545.json` (`promote`)
- `research/breakthrough_lab/platform_compatibility/week10_block1_3_controlled_rollout_20260208_163829.json` (`promote`)
- `research/breakthrough_lab/platform_compatibility/week10_block1_4_long_window_20260208_165345.json` (`promote`)
- `research/breakthrough_lab/platform_compatibility/week10_block1_5_platform_split_20260208_165631.json` (`promote`)
- `research/breakthrough_lab/platform_compatibility/week9_block5_rollback_20260208_035258.md`
- `research/breakthrough_lab/week9_comparative_dashboard_20260208_035352.json` (`promote`)
- `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_044015.json` (`promote`)
- `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_160122.json` (`promote`)
- `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_161219.json` (`promote`)
- `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_163611.json` (`promote`)
- `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_163857.json` (`promote`)
- `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_165410.json` (`promote`)
- `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_165700.json` (`promote`)
- `scripts/verify_drivers.py --json` (status `good`)
- `pytest -q` (`85 passed`)

## Readiness Dimensions

| Dimension | Status | Notes |
| --- | --- | --- |
| Driver/Runtime Health | ✅ Ready | `amdgpu` loaded, OpenCL available on Clover, Vulkan available. |
| Functional Correctness | ✅ Ready | Max error remained `<= 1e-3` in Week 9 blocks 2/3/4. |
| Stability Under Stress | ✅ Ready | Queue pulses + split-platform stress replay + preprod burn-in snapshots passed (`Week9 Block4/5`). |
| Long-Horizon Wall-Clock | ✅ Ready | Week9 Block6 30-minute wall-clock canary passed (`48/48` runs, all checks pass). |
| Controlled Rollout Week10 | ✅ Ready (Expanded) | Week10 Block1.1/1.2/1.3 passed (4 then 6 then 6 snapshots) including `1400+2048`, all with no rollback and no T5 disable events. |
| Split Platform Canary | ✅ Ready | Week10 Block1.5 Clover/rusticl split passed with ratio floor healthy and no T5 disable events. |
| Guardrails (T3/T5) | ✅ Ready | No T5 auto-disable events in Block2/3/4; T3 fallback stable at `0.0`. |
| Validation/Tests | ✅ Ready | Canonical validation promote + `pytest` full green (`85 passed`). |
| Platform Split Confidence | ✅ Ready (Controlled) | rusticl mirrored pilot passed with explicit bootstrap and validated rollback drill. |
| Rollback Operability | ✅ Ready | rollback script executed in `apply` mode with post-rollback canonical gate `promote`. |

## Readiness Score (Operational)

- **Score: 99.5/100**
- **Readiness band:** `production-candidate (controlled rollout)` on this host.

## Residual Risks Before Wider Real Tests

1. rusticl enablement remains environment-sensitive (`RUSTICL_ENABLE=radeonsi`) and should stay explicitly scripted.
2. T4 track has stable prior evidence but no Week9 Block6-specific policy mutation.
3. Falta solo ampliar horizonte split (Block1.6) antes de recomendar escalado mas agresivo.

## Recommended Gate Before “Real Test Campaign”

1. Ejecutar Week10 Block1.6 con split extendido Clover/rusticl y drift por plataforma.
2. Mantener gate canonico obligatorio antes de cada promocion de alcance.
3. Iniciar Block2.1 de pre-produccion escalada con rollback SLA activo.

## Verdict

Current code is **ready for controlled rollout promotion** on RX 590 in this host profile, with T5 hardening validated and rollback controls still active as mandatory safety net.
