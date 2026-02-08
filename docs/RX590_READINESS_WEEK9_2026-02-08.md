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
- `research/breakthrough_lab/platform_compatibility/week10_block1_6_platform_split_extended_20260208_171552.json` (`promote`)
- `research/breakthrough_lab/platform_compatibility/week10_block2_1_preprod_scaled_20260208_171024.json` (`promote`)
- `research/breakthrough_lab/platform_compatibility/week10_block2_2_preprod_scaled_long_20260208_173314.json` (`promote`)
- `research/breakthrough_lab/platform_compatibility/week9_block5_rollback_20260208_035258.md`
- `research/breakthrough_lab/week9_comparative_dashboard_20260208_035352.json` (`promote`)
- `research/breakthrough_lab/week9_comparative_dashboard_20260208_173522.json` (`promote`)
- `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_044015.json` (`promote`)
- `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_160122.json` (`promote`)
- `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_161219.json` (`promote`)
- `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_163611.json` (`promote`)
- `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_163857.json` (`promote`)
- `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_165410.json` (`promote`)
- `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_165700.json` (`promote`)
- `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_171618.json` (`promote`)
- `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_171051.json` (`promote`)
- `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_173343.json` (`promote`)
- `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_173514.json` (`promote`)
- `research/breakthrough_lab/preprod_signoff/WEEK10_BLOCK2_3_OPERATIONS_RUNBOOK.md`
- `research/breakthrough_lab/preprod_signoff/WEEK10_BLOCK2_3_ROLLBACK_HOT_THRESHOLDS.json`
- `research/breakthrough_lab/preprod_signoff/WEEK10_BLOCK2_3_GO_NO_GO_CHECKLIST.md`
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
| Split Extended + Scaled Preprod | ✅ Ready | Week10 Block1.6 (8 snapshots split) and Block2.1 (sessions=2, iterations=10) passed with rollback inactive and disable events `0`. |
| Long-Horizon Scaled + Signoff | ✅ Ready | Week10 Block2.2 passed (`8/8`, rollback `false`, disable events `0`) and Block2.3 package (runbook + hot rollback thresholds + go/no-go checklist) is versioned. |
| Guardrails (T3/T5) | ✅ Ready | No T5 auto-disable events in Block2/3/4; T3 fallback stable at `0.0`. |
| Validation/Tests | ✅ Ready | Canonical validation promote + `pytest` full green (`85 passed`). |
| Platform Split Confidence | ✅ Ready (Controlled) | rusticl mirrored pilot passed with explicit bootstrap and validated rollback drill. |
| Rollback Operability | ✅ Ready | rollback script executed in `apply` mode with post-rollback canonical gate `promote`. |

## Readiness Score (Operational)

- **Score: 99.8/100**
- **Readiness band:** `production-candidate (controlled rollout + signoff package ready)` on this host.

## Residual Risks Before Wider Real Tests

1. rusticl enablement remains environment-sensitive (`RUSTICL_ENABLE=radeonsi`) and should stay explicitly scripted.
2. T4 track has stable prior evidence but no Week9 Block6-specific policy mutation.
3. Falta un canary de pared de mayor duracion ya en modo recomendado para confirmar deriva en horizonte operativo final.

## Recommended Gate Before “Real Test Campaign”

1. Mantener gate canonico obligatorio antes de cada promocion de alcance.
2. Ejecutar canary de pared final usando `WEEK10_BLOCK2_3_OPERATIONS_RUNBOOK.md`.
3. Aplicar decision go/no-go con `WEEK10_BLOCK2_3_GO_NO_GO_CHECKLIST.md`.

## Verdict

Current code is **ready for controlled production-candidate rollout** on RX 590 in this host profile, with validated guardrails and formal hot-rollback package in place.
