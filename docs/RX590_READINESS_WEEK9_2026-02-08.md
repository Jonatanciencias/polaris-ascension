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
- `research/breakthrough_lab/platform_compatibility/week9_block5_rollback_20260208_035258.md`
- `research/breakthrough_lab/week9_comparative_dashboard_20260208_035352.json` (`promote`)
- `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_044015.json` (`promote`)
- `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_160122.json` (`promote`)
- `scripts/verify_drivers.py --json` (status `good`)
- `pytest -q` (`85 passed`)

## Readiness Dimensions

| Dimension | Status | Notes |
| --- | --- | --- |
| Driver/Runtime Health | ✅ Ready | `amdgpu` loaded, OpenCL available on Clover, Vulkan available. |
| Functional Correctness | ✅ Ready | Max error remained `<= 1e-3` in Week 9 blocks 2/3/4. |
| Stability Under Stress | ✅ Ready | Queue pulses + split-platform stress replay + preprod burn-in snapshots passed (`Week9 Block4/5`). |
| Long-Horizon Wall-Clock | ✅ Ready | Week9 Block6 30-minute wall-clock canary passed (`48/48` runs, all checks pass). |
| Controlled Rollout Week10 | ⚠️ Iterate | Week10 Block1 triggered auto rollback on T5 hard guardrail at snapshot 2; rollback and post-gate worked as designed. |
| Guardrails (T3/T5) | ✅ Ready | No T5 auto-disable events in Block2/3/4; T3 fallback stable at `0.0`. |
| Validation/Tests | ✅ Ready | Canonical validation promote + `pytest` full green (`85 passed`). |
| Platform Split Confidence | ✅ Ready (Controlled) | rusticl mirrored pilot passed with explicit bootstrap and validated rollback drill. |
| Rollback Operability | ✅ Ready | rollback script executed in `apply` mode with post-rollback canonical gate `promote`. |

## Readiness Score (Operational)

- **Score: 96/100**
- **Readiness band:** `controlled-rollout candidate (iterate with guardrails)` on this host.

## Residual Risks Before Wider Real Tests

1. rusticl enablement remains environment-sensitive (`RUSTICL_ENABLE=radeonsi`) and should stay explicitly scripted.
2. T4 track has stable prior evidence but no Week9 Block6-specific policy mutation.
3. T5 requires additional hardening for rollout scope after disable event observed in Week10 Block1.

## Recommended Gate Before “Real Test Campaign”

1. Harden T5 rollout profile and rerun Week10 Block1 to move from `iterate` to `promote`.
2. Keep canonical gate mandatory before each scope increase.
3. Continue staged rollout with rollback SLA and weekly drift tracking.

## Verdict

Current code is **ready for controlled rollout in iterate mode** on RX 590 in this host profile: rollback controls are validated, but T5 rollout hardening is required before broader promotion.
