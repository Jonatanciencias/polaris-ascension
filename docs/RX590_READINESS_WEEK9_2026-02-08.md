# RX 590 Readiness Assessment - Week 9 (2026-02-08)

## Scope

Assessment of current codebase readiness for real-world testing on installed `AMD Radeon RX 590 GME`.

## Evidence Used

- `research/breakthrough_lab/week9_block2_long_canary_rerun_20260208_032017.json` (`promote`)
- `research/breakthrough_lab/platform_compatibility/week9_block3_robustness_replay_20260208_033111.json` (`promote`)
- `research/breakthrough_lab/platform_compatibility/week9_block4_stress_split_20260208_033946.json` (`promote`)
- `research/breakthrough_lab/week9_comparative_dashboard_20260208_034022.json` (`promote`)
- `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_034047.json` (`promote`)
- `scripts/verify_drivers.py --json` (status `good`)
- `pytest -q` (`85 passed`)

## Readiness Dimensions

| Dimension | Status | Notes |
| --- | --- | --- |
| Driver/Runtime Health | ✅ Ready | `amdgpu` loaded, OpenCL available on Clover, Vulkan available. |
| Functional Correctness | ✅ Ready | Max error remained `<= 1e-3` in Week 9 blocks 2/3/4. |
| Stability Under Stress | ✅ Ready | Queue pulses + split-platform stress replay passed (`Week9 Block4`). |
| Guardrails (T3/T5) | ✅ Ready | No T5 auto-disable events in Block2/3/4; T3 fallback stable at `0.0`. |
| Validation/Tests | ✅ Ready | Canonical validation promote + `pytest` full green (`85 passed`). |
| Platform Split Confidence | ⚠️ Conditional | rusticl path validated in canary mode, but inventory may need explicit `RUSTICL_ENABLE=radeonsi`. |

## Readiness Score (Operational)

- **Score: 90/100**
- **Readiness band:** `pre-production ready` for controlled real tests on this host.

## Residual Risks Before Wider Real Tests

1. rusticl enablement remains environment-sensitive (`RUSTICL_ENABLE=radeonsi`).
2. Evidence horizon for stress is short-to-medium; no multi-hour burn-in yet in Week 9.
3. T4 track has stable prior evidence but no new Week 9 policy mutation.

## Recommended Gate Before “Real Test Campaign”

1. Run a 6h burn-in canary on Clover with periodic queue pulses and hourly snapshots.
2. Run mirrored 2h rusticl canary with explicit env bootstrap and rollback checklist.
3. Freeze runtime policy bundle (`T3/T5`) and collect single consolidated report for sign-off.

## Verdict

Current code is **close and practically ready** for real RX 590 tests in controlled mode, with no blocking technical issue detected in Week 9 evidence.
