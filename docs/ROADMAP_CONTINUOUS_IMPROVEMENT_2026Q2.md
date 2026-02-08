# Roadmap 2026Q2 - Continuous Improvement

## Mission

Advance from roadmap closure to continuous, low-risk production improvement:

- preserve deterministic throughput at target sizes (`1400`, `2048`, `3072`)
- reduce operational variance and diagnostics drift
- improve maintainability and CI confidence without destabilizing production kernels

## Baseline at Kickoff (2026-02-08)

- 2026Q1 closure status: **promote**
- Week6 final strict rerun: **promote**
- Canonical suite: `pytest -q` green (`83 passed`)
- Primary production evidence: `research/breakthrough_lab/week6_final_suite_20260208_011347.json`

## Execution Status

- Week 8 - Block 1 (Validation Discipline Hardening): **promote**
  - Acta: `research/breakthrough_lab/ACTA_WEEK8_BLOCK1_VALIDATION_DISCIPLINE_2026-02-08.md`
  - Decision: `research/breakthrough_lab/week8_block1_validation_discipline_decision.json`
  - Evidence dir: `research/breakthrough_lab/week8_validation_discipline/`
- Week 8 - Block 2 (Local/CI parity hardening): **promote**
  - Acta: `research/breakthrough_lab/ACTA_WEEK8_BLOCK2_LOCAL_CI_PARITY_2026-02-08.md`
  - Decision: `research/breakthrough_lab/week8_block2_local_ci_parity_decision.json`
  - Key changes: `ci.yml` primary gate + runner unit tests
- Week 8 - Block 3 (T3 drift robustness under controlled pressure): **promote**
  - Acta: `research/breakthrough_lab/ACTA_WEEK8_BLOCK3_T3_DRIFT_CONTROLLED_2026-02-08.md`
  - Decision: `research/breakthrough_lab/week8_block3_t3_drift_controlled_decision.json`
  - Evidence:
    - `research/breakthrough_lab/t3_online_control/week8_t3_drift_campaign_20260208_020148.json`
    - `research/breakthrough_lab/t3_online_control/policy_hardening_block3.json`
- Week 8 - Block 4 (T4 mixed activation refinement): **promote**
  - Acta: `research/breakthrough_lab/ACTA_WEEK8_BLOCK4_T4_MIXED_POLICY_REFINEMENT_2026-02-08.md`
  - Decision: `research/breakthrough_lab/week8_block4_t4_mixed_policy_refinement_decision.json`
  - Evidence:
    - `research/breakthrough_lab/t4_approximate_gemm/week8_t4_mixed_campaign_20260208_021541.json`
    - `research/breakthrough_lab/t4_approximate_gemm/policy_activation_block4.json`
    - `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_021724.json`
- Week 8 - Block 5 (T5 reliability maturation with fault-injection tuning): **promote**
  - Acta: `research/breakthrough_lab/ACTA_WEEK8_BLOCK5_T5_RELIABILITY_MATURATION_2026-02-08.md`
  - Decision: `research/breakthrough_lab/week8_block5_t5_reliability_maturation_decision.json`
  - Evidence:
    - `research/breakthrough_lab/t5_reliability_abft/week8_t5_maturation_20260208_022633.json`
    - `research/breakthrough_lab/t5_reliability_abft/policy_hardening_block5.json`
    - `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_022835.json`
- Week 8 - Block 6 (Integrated consolidation + interaction + critical canary): **promote**
  - Acta: `research/breakthrough_lab/ACTA_WEEK8_BLOCK6_CONSOLIDATION_2026-02-08.md`
  - Decision: `research/breakthrough_lab/week8_block6_consolidation_decision.json`
  - Evidence:
    - `research/breakthrough_lab/week8_block6_integrated_consolidation_20260208_024445.json`
    - `research/breakthrough_lab/week8_block6_t4_t5_interaction_20260208_024510.json`
    - `research/breakthrough_lab/platform_compatibility/week8_platform_canary_critical_20260208_024625.json`
    - `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_024700.json`
- Week 9 - Block 1 (Long mixed canary under queue pressure): **iterate**
  - Acta: `research/breakthrough_lab/ACTA_WEEK9_BLOCK1_LONG_MIXED_CANARY_2026-02-08.md`
  - Decision: `research/breakthrough_lab/week9_block1_long_mixed_canary_decision.json`
  - Evidence:
    - `research/breakthrough_lab/week9_block1_long_canary_20260208_030816.json`
    - `research/breakthrough_lab/week9_block1_long_canary_20260208_030816.md`
    - `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_030950.json`
  - Key finding: all checks passed except `t5_disable_events_zero` (1 auto-disable event in T5 path).
- Week 9 - Block 2 (T5 hardening + strict rerun of long canary): **promote**
  - Acta: `research/breakthrough_lab/ACTA_WEEK9_BLOCK2_T5_HARDENING_2026-02-08.md`
  - Decision: `research/breakthrough_lab/week9_block2_t5_hardening_decision.json`
  - Evidence:
    - `research/breakthrough_lab/t5_reliability_abft/policy_hardening_week9_block2.json`
    - `research/breakthrough_lab/week9_block2_long_canary_rerun_20260208_032017.json`
    - `research/breakthrough_lab/week9_block2_long_canary_rerun_20260208_032017.md`
    - `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_032043.json`
  - Key finding: strict rerun closed Block 1 debt (`t5_disable_events_zero` now pass, observed=0).
- Week 9 - Block 3 (Robustness replay with alternate seeds + short platform split): **promote**
  - Acta: `research/breakthrough_lab/ACTA_WEEK9_BLOCK3_ROBUSTNESS_REPLAY_2026-02-08.md`
  - Decision: `research/breakthrough_lab/week9_block3_robustness_replay_decision.json`
  - Evidence:
    - `research/breakthrough_lab/platform_compatibility/week9_block3_robustness_replay_20260208_033111.json`
    - `research/breakthrough_lab/platform_compatibility/week9_block3_robustness_replay_20260208_033111.md`
    - `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_033147.json`
  - Key finding: no post-hardening regressions in T5; split Clover/rusticl passed with min ratio 0.9209.
- Week 9 - Block 4 (Short stress replay with queue pulses + platform split): **promote**
  - Acta: `research/breakthrough_lab/ACTA_WEEK9_BLOCK4_STRESS_SPLIT_2026-02-08.md`
  - Decision: `research/breakthrough_lab/week9_block4_stress_split_decision.json`
  - Evidence:
    - `research/breakthrough_lab/platform_compatibility/week9_block4_stress_split_20260208_033946.json`
    - `research/breakthrough_lab/platform_compatibility/week9_block4_stress_split_20260208_033946.md`
    - `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_034047.json`
  - Key finding: stress with queue pulses preserved correctness and guardrails (T5 disable events=0).
- Week 9 - Comparative Dashboard (T3/T4/T5 with Week9 deltas): **promote**
  - Artifact:
    - `research/breakthrough_lab/week9_comparative_dashboard_20260208_034022.json`
    - `research/breakthrough_lab/week9_comparative_dashboard_20260208_034022.md`
  - Key finding: Block1 iterate is superseded by Block2 hardening; active chain Block2/3/4 remains promote.

## Governance Rules

1. No promotion claim without machine-readable artifact.
2. All new experiments stay in `research/breakthrough_lab/*` until promoted.
3. Every block must end with:
   - evidence file(s),
   - acta,
   - formal decision (`promote|iterate|refine|stop`).
4. Production path changes require fallback and deterministic seed protocol.

## 2026Q2 Execution Blocks

### Block 1 - Validation Discipline Hardening (Week 1)

- Goal: stabilize CI/local parity and reduce hidden regressions.
- Tasks:
  - finalize single validation entrypoint for local + CI
  - enforce results schema validation in CI path
  - add smoke checks for driver diagnostics script
- Exit gate:
  - canonical suite green in CI and local
  - no schema drift in breakthrough artifacts

### Block 2 - Selector Robustness Under Drift (Week 2)

- Goal: improve resilience of `auto`/`auto_t3_controlled` under thermal/load drift.
- Tasks:
  - controlled drift scenarios (warm/cold, queue pressure)
  - guardrail threshold calibration with rollback-safe defaults
- Exit gate:
  - correctness violations = 0
  - fallback rate within policy limits

### Block 3 - Approximate Mode Operational Envelope (Week 3)

- Goal: reduce unnecessary fallback in T4 with explicit gating confidence.
- Tasks:
  - refine compressibility/predictive gating
  - expand contract evaluation over mixed workload set
- Exit gate:
  - contract satisfaction >= 95%
  - no regression in production exact mode

### Block 4 - Reliability Guardrails Maturation (Week 4)

- Goal: increase fault-detection robustness while keeping low overhead.
- Tasks:
  - ABFT-lite coverage tuning for edge distributions
  - shadow canary replay with auto-disable verification
- Exit gate:
  - overhead remains below configured limit
  - false positive rate below policy threshold

### Block 5 - Platform Compatibility Closure (Week 5)

- Goal: formal platform policy for Clover/Rusticl compatibility and canary rollout.
- Tasks:
  - compatibility matrix refresh
  - explicit runtime selection policy publication
- Exit gate:
  - matrix artifact published
  - canary policy approved with rollback path

### Block 6 - Q2 Final Consolidation (Week 6)

- Goal: close quarter with reproducible evidence and comparative report.
- Tasks:
  - full final suite execution
  - final acta and comparative report update
- Exit gate:
  - final suite promote
  - no unresolved high-priority debt

## Immediate Backlog (Next Actions)

1. Execute Week 9 - Block 5: controlled pre-production pilot on RX 590 (extended burn-in window + hourly evidence snapshots).
2. Define and execute mirrored rusticl pilot profile with explicit environment bootstrap and rollback script.
3. Keep `scripts/run_validation_suite.py --tier canonical --driver-smoke` as required gate before each block closure.
