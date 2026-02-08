# Roadmap 2026Q2 - Continuous Improvement

## Mission

Advance from roadmap closure to continuous, low-risk production improvement:

- preserve deterministic throughput at target sizes (`1400`, `2048`, `3072`)
- reduce operational variance and diagnostics drift
- improve maintainability and CI confidence without destabilizing production kernels

## Baseline at Kickoff (2026-02-08)

- 2026Q1 closure status: **promote**
- Week6 final strict rerun: **promote**
- Canonical suite: `pytest -q` green (`78 passed`)
- Primary production evidence: `research/breakthrough_lab/week6_final_suite_20260208_011347.json`

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

1. Create `research/breakthrough_lab/week8_validation_discipline/` scaffold.
2. Add CI-facing smoke test for `scripts/verify_drivers.py --json`.
3. Pin and document validation command matrix (`tests`, full repo, schema).
4. Register first Week 8 acta/decision template.
