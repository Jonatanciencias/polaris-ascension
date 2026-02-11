# Week 18 Block 1 - Dependent Adoption Runbook (v0.15.0)

1. Validate stable manifest and referenced evidence paths.
2. Run canonical gate (`run_validation_suite.py --tier canonical --driver-smoke`).
3. Execute dependent plugin pilot using stable profile (1400/2048/3072).
4. Store JSON + MD evidence and close local decision.
5. If guardrails fail, rollback to last known-good policy and stop expansion.

