# Breakthrough Roadmap 2026Q1

## 1) Mission

Transform the project from a strong handcrafted GEMM optimization into a repeatable breakthrough pipeline:
- higher sustained performance on RX 590 class hardware
- measurable robustness and reproducibility
- faster iteration through automated search and online adaptation

## 2) Operating Philosophy

Use a dual-lane model to protect production quality while enabling aggressive experimentation:

- `Lane A: Production Core`
  - only validated kernels and selectors
  - strict reproducibility protocol
  - no speculative methods

- `Lane B: Breakthrough Lab`
  - high-risk, high-reward experiments
  - clear promotion gates to production
  - hard stop for ideas that fail objective thresholds

Promotion gate from Lab -> Production:
- speedup >= +10% vs current production baseline on target sizes
- correctness within defined error budget
- stability: low variance across repeated sessions
- operational cost acceptable (build time, complexity, maintenance)

## 3) Technical North Stars

Primary KPIs:
- `KPI-1`: Peak and sustained GFLOPS at 1400, 2048, 3072
- `KPI-2`: Stability (std/cv across sessions)
- `KPI-3`: Accuracy envelope (max error, relative error)
- `KPI-4`: End-to-end latency (kernel + host overhead)
- `KPI-5`: Engineering cost (compile time, code complexity)

Secondary KPIs:
- reproducibility score
- thermal/load sensitivity
- selector decision quality vs oracle

## 4) Innovation Tracks (Cross-Disciplinary)

### Track T1: Math + Physics (IO-aware kernel design)
- Build kernels from a communication-minimization model, not only FLOP count.
- Add roofline and arithmetic-intensity guidance to kernel selection.
- Expected value: real +10% to +20% improvements on sustained throughput.

### Track T2: Computer Science (Auto-scheduler sidecar)
- Build an automated schedule search sidecar for tile/workgroup/vector layouts.
- Distill best schedules into production kernels.
- Expected value: faster discovery of non-obvious high-performance points.

### Track T3: Control/Engineering (Online adaptive selector)
- Evolve static selector to contextual online learner (bandit-style policy).
- Inputs: matrix shape, runtime drift, queue behavior, thermal proxy.
- Expected value: better real-world performance consistency under varying load.

### Track T4: Applied Math (Approximate GEMM with contracts)
- Add optional approximate mode with explicit error budget contract.
- Candidate methods: low-rank randomized approximations with fallback to exact.
- Expected value: large speedups for suitable workloads with bounded error.

### Track T5: Reliability Engineering (ABFT-lite)
- Add algorithm-based fault tolerance checks (checksums / lightweight verification).
- Target production trustworthiness for long runs and unstable environments.

### Track T6: Quantum-inspired (offline design, not runtime compute)
- Use quantum-inspired optimization for offline search over schedules/strategies.
- Do not position as runtime quantum acceleration on this hardware stack.

## 5) Language Strategy (What to Add and Why)

### Current baseline
- `Python`: orchestration, benchmarking, reporting, experiments
- `OpenCL C`: GPU kernel implementation

### Proposed additions
- `Rust` (recommended, priority high)
  - Best fit for: autotuner engine, telemetry pipeline, robust runtime services
  - Why: strong safety + good performance + clean FFI boundaries
  - Integration path: `pyo3`/`maturin` modules for Python-facing APIs

- `C/C++` (recommended, targeted use)
  - Best fit for: microbench kernels, low-level host runtime hot paths
  - Why: mature low-level performance tooling, broad ecosystem
  - Constraint: keep boundary small to avoid maintenance sprawl

- `Assembly` (limited and tactical)
  - Best fit for: narrow, measured hotspots only
  - Why: max control
  - Constraint: high maintenance and portability cost; use only with hard evidence

### Language Decision Matrix

| Language | Perf Potential | Dev Velocity | Safety | Recommended Scope |
|---|---:|---:|---:|---|
| Python | Medium | High | Medium | orchestration, experimentation |
| OpenCL C | High | Medium | Medium | production kernels |
| Rust | High | Medium | High | tuner/runtime/telemetry |
| C/C++ | High | Medium | Low-Med | targeted host hot paths |
| Assembly | Very High (local) | Low | Low | only proven micro-hotspots |

Decision:
- Add `Rust` first.
- Keep `C/C++` optional and minimal.
- Avoid broad assembly adoption.

## 6) Phased Execution Plan

### Phase 0 - Baseline and Governance (1 week)
Deliverables:
- single benchmark harness as source of truth
- promotion gate checklist
- lab-vs-core repository conventions

Exit criteria:
- reproducible baseline artifact generation is automatic
- all experiments write comparable metrics schema

### Phase 1 - IO-aware Kernel Program (3 weeks)
Deliverables:
- roofline-assisted kernel diagnostics
- first batch of communication-aware kernel variants
- track-level report with pass/fail decisions

Exit criteria:
- at least one candidate with >= +10% sustained gain on one target size
- no regression in correctness

### Phase 2 - Auto-scheduler + Rust Sidecar (4 weeks)
Deliverables:
- schedule search service (Rust core + Python API)
- search space definition for tile/workgroup/vector configs
- schedule replay and distillation pipeline

Exit criteria:
- reproducible discovery of top schedules
- lower manual tuning time per kernel cycle

### Phase 3 - Online Selector + Approximate Mode (4 weeks)
Deliverables:
- contextual online selector prototype
- optional approximate GEMM mode with error contract and fallback
- safety policy for auto-disable on drift

Exit criteria:
- selector outperforms static heuristic on mixed workload suite
- approximate mode respects error budget in >= 95% runs

### Phase 4 - Reliability + Platform Hardening (4 weeks)
Deliverables:
- ABFT-lite validation layer
- long-run stability campaign
- migration feasibility report for Rusticl/ROCm validation path

Exit criteria:
- no critical correctness escapes in stress campaign
- platform compatibility plan approved

### Phase 5 - Production Integration and Publication (2 weeks)
Deliverables:
- promoted techniques merged into production path
- updated docs, benchmarking protocol, and reproducibility artifacts
- public technical report with honest claims

Exit criteria:
- full validation suite green
- measurable production uplift documented

## 7) Repo Structure for Breakthrough Work

Recommended structure:
- `research/breakthrough_lab/`
  - `t1_io_aware/`
  - `t2_auto_scheduler/`
  - `t3_online_control/`
  - `t4_approximate_gemm/`
  - `t5_reliability_abft/`
  - `t6_quantum_offline/`
- `src/benchmarking/` (shared harness and report schema)
- `src/optimization_engines/` (only promoted, production-ready components)

## 8) Engineering Rules

- No performance claim without reproducible protocol.
- No merge to production lane without promotion gate.
- Keep experiment logs and artifacts machine-readable.
- Prefer small, reversible increments.
- Commit by track and phase; avoid mixed concerns.

## 9) Immediate Next Actions (Week 1)

1. Create breakthrough lab folders and templates.
2. Freeze baseline protocol and reference datasets.
3. Define track-specific experiment cards (hypothesis, method, metric, stop rule).
4. Start Rust sidecar scaffold for auto-scheduler APIs.
5. Run first IO-aware variant campaign on 1400/2048/3072.
