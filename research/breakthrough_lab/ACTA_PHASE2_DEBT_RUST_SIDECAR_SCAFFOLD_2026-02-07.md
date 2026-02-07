# Acta - Phase 2 Debt Closure (Rust Sidecar Scaffold)

- Date: 2026-02-07
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope: cierre de deuda de roadmap Phase 2 con scaffold minimo de sidecar Rust para `t2_auto_scheduler`.

## Objetivo

Entregar una base tecnica minima para el sidecar Rust definido en roadmap:
1. crate Rust con API inicial orientada a auto-scheduler,
2. puente Python (`pyo3/maturin`),
3. smoke local deterministico,
4. evidencia de validacion reproducible.

## Entregables

Scaffold creado en:
- `research/breakthrough_lab/t2_auto_scheduler/rust_sidecar/`

Archivos:
- `research/breakthrough_lab/t2_auto_scheduler/rust_sidecar/Cargo.toml`
- `research/breakthrough_lab/t2_auto_scheduler/rust_sidecar/pyproject.toml`
- `research/breakthrough_lab/t2_auto_scheduler/rust_sidecar/src/lib.rs`
- `research/breakthrough_lab/t2_auto_scheduler/rust_sidecar/python_client.py`
- `research/breakthrough_lab/t2_auto_scheduler/rust_sidecar/smoke_test.py`
- `research/breakthrough_lab/t2_auto_scheduler/rust_sidecar/README.md`
- `research/breakthrough_lab/t2_auto_scheduler/rust_sidecar/.gitignore`

## Capacidades Iniciales

- `candidate_enumeration`: enumeracion deterministica de candidatos de schedule.
- `deterministic_replay_plan`: construccion de plan de replay deterministico (seeds estables).
- cliente Python con dos modos:
  - `native-rust` si el modulo compilado esta disponible,
  - `python-fallback` deterministico para entornos sin build nativo.

## Validacion Ejecutada

Rust core tests:
- `cargo test --manifest-path research/breakthrough_lab/t2_auto_scheduler/rust_sidecar/Cargo.toml`
- Resultado: **pass** (`2` tests, `0` failures)

Rust module compilation path:
- `cargo check --manifest-path research/breakthrough_lab/t2_auto_scheduler/rust_sidecar/Cargo.toml --features python-module`
- Resultado: **pass**

Python smoke:
- `./venv/bin/python research/breakthrough_lab/t2_auto_scheduler/rust_sidecar/smoke_test.py`
- Resultado: **pass** (modo `python-fallback` en este entorno)

## Decision

Estado de deuda Phase 2 (sidecar scaffold): **closed (scaffold complete)**.

Siguiente gate tecnico:
- integrar una primera llamada real desde runner T2 hacia el sidecar (modo shadow) y comparar consistencia frente a path Python.
