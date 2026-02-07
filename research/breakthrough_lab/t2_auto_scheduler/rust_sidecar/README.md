# T2 Rust Sidecar Scaffold (Phase 2 Debt Closure)

Minimal Rust sidecar scaffold for `t2_auto_scheduler`:
- deterministic candidate enumeration API
- deterministic replay-plan API
- Python bridge via `pyo3/maturin`
- Python fallback client for environments without compiled extension

## Layout

- `Cargo.toml` - Rust crate definition (`cdylib` + `rlib`)
- `src/lib.rs` - `pyo3` module (`t2_rust_sidecar`)
- `pyproject.toml` - maturin build backend
- `python_client.py` - Python API with native/fallback mode
- `smoke_test.py` - deterministic smoke test

## Native Build (optional)

From repository root:

```bash
python -m pip install maturin
maturin develop --manifest-path research/breakthrough_lab/t2_auto_scheduler/rust_sidecar/Cargo.toml --release
```

Then run:

```bash
python research/breakthrough_lab/t2_auto_scheduler/rust_sidecar/smoke_test.py
```

## Local Validation Without Native Build

Rust-only checks:

```bash
cargo test --manifest-path research/breakthrough_lab/t2_auto_scheduler/rust_sidecar/Cargo.toml
cargo check --manifest-path research/breakthrough_lab/t2_auto_scheduler/rust_sidecar/Cargo.toml --features python-module
```

Python fallback smoke:

```bash
python research/breakthrough_lab/t2_auto_scheduler/rust_sidecar/smoke_test.py
```

The smoke test is deterministic and validates API shape/cardinality.
