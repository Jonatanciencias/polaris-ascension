from __future__ import annotations

import numpy as np
import pytest

from src.optimization_engines import quantum_annealing_optimizer as qa_module
from src.optimization_engines.quantum_annealing_optimizer import QuantumAnnealingMatrixOptimizer


def _make_optimizer() -> QuantumAnnealingMatrixOptimizer:
    optimizer = QuantumAnnealingMatrixOptimizer.__new__(QuantumAnnealingMatrixOptimizer)
    optimizer.num_spins = 16
    optimizer.beta_init = 0.1
    optimizer.beta_final = 2.0
    optimizer.opencl_available = False
    return optimizer


def test_matrix_to_ising_shape_and_symmetry() -> None:
    optimizer = _make_optimizer()
    A = np.arange(12, dtype=np.float32).reshape(3, 4)
    B = np.arange(20, dtype=np.float32).reshape(4, 5)

    h = optimizer._matrix_multiplication_to_ising(A, B)

    assert h.shape == (15, 15)
    assert np.allclose(h, h.T)


def test_energy_changes_cpu_returns_expected_shape() -> None:
    optimizer = _make_optimizer()
    h = np.eye(4, dtype=np.float32)
    state = np.array([1, -1, 1, -1], dtype=np.int32)

    changes = optimizer._calculate_energy_changes_cpu(h, state)

    assert changes.shape == (4,)
    assert changes.dtype == np.float32


def test_batch_and_total_energy_use_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    optimizer = _make_optimizer()
    h = np.eye(3, dtype=np.float32)
    state = np.array([1, 1, -1], dtype=np.int32)

    called = {"batch": False, "total": False}

    def _fake_changes(hamiltonian: np.ndarray, s: np.ndarray):
        _ = (hamiltonian, s)
        called["batch"] = True
        return np.array([1.0, 2.0, 3.0], dtype=np.float32)

    def _fake_total(hamiltonian: np.ndarray, s: np.ndarray) -> float:
        _ = (hamiltonian, s)
        called["total"] = True
        return -7.0

    monkeypatch.setattr(optimizer, "_calculate_energy_changes_cpu", _fake_changes)
    monkeypatch.setattr(optimizer, "_calculate_total_energy_cpu", _fake_total)

    changes = optimizer._calculate_energy_changes_batch(h, state)
    total = optimizer._calculate_total_energy_optimized(h, state)

    assert called["batch"] is True
    assert called["total"] is True
    assert np.allclose(changes, [1.0, 2.0, 3.0])
    assert total == -7.0


def test_ising_to_matrix_result_shape_and_nonzero() -> None:
    optimizer = _make_optimizer()
    state = np.array([1, -1, 1, -1], dtype=np.int32)

    result = optimizer._ising_to_matrix_result(state, 2, 3)

    assert result.shape == (2, 3)
    assert np.count_nonzero(result) > 0


def test_run_quantum_annealing_early_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    optimizer = _make_optimizer()
    optimizer.beta_final = 3.0

    def _zero_changes(hamiltonian: np.ndarray, state: np.ndarray) -> np.ndarray:
        _ = hamiltonian
        return np.zeros(len(state), dtype=np.float32)

    monkeypatch.setattr(optimizer, "_calculate_energy_changes_batch", _zero_changes)
    monkeypatch.setattr(optimizer, "_calculate_total_energy_optimized", lambda h, s: 1.0)
    monkeypatch.setattr(qa_module.np.random, "random", lambda: 1.0)

    h = np.eye(8, dtype=np.float32)
    state, history = optimizer._run_quantum_annealing(h, num_sweeps=60)

    assert state.shape == (8,)
    assert 21 <= len(history) < 60


def test_quantum_annealing_optimization_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    optimizer = _make_optimizer()

    monkeypatch.setattr(
        optimizer,
        "_matrix_multiplication_to_ising",
        lambda A, B: np.eye(4, dtype=np.float32),
    )
    monkeypatch.setattr(
        optimizer,
        "_run_quantum_annealing",
        lambda h, sweeps: (np.array([1, -1, 1, -1], dtype=np.int32), [-2.0, -2.0]),
    )
    monkeypatch.setattr(
        optimizer,
        "_ising_to_matrix_result",
        lambda state, m, n: np.ones((m, n), dtype=np.float32),
    )

    A = np.ones((2, 2), dtype=np.float32)
    B = np.ones((2, 2), dtype=np.float32)
    result, metrics = optimizer.quantum_annealing_optimization(A, B, num_sweeps=5)

    assert result.shape == (2, 2)
    assert "gflops_achieved" in metrics
    assert metrics["convergence"] is True


def test_hybrid_quantum_classical_gemm(monkeypatch: pytest.MonkeyPatch) -> None:
    optimizer = _make_optimizer()

    monkeypatch.setattr(
        optimizer,
        "quantum_annealing_optimization",
        lambda A, B, num_sweeps=50: (
            np.zeros_like(A @ B),
            {
                "computation_time": 0.01,
                "gflops_achieved": 10.0,
                "convergence": True,
            },
        ),
    )

    A = np.ones((4, 4), dtype=np.float32)
    B = np.ones((4, 4), dtype=np.float32)
    result, metrics = optimizer.hybrid_quantum_classical_gemm(A, B)

    assert result.shape == (4, 4)
    assert metrics["qa_convergence"] is True
    assert "gflops_hybrid" in metrics


def test_benchmark_quantum_techniques_with_stubbed_optimizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeOptimizer:
        def __init__(self):
            pass

        def quantum_annealing_optimization(self, A, B, num_sweeps=50):
            _ = (A, B, num_sweeps)
            return np.zeros((2, 2), dtype=np.float32), {
                "gflops_achieved": 1.0,
                "computation_time": 0.01,
                "convergence": True,
            }

        def hybrid_quantum_classical_gemm(self, A, B):
            _ = (A, B)
            return np.zeros((2, 2), dtype=np.float32), {"gflops_hybrid": 2.0}

    monkeypatch.setattr(qa_module, "QuantumAnnealingMatrixOptimizer", _FakeOptimizer)

    results = qa_module.benchmark_quantum_techniques()

    assert set(results.keys()) == {128, 256, 512}
    assert "quantum_direct" in results[128]


def test_main_success_and_failure_paths(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    class _FakeOptimizer:
        def quantum_annealing_optimization(self, A, B, num_sweeps=100):
            _ = (A, B, num_sweeps)
            return np.ones((2, 2), dtype=np.float32), {
                "gflops_achieved": 1.0,
                "relative_error": 0.1,
                "convergence": True,
            }

        def hybrid_quantum_classical_gemm(self, A, B):
            _ = (A, B)
            return np.ones((2, 2), dtype=np.float32), {"gflops_hybrid": 2.0}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(qa_module, "QuantumAnnealingMatrixOptimizer", _FakeOptimizer)
    monkeypatch.setattr(qa_module, "benchmark_quantum_techniques", lambda: {"ok": True})

    ok = qa_module.main()
    assert ok == 0

    def _boom():
        raise RuntimeError("forced main failure")

    monkeypatch.setattr(qa_module, "QuantumAnnealingMatrixOptimizer", _boom)
    fail = qa_module.main()
    assert fail == 1
