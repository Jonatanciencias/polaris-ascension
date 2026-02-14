from __future__ import annotations

import numpy as np

import src.optimization_engines.optimized_kernel_engine as engine_module
from src.optimization_engines.optimized_kernel_engine import (
    BufferPool,
    KernelMetrics,
    KernelType,
    OperationResult,
    OptimizedKernelEngine,
    TransferMetrics,
)


class _DummyBuffer:
    def __init__(self, size: int) -> None:
        self.size = size
        self.released = False

    def release(self) -> None:
        self.released = True


def test_transfer_metrics_and_operation_summary() -> None:
    transfer = TransferMetrics(
        h2d_time_ms=2.0, d2h_time_ms=4.0, h2d_bytes=2_000_000_000, d2h_bytes=4_000_000_000
    )
    kernel = KernelMetrics(
        kernel_name="k0",
        exec_time_ms=1.0,
        gflops=100.0,
        efficiency=0.5,
        work_groups=8,
    )
    op = OperationResult(
        result=np.zeros((2, 2), dtype=np.float32),
        transfer_metrics=transfer,
        kernel_metrics=kernel,
        total_time_ms=3.0,
    )

    assert transfer.h2d_bandwidth_gbps == 1000.0
    assert transfer.d2h_bandwidth_gbps == 1000.0
    summary = op.summary()
    assert "Kernel: k0" in summary
    assert "Rendimiento" in summary


def test_buffer_pool_return_hit_rate_and_clear() -> None:
    pool = BufferPool.__new__(BufferPool)
    pool.max_pool_size = 1
    pool._read_buffers = {}
    pool._write_buffers = {}
    pool._stats = {"hits": 2, "misses": 1}

    r1 = _DummyBuffer(64)
    r2 = _DummyBuffer(64)
    w1 = _DummyBuffer(32)

    pool.return_buffer(r1, is_write=False)
    pool.return_buffer(r2, is_write=False)
    pool.return_buffer(w1, is_write=True)

    assert r1.released is False
    assert r2.released is True
    assert pool.hit_rate == 2 / 3

    pool.clear()
    assert pool._read_buffers == {}
    assert pool._write_buffers == {}


def test_select_best_kernel_branches() -> None:
    engine = OptimizedKernelEngine.__new__(OptimizedKernelEngine)

    assert (
        engine.select_best_kernel(64, 64, 64, fused_op="transpose_b")
        == KernelType.GEMM_FUSED_TRANSPOSE
    )
    assert (
        engine.select_best_kernel(64, 64, 64, fused_op="relu_bias")
        == KernelType.GEMM_FUSED_RELU_BIAS
    )
    assert engine.select_best_kernel(100, 120, 96) == KernelType.GEMM_FLOAT4_SMALL
    assert engine.select_best_kernel(256, 256, 256) == KernelType.GEMM_FLOAT4_SMALL
    assert engine.select_best_kernel(700, 700, 700) == KernelType.GEMM_FLOAT4_VEC
    assert engine.select_best_kernel(700, 702, 700) == KernelType.GEMM_FLOAT4_CLOVER
    assert engine.select_best_kernel(3000, 3000, 3000) == KernelType.GEMM_FLOAT4_VEC
    assert engine.select_best_kernel(3000, 3001, 3000) == KernelType.GEMM_GCN4_STREAMING
    assert engine.select_best_kernel(1800, 1801, 1799) == KernelType.GEMM_FLOAT4_CLOVER


def test_get_optimal_work_size_paths() -> None:
    engine = OptimizedKernelEngine.__new__(OptimizedKernelEngine)
    engine.max_work_group_size = 32
    engine.KERNEL_CONFIGS = OptimizedKernelEngine.KERNEL_CONFIGS

    global_size, local_size = engine._get_optimal_work_size(KernelType.GEMM_BASIC, 100, 130)
    assert local_size[0] * local_size[1] <= 32
    assert global_size[0] >= 100
    assert global_size[1] >= 130

    gcn_global, gcn_local = engine._get_optimal_work_size(KernelType.GEMM_GCN4_ULTRA, 130, 130)
    assert gcn_local[0] * gcn_local[1] <= engine.max_work_group_size
    assert gcn_global[0] > 0 and gcn_global[1] > 0

    vec_global, vec_local = engine._get_optimal_work_size(KernelType.GEMM_FLOAT4_VEC, 128, 256)
    assert vec_local[0] * vec_local[1] <= engine.max_work_group_size
    assert vec_global[0] >= 128
    assert vec_global[1] >= 256 // 4


def test_get_statistics_memory_and_cleanup() -> None:
    engine = OptimizedKernelEngine.__new__(OptimizedKernelEngine)
    engine._operation_history = []
    assert engine.get_statistics() == {"message": "No operations recorded"}

    transfer = TransferMetrics(h2d_time_ms=1.0, d2h_time_ms=1.0, h2d_bytes=10, d2h_bytes=10)
    kernel = KernelMetrics("k", 1.0, 10.0, 0.1, 1)
    op = OperationResult(np.zeros((1, 1), dtype=np.float32), transfer, kernel, 2.0)
    engine._operation_history = [op]

    class _Pool:
        hit_rate = 0.75

        def __init__(self) -> None:
            self.cleared = False

        def clear(self) -> None:
            self.cleared = True

    class _MemManager:
        def __init__(self) -> None:
            self.cleared = False

        def get_stats(self):
            from src.optimization_engines.advanced_memory_manager import MemoryStats

            return MemoryStats(
                peak_usage=100,
                current_usage=50,
                pool_hits=3,
                pool_misses=1,
                evictions=1,
                prefetch_hits=2,
                tiles_created=4,
                compression_savings=20,
                total_allocated=200,
            )

        def clear(self) -> None:
            self.cleared = True

    pool = _Pool()
    mem = _MemManager()
    engine.buffer_pool = pool
    engine.memory_manager = mem

    engine.enable_buffer_pool = True
    engine.enable_advanced_memory = False
    stats = engine.get_statistics()
    assert stats["buffer_pool_hit_rate"] == 0.75

    engine.enable_advanced_memory = True
    stats2 = engine.get_statistics()
    assert "memory" in stats2
    assert stats2["memory"]["evictions"] == 1

    assert engine.get_memory_stats() is not None
    engine.cleanup()
    assert mem.cleared is True
    assert engine._operation_history == []


def test_optimized_gemm_convenience_function(monkeypatch) -> None:
    called = {"cleanup": False}

    class _FakeResult:
        def __init__(self) -> None:
            self.result = np.array([[42.0]], dtype=np.float32)

    class _FakeEngine:
        def __init__(self, enable_profiling: bool = False) -> None:
            _ = enable_profiling

        def gemm(self, A: np.ndarray, B: np.ndarray):
            _ = (A, B)
            return _FakeResult()

        def cleanup(self) -> None:
            called["cleanup"] = True

    monkeypatch.setattr(engine_module, "OptimizedKernelEngine", _FakeEngine)

    out = engine_module.optimized_gemm(
        np.ones((1, 1), dtype=np.float32), np.ones((1, 1), dtype=np.float32)
    )
    assert out.shape == (1, 1)
    assert out[0, 0] == 42.0
    assert called["cleanup"] is True
