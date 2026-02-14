from __future__ import annotations

import threading

import numpy as np
import pytest

from src.optimization_engines.adaptive_kernel_selector import ProductionKernelSelector
from src.optimization_engines.advanced_polaris_opencl_engine import (
    AdvancedPolarisOpenCLEngine,
    PolarisOptimizationConfig,
    PolarisPerformanceMetrics,
    TransferMetrics,
)
from src.optimization_engines.advanced_memory_manager import (
    AdvancedBufferPool,
    BufferInfo,
    BufferTier,
    MemoryStats,
    TilingManager,
    TilingStrategy,
    TileDescriptor,
)
from src.optimization_engines.optimized_opencl_engine import (
    OpenCLOptimizationConfig,
    OptimizedOpenCLEngine,
    PerformanceMetrics,
)
import src.optimization_engines.optimized_opencl_engine as optimized_opencl_module


def test_adaptive_selector_promoted_scope_and_fallback() -> None:
    selector = ProductionKernelSelector(model_path="/tmp/nonexistent_model.pkl")

    in_scope = selector.select_kernel(1400, 1400, 1400)
    out_scope = selector.select_kernel(2048, 2048, 2048)

    assert in_scope["kernel_key"] == "tile20_v3_1400"
    assert out_scope["kernel_key"] != "tile20_v3_1400"


def test_adaptive_selector_predictions_report_t2_none_outside_scope() -> None:
    selector = ProductionKernelSelector(model_path="/tmp/nonexistent_model.pkl")

    predictions = selector.get_all_predictions(1024, 1024, 1024)

    assert predictions["tile20_v3_1400"] is None
    assert predictions["selected"] in {"tile20", "tile24", "tile16", "tile20_v3_1400"}


def test_adaptive_selector_runtime_feedback_without_policy() -> None:
    selector = ProductionKernelSelector(model_path="/tmp/nonexistent_model.pkl")

    feedback = selector.record_runtime_feedback(
        M=1024,
        N=1024,
        K=1024,
        static_arm="tile20",
        online_arm="tile20",
        online_gflops=800.0,
        static_gflops=790.0,
        online_max_error=1e-4,
    )

    assert feedback["policy_enabled"] is False
    assert feedback["executed_arm"] == "tile20"


def test_polaris_workgroup_rounding_and_summary() -> None:
    engine = AdvancedPolarisOpenCLEngine.__new__(AdvancedPolarisOpenCLEngine)
    engine.config = PolarisOptimizationConfig(tile_size=16)

    global_ws, local_ws = engine._calculate_polaris_work_groups(1001, 997)

    assert local_ws == (16, 16)
    assert global_ws[0] % local_ws[0] == 0
    assert global_ws[1] % local_ws[1] == 0

    engine.performance_history = []
    empty = engine.get_performance_summary()
    assert empty["message"] == "No performance data available"

    tm = TransferMetrics(
        host_to_device_time=0.1,
        device_to_host_time=0.1,
        overlap_efficiency=0.9,
        bandwidth_achieved=100.0,
        zero_copy_used=True,
    )
    engine.performance_history = [
        PolarisPerformanceMetrics(
            gflops_achieved=400.0,
            memory_bandwidth=120.0,
            kernel_efficiency=11.1,
            wavefront_occupancy=0.8,
            lds_utilization=0.4,
            transfer_metrics=tm,
        ),
        PolarisPerformanceMetrics(
            gflops_achieved=500.0,
            memory_bandwidth=140.0,
            kernel_efficiency=13.9,
            wavefront_occupancy=0.9,
            lds_utilization=0.5,
            transfer_metrics=tm,
        ),
    ]

    summary = engine.get_performance_summary()
    assert summary["max_gflops"] == 500.0
    assert summary["total_runs"] == 2


def test_polaris_cleanup_is_safe_without_gpu_runtime() -> None:
    class _DummyExecutor:
        def __init__(self) -> None:
            self.called = False

        def shutdown(self, wait: bool = True) -> None:
            self.called = wait

    engine = AdvancedPolarisOpenCLEngine.__new__(AdvancedPolarisOpenCLEngine)
    engine.pinned_buffers = [object()]
    engine.zero_copy_buffers = [object()]
    engine.transfer_events = [object()]
    dummy = _DummyExecutor()
    engine.executor = dummy

    engine.cleanup()

    assert engine.pinned_buffers == []
    assert engine.zero_copy_buffers == []
    assert engine.transfer_events == []
    assert dummy.called is True


def test_optimized_opencl_summary_handles_empty_and_populated_history() -> None:
    engine = OptimizedOpenCLEngine.__new__(OptimizedOpenCLEngine)
    engine.metrics_history = []
    assert engine.get_performance_summary() == {}

    engine.metrics_history = [
        PerformanceMetrics(
            gflops=100.0,
            bandwidth_gb_s=50.0,
            kernel_time_ms=2.0,
            total_time_ms=3.0,
            efficiency_percent=10.0,
        ),
        PerformanceMetrics(
            gflops=200.0,
            bandwidth_gb_s=70.0,
            kernel_time_ms=1.0,
            total_time_ms=2.0,
            efficiency_percent=20.0,
        ),
    ]

    summary = engine.get_performance_summary()
    assert summary["total_operations"] == 2
    assert summary["peak_gflops"] == 200.0
    assert summary["average_gflops"] == pytest.approx(150.0)


def test_create_polaris_engine_passes_config(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def _fake_init(self, config: PolarisOptimizationConfig) -> None:
        captured["config"] = config

    monkeypatch.setattr(AdvancedPolarisOpenCLEngine, "__init__", _fake_init)

    from src.optimization_engines.advanced_polaris_opencl_engine import create_polaris_engine

    _ = create_polaris_engine(use_zero_copy=False, use_async=True)

    cfg = captured["config"]
    assert cfg.use_zero_copy is False
    assert cfg.use_async_transfers is True
    assert cfg.use_pinned_memory is True


def test_optimized_opencl_benchmark_optimization_aggregates_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    engine = OptimizedOpenCLEngine.__new__(OptimizedOpenCLEngine)
    engine.config = OpenCLOptimizationConfig()
    engine.metrics_history = []

    def _mk_metrics(gflops: float) -> PerformanceMetrics:
        return PerformanceMetrics(
            gflops=gflops,
            bandwidth_gb_s=50.0,
            kernel_time_ms=1.0,
            total_time_ms=2.0,
            efficiency_percent=10.0,
        )

    def _fake_optimized_gemm(A: np.ndarray, B: np.ndarray):
        current = 120.0 if engine.config.use_vectorization else 95.0
        return np.zeros((A.shape[0], B.shape[1]), dtype=np.float32), _mk_metrics(current)

    def _fake_ultra(A: np.ndarray, B: np.ndarray):
        return np.zeros((A.shape[0], B.shape[1]), dtype=np.float32), _mk_metrics(180.0)

    def _fake_cw(A: np.ndarray, B: np.ndarray):
        return np.zeros((A.shape[0], B.shape[1]), dtype=np.float32), _mk_metrics(80.0)

    def _fake_low_rank(A: np.ndarray, B: np.ndarray):
        return np.zeros((A.shape[0], B.shape[1]), dtype=np.float32), _mk_metrics(140.0)

    monkeypatch.setattr(engine, "optimized_gemm", _fake_optimized_gemm)
    monkeypatch.setattr(engine, "optimized_gemm_ultra", _fake_ultra)
    monkeypatch.setattr(engine, "optimized_cw_gemm", _fake_cw)
    monkeypatch.setattr(engine, "optimized_low_rank_gemm", _fake_low_rank)
    monkeypatch.chdir(tmp_path)

    results = engine.benchmark_optimization(sizes=[128, 256])

    assert len(results["vectorized_gemm"]) == 2
    assert len(results["shared_memory_gemm"]) == 2
    assert len(results["ultra_optimized_gemm"]) == 2
    assert results["best_results"]["ultra_optimized_gemm"]["peak_gflops"] == 180.0
    assert (tmp_path / "opencl_optimization_benchmark.npz").exists()


def test_adaptive_selector_summary_snapshot_and_convenience_entrypoint() -> None:
    selector = ProductionKernelSelector(model_path="/tmp/nonexistent_model.pkl")

    summary = selector.benchmark_summary()
    assert "PRODUCTION KERNEL SELECTOR" in summary
    assert "tile20_v3_1400" in summary

    assert selector.get_t3_policy_snapshot() is None

    from src.optimization_engines.adaptive_kernel_selector import select_optimal_kernel

    recommendation = select_optimal_kernel(512, 512, 512)
    assert "kernel_key" in recommendation
    assert "predicted_gflops" in recommendation


def test_adaptive_selector_hybrid_model_and_heuristic_override_paths() -> None:
    selector = ProductionKernelSelector(model_path="/tmp/nonexistent_model.pkl")

    class _FakeModel:
        def predict(self, features: np.ndarray) -> np.ndarray:
            tile_size = int(features[0, 3])
            if tile_size == 24:
                return np.array([500.0], dtype=np.float32)
            return np.array([700.0], dtype=np.float32)

    selector.model_available = True
    selector.model = _FakeModel()

    medium = selector.select_kernel(1024, 1024, 1024)
    assert medium["selection_method"] == "hybrid (ml primary)"

    large = selector.select_kernel(3072, 3072, 3072)
    assert large["selection_method"] == "hybrid (heuristic override)"


def test_adaptive_selector_t3_policy_fallbacks_to_static_if_online_arm_not_predicted() -> None:
    selector = ProductionKernelSelector(model_path="/tmp/nonexistent_model.pkl")

    class _FakePolicy:
        def select(self, *, size: int, static_arm: str, eligible_arms: list[str]):
            _ = (size, static_arm, eligible_arms)
            return {
                "online_arm": "tile16",
                "selection_reason": "forced_invalid_arm",
            }

        def snapshot(self) -> dict[str, str]:
            return {"state": "ok"}

    selector.t3_policy = _FakePolicy()
    rec = selector.select_kernel(1024, 1024, 1024)

    assert rec["kernel_key"] == rec["static_kernel_key"]
    assert rec["selection_method"] == rec["static_selection_method"]
    assert rec["policy"] is None


def test_adaptive_selector_init_handles_corrupt_model_file(tmp_path) -> None:
    bad_model = tmp_path / "broken_model.pkl"
    bad_model.write_text("not-a-joblib-model", encoding="utf-8")

    selector = ProductionKernelSelector(model_path=str(bad_model))
    assert selector.model_available is False


def test_adaptive_selector_t3_enable_failure_path(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.optimization_engines import t3_controlled_policy as t3_policy_module

    def _raise(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("forced policy load error")

    monkeypatch.setattr(t3_policy_module.ControlledT3Policy, "from_policy_file", _raise)

    selector = ProductionKernelSelector(
        model_path="/tmp/nonexistent_model.pkl",
        enable_t3_controlled=True,
    )
    assert selector.t3_policy is None
    assert selector.t3_controlled_enabled is False


def test_adaptive_selector_predict_performance_falls_back_when_model_fails() -> None:
    selector = ProductionKernelSelector(model_path="/tmp/nonexistent_model.pkl")

    class _BrokenModel:
        def predict(self, features: np.ndarray) -> np.ndarray:
            _ = features
            raise RuntimeError("forced predict error")

    selector.model_available = True
    selector.model = _BrokenModel()

    predicted = selector._predict_performance(1024, 1024, 1024, "tile20")
    assert predicted == 867.0


def test_adaptive_selector_summary_feedback_and_snapshot_with_policy() -> None:
    selector = ProductionKernelSelector(model_path="/tmp/nonexistent_model.pkl")
    selector.model_available = True
    selector.metrics = {"r2_train": 0.99, "mae_train": 1.2}

    class _FakePolicy:
        def record_feedback(self, **kwargs):  # noqa: ANN003
            _ = kwargs
            return {"fallback_triggered": False}

        def snapshot(self) -> dict[str, str]:
            return {"state": "ok"}

    selector.t3_policy = _FakePolicy()

    summary = selector.benchmark_summary()
    assert "R² Score" in summary
    assert "MAE" in summary

    feedback = selector.record_runtime_feedback(
        M=1536,
        N=1536,
        K=1536,
        static_arm="tile20",
        online_arm="tile20_v3_1400",
        online_gflops=800.0,
        static_gflops=790.0,
        online_max_error=1e-4,
    )
    assert feedback["policy_enabled"] is True
    assert selector.get_t3_policy_snapshot() == {"state": "ok"}


def test_adaptive_selector_heuristic_promoted_and_peak_zone_branches() -> None:
    selector = ProductionKernelSelector(model_path="/tmp/nonexistent_model.pkl")
    assert selector._heuristic_selection(1400, 1400, 1400) == "tile20_v3_1400"
    assert selector._heuristic_selection(1600, 1600, 1600) == "tile20"


def test_optimized_opencl_benchmark_optimization_covers_error_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    engine = OptimizedOpenCLEngine.__new__(OptimizedOpenCLEngine)
    engine.config = OpenCLOptimizationConfig()
    engine.metrics_history = []

    def _raise(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("forced benchmark failure")

    def _ok(*args, **kwargs):  # noqa: ANN002, ANN003
        return np.zeros((2, 2), dtype=np.float32), PerformanceMetrics(
            gflops=55.0,
            bandwidth_gb_s=10.0,
            kernel_time_ms=1.0,
            total_time_ms=2.0,
            efficiency_percent=5.0,
        )

    monkeypatch.setattr(engine, "optimized_gemm", _raise)
    monkeypatch.setattr(engine, "optimized_gemm_ultra", _raise)
    monkeypatch.setattr(engine, "optimized_cw_gemm", _ok)
    monkeypatch.setattr(engine, "optimized_low_rank_gemm", _raise)
    monkeypatch.chdir(tmp_path)

    # Size > 1024 to exercise CW skip branch and all exception fallbacks.
    results = engine.benchmark_optimization(sizes=[1536])

    assert results["vectorized_gemm"] == [0.0]
    assert results["shared_memory_gemm"] == [0.0]
    assert results["ultra_optimized_gemm"] == [0.0]
    assert results["cw_gemm"] == [0.0]
    assert results["low_rank_gemm"] == [0.0]
    assert "best_results" in results


def test_optimized_opencl_benchmark_optimization_covers_cw_failure_branch(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    engine = OptimizedOpenCLEngine.__new__(OptimizedOpenCLEngine)
    engine.config = OpenCLOptimizationConfig()
    engine.metrics_history = []

    def _ok(*args, **kwargs):  # noqa: ANN002, ANN003
        return np.zeros((2, 2), dtype=np.float32), PerformanceMetrics(
            gflops=42.0,
            bandwidth_gb_s=10.0,
            kernel_time_ms=1.0,
            total_time_ms=2.0,
            efficiency_percent=5.0,
        )

    def _raise(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("forced cw failure")

    monkeypatch.setattr(engine, "optimized_gemm", _ok)
    monkeypatch.setattr(engine, "optimized_gemm_ultra", _ok)
    monkeypatch.setattr(engine, "optimized_cw_gemm", _raise)
    monkeypatch.setattr(engine, "optimized_low_rank_gemm", _ok)
    monkeypatch.chdir(tmp_path)

    # Size <= 1024 to execute CW path and its exception branch.
    results = engine.benchmark_optimization(sizes=[256])
    assert results["cw_gemm"] == [0.0]


def test_optimized_opencl_benchmark_default_sizes_branch(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    engine = OptimizedOpenCLEngine.__new__(OptimizedOpenCLEngine)
    engine.config = OpenCLOptimizationConfig()
    engine.metrics_history = []

    monkeypatch.setattr(
        optimized_opencl_module.np.random,
        "randn",
        lambda *shape: np.ones((2, 2), dtype=np.float32),
    )

    def _ok(*args, **kwargs):  # noqa: ANN002, ANN003
        return np.ones((2, 2), dtype=np.float32), PerformanceMetrics(
            gflops=10.0,
            bandwidth_gb_s=1.0,
            kernel_time_ms=1.0,
            total_time_ms=1.0,
            efficiency_percent=1.0,
        )

    monkeypatch.setattr(engine, "optimized_gemm", _ok)
    monkeypatch.setattr(engine, "optimized_gemm_ultra", _ok)
    monkeypatch.setattr(engine, "optimized_cw_gemm", _ok)
    monkeypatch.setattr(engine, "optimized_low_rank_gemm", _ok)
    monkeypatch.chdir(tmp_path)

    # sizes=None hits the default-sizes branch.
    results = engine.benchmark_optimization()
    assert len(results["sizes"]) == 4


def test_optimized_opencl_runtime_guards_without_opencl_context() -> None:
    engine = OptimizedOpenCLEngine.__new__(OptimizedOpenCLEngine)
    engine.config = OpenCLOptimizationConfig()
    engine.context = None
    engine.queue = None
    engine.program = None
    engine.device = None

    A = np.ones((2, 2), dtype=np.float32)
    B = np.ones((2, 2), dtype=np.float32)

    with pytest.raises(RuntimeError, match="OpenCL runtime not initialized"):
        engine.optimized_gemm(A, B)

    with pytest.raises(RuntimeError, match="OpenCL runtime not initialized"):
        engine.optimized_cw_gemm(A, B)

    with pytest.raises(RuntimeError, match="OpenCL runtime not initialized"):
        engine.optimized_gemm_ultra(A, B)

    with pytest.raises(RuntimeError, match="OpenCL runtime not initialized"):
        engine.optimized_low_rank_gemm(A, B)


def test_optimized_opencl_get_optimal_work_groups_rounding() -> None:
    engine = OptimizedOpenCLEngine.__new__(OptimizedOpenCLEngine)
    engine.config = OpenCLOptimizationConfig(tile_size=32, local_work_size=(16, 16))

    global_ws, local_ws = engine._get_optimal_work_groups(65, 97)
    assert local_ws == (16, 16)
    assert global_ws == (128, 96)


def test_advanced_memory_manager_dataclass_behaviors() -> None:
    info = BufferInfo(
        buffer=object(),
        size=1024,
        tier=BufferTier.COLD,
        last_access=0.0,
        access_count=3,
    )
    info.touch()
    assert info.tier == BufferTier.WARM

    info.access_count = 10
    info.touch()
    assert info.tier == BufferTier.HOT

    tile = TileDescriptor(
        row_start=2,
        row_end=6,
        col_start=3,
        col_end=8,
        size_bytes=80,
    )
    assert tile.shape == (4, 5)
    assert tile.rows == 4
    assert tile.cols == 5

    stats = MemoryStats(total_allocated=200, peak_usage=150, pool_hits=3, pool_misses=1)
    assert stats.hit_rate == pytest.approx(0.75)
    assert stats.memory_efficiency == pytest.approx(0.25)
    summary = stats.summary()
    assert "MEMORY STATISTICS" in summary
    assert "Pool Hit Rate" in summary


def test_advanced_buffer_pool_internal_branches() -> None:
    class _DummyBuffer:
        def __init__(self, size: int) -> None:
            self.size = size
            self.released = False

        def release(self) -> None:
            self.released = True

    pool = AdvancedBufferPool.__new__(AdvancedBufferPool)
    pool.max_buffers_per_size = 1
    pool._read_pool = {}
    pool._write_pool = {}
    pool._pinned_buffers = set()
    pool.stats = MemoryStats()
    pool._current_pool_size = 0
    pool._lock = threading.Lock()

    exact = BufferInfo(object(), 128, BufferTier.WARM, 0.0, 1)
    larger = BufferInfo(object(), 150, BufferTier.WARM, 0.0, 1)
    candidates = {128: [exact], 150: [larger]}
    assert pool._find_best_match(candidates, 128) is exact
    assert pool._find_best_match(candidates, 140) is larger
    assert pool._find_best_match(candidates, 300) is None

    old = _DummyBuffer(64)
    new = _DummyBuffer(64)
    pool._read_pool = {64: [BufferInfo(old, 64, BufferTier.WARM, 0.0, 1)]}
    pool._write_pool = {}
    pool._current_pool_size = 64
    pool.return_buffer(new, read_only=True)
    assert old.released is True
    assert len(pool._read_pool[64]) == 1

    pinned = _DummyBuffer(32)
    pool._pinned_buffers.add(id(pinned))
    pool.return_buffer(pinned, read_only=True)
    assert id(pinned) in pool._pinned_buffers

    cold = _DummyBuffer(40)
    warm = _DummyBuffer(30)
    pool._read_pool = {
        40: [BufferInfo(cold, 40, BufferTier.COLD, 0.0, 1, is_pinned=False)],
        30: [BufferInfo(warm, 30, BufferTier.WARM, 1.0, 1, is_pinned=False)],
    }
    pool._write_pool = {}
    pool._current_pool_size = 70
    pool._evict_cold_buffers(needed_size=35)
    assert cold.released is True
    assert pool.stats.evictions >= 1

    pool.pin_buffer(new)
    assert id(new) in pool._pinned_buffers
    pool.unpin_buffer(new)
    assert id(new) not in pool._pinned_buffers

    pool.clear()
    assert pool._current_pool_size == 0
    assert pool._pinned_buffers == set()
    assert pool.get_stats().current_usage == 0


def test_tiling_manager_strategy_selection() -> None:
    manager = TilingManager.__new__(TilingManager)

    small = manager.select_strategy((16, 16), np.float32)
    medium = manager.select_strategy((2048, 2048), np.float32)
    large = manager.select_strategy((8192, 8192), np.float32)

    assert small == (TilingStrategy.NONE, 0)
    assert medium == (TilingStrategy.SQUARE, TilingManager.MEDIUM_TILE_SIZE)
    assert large == (TilingStrategy.SQUARE, TilingManager.LARGE_TILE_SIZE)


def test_tiling_manager_tile_creation_and_cache_lifecycle() -> None:
    class _DummyPool:
        def __init__(self) -> None:
            self.stats = MemoryStats()
            self.returned = 0

        def get_buffer(self, size: int, read_only: bool, data: np.ndarray):
            _ = (size, read_only, data)
            return object()

        def return_buffer(self, buf, read_only: bool = True):  # noqa: ANN001
            _ = (buf, read_only)
            self.returned += 1

    manager = TilingManager.__new__(TilingManager)
    manager.buffer_pool = _DummyPool()
    manager.queue = None
    manager.max_tiles_in_memory = 1
    manager._loaded_tiles = __import__("collections").OrderedDict()

    matrix = np.arange(16, dtype=np.float32).reshape(4, 4)
    tiles = manager.create_tiles(matrix, tile_size=2)
    assert len(tiles) == 4
    assert manager.buffer_pool.stats.tiles_created == 4

    _ = manager.load_tile(matrix, tiles[0], read_only=True)
    _ = manager.load_tile(matrix, tiles[1], read_only=True)
    assert manager.buffer_pool.returned >= 1

    manager.unload_tile(tiles[1], read_only=True)
    manager.clear_cache()
