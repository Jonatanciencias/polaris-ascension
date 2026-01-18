"""
Tests for sparse operations and pruning algorithms.

Session 10: Sparse Networks Implementation
- Magnitude pruning
- Structured pruning (channel/filter/head)
- Gradual pruning with polynomial schedule
"""

import pytest
import numpy as np
from src.compute.sparse import (
    SparseTensorConfig,
    SparseOperations,
    MagnitudePruner,
    StructuredPruner,
    GradualPruner,
    create_sparse_layer,
)


class TestSparseTensorConfig:
    """Test sparse tensor configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SparseTensorConfig()
        assert config.target_sparsity == 0.7
        assert config.block_size == 64
        assert config.wavefront_size == 64
        
    def test_custom_config(self):
        """Test custom configuration."""
        config = SparseTensorConfig(
            target_sparsity=0.9,
            block_size=32,
            wavefront_size=32
        )
        assert config.target_sparsity == 0.9
        assert config.block_size == 32


class TestSparseOperations:
    """Test sparse matrix operations."""
    
    def test_analyze_sparsity_dense(self):
        """Test sparsity analysis on dense matrix."""
        ops = SparseOperations()
        dense_matrix = np.ones((100, 100))
        
        result = ops.analyze_sparsity(dense_matrix)
        
        assert result["sparsity"] == 0.0
        assert result["total_elements"] == 10000
        assert result["nonzero_elements"] == 10000
        assert result["zero_elements"] == 0
        
    def test_analyze_sparsity_sparse(self):
        """Test sparsity analysis on 70% sparse matrix."""
        ops = SparseOperations()
        sparse_matrix = np.zeros((100, 100))
        sparse_matrix[:30, :] = 1.0  # Only 30% non-zero
        
        result = ops.analyze_sparsity(sparse_matrix)
        
        assert abs(result["sparsity"] - 0.7) < 0.01
        assert result["total_elements"] == 10000
        assert result["nonzero_elements"] == 3000
        
    def test_to_csr_format(self):
        """Test conversion to CSR format."""
        ops = SparseOperations()
        matrix = np.array([
            [1, 0, 2],
            [0, 0, 3],
            [4, 5, 0]
        ])
        
        csr = ops.to_csr(matrix)
        
        assert len(csr["values"]) == 5
        assert len(csr["col_indices"]) == 5
        assert len(csr["row_pointers"]) == 4
        assert np.array_equal(csr["values"], [1, 2, 3, 4, 5])
        assert np.array_equal(csr["col_indices"], [0, 2, 2, 0, 1])
        assert np.array_equal(csr["row_pointers"], [0, 2, 3, 5])
        
    def test_sparse_matmul_basic(self):
        """Test sparse matrix multiplication."""
        ops = SparseOperations()
        A = np.array([[1, 0], [0, 2]])
        B = np.array([[3, 0], [0, 4]])
        
        result = ops.sparse_matmul(A, B)
        expected = np.array([[3, 0], [0, 8]])
        
        assert np.array_equal(result, expected)


class TestMagnitudePruner:
    """Test magnitude-based pruning."""
    
    def test_initialization(self):
        """Test pruner initialization."""
        pruner = MagnitudePruner(sparsity=0.8)
        assert pruner.sparsity == 0.8
        assert pruner.scope == "local"
        
    def test_invalid_sparsity(self):
        """Test invalid sparsity values."""
        with pytest.raises(ValueError):
            MagnitudePruner(sparsity=1.5)
        with pytest.raises(ValueError):
            MagnitudePruner(sparsity=-0.1)
            
    def test_prune_layer_70_percent(self):
        """Test pruning to 70% sparsity."""
        pruner = MagnitudePruner(sparsity=0.7)
        weights = np.random.randn(100, 100)
        
        pruned_weights, mask = pruner.prune_layer(weights)
        
        actual_sparsity = pruner.measure_sparsity(pruned_weights)
        assert abs(actual_sparsity - 0.7) < 0.05  # Within 5% tolerance
        assert mask.shape == weights.shape
        assert np.all((mask == 0) | (mask == 1))
        
    def test_prune_layer_preserves_large_weights(self):
        """Test that pruning preserves largest weights."""
        pruner = MagnitudePruner(sparsity=0.5)
        weights = np.array([[1.0, 0.1], [10.0, 0.2]])
        
        pruned_weights, mask = pruner.prune_layer(weights)
        
        # Large weights should be preserved
        assert pruned_weights[0, 0] != 0  # 1.0 should remain
        assert pruned_weights[1, 0] != 0  # 10.0 should remain
        # Small weights should be pruned
        assert np.sum(mask) == 2  # Only 2 weights remain (50% pruned)
        
    def test_measure_sparsity(self):
        """Test sparsity measurement."""
        pruner = MagnitudePruner()
        weights = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        
        sparsity = pruner.measure_sparsity(weights)
        
        assert abs(sparsity - 0.666) < 0.01  # 6/9 zeros
        
    def test_prune_model_local(self):
        """Test local model pruning (per-layer)."""
        pruner = MagnitudePruner(sparsity=0.5, scope="local")
        model_weights = {
            "layer1": np.random.randn(50, 50),
            "layer2": np.random.randn(30, 30),
        }
        
        pruned_weights, masks = pruner.prune_model(model_weights)
        
        assert len(pruned_weights) == 2
        assert len(masks) == 2
        # Each layer should be ~50% sparse
        for name in model_weights:
            sparsity = pruner.measure_sparsity(pruned_weights[name])
            assert abs(sparsity - 0.5) < 0.1
            
    def test_prune_model_global(self):
        """Test global model pruning (across all layers)."""
        pruner = MagnitudePruner(sparsity=0.7, scope="global")
        model_weights = {
            "layer1": np.random.randn(50, 50),
            "layer2": np.random.randn(30, 30),
        }
        
        pruned_weights, masks = pruner.prune_model(model_weights)
        
        # Total sparsity should be ~70%
        total_params = sum(w.size for w in model_weights.values())
        total_nonzero = sum(np.count_nonzero(w) for w in pruned_weights.values())
        global_sparsity = 1.0 - (total_nonzero / total_params)
        
        assert abs(global_sparsity - 0.7) < 0.05
        
    def test_compression_stats(self):
        """Test compression statistics calculation."""
        pruner = MagnitudePruner(sparsity=0.9)
        model_weights = {
            "layer1": np.random.randn(100, 100),
        }
        
        _, masks = pruner.prune_model(model_weights)
        stats = pruner.get_compression_stats(masks)
        
        assert stats["total_parameters"] == 10000
        assert stats["sparsity"] > 0.85  # Close to 90%
        assert stats["compression_ratio"] > 5  # Should be ~10x
        assert "memory_reduction" in stats
        
    def test_pruning_history(self):
        """Test that pruning history is recorded."""
        pruner = MagnitudePruner(sparsity=0.8)
        weights = np.random.randn(100, 100)
        
        pruner.prune_layer(weights)
        
        assert len(pruner.pruning_history) == 1
        history = pruner.pruning_history[0]
        assert history["target_sparsity"] == 0.8
        assert "actual_sparsity" in history
        assert "threshold" in history
        assert history["layer_shape"] == (100, 100)


class TestStructuredPruner:
    """Test structured pruning (channels/filters/heads)."""
    
    def test_initialization(self):
        """Test structured pruner initialization."""
        pruner = StructuredPruner(sparsity=0.5, granularity="channel")
        assert pruner.sparsity == 0.5
        assert pruner.granularity == "channel"
        assert pruner.importance_metric == "l1"
        
    def test_prune_channels_conv(self):
        """Test channel pruning for convolutional layer."""
        pruner = StructuredPruner(sparsity=0.5, importance_metric="l1")
        # Conv weights: (out_channels=64, in_channels=32, H=3, W=3)
        weights = np.random.randn(64, 32, 3, 3)
        
        pruned_weights, kept_indices = pruner.prune_channels(weights, channel_axis=0)
        
        assert pruned_weights.shape[0] == 32  # 50% of 64 channels removed
        assert len(kept_indices) == 32
        assert pruned_weights.shape[1:] == (32, 3, 3)  # Other dims unchanged
        
    def test_prune_channels_preserves_important(self):
        """Test that important channels are preserved."""
        pruner = StructuredPruner(sparsity=0.5, importance_metric="l1")
        # Create weights where first channel has high magnitude
        weights = np.random.randn(10, 5) * 0.1
        weights[0, :] = 10.0  # First channel very important
        
        pruned_weights, kept_indices = pruner.prune_channels(weights, channel_axis=0)
        
        assert 0 in kept_indices  # First channel should be kept
        assert len(kept_indices) == 5  # 50% kept
        
    def test_prune_filters(self):
        """Test filter pruning (input channels)."""
        pruner = StructuredPruner(sparsity=0.5)
        weights = np.random.randn(64, 32, 3, 3)
        
        pruned_weights, kept_indices = pruner.prune_filters(weights, filter_axis=1)
        
        assert pruned_weights.shape[1] == 16  # 50% of 32 filters removed
        assert pruned_weights.shape[0] == 64  # Out channels unchanged
        
    def test_importance_metric_l1(self):
        """Test L1 importance metric."""
        pruner = StructuredPruner(importance_metric="l1")
        weights = np.array([[1, 2], [3, 4], [5, 6]])
        
        scores = pruner._compute_importance_scores(weights, axis=0)
        
        assert len(scores) == 3
        assert scores[0] == 3  # |1| + |2|
        assert scores[1] == 7  # |3| + |4|
        assert scores[2] == 11  # |5| + |6|
        
    def test_importance_metric_l2(self):
        """Test L2 importance metric."""
        pruner = StructuredPruner(importance_metric="l2")
        weights = np.array([[3, 4], [5, 12]])
        
        scores = pruner._compute_importance_scores(weights, axis=0)
        
        assert len(scores) == 2
        assert abs(scores[0] - 5.0) < 0.01  # sqrt(9 + 16)
        assert abs(scores[1] - 13.0) < 0.01  # sqrt(25 + 144)
        
    def test_prune_attention_heads(self):
        """Test attention head pruning."""
        pruner = StructuredPruner(sparsity=0.5)
        # 8 heads, 64-dim model
        d_model = 64
        num_heads = 8
        attention_weights = np.random.randn(d_model, d_model)
        
        pruned_weights, kept_indices = pruner.prune_attention_heads(
            attention_weights, 
            num_heads
        )
        
        assert len(kept_indices) == 4  # 50% of 8 heads
        assert pruned_weights.shape[0] == 32  # 4 heads * 8 dim per head
        
    def test_cannot_prune_all_channels(self):
        """Test that at least 1 channel is always kept."""
        pruner = StructuredPruner(sparsity=0.99)
        weights = np.random.randn(10, 5)
        
        # Should keep at least 1 channel even with 99% sparsity
        pruned_weights, kept_indices = pruner.prune_channels(weights)
        
        assert pruned_weights.shape[0] >= 1
        assert len(kept_indices) >= 1
            
    def test_pruning_history_tracking(self):
        """Test that pruning history is tracked."""
        pruner = StructuredPruner(sparsity=0.5)
        weights = np.random.randn(64, 32, 3, 3)
        
        pruner.prune_channels(weights)
        
        assert len(pruner.pruning_history) == 1
        history = pruner.pruning_history[0]
        assert history["granularity"] == "channel"
        assert history["original_channels"] == 64
        assert history["remaining_channels"] == 32
        assert history["original_shape"] == (64, 32, 3, 3)


class TestGradualPruner:
    """Test gradual iterative pruning with schedule."""
    
    def test_initialization(self):
        """Test gradual pruner initialization."""
        pruner = GradualPruner(
            initial_sparsity=0.0,
            final_sparsity=0.9,
            begin_step=1000,
            end_step=10000
        )
        assert pruner.initial_sparsity == 0.0
        assert pruner.final_sparsity == 0.9
        assert pruner.begin_step == 1000
        assert pruner.end_step == 10000
        
    def test_compute_sparsity_before_begin(self):
        """Test sparsity before pruning begins."""
        pruner = GradualPruner(
            initial_sparsity=0.0,
            final_sparsity=0.9,
            begin_step=1000,
            end_step=10000
        )
        
        sparsity = pruner.compute_sparsity(500)
        
        assert sparsity == 0.0
        
    def test_compute_sparsity_after_end(self):
        """Test sparsity after pruning ends."""
        pruner = GradualPruner(
            initial_sparsity=0.0,
            final_sparsity=0.9,
            begin_step=1000,
            end_step=10000
        )
        
        sparsity = pruner.compute_sparsity(15000)
        
        assert sparsity == 0.9
        
    def test_compute_sparsity_polynomial_decay(self):
        """Test polynomial decay schedule."""
        pruner = GradualPruner(
            initial_sparsity=0.0,
            final_sparsity=0.9,
            begin_step=0,
            end_step=1000
        )
        
        # At 50% progress
        sparsity_mid = pruner.compute_sparsity(500)
        # Should be between 0 and 0.9, but closer to 0.9 due to cubic decay
        assert 0.7 < sparsity_mid < 0.9
        
        # At 75% progress
        sparsity_75 = pruner.compute_sparsity(750)
        assert sparsity_75 > sparsity_mid
        assert sparsity_75 > 0.85
        
    def test_should_prune_frequency(self):
        """Test pruning frequency check."""
        pruner = GradualPruner(
            begin_step=1000,
            end_step=10000,
            frequency=100
        )
        
        assert pruner.should_prune(1000) == True
        assert pruner.should_prune(1100) == True
        assert pruner.should_prune(1050) == False
        assert pruner.should_prune(500) == False  # Before begin
        assert pruner.should_prune(15000) == False  # After end
        
    def test_prune_step_magnitude(self):
        """Test gradual pruning step with magnitude method."""
        pruner = GradualPruner(
            initial_sparsity=0.0,
            final_sparsity=0.9,
            begin_step=0,
            end_step=1000,
            pruning_method="magnitude"
        )
        weights = np.random.randn(100, 100)
        
        pruned_weights, mask = pruner.prune_step(weights, step=500)
        
        sparsity = pruner.base_pruner.measure_sparsity(pruned_weights)
        assert sparsity > 0.5  # Should be pruned by this point
        assert len(pruner.pruning_schedule) == 1
        
    def test_prune_step_structured(self):
        """Test gradual pruning with structured method."""
        pruner = GradualPruner(
            initial_sparsity=0.0,
            final_sparsity=0.5,
            begin_step=0,
            end_step=1000,
            pruning_method="structured"
        )
        weights = np.random.randn(64, 32, 3, 3)
        
        pruned_weights, mask = pruner.prune_step(weights, step=500)
        
        # Some channels should be removed
        assert pruned_weights.shape[0] < 64
        
    def test_get_schedule(self):
        """Test schedule generation for visualization."""
        pruner = GradualPruner(
            initial_sparsity=0.0,
            final_sparsity=0.9,
            begin_step=0,
            end_step=1000,
            frequency=100
        )
        
        schedule = pruner.get_schedule(num_steps=1500)
        
        assert len(schedule) > 0
        # Check monotonic increase
        sparsities = [s for _, s in schedule]
        assert all(sparsities[i] <= sparsities[i+1] for i in range(len(sparsities)-1))
        
    def test_gradual_vs_oneshot_comparison(self):
        """Test that gradual pruning differs from one-shot."""
        weights = np.random.randn(100, 100)
        
        # One-shot pruning
        oneshot = MagnitudePruner(sparsity=0.9)
        pruned_oneshot, _ = oneshot.prune_layer(weights.copy())
        
        # Gradual pruning at midpoint
        gradual = GradualPruner(
            initial_sparsity=0.0,
            final_sparsity=0.9,
            begin_step=0,
            end_step=1000
        )
        pruned_gradual, _ = gradual.prune_step(weights.copy(), step=500)
        
        sparsity_oneshot = oneshot.measure_sparsity(pruned_oneshot)
        sparsity_gradual = gradual.base_pruner.measure_sparsity(pruned_gradual)
        
        # Gradual should have lower sparsity at midpoint
        assert sparsity_gradual < sparsity_oneshot


class TestFactoryFunctions:
    """Test factory and utility functions."""
    
    def test_create_sparse_layer_default(self):
        """Test sparse layer creation with defaults."""
        layer = create_sparse_layer(1024, 512)
        
        assert layer["type"] == "sparse_linear"
        assert layer["in_features"] == 1024
        assert layer["out_features"] == 512
        assert layer["sparsity"] == 0.9
        assert layer["gpu_family"] == "polaris"
        
    def test_create_sparse_layer_custom(self):
        """Test sparse layer with custom parameters."""
        layer = create_sparse_layer(
            2048, 1024, 
            sparsity=0.8, 
            gpu_family="navi"
        )
        
        assert layer["sparsity"] == 0.8
        assert layer["gpu_family"] == "navi"
        
    def test_wavefront_aligned_detection(self):
        """Test wavefront alignment detection."""
        # Aligned (multiples of 64)
        layer_aligned = create_sparse_layer(1024, 512)
        assert layer_aligned["wavefront_aligned"] == True
        
        # Not aligned
        layer_unaligned = create_sparse_layer(1000, 500)
        assert layer_unaligned["wavefront_aligned"] == False


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_prune_zero_tensor(self):
        """Test pruning all-zero tensor."""
        pruner = MagnitudePruner(sparsity=0.5)
        weights = np.zeros((10, 10))
        
        pruned_weights, mask = pruner.prune_layer(weights)
        
        # All zeros have same magnitude, so result is all zeros
        assert np.allclose(pruned_weights, 0)
        sparsity = pruner.measure_sparsity(pruned_weights)
        # Resultado esperado: 100% sparse (todos zeros)
        assert sparsity == 1.0
        
    def test_prune_single_element(self):
        """Test pruning single-element tensor."""
        pruner = MagnitudePruner(sparsity=0.5)
        weights = np.array([[5.0]])
        
        pruned_weights, mask = pruner.prune_layer(weights)
        
        # Single element should be kept (can't achieve 50% sparsity)
        assert pruned_weights[0, 0] != 0
        
    def test_structured_prune_small_layer(self):
        """Test structured pruning on very small layer."""
        pruner = StructuredPruner(sparsity=0.9)
        weights = np.random.randn(5, 10)
        
        # Should keep at least 1 channel
        pruned_weights, indices = pruner.prune_channels(weights)
        assert pruned_weights.shape[0] >= 1
        
    def test_negative_weights(self):
        """Test pruning with negative weights."""
        pruner = MagnitudePruner(sparsity=0.5)
        weights = np.array([[-10, -1], [2, 0.5]])
        
        pruned_weights, mask = pruner.prune_layer(weights)
        
        # -10 and 2 should be kept (highest magnitude)
        assert pruned_weights[0, 0] != 0  # -10
        assert pruned_weights[1, 0] != 0  # 2
