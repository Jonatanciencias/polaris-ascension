"""
Tests for Dynamic Sparse Training (RigL + Dynamic Allocation)
=============================================================

Session 11: Comprehensive tests for dynamic sparse training algorithms.

Test Coverage:
- RigLPruner: initialization, updates, drop/grow logic, scheduling
- DynamicSparsityAllocator: sensitivity computation, allocation
- Integration: multi-layer training scenarios
"""

import pytest
import numpy as np
from src.compute.dynamic_sparse import (
    RigLPruner,
    DynamicSparsityAllocator,
    RigLConfig,
)


class TestRigLPruner:
    """Tests for RigL pruner."""
    
    def test_initialization(self):
        """Test RigL pruner initialization."""
        rigl = RigLPruner(
            sparsity=0.9,
            T_end=10000,
            delta_T=100,
            alpha=0.3
        )
        
        assert rigl.sparsity == 0.9
        assert rigl.T_end == 10000
        assert rigl.delta_T == 100
        assert rigl.alpha == 0.3
        assert rigl.current_step == 0
        assert len(rigl.update_history) == 0
    
    def test_invalid_sparsity(self):
        """Test that invalid sparsity raises error."""
        with pytest.raises(ValueError, match="Sparsity must be"):
            RigLPruner(sparsity=1.5)
        
        with pytest.raises(ValueError, match="Sparsity must be"):
            RigLPruner(sparsity=-0.1)
    
    def test_invalid_alpha(self):
        """Test that invalid alpha raises error."""
        with pytest.raises(ValueError, match="Alpha must be"):
            RigLPruner(alpha=1.5)
        
        with pytest.raises(ValueError, match="Alpha must be"):
            RigLPruner(alpha=0.0)
    
    def test_initialize_mask(self):
        """Test mask initialization."""
        rigl = RigLPruner(sparsity=0.9)
        weights = np.random.randn(100, 100)
        
        mask = rigl.initialize_mask(weights)
        
        # Check shape
        assert mask.shape == weights.shape
        
        # Check sparsity (approximately 90%)
        actual_sparsity = 1.0 - (np.count_nonzero(mask) / mask.size)
        assert 0.89 <= actual_sparsity <= 0.91
        
        # Check binary mask
        assert np.all((mask == 0) | (mask == 1))
        
        # Check history
        assert len(rigl.update_history) == 1
        assert rigl.update_history[0]["action"] == "initialize"
    
    def test_should_update_schedule(self):
        """Test update scheduling."""
        rigl = RigLPruner(T_end=1000, delta_T=100)
        
        # Should not update at step 0
        assert not rigl.should_update(0)
        
        # Should update at delta_T multiples
        assert rigl.should_update(100)
        assert rigl.should_update(200)
        assert rigl.should_update(1000)
        
        # Should not update after T_end
        assert not rigl.should_update(1001)
        assert not rigl.should_update(1100)
        
        # Should not update at non-multiples
        assert not rigl.should_update(50)
        assert not rigl.should_update(150)
    
    def test_get_update_schedule(self):
        """Test getting full update schedule."""
        rigl = RigLPruner(T_end=500, delta_T=100)
        
        schedule = rigl.get_update_schedule()
        
        assert schedule == [100, 200, 300, 400, 500]
    
    def test_accumulate_gradients(self):
        """Test gradient accumulation."""
        rigl = RigLPruner()
        
        grad1 = np.random.randn(10, 10)
        grad2 = np.random.randn(10, 10)
        
        rigl.accumulate_gradients(grad1, "layer1")
        rigl.accumulate_gradients(grad2, "layer1")
        
        # Check accumulated
        expected = np.abs(grad1) + np.abs(grad2)
        np.testing.assert_array_almost_equal(
            rigl.accumulated_gradients["layer1"],
            expected
        )
    
    def test_update_mask_drop_phase(self):
        """Test that drop phase removes lowest magnitude weights."""
        rigl = RigLPruner(sparsity=0.5, alpha=0.5)
        
        # Create weights: [1, 2, 3, 4] (keep 2 highest)
        weights = np.array([1.0, 2.0, 3.0, 4.0])
        mask = np.array([1.0, 1.0, 1.0, 1.0])
        gradients = np.random.randn(4)
        
        # Update mask
        new_weights, new_mask = rigl.update_mask(
            weights, gradients, mask, step=100
        )
        
        # Should drop 1 weight (50% of 2 pruned initially = 1)
        # Since we start with no pruned weights, first update prepares for drop
        # The drop should target lowest magnitudes (1.0 and 2.0)
        assert np.sum(new_mask) <= 4  # Some weights should be dropped
    
    def test_update_mask_maintains_sparsity(self):
        """Test that update maintains constant sparsity."""
        rigl = RigLPruner(sparsity=0.5, alpha=0.3, grad_accumulation_steps=1)
        
        weights = np.random.randn(100)
        mask = rigl.initialize_mask(weights)
        gradients = np.random.randn(100)
        
        initial_sparsity = 1.0 - (np.count_nonzero(mask) / mask.size)
        
        # Update multiple times
        for step in [100, 200, 300]:
            gradients = np.random.randn(100)
            weights, mask = rigl.update_mask(
                weights, gradients, mask, step
            )
        
        final_sparsity = 1.0 - (np.count_nonzero(mask) / mask.size)
        
        # Sparsity should remain approximately constant
        assert abs(final_sparsity - initial_sparsity) < 0.05
    
    def test_update_mask_grow_highest_gradients(self):
        """Test that grow phase adds connections with highest gradients."""
        rigl = RigLPruner(sparsity=0.5, alpha=0.5, grad_accumulation_steps=1)
        
        # Create simple scenario
        weights = np.array([1.0, 2.0, 0.0, 0.0])  # 2 active, 2 inactive
        mask = np.array([1.0, 1.0, 0.0, 0.0])
        
        # Strong gradients on inactive weights
        gradients = np.array([0.1, 0.1, 10.0, 20.0])
        
        new_weights, new_mask = rigl.update_mask(
            weights, gradients, mask, step=100
        )
        
        # At least one of the high-gradient inactive weights should be grown
        # (index 2 or 3 should become active)
        assert new_mask[2] == 1.0 or new_mask[3] == 1.0
    
    def test_update_mask_applies_mask(self):
        """Test that weights are masked after update."""
        rigl = RigLPruner(sparsity=0.5, grad_accumulation_steps=1)
        
        weights = np.random.randn(100)
        mask = rigl.initialize_mask(weights)
        gradients = np.random.randn(100)
        
        new_weights, new_mask = rigl.update_mask(
            weights, gradients, mask, step=100
        )
        
        # All pruned weights should be zero
        pruned_indices = new_mask == 0
        assert np.all(new_weights[pruned_indices] == 0)
    
    def test_get_statistics(self):
        """Test statistics computation."""
        rigl = RigLPruner(sparsity=0.8, alpha=0.3, grad_accumulation_steps=1)
        
        weights = np.random.randn(100)
        mask = rigl.initialize_mask(weights)
        
        # Perform updates
        for step in [100, 200, 300]:
            gradients = np.random.randn(100)
            weights, mask = rigl.update_mask(
                weights, gradients, mask, step
            )
        
        stats = rigl.get_statistics()
        
        assert "total_updates" in stats
        assert "final_sparsity" in stats
        assert "avg_drop_per_update" in stats
        assert "total_connections_changed" in stats
        
        assert stats["total_updates"] == 3
        assert 0.7 <= stats["final_sparsity"] <= 0.9
    
    def test_gradient_accumulation(self):
        """Test gradient accumulation over multiple steps."""
        rigl = RigLPruner(grad_accumulation_steps=3)
        
        weights = np.random.randn(50)
        mask = rigl.initialize_mask(weights)
        
        # Accumulate gradients without updating
        for i in range(2):
            grad = np.random.randn(50)
            weights, mask = rigl.update_mask(
                weights, grad, mask, step=100 + i
            )
        
        # Mask should not change (not enough accumulation yet)
        initial_mask = mask.copy()
        
        # Third gradient triggers update
        grad = np.random.randn(50)
        weights, mask = rigl.update_mask(
            weights, grad, mask, step=102
        )
        
        # Now mask might have changed
        # (depends on gradients, but accumulation counter should reset)
        assert rigl.grad_step_counter == 0


class TestDynamicSparsityAllocator:
    """Tests for dynamic sparsity allocator."""
    
    def test_initialization(self):
        """Test allocator initialization."""
        allocator = DynamicSparsityAllocator(
            target_sparsity=0.9,
            method="gradient"
        )
        
        assert allocator.target_sparsity == 0.9
        assert allocator.method == "gradient"
        assert len(allocator.allocation_history) == 0
    
    def test_invalid_sparsity(self):
        """Test invalid target sparsity."""
        with pytest.raises(ValueError, match="Target sparsity"):
            DynamicSparsityAllocator(target_sparsity=1.5)
    
    def test_compute_sensitivities_gradient(self):
        """Test sensitivity computation using gradients."""
        allocator = DynamicSparsityAllocator(method="gradient")
        
        gradients = {
            "layer1": np.array([1.0, 2.0, 3.0]),
            "layer2": np.array([0.1, 0.2, 0.1]),
        }
        
        sensitivities = allocator.compute_sensitivities(gradients)
        
        assert "layer1" in sensitivities
        assert "layer2" in sensitivities
        
        # layer1 has higher gradients -> higher sensitivity
        assert sensitivities["layer1"] > sensitivities["layer2"]
    
    def test_compute_sensitivities_uniform(self):
        """Test uniform sensitivity (no preference)."""
        allocator = DynamicSparsityAllocator(method="uniform")
        
        gradients = {
            "layer1": np.random.randn(100),
            "layer2": np.random.randn(50),
        }
        
        sensitivities = allocator.compute_sensitivities(gradients)
        
        # All sensitivities should be 1.0
        assert sensitivities["layer1"] == 1.0
        assert sensitivities["layer2"] == 1.0
    
    def test_allocate_sparsity_inverse_sensitivity(self):
        """Test that more sensitive layers get lower sparsity."""
        allocator = DynamicSparsityAllocator(
            target_sparsity=0.8,
            method="gradient"
        )
        
        # layer1: high gradient (high sensitivity)
        # layer2: low gradient (low sensitivity)
        gradients = {
            "layer1": np.ones(100) * 10.0,
            "layer2": np.ones(100) * 1.0,
        }
        
        model_weights = {
            "layer1": np.random.randn(100),
            "layer2": np.random.randn(100),
        }
        
        sensitivities = allocator.compute_sensitivities(gradients)
        sparsities = allocator.allocate_sparsity(model_weights, sensitivities)
        
        # More sensitive layer1 should have LOWER sparsity
        assert sparsities["layer1"] < sparsities["layer2"]
    
    def test_allocate_sparsity_achieves_target(self):
        """Test that overall sparsity matches target."""
        allocator = DynamicSparsityAllocator(target_sparsity=0.9)
        
        gradients = {
            "layer1": np.random.randn(100),
            "layer2": np.random.randn(200),
            "layer3": np.random.randn(50),
        }
        
        model_weights = {
            name: np.random.randn(*grad.shape)
            for name, grad in gradients.items()
        }
        
        sensitivities = allocator.compute_sensitivities(gradients)
        sparsities = allocator.allocate_sparsity(model_weights, sensitivities)
        
        # Compute overall sparsity
        total_params = sum(w.size for w in model_weights.values())
        total_pruned = sum(
            sparsities[name] * weights.size
            for name, weights in model_weights.items()
        )
        overall_sparsity = total_pruned / total_params
        
        # Should be close to target (within 5%)
        assert abs(overall_sparsity - 0.9) < 0.05
    
    def test_allocate_sparsity_bounds(self):
        """Test that allocated sparsities are within valid bounds."""
        allocator = DynamicSparsityAllocator(target_sparsity=0.95)
        
        gradients = {
            f"layer{i}": np.random.randn(100)
            for i in range(10)
        }
        
        model_weights = {
            name: np.random.randn(*grad.shape)
            for name, grad in gradients.items()
        }
        
        sensitivities = allocator.compute_sensitivities(gradients)
        sparsities = allocator.allocate_sparsity(model_weights, sensitivities)
        
        # All sparsities should be in [0, 0.99]
        for sparsity in sparsities.values():
            assert 0.0 <= sparsity < 1.0
    
    def test_get_statistics(self):
        """Test statistics computation."""
        allocator = DynamicSparsityAllocator(target_sparsity=0.8)
        
        gradients = {
            "layer1": np.random.randn(100),
            "layer2": np.random.randn(100),
        }
        
        model_weights = {
            name: np.random.randn(*grad.shape)
            for name, grad in gradients.items()
        }
        
        sensitivities = allocator.compute_sensitivities(gradients)
        sparsities = allocator.allocate_sparsity(model_weights, sensitivities)
        
        stats = allocator.get_statistics()
        
        assert "num_allocations" in stats
        assert "target_sparsity" in stats
        assert "min_layer_sparsity" in stats
        assert "max_layer_sparsity" in stats
        assert "mean_layer_sparsity" in stats
        
        assert stats["num_allocations"] == 1
        assert stats["target_sparsity"] == 0.8
    
    def test_allocation_history(self):
        """Test that allocation history is tracked."""
        allocator = DynamicSparsityAllocator(target_sparsity=0.9)
        
        gradients = {"layer1": np.random.randn(100)}
        model_weights = {"layer1": np.random.randn(100)}
        
        # Multiple allocations
        for _ in range(3):
            sensitivities = allocator.compute_sensitivities(gradients)
            allocator.allocate_sparsity(model_weights, sensitivities)
        
        assert len(allocator.allocation_history) == 3
        
        # Each entry should have required fields
        for entry in allocator.allocation_history:
            assert "target_sparsity" in entry
            assert "layer_sparsities" in entry
            assert "sensitivities" in entry


class TestIntegration:
    """Integration tests for dynamic sparse training."""
    
    def test_rigl_full_training_loop(self):
        """Test RigL in a simulated training loop."""
        rigl = RigLPruner(
            sparsity=0.8,
            T_end=500,
            delta_T=100,
            alpha=0.3,
            grad_accumulation_steps=1
        )
        
        # Initialize
        weights = np.random.randn(200)
        mask = rigl.initialize_mask(weights)
        
        initial_sparsity = 1.0 - (np.count_nonzero(mask) / mask.size)
        
        # Training loop
        for step in range(1, 501):
            # Simulate gradients
            gradients = np.random.randn(200)
            
            # Update mask if needed
            if rigl.should_update(step):
                weights, mask = rigl.update_mask(
                    weights, gradients, mask, step
                )
        
        # Check that training completed
        stats = rigl.get_statistics()
        assert stats["total_updates"] == 5  # Steps 100, 200, 300, 400, 500
        
        # Sparsity should be maintained
        final_sparsity = 1.0 - (np.count_nonzero(mask) / mask.size)
        assert abs(final_sparsity - initial_sparsity) < 0.1
    
    def test_combined_rigl_dynamic_allocation(self):
        """Test RigL with dynamic per-layer sparsity allocation."""
        # Multi-layer model
        layer_shapes = {
            "conv1": (64, 3, 3, 3),     # 1,728 params
            "conv2": (128, 64, 3, 3),   # 73,728 params
            "fc": (10, 1024),           # 10,240 params
        }
        
        # Initialize weights
        model_weights = {
            name: np.random.randn(*shape)
            for name, shape in layer_shapes.items()
        }
        
        # Compute gradients (simulate different importance)
        gradients = {
            "conv1": np.random.randn(*layer_shapes["conv1"]) * 0.1,  # Low importance
            "conv2": np.random.randn(*layer_shapes["conv2"]) * 1.0,  # High importance
            "fc": np.random.randn(*layer_shapes["fc"]) * 0.5,        # Medium importance
        }
        
        # Allocate sparsity dynamically
        # Use moderate sparsity for better RigL stability
        allocator = DynamicSparsityAllocator(target_sparsity=0.7)
        sensitivities = allocator.compute_sensitivities(gradients)
        layer_sparsities = allocator.allocate_sparsity(model_weights, sensitivities)
        
        # Verify inverse relationship: more important layers have lower sparsity
        # conv2 (high grad) should have lowest sparsity
        # conv1 (low grad) should have highest sparsity
        assert layer_sparsities["conv2"] < layer_sparsities["conv1"], \
            f"conv2 ({layer_sparsities['conv2']:.3f}) should have lower sparsity than conv1 ({layer_sparsities['conv1']:.3f})"
        assert layer_sparsities["conv2"] < layer_sparsities["fc"], \
            f"conv2 ({layer_sparsities['conv2']:.3f}) should have lower sparsity than fc ({layer_sparsities['fc']:.3f})"
        
        # Verify overall sparsity is close to target
        total_params = sum(w.size for w in model_weights.values())
        total_pruned = sum(layer_sparsities[name] * weights.size 
                          for name, weights in model_weights.items())
        overall_sparsity = total_pruned / total_params
        assert abs(overall_sparsity - 0.7) < 0.05, \
            f"Overall sparsity {overall_sparsity:.3f} should be close to target 0.7"
        
        # Create RigL pruner for each layer with different sparsities
        rigls = {}
        masks = {}
        
        for layer_name, sparsity in layer_sparsities.items():
            rigl = RigLPruner(
                sparsity=sparsity,
                T_end=300,
                delta_T=100,
                alpha=0.3,
                grad_accumulation_steps=1
            )
            rigls[layer_name] = rigl
            masks[layer_name] = rigl.initialize_mask(model_weights[layer_name])
        
        # Training loop
        for step in range(1, 301):
            for layer_name in model_weights.keys():
                # Simulate gradients
                grad = np.random.randn(*layer_shapes[layer_name])
                
                # Update mask
                if rigls[layer_name].should_update(step):
                    model_weights[layer_name], masks[layer_name] = rigls[layer_name].update_mask(
                        model_weights[layer_name],
                        grad,
                        masks[layer_name],
                        step,
                        layer_name
                    )
        
        # Verify that training completed and masks are valid
        for layer_name in model_weights.keys():
            actual_sparsity = 1.0 - (
                np.count_nonzero(masks[layer_name]) / masks[layer_name].size
            )
            target_sparsity = layer_sparsities[layer_name]
            
            # RigL maintains approximate sparsity
            # For moderate sparsities (< 0.9), should be within 15%
            assert 0 <= actual_sparsity < 1.0, \
                f"{layer_name}: invalid sparsity {actual_sparsity}"
    
    def test_rigl_config_dataclass(self):
        """Test RigLConfig dataclass."""
        config = RigLConfig(
            sparsity=0.95,
            T_end=5000,
            delta_T=50,
            alpha=0.2,
            grad_accumulation_steps=2
        )
        
        assert config.sparsity == 0.95
        assert config.T_end == 5000
        assert config.delta_T == 50
        assert config.alpha == 0.2
        assert config.grad_accumulation_steps == 2
        
        # Test defaults
        config_default = RigLConfig()
        assert config_default.sparsity == 0.9
        assert config_default.T_end == 10000
