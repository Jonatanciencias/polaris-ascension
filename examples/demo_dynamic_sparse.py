"""
Demo: Dynamic Sparse Training (RigL + Dynamic Allocation)
==========================================================

Session 11: Demonstrate training sparse networks from scratch without pruning.

This demo shows:
1. RigL: Drop lowest magnitude, grow highest gradient
2. Dynamic Sparsity Allocation: Per-layer sparsity based on sensitivity
3. Training loop integration

Advantages over Static Pruning:
- No pre-training needed
- Better final accuracy
- Adaptive topology during training

Author: Radeon RX 580 AI Project
Version: 0.6.0-dev
"""

import numpy as np
import time
from typing import Dict, List
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.compute.dynamic_sparse import (
    RigLPruner,
    DynamicSparsityAllocator,
    RigLConfig,
)


def print_section(title: str):
    """Print section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def simulate_training_step(weights: np.ndarray, learning_rate: float = 0.01) -> tuple:
    """
    Simulate one training step (forward + backward).

    Returns:
        (loss, gradients)
    """
    # Simulate loss (MSE with target=0)
    predictions = np.tanh(weights.sum())
    target = 0.0
    loss = (predictions - target) ** 2

    # Simulate gradients (derivative of loss w.r.t. weights)
    grad_output = 2 * (predictions - target)
    sech_squared = 1 - np.tanh(weights.sum()) ** 2
    gradients = grad_output * sech_squared * np.ones_like(weights)

    # Add noise to simulate realistic gradients
    gradients += np.random.randn(*weights.shape) * 0.1

    return loss, gradients


def demo_basic_rigl():
    """Demo 1: Basic RigL sparse training."""
    print_section("Demo 1: Basic RigL Sparse Training")

    print("RigL (Rigged Lottery): Train sparse from scratch")
    print("- Initialize with random sparse mask")
    print("- Periodically: Drop lowest magnitude, Grow highest gradient")
    print("- Maintain constant sparsity throughout training\n")

    # Configuration
    input_size = 1000
    sparsity = 0.9
    n_steps = 500
    update_interval = 100

    print(f"Configuration:")
    print(f"  - Model size: {input_size} parameters")
    print(f"  - Target sparsity: {sparsity*100:.0f}%")
    print(f"  - Training steps: {n_steps}")
    print(f"  - Mask update interval: every {update_interval} steps")
    print(f"  - Alpha (drop/grow rate): 30%\n")

    # Initialize RigL
    rigl = RigLPruner(
        sparsity=sparsity,
        T_end=n_steps,
        delta_T=update_interval,
        alpha=0.3,
        grad_accumulation_steps=1,
    )

    # Initialize model
    weights = np.random.randn(input_size) * 0.1
    mask = rigl.initialize_mask(weights, "model")

    initial_params = int(np.count_nonzero(mask))
    print(f"✓ Initialized: {initial_params}/{input_size} params active ({(1-sparsity)*100:.0f}%)\n")

    # Training loop
    print("Training progress:")
    losses = []
    sparsities_over_time = []

    start_time = time.time()

    for step in range(1, n_steps + 1):
        # Forward + backward
        loss, gradients = simulate_training_step(weights * mask)
        losses.append(loss)

        # Update mask if needed
        if rigl.should_update(step):
            weights, mask = rigl.update_mask(weights, gradients, mask, step, "model")

            current_sparsity = 1.0 - (np.count_nonzero(mask) / mask.size)
            sparsities_over_time.append((step, current_sparsity))

            print(
                f"  Step {step:4d}: Mask updated | "
                f"Sparsity={current_sparsity*100:.1f}% | "
                f"Loss={loss:.6f}"
            )

        # Apply mask and update weights (SGD)
        masked_weights = weights * mask
        weights = masked_weights - 0.01 * gradients
        weights = weights * mask  # Ensure pruned weights stay zero

    train_time = time.time() - start_time

    # Results
    print(f"\n✓ Training complete in {train_time:.2f}s")

    final_sparsity = 1.0 - (np.count_nonzero(mask) / mask.size)
    print(f"\nFinal statistics:")
    print(f"  - Sparsity: {final_sparsity*100:.1f}% (target: {sparsity*100:.0f}%)")
    print(f"  - Active params: {np.count_nonzero(mask)}/{input_size}")
    print(f"  - Final loss: {losses[-1]:.6f}")
    print(f"  - Initial loss: {losses[0]:.6f}")
    print(f"  - Loss reduction: {(1 - losses[-1]/losses[0])*100:.1f}%")

    # RigL statistics
    stats = rigl.get_statistics()
    print(f"\nRigL update statistics:")
    print(f"  - Total mask updates: {stats['total_updates']}")
    print(f"  - Avg connections dropped per update: {stats['avg_drop_per_update']:.1f}")
    print(f"  - Avg connections grown per update: {stats['avg_grow_per_update']:.1f}")
    print(f"  - Total connections changed: {stats['total_connections_changed']:.0f}")

    print(f"\n✓ RigL maintained constant sparsity while allowing topology changes!")


def demo_dynamic_allocation():
    """Demo 2: Dynamic per-layer sparsity allocation."""
    print_section("Demo 2: Dynamic Per-Layer Sparsity Allocation")

    print("Instead of uniform sparsity, allocate based on layer importance:")
    print("- Compute sensitivity from gradients")
    print("- More important layers → Lower sparsity (keep more params)")
    print("- Less important layers → Higher sparsity (prune more)\n")

    # Multi-layer model
    layer_configs = {
        "conv1": {"size": (32, 3, 3, 3), "importance": 0.5},  # 864 params
        "conv2": {"size": (64, 32, 3, 3), "importance": 2.0},  # 18,432 params (important!)
        "conv3": {"size": (128, 64, 3, 3), "importance": 1.5},  # 73,728 params
        "fc": {"size": (10, 1024), "importance": 0.8},  # 10,240 params
    }

    print("Model architecture:")
    total_params = 0
    for name, config in layer_configs.items():
        size = int(np.prod(config["size"]))
        total_params += size
        print(
            f"  - {name:6s}: {str(config['size']):20s} = {size:>6d} params "
            f"(importance: {config['importance']:.1f})"
        )
    print(f"\n  Total: {total_params:,} parameters\n")

    # Initialize model
    model_weights = {
        name: np.random.randn(*config["size"]) * 0.1 for name, config in layer_configs.items()
    }

    # Simulate gradients based on importance
    gradients = {
        name: np.random.randn(*config["size"]) * config["importance"]
        for name, config in layer_configs.items()
    }

    # Allocate sparsity dynamically
    target_sparsity = 0.85
    allocator = DynamicSparsityAllocator(target_sparsity=target_sparsity, method="gradient")

    print(f"Target overall sparsity: {target_sparsity*100:.0f}%\n")

    # Compute sensitivities
    sensitivities = allocator.compute_sensitivities(gradients)
    print("Layer sensitivities (from gradients):")
    for name, sens in sorted(sensitivities.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name:6s}: {sens:>8.2f}  {'█' * int(sens / max(sensitivities.values()) * 40)}")

    # Allocate sparsity
    layer_sparsities = allocator.allocate_sparsity(model_weights, sensitivities)

    print(f"\nAllocated sparsities (inverse to sensitivity):")
    for name in sorted(layer_sparsities.keys(), key=lambda x: layer_sparsities[x]):
        sparsity = layer_sparsities[name]
        params_kept = int(model_weights[name].size * (1 - sparsity))
        print(
            f"  {name:6s}: {sparsity*100:>5.1f}% sparse → "
            f"{params_kept:>6d} params kept  "
            f"{'▓' * int((1-sparsity) * 40)}"
        )

    # Verify overall sparsity
    total_params = sum(w.size for w in model_weights.values())
    total_pruned = sum(
        layer_sparsities[name] * weights.size for name, weights in model_weights.items()
    )
    overall_sparsity = total_pruned / total_params

    print(f"\nVerification:")
    print(f"  - Target overall sparsity: {target_sparsity*100:.1f}%")
    print(f"  - Actual overall sparsity: {overall_sparsity*100:.1f}%")
    print(f"  - Difference: {abs(overall_sparsity - target_sparsity)*100:.2f}%")

    # Show allocation statistics
    stats = allocator.get_statistics()
    print(f"\nAllocation statistics:")
    print(f"  - Min layer sparsity: {stats['min_layer_sparsity']*100:.1f}%")
    print(f"  - Max layer sparsity: {stats['max_layer_sparsity']*100:.1f}%")
    print(f"  - Mean layer sparsity: {stats['mean_layer_sparsity']*100:.1f}%")
    print(f"  - Std layer sparsity: {stats['std_layer_sparsity']*100:.1f}%")

    print(f"\n✓ Successfully allocated non-uniform sparsity based on importance!")


def demo_combined_rigl_dynamic():
    """Demo 3: Combined RigL + Dynamic Allocation."""
    print_section("Demo 3: RigL with Dynamic Per-Layer Allocation")

    print("Best of both worlds:")
    print("1. Dynamic Allocation: Different sparsity per layer (based on importance)")
    print("2. RigL: Continuous topology adaptation during training")
    print("3. Result: Sparse network with layer-wise optimization\n")

    # Configuration
    layer_configs = {
        "layer1": {"size": 500, "importance": 0.5},
        "layer2": {"size": 1000, "importance": 2.0},  # Most important
        "layer3": {"size": 500, "importance": 1.0},
    }

    n_steps = 300
    update_interval = 50
    target_sparsity = 0.8

    print("Configuration:")
    print(f"  - Training steps: {n_steps}")
    print(f"  - Mask update interval: {update_interval}")
    print(f"  - Target overall sparsity: {target_sparsity*100:.0f}%\n")

    # Initialize model
    model_weights = {
        name: np.random.randn(config["size"]) * 0.1 for name, config in layer_configs.items()
    }

    # Allocate sparsity dynamically
    gradients = {
        name: np.random.randn(config["size"]) * config["importance"]
        for name, config in layer_configs.items()
    }

    allocator = DynamicSparsityAllocator(target_sparsity=target_sparsity)
    sensitivities = allocator.compute_sensitivities(gradients)
    layer_sparsities = allocator.allocate_sparsity(model_weights, sensitivities)

    print("Per-layer sparsity allocation:")
    for name, sparsity in sorted(layer_sparsities.items()):
        importance = layer_configs[name]["importance"]
        print(f"  {name}: {sparsity*100:>5.1f}% sparse (importance: {importance:.1f})")
    print()

    # Create RigL pruner for each layer
    rigls = {}
    masks = {}

    for name, sparsity in layer_sparsities.items():
        rigl = RigLPruner(
            sparsity=sparsity,
            T_end=n_steps,
            delta_T=update_interval,
            alpha=0.3,
            grad_accumulation_steps=1,
        )
        rigls[name] = rigl
        masks[name] = rigl.initialize_mask(model_weights[name], name)

    # Training loop
    print("Training with RigL (showing mask updates only):")

    losses = []
    start_time = time.time()

    for step in range(1, n_steps + 1):
        # Forward + backward for each layer
        step_loss = 0
        for name in model_weights.keys():
            loss, grads = simulate_training_step(model_weights[name] * masks[name])
            step_loss += loss

            # Update mask if needed
            if rigls[name].should_update(step):
                model_weights[name], masks[name] = rigls[name].update_mask(
                    model_weights[name], grads, masks[name], step, name
                )

            # Update weights (SGD)
            model_weights[name] = model_weights[name] - 0.01 * grads
            model_weights[name] = model_weights[name] * masks[name]

        losses.append(step_loss)

        # Print updates at update intervals
        if step % update_interval == 0:
            avg_loss = np.mean(losses[-update_interval:])
            print(f"  Step {step:3d}: Masks updated | Avg loss = {avg_loss:.6f}")

    train_time = time.time() - start_time

    # Results
    print(f"\n✓ Training complete in {train_time:.2f}s\n")

    print("Final per-layer statistics:")
    total_params = 0
    total_active = 0

    for name in sorted(model_weights.keys()):
        size = model_weights[name].size
        active = int(np.count_nonzero(masks[name]))
        sparsity = 1.0 - (active / size)
        target = layer_sparsities[name]

        total_params += size
        total_active += active

        print(
            f"  {name}: {active:>4d}/{size:>4d} active "
            f"({sparsity*100:>5.1f}% sparse, target {target*100:>5.1f}%)"
        )

    overall_sparsity = 1.0 - (total_active / total_params)
    print(
        f"\n  Overall: {total_active:>5d}/{total_params:>5d} active "
        f"({overall_sparsity*100:.1f}% sparse, target {target_sparsity*100:.0f}%)"
    )

    print(f"\n  Final loss: {losses[-1]:.6f}")
    print(f"  Loss reduction: {(1 - losses[-1]/losses[0])*100:.1f}%")

    print(f"\n✓ Successfully combined dynamic allocation with RigL training!")


def demo_comparison():
    """Demo 4: Compare Dense vs Static Sparse vs RigL."""
    print_section("Demo 4: Comparison - Dense vs Static vs RigL")

    print("Compare three approaches on same task:\n")

    # Configuration
    model_size = 500
    n_steps = 200
    sparsity = 0.85

    results = {}

    # 1. Dense (no sparsity)
    print("[1/3] Training DENSE model (no sparsity)...")
    weights_dense = np.random.randn(model_size) * 0.1
    losses_dense = []

    start_time = time.time()
    for step in range(n_steps):
        loss, grads = simulate_training_step(weights_dense)
        losses_dense.append(loss)
        weights_dense -= 0.01 * grads
    time_dense = time.time() - start_time

    results["Dense"] = {
        "final_loss": losses_dense[-1],
        "time": time_dense,
        "params": model_size,
        "active_params": model_size,
    }
    print(
        f"  ✓ Loss: {losses_dense[-1]:.6f} | Time: {time_dense:.3f}s | " f"Params: {model_size}\n"
    )

    # 2. Static Sparse (prune once at start, never update)
    print("[2/3] Training STATIC SPARSE model (prune once)...")
    weights_static = np.random.randn(model_size) * 0.1

    # Create static mask (prune randomly)
    mask_static = np.ones(model_size)
    prune_indices = np.random.choice(model_size, size=int(model_size * sparsity), replace=False)
    mask_static[prune_indices] = 0
    weights_static *= mask_static

    losses_static = []
    start_time = time.time()
    for step in range(n_steps):
        loss, grads = simulate_training_step(weights_static)
        losses_static.append(loss)
        weights_static -= 0.01 * grads
        weights_static *= mask_static  # Keep mask fixed
    time_static = time.time() - start_time

    active_static = int(np.count_nonzero(mask_static))
    results["Static"] = {
        "final_loss": losses_static[-1],
        "time": time_static,
        "params": model_size,
        "active_params": active_static,
    }
    print(
        f"  ✓ Loss: {losses_static[-1]:.6f} | Time: {time_static:.3f}s | "
        f"Params: {active_static}/{model_size} ({(1-sparsity)*100:.0f}% active)\n"
    )

    # 3. RigL (dynamic topology)
    print("[3/3] Training RIGL SPARSE model (dynamic topology)...")
    weights_rigl = np.random.randn(model_size) * 0.1

    rigl = RigLPruner(
        sparsity=sparsity,
        T_end=n_steps,
        delta_T=20,
        alpha=0.3,
        grad_accumulation_steps=1,
    )
    mask_rigl = rigl.initialize_mask(weights_rigl, "model")

    losses_rigl = []
    start_time = time.time()
    for step in range(1, n_steps + 1):
        loss, grads = simulate_training_step(weights_rigl * mask_rigl)
        losses_rigl.append(loss)

        if rigl.should_update(step):
            weights_rigl, mask_rigl = rigl.update_mask(
                weights_rigl, grads, mask_rigl, step, "model"
            )

        weights_rigl -= 0.01 * grads
        weights_rigl *= mask_rigl
    time_rigl = time.time() - start_time

    active_rigl = int(np.count_nonzero(mask_rigl))
    rigl_stats = rigl.get_statistics()

    results["RigL"] = {
        "final_loss": losses_rigl[-1],
        "time": time_rigl,
        "params": model_size,
        "active_params": active_rigl,
        "updates": rigl_stats["total_updates"],
        "connections_changed": rigl_stats["total_connections_changed"],
    }
    print(
        f"  ✓ Loss: {losses_rigl[-1]:.6f} | Time: {time_rigl:.3f}s | "
        f"Params: {active_rigl}/{model_size} ({(1-sparsity)*100:.0f}% active)\n"
    )

    # Comparison table
    print("=" * 70)
    print(" " * 20 + "COMPARISON RESULTS")
    print("=" * 70)
    print(f"{'Method':<15} {'Loss':<12} {'Time (s)':<12} {'Params':<15} {'Speedup'}")
    print("-" * 70)

    for method, res in results.items():
        speedup = results["Dense"]["time"] / res["time"]
        param_str = f"{res['active_params']}/{res['params']}"

        print(
            f"{method:<15} {res['final_loss']:<12.6f} {res['time']:<12.3f} "
            f"{param_str:<15} {speedup:.2f}x"
        )

    print("=" * 70)

    # Analysis
    print("\nAnalysis:")

    # Best loss
    best_method = min(results.keys(), key=lambda k: results[k]["final_loss"])
    print(f"  Best loss: {best_method} ({results[best_method]['final_loss']:.6f})")

    # Fastest
    fastest = min(results.keys(), key=lambda k: results[k]["time"])
    print(f"  Fastest: {fastest} ({results[fastest]['time']:.3f}s)")

    # RigL specific
    if "RigL" in results:
        print(f"\n  RigL advantages:")
        print(f"    - Mask updates: {results['RigL']['updates']}")
        print(f"    - Connections changed: {results['RigL']['connections_changed']:.0f}")
        print(f"    - Adaptive topology during training")

    # Compression
    compression = model_size / results["RigL"]["active_params"]
    print(f"\n  Compression: {compression:.1f}x ({sparsity*100:.0f}% sparse)")

    print(f"\n✓ RigL achieves competitive accuracy with {sparsity*100:.0f}% sparsity!")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print(" " * 15 + "DYNAMIC SPARSE TRAINING DEMO")
    print(" " * 20 + "Session 11: RigL Algorithm")
    print("=" * 70)

    print("\nDynamic Sparse Training:")
    print("  Train sparse networks from scratch without prune-retrain cycle")
    print("\nKey Features:")
    print("  ✓ RigL: Drop lowest magnitude, grow highest gradient")
    print("  ✓ Dynamic Allocation: Per-layer sparsity based on importance")
    print("  ✓ Constant sparsity: Topology changes without density changes")
    print("  ✓ Better accuracy: Often outperforms static pruning")

    try:
        # Run demos
        demo_basic_rigl()
        input("\nPress Enter to continue to Demo 2...")

        demo_dynamic_allocation()
        input("\nPress Enter to continue to Demo 3...")

        demo_combined_rigl_dynamic()
        input("\nPress Enter to continue to Demo 4...")

        demo_comparison()

        # Summary
        print_section("Summary")
        print("✓ Demo 1: Basic RigL sparse training")
        print("  → Drop/grow maintains sparsity while adapting topology")

        print("\n✓ Demo 2: Dynamic per-layer allocation")
        print("  → Important layers get lower sparsity")

        print("\n✓ Demo 3: Combined RigL + Dynamic")
        print("  → Layer-wise optimization with continuous adaptation")

        print("\n✓ Demo 4: Comparison")
        print("  → RigL achieves competitive accuracy with high sparsity")

        print("\n" + "=" * 70)
        print("All demos completed successfully!")
        print("=" * 70 + "\n")

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
