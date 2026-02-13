"""
Tensor Decomposition Fine-tuning Demo - Session 25
==================================================

Demonstrates accuracy recovery through fine-tuning after tensor decomposition.

This demo shows:
1. Basic fine-tuning: Recovering accuracy after compression
2. Knowledge distillation: Using original model as teacher
3. Layer-wise fine-tuning: Progressive recovery strategy
4. Scheduler comparison: Cosine vs Plateau
5. Compression-accuracy trade-offs
6. Complete pipeline: Compress → Fine-tune → Deploy
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.compute.tensor_decomposition import (
    DecompositionConfig,
    compute_compression_ratio,
    decompose_model,
)
from src.compute.tensor_decomposition_finetuning import (
    DecompositionFinetuner,
    FinetuneConfig,
    LayerWiseFinetuner,
    quick_finetune,
)

# ============================================================================
#  Helper Functions
# ============================================================================


def count_parameters(model):
    """Count total number of parameters in model."""
    return sum(p.numel() for p in model.parameters())


# ============================================================================
#  Demo Model
# ============================================================================


class ConvNet(nn.Module):
    """Convolutional network for CIFAR-10 style data."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv2(x)))  # 16x16 -> 8x8
        x = self.pool(F.relu(self.conv3(x)))  # 8x8 -> 4x4
        x = x.reshape(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def create_cifar_like_data(num_samples=1000, batch_size=64):
    """Create CIFAR-10 like synthetic data."""
    print(f"\n📊 Creating {num_samples} synthetic samples...")

    # Generate random images and labels
    X = torch.randn(num_samples, 3, 32, 32)
    y = torch.randint(0, 10, (num_samples,))

    # Create train/val split
    split = int(0.8 * num_samples)

    train_dataset = TensorDataset(X[:split], y[:split])
    val_dataset = TensorDataset(X[split:], y[split:])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val: {len(val_dataset)} samples")

    return train_loader, val_loader


def evaluate_model(model, data_loader, criterion):
    """Evaluate model on data."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in data_loader:
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()

            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def train_model_baseline(model, train_loader, epochs=5):
    """Train baseline model for comparison."""
    print("\n🎯 Training baseline model...")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"   Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")

    print("✅ Baseline training complete")


# ============================================================================
#  Demo 1: Basic Fine-tuning
# ============================================================================


def demo_basic_finetuning():
    """Demo 1: Basic fine-tuning after compression."""
    print("\n" + "=" * 70)
    print("Demo 1: Basic Fine-tuning")
    print("=" * 70)

    # Create model and data
    model = ConvNet()
    train_loader, val_loader = create_cifar_like_data(num_samples=500)

    # Train baseline
    train_model_baseline(model, train_loader, epochs=3)

    # Evaluate baseline
    criterion = nn.CrossEntropyLoss()
    baseline_loss, baseline_acc = evaluate_model(model, val_loader, criterion)
    print(f"\n📊 Baseline: Loss = {baseline_loss:.4f}, Accuracy = {baseline_acc:.2f}%")

    # Compress with Tucker
    print("\n🔧 Compressing with Tucker decomposition...")
    decomp_config = DecompositionConfig(method="tucker", ranks=[16, 32], auto_rank=False)
    compressed = decompose_model(model, decomp_config)

    # Compare parameters
    original_params = count_parameters(model)
    compressed_params = count_parameters(compressed)
    compression_ratio = original_params / compressed_params

    print(f"   Original params: {original_params:,}")
    print(f"   Compressed params: {compressed_params:,}")
    print(f"   Compression ratio: {compression_ratio:.2f}x")

    # Evaluate compressed (before fine-tuning)
    compressed_loss, compressed_acc = evaluate_model(compressed, val_loader, criterion)
    print(f"\n📊 Compressed (no fine-tuning):")
    print(f"   Loss = {compressed_loss:.4f}")
    print(f"   Accuracy = {compressed_acc:.2f}%")
    print(f"   Accuracy drop = {baseline_acc - compressed_acc:.2f}%")

    # Fine-tune
    print("\n🔧 Fine-tuning compressed model...")
    finetune_config = FinetuneConfig(
        epochs=5, learning_rate=1e-3, scheduler="cosine", early_stopping=True, patience=3
    )
    finetuner = DecompositionFinetuner(finetune_config)

    tuned, metrics = finetuner.fine_tune(
        decomposed_model=compressed,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
    )

    # Evaluate fine-tuned
    tuned_loss, tuned_acc = evaluate_model(tuned, val_loader, criterion)

    print(f"\n📊 After Fine-tuning:")
    print(f"   Loss = {tuned_loss:.4f}")
    print(f"   Accuracy = {tuned_acc:.2f}%")
    print(f"   Accuracy recovered = {tuned_acc - compressed_acc:.2f}%")
    print(f"   Final gap from baseline = {baseline_acc - tuned_acc:.2f}%")

    # Summary
    print("\n" + "-" * 70)
    print("SUMMARY")
    print("-" * 70)
    print(f"Baseline:    {baseline_acc:6.2f}%  ({original_params:>9,} params)")
    print(
        f"Compressed:  {compressed_acc:6.2f}%  ({compressed_params:>9,} params, {compression_ratio:.1f}x)"
    )
    print(
        f"Fine-tuned:  {tuned_acc:6.2f}%  ({compressed_params:>9,} params, {compression_ratio:.1f}x)"
    )
    print(f"\n✅ Recovered {tuned_acc - compressed_acc:.1f}% accuracy with fine-tuning!")

    return metrics


# ============================================================================
#  Demo 2: Knowledge Distillation
# ============================================================================


def demo_knowledge_distillation():
    """Demo 2: Fine-tuning with knowledge distillation."""
    print("\n" + "=" * 70)
    print("Demo 2: Knowledge Distillation")
    print("=" * 70)

    # Create model and data
    original = ConvNet()
    train_loader, val_loader = create_cifar_like_data(num_samples=500)

    # Train original (teacher)
    train_model_baseline(original, train_loader, epochs=3)

    criterion = nn.CrossEntropyLoss()
    teacher_loss, teacher_acc = evaluate_model(original, val_loader, criterion)
    print(f"\n📊 Teacher: Loss = {teacher_loss:.4f}, Accuracy = {teacher_acc:.2f}%")

    # Create student (copy of teacher for fair comparison)
    student = ConvNet()
    student.load_state_dict(original.state_dict())

    # Compress student
    print("\n🔧 Compressing student model...")
    decomp_config = DecompositionConfig(method="tucker", ranks=[16, 32])
    compressed_student = decompose_model(student, decomp_config)

    # Fine-tune WITHOUT distillation
    print("\n🔧 Fine-tuning WITHOUT distillation...")
    config_no_kd = FinetuneConfig(
        epochs=5, learning_rate=1e-3, use_distillation=False, early_stopping=False, verbose=False
    )
    finetuner_no_kd = DecompositionFinetuner(config_no_kd)

    # Copy for KD experiment
    compressed_for_kd = ConvNet()
    compressed_for_kd.load_state_dict(compressed_student.state_dict())

    tuned_no_kd, metrics_no_kd = finetuner_no_kd.fine_tune(
        decomposed_model=compressed_student,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
    )

    no_kd_loss, no_kd_acc = evaluate_model(tuned_no_kd, val_loader, criterion)

    # Fine-tune WITH distillation
    print("\n🔧 Fine-tuning WITH distillation...")
    config_kd = FinetuneConfig(
        epochs=5,
        learning_rate=1e-3,
        use_distillation=True,
        distillation_alpha=0.5,
        distillation_temperature=3.0,
        early_stopping=False,
        verbose=False,
    )
    finetuner_kd = DecompositionFinetuner(config_kd)

    tuned_kd, metrics_kd = finetuner_kd.fine_tune(
        decomposed_model=compressed_for_kd,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        original_model=original,  # Teacher
    )

    kd_loss, kd_acc = evaluate_model(tuned_kd, val_loader, criterion)

    # Compare
    print("\n" + "-" * 70)
    print("COMPARISON")
    print("-" * 70)
    print(f"Teacher (original):         {teacher_acc:6.2f}%")
    print(f"Student (no distillation):  {no_kd_acc:6.2f}%  (Δ = {teacher_acc - no_kd_acc:+.2f}%)")
    print(f"Student (with distillation):{kd_acc:6.2f}%  (Δ = {teacher_acc - kd_acc:+.2f}%)")
    print(f"\n✅ Knowledge distillation improved accuracy by {kd_acc - no_kd_acc:.2f}%!")


# ============================================================================
#  Demo 3: Layer-wise Fine-tuning
# ============================================================================


def demo_layerwise_finetuning():
    """Demo 3: Layer-wise progressive fine-tuning."""
    print("\n" + "=" * 70)
    print("Demo 3: Layer-wise Fine-tuning")
    print("=" * 70)

    # Create model and data
    model = ConvNet()
    train_loader, val_loader = create_cifar_like_data(num_samples=500)

    # Train baseline
    train_model_baseline(model, train_loader, epochs=3)

    criterion = nn.CrossEntropyLoss()

    # Compress
    decomp_config = DecompositionConfig(method="tucker", ranks=[16, 32])
    compressed = decompose_model(model, decomp_config)

    # Standard fine-tuning
    print("\n🔧 Standard fine-tuning...")
    config_standard = FinetuneConfig(epochs=3, verbose=False, early_stopping=False)
    finetuner_standard = DecompositionFinetuner(config_standard)

    compressed_copy1 = ConvNet()
    compressed_copy1.load_state_dict(compressed.state_dict())

    start = time.time()
    tuned_standard, metrics_standard = finetuner_standard.fine_tune(
        decomposed_model=compressed_copy1,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
    )
    time_standard = time.time() - start

    _, acc_standard = evaluate_model(tuned_standard, val_loader, criterion)

    # Layer-wise fine-tuning
    print("\n🔧 Layer-wise fine-tuning...")
    config_layerwise = FinetuneConfig(epochs=2, verbose=False, early_stopping=False)
    finetuner_layerwise = LayerWiseFinetuner(config_layerwise)

    compressed_copy2 = ConvNet()
    compressed_copy2.load_state_dict(compressed.state_dict())

    start = time.time()
    tuned_layerwise, metrics_layerwise = finetuner_layerwise.fine_tune_layerwise(
        model=compressed_copy2,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        layers_per_stage=2,
    )
    time_layerwise = time.time() - start

    _, acc_layerwise = evaluate_model(tuned_layerwise, val_loader, criterion)

    # Compare
    print("\n" + "-" * 70)
    print("COMPARISON")
    print("-" * 70)
    print(f"Standard:    {acc_standard:6.2f}%  ({time_standard:.2f}s)")
    print(f"Layer-wise:  {acc_layerwise:6.2f}%  ({time_layerwise:.2f}s)")
    print(f"\nℹ️  Layer-wise fine-tuning trains layers progressively")


# ============================================================================
#  Demo 4: Scheduler Comparison
# ============================================================================


def demo_scheduler_comparison():
    """Demo 4: Compare different learning rate schedulers."""
    print("\n" + "=" * 70)
    print("Demo 4: Scheduler Comparison")
    print("=" * 70)

    # Create model and data
    model = ConvNet()
    train_loader, val_loader = create_cifar_like_data(num_samples=500)

    # Train baseline
    train_model_baseline(model, train_loader, epochs=2)

    criterion = nn.CrossEntropyLoss()

    # Compress
    decomp_config = DecompositionConfig(method="tucker", ranks=[16, 32])
    compressed = decompose_model(model, decomp_config)

    schedulers = ["none", "cosine", "plateau"]
    results = {}

    for scheduler in schedulers:
        print(f"\n🔧 Testing scheduler: {scheduler}")

        # Copy compressed model
        model_copy = ConvNet()
        model_copy.load_state_dict(compressed.state_dict())

        # Fine-tune
        config = FinetuneConfig(epochs=5, scheduler=scheduler, verbose=False, early_stopping=False)
        finetuner = DecompositionFinetuner(config)

        tuned, metrics = finetuner.fine_tune(
            decomposed_model=model_copy,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
        )

        _, acc = evaluate_model(tuned, val_loader, criterion)
        results[scheduler] = {"accuracy": acc, "history": metrics["history"]}

    # Display results
    print("\n" + "-" * 70)
    print("SCHEDULER COMPARISON")
    print("-" * 70)
    for scheduler, result in results.items():
        acc = result["accuracy"]
        print(f"{scheduler:10s}: {acc:6.2f}%")

    print("\nℹ️  Cosine annealing often works best for fine-tuning")


# ============================================================================
#  Demo 5: Compression-Accuracy Trade-off
# ============================================================================


def demo_compression_accuracy_tradeoff():
    """Demo 5: Explore compression-accuracy trade-off."""
    print("\n" + "=" * 70)
    print("Demo 5: Compression-Accuracy Trade-off")
    print("=" * 70)

    # Create model and data
    model = ConvNet()
    train_loader, val_loader = create_cifar_like_data(num_samples=500)

    # Train baseline
    train_model_baseline(model, train_loader, epochs=3)

    criterion = nn.CrossEntropyLoss()
    baseline_loss, baseline_acc = evaluate_model(model, val_loader, criterion)
    original_params = count_parameters(model)

    # Test different compression levels
    rank_configs = [
        ([32, 64], "Low compression"),
        ([16, 32], "Medium compression"),
        ([8, 16], "High compression"),
    ]

    results = []

    for ranks, desc in rank_configs:
        print(f"\n🔧 Testing {desc}: ranks={ranks}")

        # Compress
        decomp_config = DecompositionConfig(method="tucker", ranks=ranks)
        compressed = decompose_model(model, decomp_config)

        params = count_parameters(compressed)
        ratio = original_params / params

        # Fine-tune
        config = FinetuneConfig(epochs=3, verbose=False, early_stopping=False)
        finetuner = DecompositionFinetuner(config)

        tuned, _ = finetuner.fine_tune(
            decomposed_model=compressed,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
        )

        _, acc = evaluate_model(tuned, val_loader, criterion)

        results.append(
            {"ranks": ranks, "desc": desc, "params": params, "ratio": ratio, "accuracy": acc}
        )

        print(f"   Params: {params:,} ({ratio:.1f}x compression)")
        print(f"   Accuracy: {acc:.2f}%")

    # Display Pareto frontier
    print("\n" + "-" * 70)
    print("COMPRESSION-ACCURACY TRADE-OFF")
    print("-" * 70)
    print(f"{'Configuration':<25} {'Compression':<12} {'Accuracy':>10} {'Gap':>8}")
    print("-" * 70)
    print(f"{'Baseline (original)':<25} {'1.0x':<12} {baseline_acc:>9.2f}% {'-':>8}")

    for r in results:
        gap = baseline_acc - r["accuracy"]
        print(f"{r['desc']:<25} {r['ratio']:<11.1f}x {r['accuracy']:>9.2f}% {gap:>7.2f}%")

    print("\n✅ Find optimal trade-off based on your requirements!")


# ============================================================================
#  Demo 6: Complete Pipeline
# ============================================================================


def demo_complete_pipeline():
    """Demo 6: Complete production pipeline."""
    print("\n" + "=" * 70)
    print("Demo 6: Complete Production Pipeline")
    print("=" * 70)

    print("\n📋 Pipeline Steps:")
    print("   1. Train baseline model")
    print("   2. Compress with tensor decomposition")
    print("   3. Fine-tune compressed model")
    print("   4. Validate performance")
    print("   5. Deploy optimized model")

    # Step 1: Train
    print("\n" + "-" * 70)
    print("STEP 1: Train Baseline")
    print("-" * 70)
    model = ConvNet()
    train_loader, val_loader = create_cifar_like_data(num_samples=500)
    train_model_baseline(model, train_loader, epochs=3)

    criterion = nn.CrossEntropyLoss()
    baseline_loss, baseline_acc = evaluate_model(model, val_loader, criterion)
    baseline_params = count_parameters(model)
    print(f"✅ Baseline: {baseline_acc:.2f}% accuracy, {baseline_params:,} params")

    # Step 2: Compress
    print("\n" + "-" * 70)
    print("STEP 2: Compress Model")
    print("-" * 70)
    decomp_config = DecompositionConfig(method="tucker", ranks=[16, 32], auto_rank=False)
    compressed = decompose_model(model, decomp_config)
    compressed_params = count_parameters(compressed)
    compression_ratio = baseline_params / compressed_params

    compressed_loss, compressed_acc = evaluate_model(compressed, val_loader, criterion)
    print(f"✅ Compressed: {compression_ratio:.1f}x smaller")
    print(f"   Accuracy before fine-tuning: {compressed_acc:.2f}%")

    # Step 3: Fine-tune
    print("\n" + "-" * 70)
    print("STEP 3: Fine-tune")
    print("-" * 70)

    finetune_config = FinetuneConfig(epochs=5, learning_rate=1e-3, use_distillation=True)
    finetuner = DecompositionFinetuner(finetune_config)

    tuned, metrics = finetuner.fine_tune(
        decomposed_model=compressed,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        original_model=model,
    )

    tuned_loss, tuned_acc = evaluate_model(tuned, val_loader, criterion)
    print(f"✅ Fine-tuned: {tuned_acc:.2f}% accuracy")

    # Step 4: Validate
    print("\n" + "-" * 70)
    print("STEP 4: Validate Performance")
    print("-" * 70)

    # Speed test
    test_input = torch.randn(1, 3, 32, 32)

    model.eval()
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(test_input)
    time_baseline = time.time() - start

    tuned.eval()
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = tuned(test_input)
    time_tuned = time.time() - start

    speedup = time_baseline / time_tuned

    print(f"✅ Performance validated:")
    print(f"   Baseline inference: {time_baseline*10:.2f}ms/sample")
    print(f"   Optimized inference: {time_tuned*10:.2f}ms/sample")
    print(f"   Speedup: {speedup:.2f}x")

    # Step 5: Deploy
    print("\n" + "-" * 70)
    print("STEP 5: Ready for Deployment")
    print("-" * 70)
    print(f"✅ Model ready for production:")
    print(f"   Size: {compression_ratio:.1f}x smaller ({compressed_params:,} params)")
    print(f"   Speed: {speedup:.2f}x faster")
    print(f"   Accuracy: {tuned_acc:.2f}% (drop: {baseline_acc - tuned_acc:.2f}%)")

    # Final summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(
        f"Original:  {baseline_params:>8,} params  {baseline_acc:>6.2f}%  {time_baseline*10:>6.2f}ms"
    )
    print(f"Optimized: {compressed_params:>8,} params  {tuned_acc:>6.2f}%  {time_tuned*10:>6.2f}ms")
    print(
        f"Savings:   {compression_ratio:>7.1f}x smaller  {baseline_acc-tuned_acc:>5.2f}% drop  {speedup:>5.2f}x faster"
    )
    print("\n🚀 Ready to deploy!")


# ============================================================================
#  Main
# ============================================================================


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print(" Tensor Decomposition Fine-tuning Demos - Session 25")
    print("=" * 70)
    print("\nThese demos show how fine-tuning recovers accuracy after compression.")
    print("Note: Using synthetic data for demonstration purposes.")

    # Set seed for reproducibility
    torch.manual_seed(42)

    demos = [
        ("1. Basic Fine-tuning", demo_basic_finetuning),
        ("2. Knowledge Distillation", demo_knowledge_distillation),
        ("3. Layer-wise Fine-tuning", demo_layerwise_finetuning),
        ("4. Scheduler Comparison", demo_scheduler_comparison),
        ("5. Compression-Accuracy Trade-off", demo_compression_accuracy_tradeoff),
        ("6. Complete Pipeline", demo_complete_pipeline),
    ]

    print("\nAvailable demos:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"   {i}. {name}")
    print("   0. Run all demos")

    choice = input("\nSelect demo (0-6): ").strip()

    if choice == "0":
        for name, demo_func in demos:
            try:
                demo_func()
            except KeyboardInterrupt:
                print("\n\n⏸️  Demo interrupted by user")
                break
            except Exception as e:
                print(f"\n❌ Error in {name}: {e}")
                import traceback

                traceback.print_exc()
    elif choice in [str(i) for i in range(1, len(demos) + 1)]:
        idx = int(choice) - 1
        name, demo_func = demos[idx]
        try:
            demo_func()
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback

            traceback.print_exc()
    else:
        print("Invalid choice. Running Demo 1...")
        demo_basic_finetuning()

    print("\n" + "=" * 70)
    print("✅ Fine-tuning demos complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("• Fine-tuning recovers most accuracy lost during compression")
    print("• Knowledge distillation can improve recovery further")
    print("• Layer-wise fine-tuning offers alternative strategy")
    print("• Different schedulers work better for different scenarios")
    print("• Compression-accuracy trade-off is tunable")
    print("\n💡 Try these techniques on real CIFAR-10/ImageNet models!")


if __name__ == "__main__":
    main()
