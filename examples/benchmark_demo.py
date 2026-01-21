"""
Benchmarking Suite Demo - Session 25
====================================

Demonstrates comprehensive benchmarking of tensor decomposition methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, '/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580')

from src.compute.tensor_decomposition_benchmark import (
    TensorDecompositionBenchmark,
    BenchmarkConfig,
    quick_benchmark
)


class SimpleConvNet(nn.Module):
    """Simple CNN for benchmarking."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def create_data(num_samples=500, batch_size=32):
    """Create synthetic data."""
    X = torch.randn(num_samples, 3, 32, 32)
    y = torch.randint(0, 10, (num_samples,))
    
    split = int(0.8 * num_samples)
    train_dataset = TensorDataset(X[:split], y[:split])
    val_dataset = TensorDataset(X[split:], y[split:])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader


def train_baseline(model, train_loader, epochs=2):
    """Quick training."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()


def main():
    """Run benchmarking demo."""
    print("\n" + "="*70)
    print("Tensor Decomposition Benchmarking Demo")
    print("="*70)
    
    # Create model and data
    print("\n📦 Creating model and data...")
    model = SimpleConvNet()
    train_loader, val_loader = create_data(num_samples=300)
    
    # Train quickly
    print("🎯 Quick training (2 epochs)...")
    train_baseline(model, train_loader, epochs=2)
    
    # Quick benchmark
    print("\n🚀 Running quick benchmark...")
    config = BenchmarkConfig(
        methods=["tucker", "cp"],
        tucker_ranks=[[16, 32], [8, 16]],
        cp_ranks=[32, 16],
        finetune=True,
        finetune_epochs=2,
        num_inference_runs=50,
        save_results=False
    )
    
    benchmark = TensorDecompositionBenchmark(config)
    results = benchmark.benchmark_model(
        model, train_loader, val_loader, "SimpleConvNet"
    )
    
    # Print results
    benchmark.print_summary(results)
    benchmark.print_pareto_frontier(results)
    
    # Generate report
    print(benchmark.generate_report(results, "SimpleConvNet"))
    
    print("\n✅ Benchmarking complete!")
    print("\n💡 Use this suite to evaluate compression on real models!")


if __name__ == "__main__":
    main()
