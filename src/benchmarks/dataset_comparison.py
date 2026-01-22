"""
Dataset Comparison Benchmarks

Comprehensive comparison between synthetic and real datasets:
- Training performance comparison
- Accuracy metrics
- Convergence analysis
- Resource utilization
- Statistical analysis

Designed to validate synthetic data effectiveness vs real CIFAR data.

Author: AMD GPU Computing Team
Date: January 21, 2026
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path
import logging
import json
import time
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


# ============================================================================
# Benchmark Results
# ============================================================================

@dataclass
class BenchmarkResult:
    """Store benchmark results for a single experiment"""
    
    dataset_type: str  # 'synthetic' or 'real'
    model_name: str
    
    # Training metrics
    train_time: float  # Total training time (seconds)
    final_train_acc: float
    final_val_acc: float
    final_test_acc: float
    
    # Loss metrics
    final_train_loss: float
    final_val_loss: float
    final_test_loss: float
    
    # Convergence metrics
    epochs_to_converge: int  # Epochs to reach 90% of final accuracy
    best_epoch: int
    
    # Resource metrics
    peak_memory_mb: float
    avg_epoch_time: float
    
    # Training history
    train_acc_history: List[float]
    val_acc_history: List[float]
    train_loss_history: List[float]
    val_loss_history: List[float]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


# ============================================================================
# Dataset Comparison Benchmark
# ============================================================================

class DatasetComparisonBenchmark:
    """
    Compare synthetic vs real dataset performance.
    
    Metrics:
    - Training accuracy convergence
    - Final test accuracy
    - Training time
    - Resource utilization
    - Statistical significance
    """
    
    def __init__(
        self,
        output_dir: str = './outputs/dataset_comparison',
        device: str = 'cuda'
    ):
        """
        Initialize benchmark.
        
        Args:
            output_dir: Directory to save results
            device: Device to run on
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = device
        self.results = []
        
        logger.info(f"Initialized dataset comparison benchmark")
        logger.info(f"Output directory: {self.output_dir}")
    
    def benchmark_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        dataset_type: str,
        model_name: str,
        num_epochs: int = 100,
        learning_rate: float = 0.1
    ) -> BenchmarkResult:
        """
        Benchmark a model on a dataset.
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            dataset_type: 'synthetic' or 'real'
            model_name: Model name for logging
            num_epochs: Number of training epochs
            learning_rate: Learning rate
        
        Returns:
            BenchmarkResult with all metrics
        """
        logger.info("=" * 80)
        logger.info(f"Benchmarking {model_name} on {dataset_type} dataset")
        logger.info("=" * 80)
        
        # Move model to device
        model = model.to(self.device)
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs
        )
        
        # Training history
        train_acc_history = []
        val_acc_history = []
        train_loss_history = []
        val_loss_history = []
        epoch_times = []
        
        best_val_acc = 0.0
        best_epoch = 0
        epochs_to_converge = num_epochs
        
        # Track memory
        if self.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        
        start_time = time.time()
        
        # Training loop
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self._train_epoch(
                model, train_loader, criterion, optimizer
            )
            
            # Validate
            val_loss, val_acc = self._evaluate(
                model, val_loader, criterion
            )
            
            # Update scheduler
            scheduler.step()
            
            # Record history
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)
            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)
            epoch_times.append(time.time() - epoch_start)
            
            # Track best
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
            
            # Check convergence (90% of final accuracy)
            if epoch == num_epochs:
                target_acc = train_acc * 0.9
                for e, acc in enumerate(train_acc_history, 1):
                    if acc >= target_acc:
                        epochs_to_converge = e
                        break
            
            # Log progress
            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    f"Epoch {epoch}/{num_epochs} | "
                    f"Train: {train_acc:.2f}%/{train_loss:.4f} | "
                    f"Val: {val_acc:.2f}%/{val_loss:.4f}"
                )
        
        total_time = time.time() - start_time
        
        # Final test evaluation
        test_loss, test_acc = self._evaluate(model, test_loader, criterion)
        
        # Get peak memory
        if self.device == 'cuda':
            peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            peak_memory_mb = 0.0
        
        # Create result
        result = BenchmarkResult(
            dataset_type=dataset_type,
            model_name=model_name,
            train_time=total_time,
            final_train_acc=train_acc_history[-1],
            final_val_acc=val_acc_history[-1],
            final_test_acc=test_acc,
            final_train_loss=train_loss_history[-1],
            final_val_loss=val_loss_history[-1],
            final_test_loss=test_loss,
            epochs_to_converge=epochs_to_converge,
            best_epoch=best_epoch,
            peak_memory_mb=peak_memory_mb,
            avg_epoch_time=np.mean(epoch_times),
            train_acc_history=train_acc_history,
            val_acc_history=val_acc_history,
            train_loss_history=train_loss_history,
            val_loss_history=val_loss_history
        )
        
        self.results.append(result)
        
        logger.info("\n" + "=" * 80)
        logger.info("Benchmark Complete!")
        logger.info(f"Total time: {total_time / 60:.1f} minutes")
        logger.info(f"Final test accuracy: {test_acc:.2f}%")
        logger.info(f"Peak memory: {peak_memory_mb:.1f} MB")
        logger.info("=" * 80 + "\n")
        
        return result
    
    def _train_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer
    ) -> Tuple[float, float]:
        """Train for one epoch"""
        
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def _evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """Evaluate model"""
        
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def compare_results(
        self,
        synthetic_result: BenchmarkResult,
        real_result: BenchmarkResult
    ) -> Dict:
        """
        Compare synthetic vs real dataset results.
        
        Returns:
            Dictionary with comparison metrics
        """
        logger.info("=" * 80)
        logger.info("Dataset Comparison Analysis")
        logger.info("=" * 80)
        
        # Accuracy comparison
        acc_diff = real_result.final_test_acc - synthetic_result.final_test_acc
        acc_ratio = real_result.final_test_acc / synthetic_result.final_test_acc
        
        # Time comparison
        time_diff = real_result.train_time - synthetic_result.train_time
        time_ratio = real_result.train_time / synthetic_result.train_time
        
        # Convergence comparison
        conv_diff = real_result.epochs_to_converge - synthetic_result.epochs_to_converge
        
        # Memory comparison
        mem_diff = real_result.peak_memory_mb - synthetic_result.peak_memory_mb
        
        comparison = {
            'accuracy': {
                'synthetic': synthetic_result.final_test_acc,
                'real': real_result.final_test_acc,
                'difference': acc_diff,
                'ratio': acc_ratio,
                'synthetic_is_better': acc_diff < 0
            },
            'training_time': {
                'synthetic_seconds': synthetic_result.train_time,
                'real_seconds': real_result.train_time,
                'difference_seconds': time_diff,
                'ratio': time_ratio
            },
            'convergence': {
                'synthetic_epochs': synthetic_result.epochs_to_converge,
                'real_epochs': real_result.epochs_to_converge,
                'difference_epochs': conv_diff
            },
            'memory': {
                'synthetic_mb': synthetic_result.peak_memory_mb,
                'real_mb': real_result.peak_memory_mb,
                'difference_mb': mem_diff
            }
        }
        
        # Log comparison
        logger.info("\nAccuracy Comparison:")
        logger.info(f"  Synthetic: {synthetic_result.final_test_acc:.2f}%")
        logger.info(f"  Real: {real_result.final_test_acc:.2f}%")
        logger.info(f"  Difference: {acc_diff:+.2f}%")
        logger.info(f"  Ratio: {acc_ratio:.3f}x")
        
        logger.info("\nTraining Time:")
        logger.info(f"  Synthetic: {synthetic_result.train_time / 60:.1f} min")
        logger.info(f"  Real: {real_result.train_time / 60:.1f} min")
        logger.info(f"  Difference: {time_diff / 60:+.1f} min")
        
        logger.info("\nConvergence:")
        logger.info(f"  Synthetic: {synthetic_result.epochs_to_converge} epochs")
        logger.info(f"  Real: {real_result.epochs_to_converge} epochs")
        
        logger.info("=" * 80)
        
        return comparison
    
    def plot_comparison(
        self,
        synthetic_result: BenchmarkResult,
        real_result: BenchmarkResult,
        save_path: Optional[Path] = None
    ):
        """Plot comparison graphs"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training accuracy
        ax = axes[0, 0]
        ax.plot(synthetic_result.train_acc_history, label='Synthetic', linewidth=2)
        ax.plot(real_result.train_acc_history, label='Real', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Training Accuracy (%)')
        ax.set_title('Training Accuracy Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Validation accuracy
        ax = axes[0, 1]
        ax.plot(synthetic_result.val_acc_history, label='Synthetic', linewidth=2)
        ax.plot(real_result.val_acc_history, label='Real', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Accuracy (%)')
        ax.set_title('Validation Accuracy Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Training loss
        ax = axes[1, 0]
        ax.plot(synthetic_result.train_loss_history, label='Synthetic', linewidth=2)
        ax.plot(real_result.train_loss_history, label='Real', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Training Loss')
        ax.set_title('Training Loss Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Validation loss
        ax = axes[1, 1]
        ax.plot(synthetic_result.val_loss_history, label='Synthetic', linewidth=2)
        ax.plot(real_result.val_loss_history, label='Real', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Loss')
        ax.set_title('Validation Loss Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved comparison plot to {save_path}")
        
        plt.close()
    
    def save_results(self, filename: str = 'comparison_results.json'):
        """Save all results to JSON"""
        
        results_dict = {
            'results': [r.to_dict() for r in self.results],
            'summary': self._generate_summary()
        }
        
        results_path = self.output_dir / filename
        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Saved results to {results_path}")
    
    def _generate_summary(self) -> Dict:
        """Generate summary statistics"""
        
        if not self.results:
            return {}
        
        # Group by dataset type
        synthetic_results = [r for r in self.results if r.dataset_type == 'synthetic']
        real_results = [r for r in self.results if r.dataset_type == 'real']
        
        summary = {
            'total_benchmarks': len(self.results),
            'synthetic_count': len(synthetic_results),
            'real_count': len(real_results)
        }
        
        if synthetic_results:
            summary['synthetic_avg_acc'] = np.mean([r.final_test_acc for r in synthetic_results])
            summary['synthetic_avg_time'] = np.mean([r.train_time for r in synthetic_results])
        
        if real_results:
            summary['real_avg_acc'] = np.mean([r.final_test_acc for r in real_results])
            summary['real_avg_time'] = np.mean([r.train_time for r in real_results])
        
        return summary


# ============================================================================
# Demo Code
# ============================================================================

if __name__ == "__main__":
    import sys
    sys.path.append('.')
    
    from src.models.resnet import ResNet18
    from src.data.cifar_dataset import CIFARDataset
    from src.data.synthetic_data import SyntheticDataset
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 80)
    print("Dataset Comparison Demo")
    print("=" * 80)
    
    # Initialize benchmark
    benchmark = DatasetComparisonBenchmark(
        output_dir='./outputs/comparison_demo',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Create models
    print("\n1. Creating models...")
    synthetic_model = ResNet18(num_classes=10)
    real_model = ResNet18(num_classes=10)
    
    # Load synthetic dataset
    print("\n2. Loading synthetic dataset...")
    synthetic_dataset = SyntheticDataset(
        num_classes=10,
        samples_per_class=500,
        image_size=32
    )
    synthetic_train = synthetic_dataset.get_dataloader(batch_size=128, split='train')
    synthetic_val = synthetic_dataset.get_dataloader(batch_size=128, split='val')
    synthetic_test = synthetic_dataset.get_dataloader(batch_size=128, split='test')
    
    # Load real dataset
    print("\n3. Loading CIFAR-10 dataset...")
    cifar = CIFARDataset(dataset_name='cifar10')
    real_train, real_val, real_test = cifar.get_loaders(batch_size=128)
    
    # Benchmark synthetic
    print("\n4. Benchmarking synthetic dataset...")
    synthetic_result = benchmark.benchmark_model(
        model=synthetic_model,
        train_loader=synthetic_train,
        val_loader=synthetic_val,
        test_loader=synthetic_test,
        dataset_type='synthetic',
        model_name='ResNet18',
        num_epochs=10  # Short demo
    )
    
    # Benchmark real
    print("\n5. Benchmarking real dataset...")
    real_result = benchmark.benchmark_model(
        model=real_model,
        train_loader=real_train,
        val_loader=real_val,
        test_loader=real_test,
        dataset_type='real',
        model_name='ResNet18',
        num_epochs=10  # Short demo
    )
    
    # Compare
    print("\n6. Comparing results...")
    comparison = benchmark.compare_results(synthetic_result, real_result)
    
    # Plot
    print("\n7. Generating plots...")
    benchmark.plot_comparison(
        synthetic_result,
        real_result,
        save_path=benchmark.output_dir / 'comparison_plot.png'
    )
    
    # Save results
    print("\n8. Saving results...")
    benchmark.save_results()
    
    print("\n" + "=" * 80)
    print("✓ Demo complete!")
    print(f"Results saved to: {benchmark.output_dir}")
    print("=" * 80)
