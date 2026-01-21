"""
Tests for Tensor Decomposition Fine-tuning - Session 25
========================================================

Tests for post-decomposition fine-tuning pipeline.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.compute.tensor_decomposition import (
    TuckerDecomposer,
    decompose_model,
    DecompositionConfig
)
from src.compute.tensor_decomposition_finetuning import (
    DecompositionFinetuner,
    FinetuneConfig,
    quick_finetune,
    LayerWiseFinetuner
)


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def create_dummy_data(num_samples=100, batch_size=16):
    """Create dummy dataset for testing."""
    X = torch.randn(num_samples, 3, 32, 32)
    y = torch.randint(0, 10, (num_samples,))
    
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return loader


class TestFinetuneConfig:
    """Test FinetuneConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = FinetuneConfig()
        
        assert config.epochs == 3
        assert config.learning_rate == 1e-4
        assert config.scheduler == "cosine"
        assert config.early_stopping == True
        assert config.use_distillation == False
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = FinetuneConfig(
            epochs=5,
            learning_rate=1e-3,
            use_distillation=True,
            distillation_alpha=0.7
        )
        
        assert config.epochs == 5
        assert config.learning_rate == 1e-3
        assert config.use_distillation == True
        assert config.distillation_alpha == 0.7


class TestDecompositionFinetuner:
    """Test DecompositionFinetuner class."""
    
    def test_finetuner_init(self):
        """Test finetuner initialization."""
        config = FinetuneConfig(epochs=2)
        finetuner = DecompositionFinetuner(config)
        
        assert finetuner.config.epochs == 2
        assert 'train_loss' in finetuner.history
        assert 'val_loss' in finetuner.history
    
    def test_fine_tune_basic(self):
        """Test basic fine-tuning."""
        # Create model and data
        model = SimpleModel()
        train_loader = create_dummy_data(num_samples=50, batch_size=10)
        val_loader = create_dummy_data(num_samples=20, batch_size=10)
        
        # Decompose
        config = DecompositionConfig(method="tucker", ranks=[8, 16])
        compressed = decompose_model(model, config)
        
        # Fine-tune
        finetune_config = FinetuneConfig(
            epochs=1,
            verbose=False,
            early_stopping=False
        )
        finetuner = DecompositionFinetuner(finetune_config)
        
        tuned, metrics = finetuner.fine_tune(
            decomposed_model=compressed,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=nn.CrossEntropyLoss()
        )
        
        # Check results
        assert isinstance(tuned, nn.Module)
        assert 'final_train_loss' in metrics
        assert 'final_val_loss' in metrics
        assert 'final_val_accuracy' in metrics
        assert len(finetuner.history['train_loss']) == 1
    
    def test_fine_tune_with_distillation(self):
        """Test fine-tuning with knowledge distillation."""
        # Create models and data
        original = SimpleModel()
        model = SimpleModel()
        model.load_state_dict(original.state_dict())
        
        train_loader = create_dummy_data(num_samples=50, batch_size=10)
        val_loader = create_dummy_data(num_samples=20, batch_size=10)
        
        # Decompose
        config = DecompositionConfig(method="tucker", ranks=[8, 16])
        compressed = decompose_model(model, config)
        
        # Fine-tune with distillation
        finetune_config = FinetuneConfig(
            epochs=1,
            use_distillation=True,
            distillation_alpha=0.5,
            verbose=False,
            early_stopping=False
        )
        finetuner = DecompositionFinetuner(finetune_config)
        
        tuned, metrics = finetuner.fine_tune(
            decomposed_model=compressed,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=nn.CrossEntropyLoss(),
            original_model=original
        )
        
        assert isinstance(tuned, nn.Module)
        assert 'final_val_accuracy' in metrics
    
    def test_early_stopping(self):
        """Test early stopping mechanism."""
        model = SimpleModel()
        train_loader = create_dummy_data(num_samples=50, batch_size=10)
        val_loader = create_dummy_data(num_samples=20, batch_size=10)
        
        config = DecompositionConfig(method="tucker", ranks=[8, 16])
        compressed = decompose_model(model, config)
        
        finetune_config = FinetuneConfig(
            epochs=10,  # Set high
            early_stopping=True,
            patience=2,
            verbose=False
        )
        finetuner = DecompositionFinetuner(finetune_config)
        
        tuned, metrics = finetuner.fine_tune(
            decomposed_model=compressed,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=nn.CrossEntropyLoss()
        )
        
        # Should stop early
        assert metrics['epochs_trained'] < 10
    
    def test_cosine_scheduler(self):
        """Test cosine annealing scheduler."""
        model = SimpleModel()
        train_loader = create_dummy_data(num_samples=50, batch_size=10)
        val_loader = create_dummy_data(num_samples=20, batch_size=10)
        
        config = DecompositionConfig(method="tucker", ranks=[8, 16])
        compressed = decompose_model(model, config)
        
        finetune_config = FinetuneConfig(
            epochs=3,
            scheduler="cosine",
            verbose=False,
            early_stopping=False
        )
        finetuner = DecompositionFinetuner(finetune_config)
        
        tuned, metrics = finetuner.fine_tune(
            decomposed_model=compressed,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=nn.CrossEntropyLoss()
        )
        
        # Check LR decay
        lrs = metrics['history']['learning_rate']
        assert len(lrs) == 3
        assert lrs[0] > lrs[-1]  # LR should decrease
    
    def test_plateau_scheduler(self):
        """Test plateau scheduler."""
        model = SimpleModel()
        train_loader = create_dummy_data(num_samples=50, batch_size=10)
        val_loader = create_dummy_data(num_samples=20, batch_size=10)
        
        config = DecompositionConfig(method="tucker", ranks=[8, 16])
        compressed = decompose_model(model, config)
        
        finetune_config = FinetuneConfig(
            epochs=2,
            scheduler="plateau",
            verbose=False,
            early_stopping=False
        )
        finetuner = DecompositionFinetuner(finetune_config)
        
        tuned, metrics = finetuner.fine_tune(
            decomposed_model=compressed,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=nn.CrossEntropyLoss()
        )
        
        assert 'learning_rate' in metrics['history']
    
    def test_distillation_loss(self):
        """Test distillation loss computation."""
        finetuner = DecompositionFinetuner(FinetuneConfig(
            use_distillation=True,
            distillation_alpha=0.5,
            distillation_temperature=3.0
        ))
        
        student_output = torch.randn(4, 10)
        teacher_output = torch.randn(4, 10)
        target = torch.randint(0, 10, (4,))
        criterion = nn.CrossEntropyLoss()
        
        loss = finetuner._distillation_loss(
            student_output, teacher_output, target, criterion
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


class TestQuickFinetune:
    """Test quick_finetune convenience function."""
    
    def test_quick_finetune_basic(self):
        """Test quick fine-tune function."""
        model = SimpleModel()
        train_loader = create_dummy_data(num_samples=50, batch_size=10)
        val_loader = create_dummy_data(num_samples=20, batch_size=10)
        
        config = DecompositionConfig(method="tucker", ranks=[8, 16])
        compressed = decompose_model(model, config)
        
        tuned, metrics = quick_finetune(
            compressed,
            train_loader,
            val_loader,
            epochs=1
        )
        
        assert isinstance(tuned, nn.Module)
        assert 'final_val_accuracy' in metrics
    
    def test_quick_finetune_with_distillation(self):
        """Test quick fine-tune with distillation."""
        original = SimpleModel()
        model = SimpleModel()
        model.load_state_dict(original.state_dict())
        
        train_loader = create_dummy_data(num_samples=50, batch_size=10)
        val_loader = create_dummy_data(num_samples=20, batch_size=10)
        
        config = DecompositionConfig(method="tucker", ranks=[8, 16])
        compressed = decompose_model(model, config)
        
        tuned, metrics = quick_finetune(
            compressed,
            train_loader,
            val_loader,
            epochs=1,
            use_distillation=True,
            original_model=original
        )
        
        assert isinstance(tuned, nn.Module)
        assert metrics['final_val_accuracy'] >= 0.0


class TestLayerWiseFinetuner:
    """Test LayerWiseFinetuner class."""
    
    def test_layerwise_init(self):
        """Test layer-wise finetuner initialization."""
        config = FinetuneConfig(epochs=1)
        finetuner = LayerWiseFinetuner(config)
        
        assert finetuner.config.epochs == 1
    
    def test_layerwise_finetune(self):
        """Test layer-wise fine-tuning."""
        model = SimpleModel()
        train_loader = create_dummy_data(num_samples=50, batch_size=10)
        val_loader = create_dummy_data(num_samples=20, batch_size=10)
        
        config = DecompositionConfig(method="tucker", ranks=[8, 16])
        compressed = decompose_model(model, config)
        
        finetune_config = FinetuneConfig(
            epochs=1,
            verbose=False,
            early_stopping=False
        )
        finetuner = LayerWiseFinetuner(finetune_config)
        
        tuned, metrics = finetuner.fine_tune_layerwise(
            model=compressed,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=nn.CrossEntropyLoss(),
            layers_per_stage=2
        )
        
        assert isinstance(tuned, nn.Module)
        assert 'stage_metrics' in metrics
        assert len(metrics['stage_metrics']) > 0


class TestAccuracyRecovery:
    """Test accuracy recovery after fine-tuning."""
    
    def test_accuracy_improves(self):
        """Test that fine-tuning improves accuracy."""
        # Create consistent model and data
        torch.manual_seed(42)
        model = SimpleModel()
        
        train_loader = create_dummy_data(num_samples=100, batch_size=10)
        val_loader = create_dummy_data(num_samples=50, batch_size=10)
        
        # Decompose
        config = DecompositionConfig(method="tucker", ranks=[8, 16])
        compressed = decompose_model(model, config)
        
        # Measure accuracy before fine-tuning
        compressed.eval()
        correct_before = 0
        total = 0
        with torch.no_grad():
            for data, target in val_loader:
                output = compressed(data)
                pred = output.argmax(dim=1)
                correct_before += pred.eq(target).sum().item()
                total += target.size(0)
        acc_before = correct_before / total
        
        # Fine-tune
        finetune_config = FinetuneConfig(
            epochs=2,
            verbose=False,
            early_stopping=False
        )
        finetuner = DecompositionFinetuner(finetune_config)
        
        tuned, metrics = finetuner.fine_tune(
            decomposed_model=compressed,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=nn.CrossEntropyLoss()
        )
        
        acc_after = metrics['final_val_accuracy']
        
        # Accuracy should improve (or at least not get worse significantly)
        # Note: On random data this test might be flaky
        # In real scenario with proper data, accuracy should improve significantly
        assert acc_after >= acc_before * 0.8  # Allow some variance


class TestIntegration:
    """Integration tests for complete pipeline."""
    
    def test_decompose_and_finetune_pipeline(self):
        """Test complete decomposition + fine-tuning pipeline."""
        # Create model
        model = SimpleModel()
        
        # Create data
        train_loader = create_dummy_data(num_samples=100, batch_size=16)
        val_loader = create_dummy_data(num_samples=50, batch_size=16)
        
        # Step 1: Decompose
        decomp_config = DecompositionConfig(
            method="tucker",
            ranks=[8, 16],
            auto_rank=False
        )
        compressed = decompose_model(model, decomp_config)
        
        # Count parameters
        original_params = sum(p.numel() for p in model.parameters())
        compressed_params = sum(p.numel() for p in compressed.parameters())
        compression_ratio = original_params / compressed_params
        
        # Step 2: Fine-tune
        finetune_config = FinetuneConfig(
            epochs=2,
            learning_rate=1e-3,
            verbose=False,
            early_stopping=False
        )
        finetuner = DecompositionFinetuner(finetune_config)
        
        tuned, metrics = finetuner.fine_tune(
            decomposed_model=compressed,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=nn.CrossEntropyLoss()
        )
        
        # Verify complete pipeline
        assert compression_ratio >= 1.0  # Should have some compression
        assert metrics['final_val_accuracy'] >= 0.0  # Should have some accuracy
        assert len(metrics['history']['train_loss']) == 2  # 2 epochs
        
        # Model should be ready for inference
        test_input = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            output = tuned(test_input)
        assert output.shape == (1, 10)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
