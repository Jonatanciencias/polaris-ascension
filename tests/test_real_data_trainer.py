"""
Tests for Real Data Trainer

Comprehensive test suite for training loop with real datasets.

Author: AMD GPU Computing Team
Date: January 21, 2026
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import tempfile

from src.training.real_data_trainer import (
    TrainingConfig,
    RealDataTrainer
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def temp_output_dir():
    """Create temporary output directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def simple_model():
    """Create simple test model"""
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(16, 10)
        
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    return SimpleNet()


@pytest.fixture
def dummy_loaders():
    """Create dummy data loaders"""
    from torch.utils.data import TensorDataset, DataLoader
    
    # Generate dummy data
    train_images = torch.randn(100, 3, 32, 32)
    train_labels = torch.randint(0, 10, (100,))
    
    val_images = torch.randn(50, 3, 32, 32)
    val_labels = torch.randint(0, 10, (50,))
    
    test_images = torch.randn(50, 3, 32, 32)
    test_labels = torch.randint(0, 10, (50,))
    
    train_dataset = TensorDataset(train_images, train_labels)
    val_dataset = TensorDataset(val_images, val_labels)
    test_dataset = TensorDataset(test_images, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    return train_loader, val_loader, test_loader


# ============================================================================
# Test Training Config
# ============================================================================

class TestTrainingConfig:
    """Test training configuration"""
    
    def test_default_config(self, temp_output_dir):
        """Test default configuration"""
        config = TrainingConfig(
            output_dir=temp_output_dir,
            experiment_name='test'
        )
        
        assert config.num_epochs == 100
        assert config.batch_size == 128
        assert config.learning_rate == 0.1
        assert config.optimizer == 'sgd'
    
    def test_custom_config(self, temp_output_dir):
        """Test custom configuration"""
        config = TrainingConfig(
            num_epochs=50,
            batch_size=64,
            learning_rate=0.01,
            optimizer='adam',
            output_dir=temp_output_dir,
            experiment_name='test'
        )
        
        assert config.num_epochs == 50
        assert config.batch_size == 64
        assert config.learning_rate == 0.01
        assert config.optimizer == 'adam'
    
    def test_config_save(self, temp_output_dir):
        """Test config saving"""
        config = TrainingConfig(
            output_dir=temp_output_dir,
            experiment_name='test'
        )
        
        config_path = config.exp_dir / 'config.json'
        assert config_path.exists()
    
    def test_lr_schedules(self, temp_output_dir):
        """Test different LR schedules"""
        for schedule in ['step', 'cosine', 'exponential']:
            config = TrainingConfig(
                lr_schedule=schedule,
                output_dir=temp_output_dir,
                experiment_name=f'test_{schedule}'
            )
            assert config.lr_schedule == schedule


# ============================================================================
# Test Trainer Initialization
# ============================================================================

class TestTrainerInit:
    """Test trainer initialization"""
    
    def test_trainer_creation(self, simple_model, dummy_loaders, temp_output_dir):
        """Test creating trainer"""
        train_loader, val_loader, test_loader = dummy_loaders
        
        config = TrainingConfig(
            num_epochs=5,
            output_dir=temp_output_dir,
            experiment_name='test'
        )
        
        trainer = RealDataTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=config,
            device='cpu'
        )
        
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None
        assert trainer.criterion is not None
    
    def test_optimizers(self, simple_model, dummy_loaders, temp_output_dir):
        """Test different optimizers"""
        train_loader, val_loader, _ = dummy_loaders
        
        for optimizer_name in ['sgd', 'adam', 'adamw']:
            config = TrainingConfig(
                optimizer=optimizer_name,
                output_dir=temp_output_dir,
                experiment_name=f'test_{optimizer_name}'
            )
            
            trainer = RealDataTrainer(
                model=simple_model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                device='cpu'
            )
            
            assert trainer.optimizer is not None


# ============================================================================
# Test Training
# ============================================================================

class TestTraining:
    """Test training functionality"""
    
    def test_train_epoch(self, simple_model, dummy_loaders, temp_output_dir):
        """Test training for one epoch"""
        train_loader, val_loader, _ = dummy_loaders
        
        config = TrainingConfig(
            num_epochs=1,
            output_dir=temp_output_dir,
            experiment_name='test',
            use_tensorboard=False
        )
        
        trainer = RealDataTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device='cpu'
        )
        
        trainer.current_epoch = 1
        loss, acc = trainer.train_epoch()
        
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert 0 <= acc <= 100
    
    def test_validate(self, simple_model, dummy_loaders, temp_output_dir):
        """Test validation"""
        train_loader, val_loader, _ = dummy_loaders
        
        config = TrainingConfig(
            output_dir=temp_output_dir,
            experiment_name='test',
            use_tensorboard=False
        )
        
        trainer = RealDataTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device='cpu'
        )
        
        loss, acc = trainer.validate()
        
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert 0 <= acc <= 100
    
    def test_full_training(self, simple_model, dummy_loaders, temp_output_dir):
        """Test full training loop"""
        train_loader, val_loader, test_loader = dummy_loaders
        
        config = TrainingConfig(
            num_epochs=2,
            early_stopping=False,
            save_best=False,
            save_last=False,
            output_dir=temp_output_dir,
            experiment_name='test',
            use_tensorboard=False
        )
        
        trainer = RealDataTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=config,
            device='cpu'
        )
        
        history = trainer.train()
        
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert 'train_acc' in history
        assert 'val_acc' in history
        assert len(history['train_loss']) == 2


# ============================================================================
# Test Checkpointing
# ============================================================================

class TestCheckpointing:
    """Test model checkpointing"""
    
    def test_save_checkpoint(self, simple_model, dummy_loaders, temp_output_dir):
        """Test saving checkpoint"""
        train_loader, val_loader, _ = dummy_loaders
        
        config = TrainingConfig(
            output_dir=temp_output_dir,
            experiment_name='test',
            use_tensorboard=False
        )
        
        trainer = RealDataTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device='cpu'
        )
        
        trainer.current_epoch = 1
        trainer.save_checkpoint('test_checkpoint.pth')
        
        checkpoint_path = config.exp_dir / 'test_checkpoint.pth'
        assert checkpoint_path.exists()
    
    def test_load_checkpoint(self, simple_model, dummy_loaders, temp_output_dir):
        """Test loading checkpoint"""
        train_loader, val_loader, _ = dummy_loaders
        
        config = TrainingConfig(
            output_dir=temp_output_dir,
            experiment_name='test',
            use_tensorboard=False
        )
        
        # Create and train
        trainer1 = RealDataTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device='cpu'
        )
        
        trainer1.current_epoch = 1
        trainer1.save_checkpoint('test_checkpoint.pth')
        
        # Load in new trainer
        trainer2 = RealDataTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device='cpu'
        )
        
        checkpoint_path = config.exp_dir / 'test_checkpoint.pth'
        trainer2.load_checkpoint(str(checkpoint_path))
        
        assert trainer2.current_epoch == 1


# ============================================================================
# Test Early Stopping
# ============================================================================

class TestEarlyStopping:
    """Test early stopping"""
    
    def test_early_stopping_trigger(self, simple_model, dummy_loaders, temp_output_dir):
        """Test early stopping triggers"""
        train_loader, val_loader, _ = dummy_loaders
        
        config = TrainingConfig(
            early_stopping=True,
            patience=3,
            min_delta=0.001,
            output_dir=temp_output_dir,
            experiment_name='test',
            use_tensorboard=False
        )
        
        trainer = RealDataTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device='cpu'
        )
        
        # Simulate no improvement
        trainer.best_val_loss = 1.0
        
        should_stop = trainer.check_early_stopping(1.0)
        assert not should_stop
        
        should_stop = trainer.check_early_stopping(1.0)
        assert not should_stop
        
        should_stop = trainer.check_early_stopping(1.0)
        assert not should_stop
        
        should_stop = trainer.check_early_stopping(1.0)
        assert should_stop


# ============================================================================
# Test Metrics Logging
# ============================================================================

class TestMetricsLogging:
    """Test metrics logging"""
    
    def test_log_metrics(self, simple_model, dummy_loaders, temp_output_dir):
        """Test logging metrics"""
        train_loader, val_loader, _ = dummy_loaders
        
        config = TrainingConfig(
            output_dir=temp_output_dir,
            experiment_name='test',
            use_tensorboard=False
        )
        
        trainer = RealDataTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device='cpu'
        )
        
        trainer.log_metrics(
            train_loss=1.0,
            train_acc=80.0,
            val_loss=1.2,
            val_acc=75.0
        )
        
        assert len(trainer.history['train_loss']) == 1
        assert len(trainer.history['val_loss']) == 1
        assert trainer.history['train_loss'][0] == 1.0
        assert trainer.history['val_acc'][0] == 75.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
