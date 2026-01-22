"""
Real Data Training Loop

Professional training implementation for CIFAR-10/100 datasets with:
- Learning rate scheduling
- Checkpointing and model saving
- TensorBoard logging
- Early stopping
- Mixed precision training (optional)
- Progress tracking
- Integration with existing models

Optimized for AMD Radeon RX 580.

Author: AMD GPU Computing Team
Date: January 21, 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Callable, Tuple
from pathlib import Path
import logging
import time
from datetime import datetime
import json

logger = logging.getLogger(__name__)


# ============================================================================
# Training Configuration
# ============================================================================

class TrainingConfig:
    """Configuration for training"""
    
    def __init__(
        self,
        # Training parameters
        num_epochs: int = 100,
        batch_size: int = 128,
        learning_rate: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        
        # Learning rate schedule
        lr_schedule: str = 'cosine',  # 'step', 'cosine', 'exponential'
        lr_milestones: Optional[List[int]] = None,
        lr_gamma: float = 0.1,
        
        # Optimization
        optimizer: str = 'sgd',  # 'sgd', 'adam', 'adamw'
        
        # Regularization
        label_smoothing: float = 0.0,
        
        # Early stopping
        early_stopping: bool = True,
        patience: int = 15,
        min_delta: float = 1e-4,
        
        # Checkpointing
        save_best: bool = True,
        save_last: bool = True,
        checkpoint_freq: int = 10,
        
        # Logging
        log_interval: int = 10,  # Log every N batches
        use_tensorboard: bool = True,
        
        # Paths
        output_dir: str = './outputs/training',
        experiment_name: Optional[str] = None
    ):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        self.lr_schedule = lr_schedule
        self.lr_milestones = lr_milestones or [60, 120, 160]
        self.lr_gamma = lr_gamma
        
        self.optimizer = optimizer
        self.label_smoothing = label_smoothing
        
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        
        self.save_best = save_best
        self.save_last = save_last
        self.checkpoint_freq = checkpoint_freq
        
        self.log_interval = log_interval
        self.use_tensorboard = use_tensorboard
        
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        self.exp_dir = self.output_dir / self.experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        self.save_config()
    
    def save_config(self):
        """Save configuration to JSON"""
        config_path = self.exp_dir / 'config.json'
        config_dict = {k: v for k, v in self.__dict__.items() 
                      if not k.startswith('_') and k not in ['exp_dir', 'output_dir']}
        config_dict['output_dir'] = str(self.output_dir)
        config_dict['exp_dir'] = str(self.exp_dir)
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Saved config to {config_path}")


# ============================================================================
# Real Data Trainer
# ============================================================================

class RealDataTrainer:
    """
    Professional trainer for CIFAR-10/100 datasets.
    
    Features:
    - Automatic optimization setup
    - Learning rate scheduling
    - Checkpointing and model saving
    - TensorBoard logging
    - Early stopping
    - Progress tracking
    - Metric computation
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        config: Optional[TrainingConfig] = None,
        device: str = 'cuda'
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader (optional)
            config: Training configuration
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        # Configuration
        self.config = config or TrainingConfig()
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=self.config.label_smoothing
        )
        
        # Setup learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Setup TensorBoard
        if self.config.use_tensorboard:
            self.writer = SummaryWriter(log_dir=str(self.config.exp_dir / 'tensorboard'))
        else:
            self.writer = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # History
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        logger.info(f"Initialized trainer for {self.config.num_epochs} epochs")
        logger.info(f"Output directory: {self.config.exp_dir}")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer"""
        
        if self.config.optimizer == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
                nesterov=True
            )
        elif self.config.optimizer == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        
        logger.info(f"Created optimizer: {self.config.optimizer}")
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        
        if self.config.lr_schedule == 'step':
            scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=self.config.lr_milestones,
                gamma=self.config.lr_gamma
            )
        elif self.config.lr_schedule == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs
            )
        elif self.config.lr_schedule == 'exponential':
            scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=0.95
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.config.lr_schedule}")
        
        logger.info(f"Created scheduler: {self.config.lr_schedule}")
        return scheduler
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Logging
            if (batch_idx + 1) % self.config.log_interval == 0:
                avg_loss = total_loss / (batch_idx + 1)
                acc = 100.0 * correct / total
                logger.debug(
                    f"Epoch {self.current_epoch} | "
                    f"Batch {batch_idx + 1}/{len(self.train_loader)} | "
                    f"Loss: {avg_loss:.4f} | Acc: {acc:.2f}%"
                )
        
        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total
        
        logger.info(
            f"Train Epoch {self.current_epoch}: "
            f"Loss={avg_loss:.4f}, Acc={accuracy:.2f}%, Time={epoch_time:.1f}s"
        )
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """
        Validate model.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in self.val_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total
        
        logger.info(
            f"Val Epoch {self.current_epoch}: "
            f"Loss={avg_loss:.4f}, Acc={accuracy:.2f}%"
        )
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def test(self) -> Tuple[float, float]:
        """
        Test model.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        if self.test_loader is None:
            logger.warning("No test loader provided")
            return 0.0, 0.0
        
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in self.test_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(self.test_loader)
        accuracy = 100.0 * correct / total
        
        logger.info(f"Test: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%")
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint"""
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }
        
        checkpoint_path = self.config.exp_dir / filename
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.config.exp_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"✓ Saved best model (val_acc={self.best_val_acc:.2f}%)")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint['best_val_acc']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def check_early_stopping(self, val_loss: float) -> bool:
        """
        Check if training should stop early.
        
        Returns:
            True if training should stop
        """
        if not self.config.early_stopping:
            return False
        
        if val_loss < (self.best_val_loss - self.config.min_delta):
            self.epochs_without_improvement = 0
            self.best_val_loss = val_loss
        else:
            self.epochs_without_improvement += 1
        
        if self.epochs_without_improvement >= self.config.patience:
            logger.info(
                f"Early stopping triggered after {self.current_epoch} epochs "
                f"(patience={self.config.patience})"
            )
            return True
        
        return False
    
    def log_metrics(
        self,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float
    ):
        """Log metrics to TensorBoard and history"""
        
        # Update history
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
        
        # Log to TensorBoard
        if self.writer:
            self.writer.add_scalar('Loss/train', train_loss, self.current_epoch)
            self.writer.add_scalar('Loss/val', val_loss, self.current_epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, self.current_epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, self.current_epoch)
            self.writer.add_scalar('LearningRate', self.history['lr'][-1], self.current_epoch)
    
    def train(self) -> Dict:
        """
        Main training loop.
        
        Returns:
            Training history dictionary
        """
        logger.info("=" * 80)
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        for epoch in range(1, self.config.num_epochs + 1):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Log metrics
            self.log_metrics(train_loss, train_acc, val_loss, val_acc)
            
            # Update learning rate
            self.scheduler.step()
            
            # Check for best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
            
            # Save checkpoints
            if self.config.save_best and is_best:
                self.save_checkpoint('checkpoint.pth', is_best=True)
            
            if epoch % self.config.checkpoint_freq == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')
            
            # Check early stopping
            if self.check_early_stopping(val_loss):
                break
        
        # Save final model
        if self.config.save_last:
            self.save_checkpoint('last_model.pth')
        
        # Final test
        if self.test_loader:
            logger.info("\n" + "=" * 80)
            logger.info("Running final test evaluation...")
            test_loss, test_acc = self.test()
            self.history['test_loss'] = test_loss
            self.history['test_acc'] = test_acc
        
        total_time = time.time() - start_time
        
        logger.info("\n" + "=" * 80)
        logger.info("Training Complete!")
        logger.info(f"Total time: {total_time / 3600:.2f} hours")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        if self.test_loader:
            logger.info(f"Final test accuracy: {test_acc:.2f}%")
        logger.info("=" * 80)
        
        # Close TensorBoard writer
        if self.writer:
            self.writer.close()
        
        # Save history
        history_path = self.config.exp_dir / 'history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return self.history


# ============================================================================
# Demo Code
# ============================================================================

if __name__ == "__main__":
    import sys
    sys.path.append('.')
    
    from src.data.cifar_dataset import CIFARDataset
    from src.models.resnet import ResNet18
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 80)
    print("Real Data Training Demo")
    print("=" * 80)
    
    # Load dataset
    print("\n1. Loading CIFAR-10 dataset...")
    dataset = CIFARDataset(
        dataset_name='cifar10',
        data_root='./data',
        val_split=0.1,
        augment_train=True,
        augment_strength='medium'
    )
    
    train_loader, val_loader, test_loader = dataset.get_loaders(batch_size=128)
    
    # Create model
    print("\n2. Creating ResNet-18 model...")
    model = ResNet18(num_classes=10)
    
    # Setup training
    print("\n3. Setting up trainer...")
    config = TrainingConfig(
        num_epochs=5,  # Short demo
        batch_size=128,
        learning_rate=0.1,
        lr_schedule='cosine',
        early_stopping=False,
        experiment_name='demo_training'
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    trainer = RealDataTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=device
    )
    
    # Train
    print("\n4. Starting training...")
    history = trainer.train()
    
    print("\n" + "=" * 80)
    print("✓ Demo complete!")
    print(f"Best validation accuracy: {trainer.best_val_acc:.2f}%")
    print(f"Output directory: {config.exp_dir}")
    print("=" * 80)
