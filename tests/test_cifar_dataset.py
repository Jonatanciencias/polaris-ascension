"""
Tests for CIFAR Dataset

Comprehensive test suite for CIFAR-10/100 dataset loading.

Author: AMD GPU Computing Team
Date: January 21, 2026
"""

import pytest
import torch
from pathlib import Path
import tempfile

from src.data.cifar_dataset import (
    CIFARDataset,
    CIFARDataAugmentation,
    CIFAR10_MEAN,
    CIFAR10_STD
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def temp_data_dir():
    """Create temporary data directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# ============================================================================
# Test Data Augmentation
# ============================================================================

class TestDataAugmentation:
    """Test data augmentation pipeline"""
    
    def test_cifar10_augmentation(self):
        """Test CIFAR-10 augmentation"""
        aug = CIFARDataAugmentation(dataset_name='cifar10', augment_train=True)
        
        assert aug.mean == CIFAR10_MEAN
        assert aug.std == CIFAR10_STD
    
    def test_train_transform_with_augmentation(self):
        """Test training transform with augmentation"""
        aug = CIFARDataAugmentation(
            dataset_name='cifar10',
            augment_train=True,
            augment_strength='medium'
        )
        
        transform = aug.get_train_transform()
        assert transform is not None
    
    def test_test_transform(self):
        """Test test transform (no augmentation)"""
        aug = CIFARDataAugmentation(dataset_name='cifar10')
        
        transform = aug.get_test_transform()
        assert transform is not None
    
    def test_augmentation_strengths(self):
        """Test different augmentation strengths"""
        for strength in ['light', 'medium', 'heavy']:
            aug = CIFARDataAugmentation(
                dataset_name='cifar10',
                augment_strength=strength
            )
            transform = aug.get_train_transform()
            assert transform is not None


# ============================================================================
# Test CIFAR Dataset
# ============================================================================

class TestCIFARDataset:
    """Test CIFAR dataset loading"""
    
    def test_load_cifar10(self, temp_data_dir):
        """Test loading CIFAR-10"""
        dataset = CIFARDataset(
            dataset_name='cifar10',
            data_root=temp_data_dir,
            download=True
        )
        
        assert dataset.num_classes == 10
        assert len(dataset.class_names) == 10
    
    def test_load_cifar100(self, temp_data_dir):
        """Test loading CIFAR-100"""
        dataset = CIFARDataset(
            dataset_name='cifar100',
            data_root=temp_data_dir,
            download=True
        )
        
        assert dataset.num_classes == 100
    
    def test_train_val_split(self, temp_data_dir):
        """Test train/validation split"""
        dataset = CIFARDataset(
            dataset_name='cifar10',
            data_root=temp_data_dir,
            val_split=0.1,
            download=True
        )
        
        # Check split ratio
        total_train = len(dataset.train_indices) + len(dataset.val_indices)
        val_ratio = len(dataset.val_indices) / total_train
        
        assert abs(val_ratio - 0.1) < 0.01  # Allow 1% tolerance
    
    def test_get_train_loader(self, temp_data_dir):
        """Test getting training data loader"""
        dataset = CIFARDataset(
            dataset_name='cifar10',
            data_root=temp_data_dir,
            download=True
        )
        
        loader = dataset.get_train_loader(batch_size=32, num_workers=0)
        
        assert loader is not None
        assert len(loader) > 0
        
        # Check batch
        images, labels = next(iter(loader))
        assert images.shape[0] == 32
        assert images.shape[1:] == (3, 32, 32)
        assert labels.shape[0] == 32
    
    def test_get_val_loader(self, temp_data_dir):
        """Test getting validation data loader"""
        dataset = CIFARDataset(
            dataset_name='cifar10',
            data_root=temp_data_dir,
            download=True
        )
        
        loader = dataset.get_val_loader(batch_size=32, num_workers=0)
        
        assert loader is not None
        assert len(loader) > 0
    
    def test_get_test_loader(self, temp_data_dir):
        """Test getting test data loader"""
        dataset = CIFARDataset(
            dataset_name='cifar10',
            data_root=temp_data_dir,
            download=True
        )
        
        loader = dataset.get_test_loader(batch_size=32, num_workers=0)
        
        assert loader is not None
        assert len(loader) > 0
    
    def test_get_all_loaders(self, temp_data_dir):
        """Test getting all loaders at once"""
        dataset = CIFARDataset(
            dataset_name='cifar10',
            data_root=temp_data_dir,
            download=True
        )
        
        train_loader, val_loader, test_loader = dataset.get_loaders(
            batch_size=32,
            num_workers=0
        )
        
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None
    
    def test_get_statistics(self, temp_data_dir):
        """Test getting dataset statistics"""
        dataset = CIFARDataset(
            dataset_name='cifar10',
            data_root=temp_data_dir,
            download=True
        )
        
        stats = dataset.get_statistics()
        
        assert 'dataset_name' in stats
        assert 'num_classes' in stats
        assert 'train_samples' in stats
        assert 'val_samples' in stats
        assert 'test_samples' in stats
        assert stats['num_classes'] == 10
    
    def test_get_sample_batch(self, temp_data_dir):
        """Test getting sample batch"""
        dataset = CIFARDataset(
            dataset_name='cifar10',
            data_root=temp_data_dir,
            download=True
        )
        
        images, labels = dataset.get_sample_batch('train')
        
        assert images.shape[0] == 8
        assert images.shape[1:] == (3, 32, 32)
        assert labels.shape[0] == 8
    
    def test_invalid_dataset_name(self, temp_data_dir):
        """Test invalid dataset name"""
        with pytest.raises(ValueError):
            CIFARDataset(
                dataset_name='invalid',
                data_root=temp_data_dir
            )


# ============================================================================
# Test Data Loading
# ============================================================================

class TestDataLoading:
    """Test data loading functionality"""
    
    def test_batch_shapes(self, temp_data_dir):
        """Test batch shapes are correct"""
        dataset = CIFARDataset(
            dataset_name='cifar10',
            data_root=temp_data_dir,
            download=True
        )
        
        loader = dataset.get_train_loader(batch_size=64, num_workers=0)
        images, labels = next(iter(loader))
        
        assert images.shape == (64, 3, 32, 32)
        assert labels.shape == (64,)
    
    def test_label_range(self, temp_data_dir):
        """Test labels are in correct range"""
        dataset = CIFARDataset(
            dataset_name='cifar10',
            data_root=temp_data_dir,
            download=True
        )
        
        loader = dataset.get_train_loader(batch_size=32, num_workers=0)
        _, labels = next(iter(loader))
        
        assert labels.min() >= 0
        assert labels.max() < 10
    
    def test_normalization(self, temp_data_dir):
        """Test images are normalized"""
        dataset = CIFARDataset(
            dataset_name='cifar10',
            data_root=temp_data_dir,
            download=True
        )
        
        loader = dataset.get_train_loader(batch_size=32, num_workers=0)
        images, _ = next(iter(loader))
        
        # Normalized images should have values roughly in [-3, 3]
        assert images.min() > -5
        assert images.max() < 5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
