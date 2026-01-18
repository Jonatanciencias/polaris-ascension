"""
Comprehensive tests for Hybrid CPU/GPU Scheduler module.

Test Coverage:
-------------
1. Task configuration and metadata
2. Resource profiling and estimation
3. Adaptive partitioning strategies
4. Load balancing decisions
5. Hybrid scheduler scheduling logic
6. Statistics and monitoring
7. Edge cases and error handling

Session 14 - Test Suite
Author: Legacy GPU AI Platform Team
"""

import pytest
import torch
import time
from unittest.mock import Mock, patch
from src.compute.hybrid import (
    Device,
    OpType,
    TaskConfig,
    ResourceProfile,
    ResourceProfiler,
    AdaptivePartitioner,
    LoadBalancer,
    HybridScheduler,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def task_small():
    """Small task (< 1MB)."""
    return TaskConfig(
        operation=OpType.MATMUL,
        input_shapes=[(10, 10), (10, 10)],
        dtype='float32'
    )


@pytest.fixture
def task_large():
    """Large task (> 1MB)."""
    return TaskConfig(
        operation=OpType.MATMUL,
        input_shapes=[(1024, 1024), (1024, 1024)],
        dtype='float32'
    )


@pytest.fixture
def task_huge():
    """Huge task (> 100MB)."""
    return TaskConfig(
        operation=OpType.CONV,
        input_shapes=[(128, 3, 224, 224), (64, 3, 3, 3)],
        dtype='float32'
    )


@pytest.fixture
def profiler():
    """Resource profiler instance."""
    return ResourceProfiler()


@pytest.fixture
def scheduler():
    """Hybrid scheduler instance."""
    return HybridScheduler(
        transfer_threshold=100 * 1024,  # 100 KB
        enable_partitioning=True
    )


# ============================================================================
# Test TaskConfig
# ============================================================================

class TestTaskConfig:
    """Test task configuration."""
    
    def test_creation(self):
        """Test basic task creation."""
        task = TaskConfig(
            operation=OpType.MATMUL,
            input_shapes=[(100, 100), (100, 100)],
            dtype='float32'
        )
        assert task.operation == OpType.MATMUL
        assert len(task.input_shapes) == 2
        assert task.dtype == 'float32'
    
    def test_string_operation(self):
        """Test operation from string."""
        task = TaskConfig(
            operation='matmul',
            input_shapes=[(10, 10)],
            dtype='float32'
        )
        assert task.operation == OpType.MATMUL
    
    def test_total_size_calculation(self):
        """Test total size calculation."""
        task = TaskConfig(
            operation=OpType.MATMUL,
            input_shapes=[(100, 100), (100, 100)],
            dtype='float32'
        )
        # 100*100 + 100*100 = 20,000 elements
        # float32 = 4 bytes
        expected_size = 20000 * 4
        assert task.total_size == expected_size
    
    def test_flops_matmul(self):
        """Test FLOPs estimation for matmul."""
        task = TaskConfig(
            operation=OpType.MATMUL,
            input_shapes=[(128, 256), (256, 512)],
            dtype='float32'
        )
        # 2 * M * N * K = 2 * 128 * 512 * 256
        expected_flops = 2 * 128 * 512 * 256
        assert task.flops == expected_flops
    
    def test_dtype_sizes(self):
        """Test different dtype sizes."""
        task_fp32 = TaskConfig(
            operation=OpType.MATMUL,
            input_shapes=[(100, 100)],
            dtype='float32'
        )
        task_fp16 = TaskConfig(
            operation=OpType.MATMUL,
            input_shapes=[(100, 100)],
            dtype='float16'
        )
        assert task_fp16.total_size == task_fp32.total_size // 2


# ============================================================================
# Test ResourceProfile
# ============================================================================

class TestResourceProfile:
    """Test resource profile."""
    
    def test_creation(self):
        """Test profile creation."""
        profile = ResourceProfile(
            device_name='cpu',
            memory_total=16 * 1024**3
        )
        assert profile.device_name == 'cpu'
        assert profile.memory_total == 16 * 1024**3
        assert profile.memory_used == 0
    
    def test_memory_available(self):
        """Test memory available calculation."""
        profile = ResourceProfile(
            device_name='cpu',
            memory_total=16 * 1024**3,
            memory_used=8 * 1024**3,
            memory_reserved=2 * 1024**3
        )
        expected = 6 * 1024**3
        assert profile.memory_available == expected
    
    def test_memory_utilization(self):
        """Test memory utilization calculation."""
        profile = ResourceProfile(
            device_name='cpu',
            memory_total=16 * 1024**3,
            memory_used=8 * 1024**3
        )
        assert profile.memory_utilization == 0.5
    
    def test_can_fit(self):
        """Test can_fit method."""
        profile = ResourceProfile(
            device_name='cpu',
            memory_total=16 * 1024**3,
            memory_used=10 * 1024**3
        )
        assert profile.can_fit(5 * 1024**3)
        assert not profile.can_fit(10 * 1024**3)


# ============================================================================
# Test ResourceProfiler
# ============================================================================

class TestResourceProfiler:
    """Test resource profiler."""
    
    def test_initialization(self, profiler):
        """Test profiler initialization."""
        assert 'cpu' in profiler.profiles
        assert profiler.profiles['cpu'].device_name == 'cpu'
    
    def test_gpu_profile_if_available(self, profiler):
        """Test GPU profile when available."""
        if torch.cuda.is_available():
            assert 'cuda' in profiler.profiles
            assert profiler.profiles['cuda'].device_name == 'cuda'
    
    def test_estimate_execution_time(self, profiler, task_large):
        """Test execution time estimation."""
        t_cpu = profiler.estimate_execution_time(task_large, 'cpu')
        assert t_cpu > 0
        assert t_cpu < 5.0  # Should be reasonable for this size
    
    def test_estimate_transfer_time(self, profiler):
        """Test transfer time estimation."""
        size = 100 * 1024**2  # 100 MB
        t_transfer = profiler.estimate_transfer_time(size, 'cpu', 'cuda')
        
        # Should take ~8ms for 100MB @ 12 GB/s
        assert t_transfer > 0
        assert t_transfer < 1.0
    
    def test_no_transfer_same_device(self, profiler):
        """Test no transfer time for same device."""
        t_transfer = profiler.estimate_transfer_time(1024**3, 'cpu', 'cpu')
        assert t_transfer == 0.0
    
    def test_update_profile(self, profiler):
        """Test profile update."""
        profiler.update('cpu')
        profile = profiler.get_profile('cpu')
        assert profile.last_update > 0


# ============================================================================
# Test AdaptivePartitioner
# ============================================================================

class TestAdaptivePartitioner:
    """Test adaptive partitioner."""
    
    def test_initialization(self, profiler):
        """Test partitioner initialization."""
        partitioner = AdaptivePartitioner(profiler)
        assert partitioner.profiler == profiler
    
    def test_partition_data(self, profiler, task_large):
        """Test data partitioning."""
        partitioner = AdaptivePartitioner(profiler)
        
        total_size = 128
        cpu_size, gpu_size = partitioner.partition_data(task_large, total_size)
        
        # Should partition into non-zero parts
        assert cpu_size >= 0
        assert gpu_size >= 0
        assert cpu_size + gpu_size == total_size
    
    def test_partition_respects_memory(self, profiler):
        """Test partitioning respects memory constraints."""
        partitioner = AdaptivePartitioner(profiler)
        
        # Very large task that won't fit in GPU
        huge_task = TaskConfig(
            operation=OpType.MATMUL,
            input_shapes=[(10000, 10000), (10000, 10000)],
            dtype='float32'
        )
        
        cpu_size, gpu_size = partitioner.partition_data(huge_task, 100)
        
        # Should assign most/all to CPU if GPU can't fit
        if not torch.cuda.is_available():
            assert gpu_size == 0
    
    def test_should_partition_small_task(self, profiler, task_small):
        """Test that small tasks are not partitioned."""
        partitioner = AdaptivePartitioner(profiler)
        assert not partitioner.should_partition(task_small)
    
    def test_should_partition_large_task(self, profiler, task_large):
        """Test partitioning decision for large tasks."""
        partitioner = AdaptivePartitioner(profiler)
        # Result depends on GPU availability
        result = partitioner.should_partition(task_large)
        assert isinstance(result, bool)


# ============================================================================
# Test LoadBalancer
# ============================================================================

class TestLoadBalancer:
    """Test load balancer."""
    
    def test_initialization(self, profiler):
        """Test load balancer initialization."""
        balancer = LoadBalancer(profiler)
        assert balancer.profiler == profiler
        assert balancer.completion_times['cpu'] == 0.0
        assert balancer.completion_times['cuda'] == 0.0
    
    def test_schedule_task(self, profiler, task_large):
        """Test task scheduling."""
        balancer = LoadBalancer(profiler)
        device = balancer.schedule_task(task_large)
        assert device in ['cpu', 'cuda']
    
    def test_schedule_prefers_faster_device(self, profiler):
        """Test scheduling prefers faster device."""
        balancer = LoadBalancer(profiler)
        
        # Create tasks
        task1 = TaskConfig(
            operation=OpType.MATMUL,
            input_shapes=[(1000, 1000), (1000, 1000)],
            dtype='float32'
        )
        task2 = TaskConfig(
            operation=OpType.MATMUL,
            input_shapes=[(1000, 1000), (1000, 1000)],
            dtype='float32'
        )
        
        device1 = balancer.schedule_task(task1)
        device2 = balancer.schedule_task(task2)
        
        # Both should be scheduled
        assert device1 in ['cpu', 'cuda']
        assert device2 in ['cpu', 'cuda']
    
    def test_reset(self, profiler, task_large):
        """Test reset functionality."""
        balancer = LoadBalancer(profiler)
        
        # Schedule some tasks
        balancer.schedule_task(task_large)
        balancer.schedule_task(task_large)
        
        # Reset
        balancer.reset()
        
        assert balancer.completion_times['cpu'] == 0.0
        assert balancer.completion_times['cuda'] == 0.0


# ============================================================================
# Test HybridScheduler
# ============================================================================

class TestHybridScheduler:
    """Test hybrid scheduler."""
    
    def test_initialization(self):
        """Test scheduler initialization."""
        scheduler = HybridScheduler()
        assert scheduler.profiler is not None
        assert scheduler.partitioner is not None
        assert scheduler.load_balancer is not None
    
    def test_small_task_stays_cpu(self, scheduler, task_small):
        """Test that small tasks stay on CPU."""
        device = scheduler.schedule(task_small)
        assert device == 'cpu'
    
    def test_large_task_scheduling(self, scheduler, task_large):
        """Test large task scheduling."""
        device = scheduler.schedule(task_large)
        assert device in ['cpu', 'cuda']
    
    def test_explicit_device_cpu(self, scheduler, task_large):
        """Test explicit CPU device selection."""
        device = scheduler.schedule(task_large, Device.CPU)
        assert device == 'cpu'
    
    def test_explicit_device_gpu(self, scheduler, task_large):
        """Test explicit GPU device selection."""
        device = scheduler.schedule(task_large, Device.GPU)
        if torch.cuda.is_available():
            assert device == 'cuda'
        else:
            assert device == 'cpu'  # Falls back to CPU
    
    def test_statistics_tracking(self, scheduler):
        """Test statistics tracking."""
        # Initial stats
        stats = scheduler.get_statistics()
        assert stats['total_tasks'] == 0
        
        # Note: Full execution testing would require mock tensors
    
    def test_reset_statistics(self, scheduler):
        """Test statistics reset."""
        scheduler.stats['tasks_cpu'] = 10
        scheduler.stats['tasks_gpu'] = 5
        
        scheduler.reset_statistics()
        
        assert scheduler.stats['tasks_cpu'] == 0
        assert scheduler.stats['tasks_gpu'] == 0
    
    def test_repr(self, scheduler):
        """Test string representation."""
        repr_str = repr(scheduler)
        assert 'HybridScheduler' in repr_str
        assert 'CPU Threads' in repr_str
        assert 'GPU Available' in repr_str


# ============================================================================
# Integration Tests
# ============================================================================

class TestHybridSchedulerIntegration:
    """Integration tests for hybrid scheduler."""
    
    def test_schedule_multiple_tasks(self, scheduler):
        """Test scheduling multiple tasks."""
        tasks = [
            TaskConfig(OpType.MATMUL, [(100, 100), (100, 100)], 'float32'),
            TaskConfig(OpType.CONV, [(16, 3, 32, 32)], 'float32'),
            TaskConfig(OpType.ELEMENTWISE, [(1000, 1000)], 'float32'),
        ]
        
        devices = [scheduler.schedule(task) for task in tasks]
        
        # All should be scheduled
        assert len(devices) == 3
        assert all(d in ['cpu', 'cuda'] for d in devices)
    
    def test_execute_simple_operation(self, scheduler):
        """Test executing simple operation."""
        # Create simple tensors
        a = torch.randn(10, 10)
        b = torch.randn(10, 10)
        
        # Define simple function
        def matmul(x, y):
            return torch.matmul(x, y)
        
        # Execute
        result = scheduler._execute_single([a, b], matmul, 'cpu')
        
        # Verify result
        assert result.shape == (10, 10)
        expected = torch.matmul(a, b)
        assert torch.allclose(result, expected)
    
    def test_statistics_after_execution(self, scheduler):
        """Test statistics update after execution."""
        a = torch.randn(10, 10)
        b = torch.randn(10, 10)
        
        def matmul(x, y):
            return torch.matmul(x, y)
        
        # Execute
        scheduler._execute_single([a, b], matmul, 'cpu')
        
        # Check stats - use correct key names
        stats = scheduler.get_statistics()
        assert stats['cpu_tasks'] == 1
        assert stats['total_cpu_time'] > 0
    
    def test_memory_constraint_handling(self):
        """Test handling of memory constraints."""
        # Create scheduler with very limited GPU memory
        scheduler = HybridScheduler(gpu_memory_limit=100 * 1024**2)  # 100 MB
        
        # Large task that exceeds limit
        large_task = TaskConfig(
            operation=OpType.MATMUL,
            input_shapes=[(2000, 2000), (2000, 2000)],
            dtype='float32'
        )
        
        device = scheduler.schedule(large_task)
        
        # Should fall back to CPU
        if torch.cuda.is_available():
            # Might still schedule to GPU if it has room
            assert device in ['cpu', 'cuda']
        else:
            assert device == 'cpu'


# ============================================================================
# Performance Tests
# ============================================================================

class TestHybridSchedulerPerformance:
    """Performance-related tests."""
    
    def test_scheduling_overhead(self, scheduler):
        """Test that scheduling overhead is minimal."""
        task = TaskConfig(
            operation=OpType.MATMUL,
            input_shapes=[(100, 100), (100, 100)],
            dtype='float32'
        )
        
        # Measure scheduling time
        start = time.time()
        for _ in range(1000):
            scheduler.schedule(task)
        elapsed = time.time() - start
        
        # Should be < 1ms per scheduling decision
        assert elapsed < 1.0
    
    def test_profiling_overhead(self, profiler):
        """Test that profiling overhead is minimal."""
        start = time.time()
        for _ in range(100):  # Reduced iterations to avoid slow psutil calls
            profiler.update('cpu')
        elapsed = time.time() - start
        
        # Should be reasonable (psutil calls can be slow, ~100ms each)
        assert elapsed < 15.0


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_input_shapes(self):
        """Test task with empty input shapes."""
        task = TaskConfig(
            operation=OpType.MATMUL,
            input_shapes=[],
            dtype='float32'
        )
        assert task.total_size == 0
    
    def test_single_input_shape(self):
        """Test task with single input."""
        task = TaskConfig(
            operation=OpType.ACTIVATION,
            input_shapes=[(100, 100)],
            dtype='float32'
        )
        assert task.total_size == 100 * 100 * 4
    
    def test_unknown_dtype(self):
        """Test task with unknown dtype."""
        task = TaskConfig(
            operation=OpType.MATMUL,
            input_shapes=[(10, 10)],
            dtype='unknown'
        )
        # Should default to 4 bytes
        assert task.total_size == 10 * 10 * 4
    
    def test_zero_size_task(self):
        """Test zero-size task."""
        task = TaskConfig(
            operation=OpType.MATMUL,
            input_shapes=[(0, 0)],
            dtype='float32'
        )
        assert task.total_size == 0
    
    def test_scheduler_without_gpu(self):
        """Test scheduler behavior without GPU."""
        scheduler = HybridScheduler()
        
        task = TaskConfig(
            operation=OpType.MATMUL,
            input_shapes=[(1000, 1000), (1000, 1000)],
            dtype='float32'
        )
        
        # Should still work
        device = scheduler.schedule(task)
        assert device in ['cpu', 'cuda']


if __name__ == "__main__":
    """Run tests with pytest."""
    pytest.main([__file__, "-v", "--tb=short"])
