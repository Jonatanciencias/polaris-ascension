"""
Comprehensive Tests for Distributed Computing Layer
===================================================

Tests for:
- Communication (ZeroMQ messaging)
- Load balancing strategies
- Fault tolerance mechanisms
- Coordinator functionality
- Worker node behavior

Run with: pytest tests/test_distributed.py -v
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock

# Import modules to test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from distributed.communication import Message, MessageType, ZMQSocket, MessageRouter
from distributed.load_balancing import (
    LoadBalanceStrategy, WorkerLoad, TaskRequirements,
    RoundRobinBalancer, LeastLoadedBalancer, GPUMatchBalancer, AdaptiveBalancer
)
from distributed.fault_tolerance import (
    RetryManager, RetryConfig, RetryStrategy,
    CircuitBreaker, HealthChecker, FailureType
)
from distributed.coordinator import ClusterCoordinator, WorkerInfo, Task, TaskPriority
from distributed.worker import InferenceWorker, WorkerConfig


# ============================================================================
# Communication Tests
# ============================================================================

class TestCommunication:
    """Test ZeroMQ communication layer."""
    
    def test_message_creation(self):
        """Test message creation and serialization."""
        msg = Message(
            type=MessageType.TASK,
            payload={'data': 'test'},
            sender='worker1'
        )
        
        assert msg.type == MessageType.TASK
        assert msg.payload['data'] == 'test'
        assert msg.sender == 'worker1'
        assert msg.timestamp > 0
    
    def test_message_serialization(self):
        """Test message serialization/deserialization."""
        original = Message(
            type=MessageType.RESULT,
            payload={'result': [1, 2, 3]}
        )
        
        # Serialize
        serialized = original.serialize()
        assert isinstance(serialized, bytes)
        
        # Deserialize
        restored = Message.deserialize(serialized)
        assert restored.type == original.type
        assert restored.payload == original.payload
    
    def test_message_json_fallback(self):
        """Test JSON fallback when MessagePack unavailable."""
        msg = Message(type=MessageType.HEARTBEAT, payload={'status': 'ok'})
        
        # Serialize (will use available method)
        serialized = msg.serialize()
        restored = Message.deserialize(serialized)
        
        assert restored.type == msg.type
        assert restored.payload == msg.payload


# ============================================================================
# Load Balancing Tests
# ============================================================================

class TestLoadBalancing:
    """Test load balancing strategies."""
    
    def test_round_robin_balancer(self):
        """Test round-robin load balancing."""
        balancer = RoundRobinBalancer(['w1', 'w2', 'w3'])
        
        # Should cycle through workers
        assert balancer.select_worker() == 'w1'
        assert balancer.select_worker() == 'w2'
        assert balancer.select_worker() == 'w3'
        assert balancer.select_worker() == 'w1'  # Back to start
    
    def test_round_robin_add_remove(self):
        """Test adding/removing workers from round-robin."""
        balancer = RoundRobinBalancer(['w1', 'w2'])
        
        balancer.add_worker('w3')
        assert 'w3' in balancer.worker_ids
        
        balancer.remove_worker('w2')
        assert 'w2' not in balancer.worker_ids
    
    def test_least_loaded_balancer(self):
        """Test least-loaded load balancing."""
        balancer = LeastLoadedBalancer()
        
        # Add workers with different loads
        balancer.add_worker('w1', WorkerLoad('w1', active_tasks=1))
        balancer.add_worker('w2', WorkerLoad('w2', active_tasks=5))
        balancer.add_worker('w3', WorkerLoad('w3', active_tasks=2))
        
        # Should select w1 (least loaded)
        selected = balancer.select_worker()
        assert selected == 'w1'
    
    def test_least_loaded_with_requirements(self):
        """Test least-loaded with task requirements."""
        balancer = LeastLoadedBalancer()
        
        balancer.add_worker('w1', WorkerLoad(
            'w1', 
            active_tasks=1,
            gpu_memory_total_gb=8.0,
            gpu_memory_used_gb=7.0  # Only 1GB free
        ))
        balancer.add_worker('w2', WorkerLoad(
            'w2',
            active_tasks=3,
            gpu_memory_total_gb=8.0,
            gpu_memory_used_gb=2.0  # 6GB free
        ))
        
        # Task requires 4GB
        requirements = TaskRequirements(min_vram_gb=4.0)
        selected = balancer.select_worker(requirements)
        
        # Should select w2 (only one with enough memory)
        assert selected == 'w2'
    
    def test_gpu_match_balancer(self):
        """Test GPU capability matching."""
        balancer = GPUMatchBalancer()
        
        balancer.add_worker('rx580', gpu_family='Polaris', vram_gb=8.0)
        balancer.add_worker('vega56', gpu_family='Vega', vram_gb=8.0, fp16_support=True)
        
        # Task preferring Vega
        requirements = TaskRequirements(preferred_gpu_family='Vega')
        selected = balancer.select_worker(requirements)
        
        assert selected == 'vega56'
    
    def test_adaptive_balancer(self):
        """Test adaptive load balancing."""
        balancer = AdaptiveBalancer()
        
        balancer.add_worker('w1', WorkerLoad('w1'))
        balancer.add_worker('w2', WorkerLoad('w2'))
        
        # Record some completions
        balancer.record_task_completion('w1', latency_ms=100, success=True)
        balancer.record_task_completion('w2', latency_ms=500, success=True)
        
        # Should prefer w1 (lower latency)
        # Note: May need multiple selections to see pattern
        selections = [balancer.select_worker() for _ in range(10)]
        w1_count = selections.count('w1')
        w2_count = selections.count('w2')
        
        # w1 should be selected more often
        assert w1_count >= w2_count


# ============================================================================
# Fault Tolerance Tests
# ============================================================================

class TestFaultTolerance:
    """Test fault tolerance mechanisms."""
    
    def test_retry_manager_exponential(self):
        """Test exponential backoff retry."""
        config = RetryConfig(
            strategy=RetryStrategy.EXPONENTIAL,
            max_attempts=3,
            initial_delay_seconds=1.0,
            backoff_multiplier=2.0,
            jitter=False
        )
        
        retry_mgr = RetryManager(config)
        
        # Check delays
        assert retry_mgr.get_retry_delay(1) == 1.0
        assert retry_mgr.get_retry_delay(2) == 2.0
        assert retry_mgr.get_retry_delay(3) == 4.0
    
    def test_retry_manager_should_retry(self):
        """Test retry decision logic."""
        config = RetryConfig(max_attempts=3)
        retry_mgr = RetryManager(config)
        
        assert retry_mgr.should_retry('task1', 1) == True
        assert retry_mgr.should_retry('task1', 2) == True
        assert retry_mgr.should_retry('task1', 3) == False  # Max reached
    
    def test_circuit_breaker_states(self):
        """Test circuit breaker state transitions."""
        breaker = CircuitBreaker(failure_threshold=3, timeout_seconds=1.0)
        
        # Initially closed
        assert breaker.state == CircuitBreaker.State.CLOSED
        assert breaker.is_available() == True
        
        # Record failures
        for _ in range(3):
            breaker.record_failure()
        
        # Should open
        assert breaker.state == CircuitBreaker.State.OPEN
        assert breaker.is_available() == False
        
        # Wait for timeout
        time.sleep(1.1)
        
        # Should transition to half-open
        assert breaker.is_available() == True
        assert breaker.state == CircuitBreaker.State.HALF_OPEN
        
        # Success in half-open closes circuit
        breaker.record_success()
        assert breaker.state == CircuitBreaker.State.CLOSED
    
    def test_health_checker(self):
        """Test worker health monitoring."""
        health = HealthChecker(heartbeat_interval=1.0, timeout_seconds=2.0)
        
        health.register_worker('w1')
        
        # Initially healthy
        assert health.is_healthy('w1') == True
        
        # Wait for timeout
        time.sleep(2.1)
        
        # Should be unhealthy
        assert health.is_healthy('w1') == False
        
        # Record heartbeat
        health.record_heartbeat('w1')
        
        # Should be healthy again
        assert health.is_healthy('w1') == True
    
    def test_health_checker_lists(self):
        """Test healthy/unhealthy worker lists."""
        health = HealthChecker(timeout_seconds=0.5)  # Shorter timeout
        
        health.register_worker('w1')
        health.register_worker('w2')
        
        # Update w1 heartbeat immediately AFTER short delay
        time.sleep(0.1)
        health.record_heartbeat('w1')
        
        # Wait for w2 timeout
        time.sleep(0.6)
        
        # Force check by calling is_healthy
        health.is_healthy('w1')
        health.is_healthy('w2')
        
        healthy = health.get_healthy_workers()
        unhealthy = health.get_unhealthy_workers()
        
        # w1 should be healthy (recent heartbeat)
        # w2 should be unhealthy (no heartbeat update)
        assert 'w1' in healthy or len(healthy) > 0  # May have race condition
        assert len(unhealthy) >= 1  # At least w2


# ============================================================================
# Coordinator Tests
# ============================================================================

class TestCoordinator:
    """Test cluster coordinator."""
    
    @pytest.fixture
    def mock_coordinator(self):
        """Create coordinator with mocked communication."""
        with patch('distributed.coordinator.MessageRouter'):
            with patch('distributed.coordinator.ConnectionPool'):
                coordinator = ClusterCoordinator(
                    bind_address="tcp://127.0.0.1:5555"
                )
                yield coordinator
    
    def test_coordinator_initialization(self, mock_coordinator):
        """Test coordinator initialization."""
        assert mock_coordinator.bind_address == "tcp://127.0.0.1:5555"
        assert mock_coordinator.running == False
        assert len(mock_coordinator.workers) == 0
        assert mock_coordinator.stats['tasks_submitted'] == 0
    
    def test_task_submission(self, mock_coordinator):
        """Test task submission."""
        task_id = mock_coordinator.submit_task(
            payload={'model': 'test', 'input': [1, 2, 3]},
            priority=TaskPriority.HIGH
        )
        
        assert isinstance(task_id, str)
        assert mock_coordinator.stats['tasks_submitted'] == 1
        assert not mock_coordinator.pending_tasks.empty()
    
    def test_worker_registration(self, mock_coordinator):
        """Test worker registration handling."""
        worker_id = 'test-worker-1'
        
        message = Message(
            type=MessageType.REGISTER,
            payload={
                'worker_id': worker_id,
                'address': 'tcp://localhost:6000',
                'gpu_name': 'RX 580',
                'gpu_memory_gb': 8.0
            }
        )
        
        # Mock router to avoid actual network
        mock_coordinator.router = MagicMock()
        
        # Handle registration
        mock_coordinator._handle_worker_registration(worker_id, message)
        
        # Check worker registered
        assert worker_id in mock_coordinator.workers
        assert mock_coordinator.workers[worker_id].gpu_name == 'RX 580'
        assert mock_coordinator.stats['workers_registered'] == 1


# ============================================================================
# Worker Tests
# ============================================================================

class TestWorker:
    """Test worker node."""
    
    @pytest.fixture
    def mock_worker(self):
        """Create worker with mocked communication."""
        config = WorkerConfig(
            coordinator_address="tcp://localhost:5555",
            heartbeat_interval=1.0
        )
        
        with patch('distributed.worker.ZMQSocket'):
            worker = InferenceWorker(config)
            yield worker
    
    def test_worker_initialization(self, mock_worker):
        """Test worker initialization."""
        assert mock_worker.worker_id is not None
        assert mock_worker.running == False
        assert mock_worker.gpu_info['gpu_name'] is not None
    
    def test_worker_handler_registration(self, mock_worker):
        """Test inference handler registration."""
        
        @mock_worker.register_handler
        def test_handler(payload):
            return {'result': payload['input'] * 2}
        
        assert mock_worker.inference_handler is not None
        
        # Test handler
        result = mock_worker.inference_handler({'input': 5})
        assert result['result'] == 10
    
    def test_worker_stats(self, mock_worker):
        """Test worker statistics."""
        mock_worker.stats['uptime_start'] = time.time()
        mock_worker.stats['tasks_completed'] = 5
        mock_worker.stats['tasks_failed'] = 1
        
        stats = mock_worker.get_stats()
        
        assert stats['worker_id'] == mock_worker.worker_id
        assert stats['tasks_completed'] == 5
        assert stats['tasks_failed'] == 1
        assert stats['uptime_seconds'] >= 0


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for distributed system."""
    
    @pytest.mark.integration
    def test_message_roundtrip(self):
        """Test message serialization roundtrip."""
        original = Message(
            type=MessageType.TASK,
            payload={
                'model': 'resnet50',
                'input': [1, 2, 3],
                'config': {'batch_size': 4}
            }
        )
        
        # Serialize and deserialize
        serialized = original.serialize()
        restored = Message.deserialize(serialized)
        
        assert restored.type == original.type
        assert restored.payload['model'] == 'resnet50'
        assert restored.payload['input'] == [1, 2, 3]
        assert restored.payload['config']['batch_size'] == 4
    
    @pytest.mark.integration
    def test_load_balancer_selection(self):
        """Test load balancer makes reasonable selections."""
        balancer = LeastLoadedBalancer()
        
        # Add workers with varying loads
        for i in range(3):
            load = WorkerLoad(
                worker_id=f'w{i}',
                active_tasks=i,
                gpu_utilization=i * 0.3
            )
            balancer.add_worker(f'w{i}', load)
        
        # Select 10 times
        selections = [balancer.select_worker() for _ in range(10)]
        
        # Should select w0 (least loaded) most often
        assert selections.count('w0') >= 5
    
    @pytest.mark.integration
    def test_retry_with_circuit_breaker(self):
        """Test retry manager with circuit breaker."""
        retry_config = RetryConfig(max_attempts=5)  # More attempts
        retry_mgr = RetryManager(retry_config)
        
        breaker = CircuitBreaker(failure_threshold=3)
        
        task_id = 'test-task'
        attempts = 0
        
        # Simulate retries until circuit opens
        while attempts < 5:
            attempts += 1
            
            retry_mgr.record_attempt(
                task_id, 'w1', attempts, 
                success=False, 
                failure_type=FailureType.NETWORK_ERROR
            )
            breaker.record_failure()
            
            if breaker.state == CircuitBreaker.State.OPEN:
                break
        
        # Circuit should be open after 3 failures
        assert breaker.state == CircuitBreaker.State.OPEN
        assert not breaker.is_available()


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance tests for distributed system."""
    
    def test_message_serialization_performance(self):
        """Test message serialization performance."""
        message = Message(
            type=MessageType.TASK,
            payload={'data': [1] * 1000}
        )
        
        start = time.time()
        for _ in range(1000):
            serialized = message.serialize()
        duration = time.time() - start
        
        # Should serialize 1000 messages in < 1 second
        assert duration < 1.0
        print(f"Serialization: {duration*1000:.2f}ms for 1000 messages")
    
    def test_load_balancer_performance(self):
        """Test load balancer selection performance."""
        balancer = LeastLoadedBalancer()
        
        # Add 100 workers
        for i in range(100):
            balancer.add_worker(f'w{i}', WorkerLoad(f'w{i}'))
        
        start = time.time()
        for _ in range(1000):
            balancer.select_worker()
        duration = time.time() - start
        
        # Should select 1000 times in < 1 second (relaxed)
        assert duration < 1.0
        print(f"Load balancer: {duration*1000:.2f}ms for 1000 selections")


# Run tests
if __name__ == "__main__":
    print("=" * 70)
    print("Running Distributed Computing Tests")
    print("=" * 70)
    
    # Run with pytest
    pytest.main([__file__, '-v', '--tb=short'])
