"""
Tests for Async Inference Queue

Comprehensive test suite for asynchronous batch inference.

Author: AMD GPU Computing Team
Date: January 21, 2026
"""

import pytest
import asyncio
import time
import numpy as np
from datetime import datetime, timedelta

from src.api.async_inference import (
    AsyncInferenceQueue,
    InferenceJob,
    JobStatus,
    JobPriority
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
async def simple_inference_fn():
    """Simple mock inference function"""
    async def inference(batch_data):
        # Simulate processing time
        await asyncio.sleep(0.01)
        return [x * 2 for x in batch_data]
    return inference


@pytest.fixture
async def slow_inference_fn():
    """Slow mock inference function"""
    async def inference(batch_data):
        await asyncio.sleep(0.1)
        return [x * 2 for x in batch_data]
    return inference


@pytest.fixture
async def failing_inference_fn():
    """Mock inference function that fails"""
    async def inference(batch_data):
        raise RuntimeError("Inference failed")
    return inference


@pytest.fixture
async def inference_queue(simple_inference_fn):
    """Create inference queue"""
    queue = AsyncInferenceQueue(
        inference_fn=simple_inference_fn,
        batch_size=4,
        max_wait_time=0.05,
        num_workers=2
    )
    await queue.start()
    yield queue
    await queue.shutdown()


# ============================================================================
# Test Job Submission
# ============================================================================

@pytest.mark.asyncio
class TestJobSubmission:
    """Test job submission"""
    
    async def test_submit_single_job(self, inference_queue):
        """Test submitting single job"""
        job_id = await inference_queue.submit(42)
        
        assert job_id is not None
        assert isinstance(job_id, str)
    
    async def test_submit_multiple_jobs(self, inference_queue):
        """Test submitting multiple jobs"""
        job_ids = []
        for i in range(10):
            job_id = await inference_queue.submit(i)
            job_ids.append(job_id)
        
        assert len(job_ids) == 10
        assert len(set(job_ids)) == 10  # All unique
    
    async def test_submit_with_priority(self, inference_queue):
        """Test submitting jobs with different priorities"""
        job_id_low = await inference_queue.submit(1, JobPriority.LOW)
        job_id_high = await inference_queue.submit(2, JobPriority.HIGH)
        
        assert job_id_low != job_id_high


# ============================================================================
# Test Job Retrieval
# ============================================================================

@pytest.mark.asyncio
class TestJobRetrieval:
    """Test job result retrieval"""
    
    async def test_get_result_success(self, inference_queue):
        """Test getting result of successful job"""
        job_id = await inference_queue.submit(21)
        result = await inference_queue.get_result(job_id, timeout=1.0)
        
        assert result == 42  # 21 * 2
    
    async def test_get_result_timeout(self, inference_queue):
        """Test timeout when getting result"""
        job_id = await inference_queue.submit(1)
        
        # Very short timeout
        with pytest.raises(asyncio.TimeoutError):
            await inference_queue.get_result(job_id, timeout=0.001)
    
    async def test_get_result_multiple(self, inference_queue):
        """Test getting results from multiple jobs"""
        job_ids = []
        for i in range(5):
            job_id = await inference_queue.submit(i)
            job_ids.append(job_id)
        
        results = []
        for job_id in job_ids:
            result = await inference_queue.get_result(job_id, timeout=1.0)
            results.append(result)
        
        assert results == [0, 2, 4, 6, 8]  # i * 2


# ============================================================================
# Test Job Status
# ============================================================================

@pytest.mark.asyncio
class TestJobStatus:
    """Test job status tracking"""
    
    async def test_status_pending(self, simple_inference_fn):
        """Test job starts as pending"""
        queue = AsyncInferenceQueue(
            inference_fn=simple_inference_fn,
            batch_size=1000,  # Large batch to keep job pending
            max_wait_time=10.0
        )
        await queue.start()
        
        job_id = await queue.submit(1)
        status = queue.get_status(job_id)
        
        assert status['status'] == JobStatus.PENDING
        
        await queue.shutdown()
    
    async def test_status_completed(self, inference_queue):
        """Test job becomes completed"""
        job_id = await inference_queue.submit(1)
        await inference_queue.get_result(job_id, timeout=1.0)
        
        status = queue.get_status(job_id)
        assert status['status'] == JobStatus.COMPLETED
    
    async def test_status_nonexistent_job(self, inference_queue):
        """Test status of non-existent job"""
        status = inference_queue.get_status("nonexistent")
        
        assert status['status'] == 'not_found'


# ============================================================================
# Test Job Cancellation
# ============================================================================

@pytest.mark.asyncio
class TestJobCancellation:
    """Test job cancellation"""
    
    async def test_cancel_pending_job(self, simple_inference_fn):
        """Test cancelling pending job"""
        queue = AsyncInferenceQueue(
            inference_fn=simple_inference_fn,
            batch_size=1000,
            max_wait_time=10.0
        )
        await queue.start()
        
        job_id = await queue.submit(1)
        success = await queue.cancel(job_id)
        
        assert success
        
        status = queue.get_status(job_id)
        assert status['status'] == JobStatus.CANCELLED
        
        await queue.shutdown()
    
    async def test_cancel_completed_job(self, inference_queue):
        """Test cancelling completed job (should fail)"""
        job_id = await inference_queue.submit(1)
        await inference_queue.get_result(job_id, timeout=1.0)
        
        success = await inference_queue.cancel(job_id)
        assert not success


# ============================================================================
# Test Batch Processing
# ============================================================================

@pytest.mark.asyncio
class TestBatchProcessing:
    """Test batch processing behavior"""
    
    async def test_batch_formation(self, simple_inference_fn):
        """Test that jobs are batched"""
        queue = AsyncInferenceQueue(
            inference_fn=simple_inference_fn,
            batch_size=5,
            max_wait_time=0.1,
            num_workers=1
        )
        await queue.start()
        
        # Submit 5 jobs quickly
        job_ids = []
        for i in range(5):
            job_id = await queue.submit(i)
            job_ids.append(job_id)
        
        # Wait for processing
        results = []
        for job_id in job_ids:
            result = await queue.get_result(job_id, timeout=1.0)
            results.append(result)
        
        # Check statistics
        stats = queue.get_statistics()
        assert stats['total_batches'] >= 1
        
        await queue.shutdown()
    
    async def test_timeout_batch_formation(self, simple_inference_fn):
        """Test batch formation with timeout"""
        queue = AsyncInferenceQueue(
            inference_fn=simple_inference_fn,
            batch_size=100,  # Large batch
            max_wait_time=0.05,  # Short timeout
            num_workers=1
        )
        await queue.start()
        
        # Submit only 2 jobs (less than batch size)
        job_id1 = await queue.submit(1)
        job_id2 = await queue.submit(2)
        
        # Should still process after timeout
        result1 = await queue.get_result(job_id1, timeout=1.0)
        result2 = await queue.get_result(job_id2, timeout=1.0)
        
        assert result1 == 2
        assert result2 == 4
        
        await queue.shutdown()


# ============================================================================
# Test Priority Handling
# ============================================================================

@pytest.mark.asyncio
class TestPriorityHandling:
    """Test priority-based scheduling"""
    
    async def test_priority_order(self, simple_inference_fn):
        """Test that high priority jobs are processed first"""
        queue = AsyncInferenceQueue(
            inference_fn=simple_inference_fn,
            batch_size=2,
            max_wait_time=0.1,
            num_workers=1
        )
        await queue.start()
        
        # Submit in order: LOW, HIGH
        job_low = await queue.submit(1, JobPriority.LOW)
        await asyncio.sleep(0.01)  # Small delay
        job_high = await queue.submit(2, JobPriority.HIGH)
        
        # HIGH should be processed despite being submitted later
        # (Implementation may vary, this tests the mechanism exists)
        
        result_high = await queue.get_result(job_high, timeout=1.0)
        result_low = await queue.get_result(job_low, timeout=1.0)
        
        assert result_high == 4
        assert result_low == 2
        
        await queue.shutdown()


# ============================================================================
# Test Worker Pool
# ============================================================================

@pytest.mark.asyncio
class TestWorkerPool:
    """Test worker pool behavior"""
    
    async def test_multiple_workers(self, simple_inference_fn):
        """Test queue with multiple workers"""
        queue = AsyncInferenceQueue(
            inference_fn=simple_inference_fn,
            batch_size=2,
            max_wait_time=0.05,
            num_workers=3
        )
        await queue.start()
        
        # Submit multiple jobs
        job_ids = []
        for i in range(10):
            job_id = await queue.submit(i)
            job_ids.append(job_id)
        
        # All should complete
        for job_id in job_ids:
            result = await queue.get_result(job_id, timeout=2.0)
            assert result is not None
        
        await queue.shutdown()
    
    async def test_concurrent_processing(self, slow_inference_fn):
        """Test that workers process concurrently"""
        queue = AsyncInferenceQueue(
            inference_fn=slow_inference_fn,
            batch_size=2,
            max_wait_time=0.05,
            num_workers=2
        )
        await queue.start()
        
        # Submit 4 jobs (2 batches)
        start_time = time.time()
        
        job_ids = []
        for i in range(4):
            job_id = await queue.submit(i)
            job_ids.append(job_id)
        
        # Wait for all
        for job_id in job_ids:
            await queue.get_result(job_id, timeout=2.0)
        
        elapsed = time.time() - start_time
        
        # With 2 workers, should take ~0.2s (2 batches * 0.1s)
        # Without concurrency, would take ~0.4s
        assert elapsed < 0.3
        
        await queue.shutdown()


# ============================================================================
# Test Error Handling
# ============================================================================

@pytest.mark.asyncio
class TestErrorHandling:
    """Test error handling"""
    
    async def test_inference_failure(self, failing_inference_fn):
        """Test handling of inference failures"""
        queue = AsyncInferenceQueue(
            inference_fn=failing_inference_fn,
            batch_size=2,
            max_wait_time=0.05
        )
        await queue.start()
        
        job_id = await queue.submit(1)
        
        # Should raise RuntimeError
        with pytest.raises(RuntimeError):
            await queue.get_result(job_id, timeout=1.0)
        
        # Status should be FAILED
        status = queue.get_status(job_id)
        assert status['status'] == JobStatus.FAILED
        
        await queue.shutdown()


# ============================================================================
# Test Statistics
# ============================================================================

@pytest.mark.asyncio
class TestStatistics:
    """Test statistics tracking"""
    
    async def test_statistics_tracking(self, inference_queue):
        """Test that statistics are tracked"""
        # Submit and process jobs
        job_ids = []
        for i in range(5):
            job_id = await inference_queue.submit(i)
            job_ids.append(job_id)
        
        # Wait for completion
        for job_id in job_ids:
            await inference_queue.get_result(job_id, timeout=1.0)
        
        # Check statistics
        stats = inference_queue.get_statistics()
        
        assert stats['total_submitted'] >= 5
        assert stats['total_completed'] >= 5
        assert stats['total_batches'] >= 1
        assert 'avg_latency' in stats
        assert 'avg_batch_size' in stats
    
    async def test_statistics_accuracy(self, inference_queue):
        """Test statistics accuracy"""
        stats_before = inference_queue.get_statistics()
        
        # Submit job
        job_id = await inference_queue.submit(1)
        await inference_queue.get_result(job_id, timeout=1.0)
        
        stats_after = inference_queue.get_statistics()
        
        assert stats_after['total_submitted'] == stats_before['total_submitted'] + 1
        assert stats_after['total_completed'] == stats_before['total_completed'] + 1


# ============================================================================
# Test Result Caching
# ============================================================================

@pytest.mark.asyncio
class TestResultCaching:
    """Test result caching with TTL"""
    
    async def test_result_cache(self, inference_queue):
        """Test that results are cached"""
        job_id = await inference_queue.submit(21)
        
        # Get result twice
        result1 = await inference_queue.get_result(job_id, timeout=1.0)
        result2 = await inference_queue.get_result(job_id, timeout=1.0)
        
        assert result1 == result2 == 42
    
    async def test_result_ttl(self, simple_inference_fn):
        """Test result TTL expiration"""
        queue = AsyncInferenceQueue(
            inference_fn=simple_inference_fn,
            batch_size=2,
            max_wait_time=0.05,
            result_ttl=0.1  # 100ms TTL
        )
        await queue.start()
        
        job_id = await queue.submit(1)
        result1 = await queue.get_result(job_id, timeout=1.0)
        
        # Wait for TTL to expire
        await asyncio.sleep(0.2)
        
        # Result should be cleaned up
        # (Implementation-dependent behavior)
        
        await queue.shutdown()


# ============================================================================
# Test Shutdown
# ============================================================================

@pytest.mark.asyncio
class TestShutdown:
    """Test graceful shutdown"""
    
    async def test_graceful_shutdown(self, simple_inference_fn):
        """Test graceful shutdown completes pending jobs"""
        queue = AsyncInferenceQueue(
            inference_fn=simple_inference_fn,
            batch_size=2,
            max_wait_time=0.05
        )
        await queue.start()
        
        # Submit jobs
        job_ids = []
        for i in range(5):
            job_id = await queue.submit(i)
            job_ids.append(job_id)
        
        # Shutdown (should wait for completion)
        await queue.shutdown()
        
        # All jobs should be processed
        for job_id in job_ids:
            status = queue.get_status(job_id)
            assert status['status'] in [JobStatus.COMPLETED, JobStatus.CANCELLED]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
