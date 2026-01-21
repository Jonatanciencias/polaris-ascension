"""
Tests for Enhanced Inference Engine (Session 15)

Comprehensive test suite for:
- ModelCompressor
- AdaptiveBatchScheduler  
- MultiModelServer
- EnhancedInferenceEngine

Tests cover:
- Compression strategies and pipelines
- Dynamic batch scheduling
- Multi-model serving with resource management
- Integration with compute primitives
"""

import pytest
import numpy as np
import time
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.inference.enhanced import (
    EnhancedInferenceEngine,
    ModelCompressor,
    AdaptiveBatchScheduler,
    MultiModelServer,
    CompressionStrategy,
    CompressionConfig,
    CompressionResult,
    BatchRequest,
    BatchResponse,
    ModelStats,
)
from src.inference import InferenceConfig
from src.inference.model_loaders import ModelMetadata


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_loader():
    """Create a mock model loader"""
    loader = Mock()
    loader.load.return_value = ModelMetadata(
        name="test_model",
        framework="onnx",
        input_names=["input"],
        output_names=["output"],
        input_shapes=[(1, 3, 224, 224)],
        output_shapes=[(1, 1000)],
        input_dtypes=["float32"],
        output_dtypes=["float32"],
        file_size_mb=50.0,
        estimated_memory_mb=100.0,
        provider="CPUExecutionProvider",
        optimization_level="all",
        extra_info={}
    )
    loader.predict.return_value = {"output": np.random.randn(1, 1000)}
    loader.unload.return_value = None
    return loader


# ============================================================================
# ModelCompressor Tests
# ============================================================================

class TestModelCompressor:
    """Tests for ModelCompressor"""
    
    def test_compressor_initialization(self):
        """Test compressor initialization"""
        config = CompressionConfig(
            strategy=CompressionStrategy.QUANTIZE_SPARSE,
            target_sparsity=0.5,
            quantization_bits=8
        )
        
        compressor = ModelCompressor(config)
        
        assert compressor.config.strategy == CompressionStrategy.QUANTIZE_SPARSE
        assert compressor.config.target_sparsity == 0.5
        assert compressor.config.quantization_bits == 8
    
    def test_compression_none_strategy(self):
        """Test no compression strategy"""
        config = CompressionConfig(strategy=CompressionStrategy.NONE)
        compressor = ModelCompressor(config)
        
        model = {"weights": np.random.randn(100, 100)}
        compressed_model, result = compressor.compress(model)
        
        assert result.compression_ratio == 1.0
        assert result.sparsity_achieved == 0.0
        assert not result.quantization_applied
        assert result.memory_savings_mb == 0.0
    
    def test_compression_quantize_only(self):
        """Test quantization-only compression"""
        config = CompressionConfig(
            strategy=CompressionStrategy.QUANTIZE_ONLY,
            quantization_bits=8
        )
        compressor = ModelCompressor(config)
        
        model = {"weights": np.random.randn(100, 100)}
        calibration_data = np.random.randn(10, 100)
        
        compressed_model, result = compressor.compress(model, calibration_data)
        
        assert result.compression_ratio >= 1.0
        assert result.quantization_applied
    
    def test_compression_sparse_only(self):
        """Test sparsity-only compression"""
        config = CompressionConfig(
            strategy=CompressionStrategy.SPARSE_ONLY,
            target_sparsity=0.7
        )
        compressor = ModelCompressor(config)
        
        model = {"weights": np.random.randn(100, 100)}
        compressed_model, result = compressor.compress(model)
        
        assert result.compression_ratio >= 1.0
        assert result.sparsity_achieved > 0.0
    
    def test_compression_quantize_sparse(self):
        """Test combined quantization + sparsity compression"""
        config = CompressionConfig(
            strategy=CompressionStrategy.QUANTIZE_SPARSE,
            target_sparsity=0.5,
            quantization_bits=8
        )
        compressor = ModelCompressor(config)
        
        model = {"weights": np.random.randn(100, 100)}
        calibration_data = np.random.randn(10, 100)
        
        compressed_model, result = compressor.compress(model, calibration_data)
        
        assert result.compression_ratio > 1.0
        assert result.quantization_applied
        assert result.sparsity_achieved > 0.0
        assert result.memory_savings_mb > 0.0
    
    def test_compression_aggressive(self):
        """Test aggressive compression with all optimizations"""
        config = CompressionConfig(
            strategy=CompressionStrategy.AGGRESSIVE,
            target_sparsity=0.8,
            quantization_bits=8,
            enable_pruning=True
        )
        compressor = ModelCompressor(config)
        
        model = {"weights": np.random.randn(100, 100)}
        compressed_model, result = compressor.compress(model)
        
        assert result.compression_ratio > 1.0
        assert result.quantization_applied
        assert result.sparsity_achieved > 0.0
    
    def test_compression_with_validation(self):
        """Test compression with accuracy validation"""
        config = CompressionConfig(strategy=CompressionStrategy.QUANTIZE_SPARSE)
        compressor = ModelCompressor(config)
        
        model = {"weights": np.random.randn(100, 100)}
        
        # Mock validation function
        def validate(model):
            return 0.015  # 1.5% accuracy loss
        
        compressed_model, result = compressor.compress(
            model,
            validation_fn=validate
        )
        
        assert result.accuracy_impact == 0.015
    
    def test_compression_result_properties(self):
        """Test compression result properties"""
        result = CompressionResult(
            original_size_mb=100.0,
            compressed_size_mb=25.0,
            compression_ratio=4.0,
            sparsity_achieved=0.75,
            quantization_applied=True,
            inference_speedup=3.5,
            memory_savings_mb=75.0,
            accuracy_impact=0.01
        )
        
        assert result.original_size_mb == 100.0
        assert result.compressed_size_mb == 25.0
        assert result.compression_ratio == 4.0
        assert result.sparsity_achieved == 0.75
        assert result.quantization_applied
        assert result.inference_speedup == 3.5
        assert result.memory_savings_mb == 75.0
        assert result.accuracy_impact == 0.01


# ============================================================================
# AdaptiveBatchScheduler Tests
# ============================================================================

class TestAdaptiveBatchScheduler:
    """Tests for AdaptiveBatchScheduler"""
    
    def test_scheduler_initialization(self):
        """Test scheduler initialization"""
        scheduler = AdaptiveBatchScheduler(
            min_batch_size=1,
            max_batch_size=32,
            max_wait_ms=50.0,
            target_latency_ms=100.0
        )
        
        assert scheduler.min_batch_size == 1
        assert scheduler.max_batch_size == 32
        assert scheduler.max_wait_ms == 50.0
        assert scheduler.target_latency_ms == 100.0
        assert scheduler.current_batch_size == 1
    
    def test_scheduler_start_stop(self):
        """Test scheduler start and stop"""
        scheduler = AdaptiveBatchScheduler()
        
        assert not scheduler._running
        
        scheduler.start()
        assert scheduler._running
        
        time.sleep(0.1)  # Let scheduler run briefly
        
        scheduler.stop()
        assert not scheduler._running
    
    def test_submit_request(self):
        """Test submitting requests"""
        scheduler = AdaptiveBatchScheduler()
        
        request = BatchRequest(
            request_id="test_1",
            inputs=np.random.randn(1, 10),
            timestamp=time.time(),
            priority=0
        )
        
        scheduler.submit_request(request)
        assert scheduler.request_queue.qsize() == 1
    
    def test_batch_collection(self):
        """Test collecting requests into batches"""
        scheduler = AdaptiveBatchScheduler(
            min_batch_size=2,
            max_batch_size=4,
            max_wait_ms=100.0
        )
        
        # Submit multiple requests
        for i in range(3):
            request = BatchRequest(
                request_id=f"test_{i}",
                inputs=np.random.randn(1, 10),
                timestamp=time.time()
            )
            scheduler.submit_request(request)
        
        # Collect batch
        batch = scheduler._collect_batch()
        
        assert len(batch) >= scheduler.min_batch_size
        assert len(batch) <= scheduler.max_batch_size
    
    def test_batch_size_adaptation_increase(self):
        """Test batch size increases when latency is low"""
        scheduler = AdaptiveBatchScheduler(
            min_batch_size=2,
            max_batch_size=16,
            target_latency_ms=100.0
        )
        
        initial_batch_size = scheduler.current_batch_size
        
        # Simulate low latency
        for _ in range(20):
            scheduler.latency_history.append(50.0)  # Below target
        
        scheduler._adapt_batch_size()
        
        assert scheduler.current_batch_size >= initial_batch_size
    
    def test_batch_size_adaptation_decrease(self):
        """Test batch size decreases when latency is high"""
        scheduler = AdaptiveBatchScheduler(
            min_batch_size=2,
            max_batch_size=16,
            target_latency_ms=100.0
        )
        
        scheduler.current_batch_size = 16  # Start high
        
        # Simulate high latency
        for _ in range(20):
            scheduler.latency_history.append(200.0)  # Above target
        
        scheduler._adapt_batch_size()
        
        assert scheduler.current_batch_size < 16
    
    def test_scheduler_stats(self):
        """Test getting scheduler statistics"""
        scheduler = AdaptiveBatchScheduler()
        
        # Add some latency history
        scheduler.latency_history.extend([10.0, 20.0, 15.0])
        
        stats = scheduler.get_stats()
        
        assert 'current_batch_size' in stats
        assert 'queue_size' in stats
        assert 'avg_latency_ms' in stats
        assert stats['avg_latency_ms'] == 15.0
    
    def test_request_callback(self):
        """Test request callback functionality"""
        scheduler = AdaptiveBatchScheduler()
        
        callback_called = [False]
        response_data = [None]
        
        def callback(response: BatchResponse):
            callback_called[0] = True
            response_data[0] = response
        
        request = BatchRequest(
            request_id="test_callback",
            inputs=np.random.randn(1, 10),
            timestamp=time.time(),
            callback=callback
        )
        
        # Process batch directly
        scheduler._process_batch([request])
        
        assert callback_called[0]
        assert response_data[0] is not None
        assert response_data[0].request_id == "test_callback"


# ============================================================================
# ModelStats Tests
# ============================================================================

class TestModelStats:
    """Tests for ModelStats"""
    
    def test_stats_initialization(self):
        """Test stats initialization"""
        stats = ModelStats(model_name="test_model")
        
        assert stats.model_name == "test_model"
        assert stats.total_requests == 0
        assert stats.avg_latency_ms == 0.0
        assert stats.min_latency_ms == float('inf')
        assert stats.max_latency_ms == 0.0
    
    def test_stats_update(self):
        """Test updating statistics"""
        stats = ModelStats(model_name="test_model")
        
        stats.update(10.0)
        assert stats.total_requests == 1
        assert stats.avg_latency_ms == 10.0
        assert stats.min_latency_ms == 10.0
        assert stats.max_latency_ms == 10.0
        
        stats.update(20.0)
        assert stats.total_requests == 2
        assert stats.avg_latency_ms == 15.0
        assert stats.min_latency_ms == 10.0
        assert stats.max_latency_ms == 20.0
        
        stats.update(5.0)
        assert stats.total_requests == 3
        assert stats.avg_latency_ms == pytest.approx(11.67, rel=0.01)
        assert stats.min_latency_ms == 5.0
        assert stats.max_latency_ms == 20.0


# ============================================================================
# MultiModelServer Tests
# ============================================================================

class TestMultiModelServer:
    """Tests for MultiModelServer"""
    
    def test_server_initialization(self):
        """Test server initialization"""
        server = MultiModelServer(
            max_models=5,
            memory_limit_mb=1000.0
        )
        
        assert server.max_models == 5
        assert server.memory_limit_mb == 1000.0
        assert len(server.models) == 0
    
    @patch('src.inference.enhanced.create_loader')
    def test_load_model(self, mock_create_loader, mock_loader):
        """Test loading a model"""
        mock_create_loader.return_value = mock_loader
        server = MultiModelServer()
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as f:
            success = server.load_model(
                "test_model",
                f.name,
                enable_batching=False
            )
        
        assert success
        assert "test_model" in server.models
        assert "test_model" in server.model_stats
    
    @patch('src.inference.enhanced.create_loader')
    def test_load_duplicate_model(self, mock_create_loader, mock_loader):
        """Test loading duplicate model fails"""
        mock_create_loader.return_value = mock_loader
        server = MultiModelServer()
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as f:
            server.load_model("test_model", f.name)
            success = server.load_model("test_model", f.name)
        
        assert not success
    
    @patch('src.inference.enhanced.create_loader')
    def test_load_model_limit(self, mock_create_loader, mock_loader):
        """Test model limit enforcement"""
        mock_create_loader.return_value = mock_loader
        server = MultiModelServer(max_models=2)
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as f:
            server.load_model("model1", f.name)
            server.load_model("model2", f.name)
            success = server.load_model("model3", f.name)
        
        assert not success
        assert len(server.models) == 2
    
    @patch('src.inference.enhanced.create_loader')
    def test_unload_model(self, mock_create_loader, mock_loader):
        """Test unloading a model"""
        mock_create_loader.return_value = mock_loader
        server = MultiModelServer()
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as f:
            server.load_model("test_model", f.name)
            success = server.unload_model("test_model")
        
        assert success
        assert "test_model" not in server.models
    
    def test_unload_nonexistent_model(self):
        """Test unloading nonexistent model"""
        server = MultiModelServer()
        success = server.unload_model("nonexistent")
        
        assert not success
    
    @patch('src.inference.enhanced.create_loader')
    def test_predict(self, mock_create_loader, mock_loader):
        """Test prediction"""
        mock_create_loader.return_value = mock_loader
        server = MultiModelServer()
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as f:
            server.load_model("test_model", f.name, enable_batching=False)
            
            inputs = np.random.randn(1, 10)
            outputs = server.predict("test_model", inputs)
        
        assert outputs is not None
        assert "output" in outputs
    
    @patch('src.inference.enhanced.create_loader')
    def test_predict_with_batching(self, mock_create_loader, mock_loader):
        """Test prediction with batching enabled"""
        mock_create_loader.return_value = mock_loader
        server = MultiModelServer()
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as f:
            server.load_model("test_model", f.name, enable_batching=True)
            
            inputs = np.random.randn(1, 10)
            outputs = server.predict("test_model", inputs, timeout_ms=1000.0)
        
        assert outputs is not None
    
    def test_predict_nonexistent_model(self):
        """Test prediction on nonexistent model"""
        server = MultiModelServer()
        
        inputs = np.random.randn(1, 10)
        outputs = server.predict("nonexistent", inputs)
        
        assert outputs is None
    
    @patch('src.inference.enhanced.create_loader')
    def test_list_models(self, mock_create_loader, mock_loader):
        """Test listing models"""
        mock_create_loader.return_value = mock_loader
        server = MultiModelServer()
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as f:
            server.load_model("model1", f.name)
            server.load_model("model2", f.name)
        
        models = server.list_models()
        
        assert len(models) == 2
        assert "model1" in models
        assert "model2" in models
    
    @patch('src.inference.enhanced.create_loader')
    def test_get_model_stats(self, mock_create_loader, mock_loader):
        """Test getting model statistics"""
        mock_create_loader.return_value = mock_loader
        server = MultiModelServer()
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as f:
            server.load_model("test_model", f.name, enable_batching=False)
            
            # Run some predictions
            inputs = np.random.randn(1, 10)
            server.predict("test_model", inputs)
            server.predict("test_model", inputs)
        
        stats = server.get_model_stats("test_model")
        
        assert stats is not None
        assert stats.total_requests == 2
        assert stats.avg_latency_ms > 0.0
    
    @patch('src.inference.enhanced.create_loader')
    def test_get_all_stats(self, mock_create_loader, mock_loader):
        """Test getting all model statistics"""
        mock_create_loader.return_value = mock_loader
        server = MultiModelServer()
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as f:
            server.load_model("model1", f.name)
            server.load_model("model2", f.name)
        
        all_stats = server.get_all_stats()
        
        assert len(all_stats) == 2
        assert "model1" in all_stats
        assert "model2" in all_stats
    
    @patch('src.inference.enhanced.create_loader')
    def test_get_server_stats(self, mock_create_loader, mock_loader):
        """Test getting server statistics"""
        mock_create_loader.return_value = mock_loader
        server = MultiModelServer()
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as f:
            server.load_model("model1", f.name)
            server.load_model("model2", f.name)
        
        stats = server.get_server_stats()
        
        assert stats['num_models'] == 2
        assert 'total_memory_mb' in stats
        assert 'memory_limit_mb' in stats
        assert 'memory_usage_pct' in stats
        assert set(stats['models']) == {"model1", "model2"}
    
    @patch('src.inference.enhanced.create_loader')
    def test_lru_eviction(self, mock_create_loader, mock_loader):
        """Test LRU model eviction"""
        mock_create_loader.return_value = mock_loader
        server = MultiModelServer(max_models=2, memory_limit_mb=200.0)
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as f:
            server.load_model("model1", f.name)
            time.sleep(0.01)  # Ensure different timestamps
            
            server.load_model("model2", f.name)
            time.sleep(0.01)
            
            # Update model2's access time
            server.predict("model2", np.random.randn(1, 10))
            
            # Try to load model3 - should evict model1 (LRU)
            success = server.load_model("model3", f.name)
            
            # One should be evicted
            assert len(server.models) <= 2


# ============================================================================
# EnhancedInferenceEngine Tests
# ============================================================================

class TestEnhancedInferenceEngine:
    """Tests for EnhancedInferenceEngine"""
    
    def test_engine_initialization(self):
        """Test engine initialization"""
        engine = EnhancedInferenceEngine()
        
        assert engine.config is not None
        assert engine.compressor is not None
        assert engine.model_server is not None
        assert engine.gpu_manager is not None
        assert engine.memory_manager is not None
    
    def test_engine_with_custom_config(self):
        """Test engine with custom configuration"""
        inference_config = InferenceConfig(
            device='opencl',
            precision='fp16',
            batch_size=4
        )
        
        compression_config = CompressionConfig(
            strategy=CompressionStrategy.AGGRESSIVE,
            target_sparsity=0.8
        )
        
        engine = EnhancedInferenceEngine(
            config=inference_config,
            compression_config=compression_config
        )
        
        assert engine.config.device == 'opencl'
        assert engine.config.precision == 'fp16'
        assert engine.compression_config.strategy == CompressionStrategy.AGGRESSIVE
    
    @patch('src.inference.enhanced.create_loader')
    def test_load_and_optimize(self, mock_create_loader, mock_loader):
        """Test loading and optimizing a model"""
        mock_create_loader.return_value = mock_loader
        engine = EnhancedInferenceEngine()
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as f:
            calibration_data = np.random.randn(10, 10)
            
            success = engine.load_and_optimize(
                "test_model",
                f.name,
                calibration_data=calibration_data,
                enable_batching=True
            )
        
        assert success
        assert "test_model" in engine.model_server.models
    
    @patch('src.inference.enhanced.create_loader')
    def test_predict(self, mock_create_loader, mock_loader):
        """Test prediction with enhanced engine"""
        mock_create_loader.return_value = mock_loader
        engine = EnhancedInferenceEngine()
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as f:
            engine.load_and_optimize("test_model", f.name, enable_batching=False)
            
            inputs = np.random.randn(2, 10)
            outputs = engine.predict("test_model", inputs)
        
        assert outputs is not None
    
    @patch('src.inference.enhanced.create_loader')
    def test_predict_with_hybrid(self, mock_create_loader, mock_loader):
        """Test prediction with hybrid scheduling"""
        mock_create_loader.return_value = mock_loader
        engine = EnhancedInferenceEngine(enable_hybrid_scheduling=True)
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as f:
            engine.load_and_optimize("test_model", f.name)
            
            inputs = np.random.randn(1, 10)
            outputs = engine.predict("test_model", inputs, use_hybrid=True)
        
        assert outputs is not None
    
    @patch('src.inference.enhanced.create_loader')
    def test_get_stats(self, mock_create_loader, mock_loader):
        """Test getting comprehensive statistics"""
        mock_create_loader.return_value = mock_loader
        engine = EnhancedInferenceEngine()
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as f:
            engine.load_and_optimize("test_model", f.name)
            
            # Run some predictions
            inputs = np.random.randn(1, 10)
            engine.predict("test_model", inputs)
        
        stats = engine.get_stats()
        
        assert 'server' in stats
        assert 'models' in stats
        assert 'test_model' in stats['models']
    
    @patch('src.inference.enhanced.create_loader')
    def test_shutdown(self, mock_create_loader, mock_loader):
        """Test engine shutdown"""
        mock_create_loader.return_value = mock_loader
        engine = EnhancedInferenceEngine()
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as f:
            engine.load_and_optimize("test_model", f.name)
        
        engine.shutdown()
        
        # Verify all models unloaded
        assert len(engine.model_server.models) == 0
    
    @patch('src.inference.enhanced.create_loader')
    def test_multiple_models(self, mock_create_loader, mock_loader):
        """Test serving multiple models simultaneously"""
        mock_create_loader.return_value = mock_loader
        engine = EnhancedInferenceEngine()
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as f:
            engine.load_and_optimize("model1", f.name)
            engine.load_and_optimize("model2", f.name)
        
        inputs = np.random.randn(1, 10)
        
        output1 = engine.predict("model1", inputs)
        output2 = engine.predict("model2", inputs)
        
        assert output1 is not None
        assert output2 is not None
        
        stats = engine.get_stats()
        assert len(stats['models']) == 2


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for complete workflows"""
    
    @patch('src.inference.enhanced.create_loader')
    def test_end_to_end_workflow(self, mock_create_loader, mock_loader):
        """Test complete end-to-end workflow"""
        mock_create_loader.return_value = mock_loader
        
        # Create engine with aggressive compression
        compression_config = CompressionConfig(
            strategy=CompressionStrategy.QUANTIZE_SPARSE,
            target_sparsity=0.5,
            quantization_bits=8
        )
        
        engine = EnhancedInferenceEngine(
            compression_config=compression_config,
            enable_hybrid_scheduling=True
        )
        
        # Load and optimize model
        with tempfile.NamedTemporaryFile(suffix='.onnx') as f:
            calibration_data = np.random.randn(10, 10)
            success = engine.load_and_optimize(
                "production_model",
                f.name,
                calibration_data=calibration_data,
                enable_batching=True
            )
        
        assert success
        
        # Run predictions
        for _ in range(5):
            inputs = np.random.randn(2, 10)
            outputs = engine.predict("production_model", inputs)
            assert outputs is not None
        
        # Check statistics
        stats = engine.get_stats()
        model_stats = stats['models']['production_model']
        
        assert model_stats['total_requests'] == 5
        assert model_stats['avg_latency_ms'] > 0.0
        
        # Cleanup
        engine.shutdown()
    
    @patch('src.inference.enhanced.create_loader')
    def test_multi_model_production_scenario(self, mock_create_loader, mock_loader):
        """Test production scenario with multiple models"""
        mock_create_loader.return_value = mock_loader
        engine = EnhancedInferenceEngine()
        
        # Load 3 different models
        model_names = ["classifier", "detector", "segmenter"]
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as f:
            for name in model_names:
                success = engine.load_and_optimize(name, f.name)
                assert success
        
        # Run mixed workload
        for i in range(10):
            model_name = model_names[i % len(model_names)]
            inputs = np.random.randn(1, 10)
            outputs = engine.predict(model_name, inputs)
            assert outputs is not None
        
        # Verify all models were used
        stats = engine.get_stats()
        for name in model_names:
            assert name in stats['models']
            assert stats['models'][name]['total_requests'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
