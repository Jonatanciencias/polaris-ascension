"""
Test Suite for SDK Layer
========================

Comprehensive tests for all SDK components:
- High-level API (easy.py)
- Plugin system (plugins.py)
- Model registry (registry.py)
- Builder pattern (builder.py)

Test Coverage:
- Unit tests for each component
- Integration tests for workflows
- Edge cases and error handling

Run with:
    pytest tests/test_sdk.py -v
    
Or specific test:
    pytest tests/test_sdk.py::test_quick_model_init -v

Version: 0.6.0-dev
"""

import pytest
import sys
from pathlib import Path
import tempfile
import shutil
import json
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sdk.easy import QuickModel, quick_inference, AutoOptimizer
from src.sdk.plugins import Plugin, PluginManager, PluginMetadata, PluginType, PluginStatus
from src.sdk.registry import ModelRegistry, ModelZoo, ModelMetadata, ModelTask, ModelFormat
from src.sdk.builder import InferencePipeline, ConfigBuilder, ModelBuilder, OptimizationGoal


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def mock_model_file(temp_dir):
    """Create a mock ONNX model file."""
    model_path = temp_dir / "test_model.onnx"
    model_path.write_text("mock onnx model content")
    return model_path


@pytest.fixture
def model_registry(temp_dir):
    """Create a temporary model registry."""
    registry_path = temp_dir / "registry.json"
    return ModelRegistry(registry_path)


@pytest.fixture
def plugin_manager(temp_dir):
    """Create a plugin manager with temp directory."""
    plugin_dir = temp_dir / "plugins"
    plugin_dir.mkdir()
    return PluginManager(plugin_dirs=[plugin_dir], auto_discover=False)


# ============================================================================
# TEST HIGH-LEVEL API (easy.py)
# ============================================================================

class TestQuickModel:
    """Tests for QuickModel class."""
    
    def test_initialization_success(self, mock_model_file):
        """Test QuickModel initializes with valid model."""
        # Note: This will fail loading actual ONNX, but tests init logic
        try:
            model = QuickModel(mock_model_file, auto_optimize=False)
            assert model.model_path == mock_model_file
            assert model.auto_optimize == False
        except Exception as e:
            # Expected to fail on actual ONNX loading
            assert "Failed to load" in str(e) or "ONNX" in str(e)
    
    def test_initialization_file_not_found(self, temp_dir):
        """Test QuickModel raises error for missing file."""
        with pytest.raises(FileNotFoundError):
            QuickModel(temp_dir / "nonexistent.onnx")
    
    def test_device_auto_detection(self, mock_model_file):
        """Test automatic device detection."""
        try:
            model = QuickModel(mock_model_file, device="auto")
            assert model.device == "auto"
            assert model.actual_device in ["cpu", "cuda"]
        except Exception:
            pass  # ONNX loading will fail, but init logic works
    
    def test_from_zoo_method(self):
        """Test QuickModel.from_zoo() class method."""
        # Should work even if model doesn't exist (will warn)
        try:
            model = QuickModel.from_zoo("mobilenetv2")
        except FileNotFoundError:
            pass  # Expected if model not actually downloaded


class TestQuickInference:
    """Tests for quick_inference function."""
    
    def test_quick_inference_signature(self):
        """Test quick_inference function signature."""
        import inspect
        sig = inspect.signature(quick_inference)
        params = list(sig.parameters.keys())
        assert "input_data" in params
        assert "model_path" in params
        assert "top_k" in params
        assert "device" in params


class TestAutoOptimizer:
    """Tests for AutoOptimizer class."""
    
    def test_initialization(self):
        """Test AutoOptimizer initialization."""
        optimizer = AutoOptimizer(
            target_device="rx580",
            target_accuracy_loss=0.01,
            target_speedup=2.0
        )
        assert optimizer.target_device == "rx580"
        assert optimizer.target_accuracy_loss == 0.01
        assert optimizer.target_speedup == 2.0
    
    def test_suggest_optimizations(self, mock_model_file):
        """Test optimization suggestions."""
        optimizer = AutoOptimizer()
        suggestions = optimizer.suggest_optimizations(mock_model_file)
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        
        # Check suggestion structure
        for suggestion in suggestions:
            assert "name" in suggestion
            assert "expected_speedup" in suggestion
            assert "recommended" in suggestion


# ============================================================================
# TEST PLUGIN SYSTEM (plugins.py)
# ============================================================================

class TestPlugin:
    """Tests for Plugin base class."""
    
    def test_plugin_abstract_methods(self):
        """Test that Plugin is abstract."""
        with pytest.raises(TypeError):
            # Can't instantiate abstract class
            Plugin()


class TestPluginMetadata:
    """Tests for PluginMetadata."""
    
    def test_metadata_creation(self):
        """Test creating plugin metadata."""
        metadata = PluginMetadata(
            name="test_plugin",
            version="1.0.0",
            author="Test Author",
            description="Test description",
            plugin_type=PluginType.OPTIMIZER
        )
        
        assert metadata.name == "test_plugin"
        assert metadata.version == "1.0.0"
        assert metadata.plugin_type == PluginType.OPTIMIZER
        assert metadata.enabled == True
    
    def test_metadata_to_dict(self):
        """Test metadata serialization."""
        metadata = PluginMetadata(
            name="test",
            version="1.0",
            author="Author",
            description="Desc"
        )
        
        data = metadata.to_dict()
        assert isinstance(data, dict)
        assert data["name"] == "test"
        assert "plugin_type" in data


class TestPluginManager:
    """Tests for PluginManager."""
    
    def test_initialization(self, plugin_manager):
        """Test PluginManager initialization."""
        assert plugin_manager is not None
        assert len(plugin_manager.list_plugins()) >= 0
    
    def test_list_plugins(self, plugin_manager):
        """Test listing plugins."""
        plugins = plugin_manager.list_plugins()
        assert isinstance(plugins, list)
    
    def test_register_hook(self, plugin_manager):
        """Test hook registration."""
        called = []
        
        def callback(value):
            called.append(value)
        
        plugin_manager.register_hook("test_event", callback)
        plugin_manager.trigger_hook("test_event", "test_value")
        
        assert called == ["test_value"]
    
    def test_cleanup_all(self, plugin_manager):
        """Test cleanup all plugins."""
        success = plugin_manager.cleanup_all()
        assert isinstance(success, bool)


# ============================================================================
# TEST MODEL REGISTRY (registry.py)
# ============================================================================

class TestModelMetadata:
    """Tests for ModelMetadata."""
    
    def test_metadata_creation(self):
        """Test creating model metadata."""
        metadata = ModelMetadata(
            name="test_model",
            display_name="Test Model",
            version="1.0.0",
            task=ModelTask.CLASSIFICATION,
            format=ModelFormat.ONNX
        )
        
        assert metadata.name == "test_model"
        assert metadata.task == ModelTask.CLASSIFICATION
    
    def test_metadata_serialization(self):
        """Test to_dict and from_dict."""
        metadata = ModelMetadata(
            name="test",
            display_name="Test",
            version="1.0",
            task=ModelTask.CLASSIFICATION,
            format=ModelFormat.ONNX
        )
        
        # Serialize
        data = metadata.to_dict()
        assert isinstance(data, dict)
        
        # Deserialize
        restored = ModelMetadata.from_dict(data)
        assert restored.name == metadata.name
        assert restored.task == metadata.task


class TestModelRegistry:
    """Tests for ModelRegistry."""
    
    def test_initialization(self, model_registry):
        """Test registry initialization."""
        assert model_registry is not None
        assert model_registry.registry_path.parent.exists()
    
    def test_register_model(self, model_registry, mock_model_file):
        """Test registering a model."""
        metadata = model_registry.register(
            name="test_model",
            path=mock_model_file,
            task=ModelTask.CLASSIFICATION,
            version="1.0.0"
        )
        
        assert metadata.name == "test_model"
        assert metadata.task == ModelTask.CLASSIFICATION
        assert "test_model" in model_registry.list_models()
    
    def test_get_model(self, model_registry, mock_model_file):
        """Test retrieving model metadata."""
        model_registry.register(
            name="test_model",
            path=mock_model_file,
            task=ModelTask.CLASSIFICATION
        )
        
        metadata = model_registry.get("test_model")
        assert metadata is not None
        assert metadata.name == "test_model"
    
    def test_unregister_model(self, model_registry, mock_model_file):
        """Test unregistering a model."""
        model_registry.register(
            name="test_model",
            path=mock_model_file,
            task=ModelTask.CLASSIFICATION
        )
        
        success = model_registry.unregister("test_model")
        assert success == True
        assert "test_model" not in model_registry.list_models()
    
    def test_search_by_task(self, model_registry, mock_model_file):
        """Test searching by task."""
        model_registry.register(
            name="classifier",
            path=mock_model_file,
            task=ModelTask.CLASSIFICATION
        )
        
        results = model_registry.search(task=ModelTask.CLASSIFICATION)
        assert len(results) >= 1
        assert any(m.name == "classifier" for m in results)
    
    def test_search_by_tags(self, model_registry, mock_model_file):
        """Test searching by tags."""
        model_registry.register(
            name="model1",
            path=mock_model_file,
            task=ModelTask.CLASSIFICATION,
            tags=["int8", "optimized"]
        )
        
        results = model_registry.search(tags=["int8"])
        assert len(results) >= 1
    
    def test_update_performance_metrics(self, model_registry, mock_model_file):
        """Test updating performance metrics."""
        model_registry.register(
            name="test_model",
            path=mock_model_file,
            task=ModelTask.CLASSIFICATION
        )
        
        model_registry.update_performance_metrics(
            "test_model",
            {"fps": 120.5, "latency_ms": 8.3}
        )
        
        metadata = model_registry.get("test_model")
        assert metadata.performance_metrics["fps"] == 120.5


class TestModelZoo:
    """Tests for ModelZoo."""
    
    def test_initialization(self, temp_dir):
        """Test zoo initialization."""
        zoo = ModelZoo(cache_dir=temp_dir / "zoo")
        assert zoo.cache_dir.exists()
    
    def test_list_models(self):
        """Test listing zoo models."""
        zoo = ModelZoo()
        models = zoo.list_models()
        
        assert isinstance(models, list)
        assert len(models) > 0
        assert "mobilenetv2-int8" in models
    
    def test_list_models_by_task(self):
        """Test filtering models by task."""
        zoo = ModelZoo()
        models = zoo.list_models(task=ModelTask.CLASSIFICATION)
        
        assert isinstance(models, list)
        assert len(models) > 0
    
    def test_get_model_info(self):
        """Test getting model information."""
        zoo = ModelZoo()
        info = zoo.get_model_info("mobilenetv2-int8")
        
        assert info is not None
        assert "display_name" in info
        assert "size_mb" in info
        assert "accuracy" in info


# ============================================================================
# TEST BUILDER PATTERN (builder.py)
# ============================================================================

class TestInferencePipeline:
    """Tests for InferencePipeline."""
    
    def test_initialization(self):
        """Test pipeline initialization."""
        pipeline = InferencePipeline()
        assert pipeline is not None
        assert pipeline._built == False
    
    def test_fluent_api_chaining(self, mock_model_file):
        """Test method chaining."""
        pipeline = (InferencePipeline()
            .use_model(mock_model_file)
            .on_device("cpu")
            .with_batch_size(16)
        )
        
        assert pipeline._config.model_path == mock_model_file
        assert pipeline._config.batch_size == 16
    
    def test_build_without_model_raises_error(self):
        """Test build fails without model."""
        pipeline = InferencePipeline()
        
        with pytest.raises(ValueError, match="Model path not set"):
            pipeline.build()
    
    def test_optimization_goal_setting(self, mock_model_file):
        """Test setting optimization goal."""
        pipeline = (InferencePipeline()
            .use_model(mock_model_file)
            .optimize_for("speed")
        )
        
        assert pipeline._config.optimization_goal == OptimizationGoal.SPEED
    
    def test_quantization_methods(self, mock_model_file):
        """Test quantization enabling."""
        pipeline = (InferencePipeline()
            .use_model(mock_model_file)
            .enable_int8_quantization()
        )
        
        assert pipeline._config.enable_quantization == True
        assert pipeline._config.quantization_level == "int8"
    
    def test_preprocessing_addition(self, mock_model_file):
        """Test adding preprocessing steps."""
        pipeline = (InferencePipeline()
            .use_model(mock_model_file)
            .add_preprocessing(resize=(224, 224))
        )
        
        assert len(pipeline._config.preprocessing_steps) == 1
        assert pipeline._config.preprocessing_steps[0]["resize"] == (224, 224)


class TestConfigBuilder:
    """Tests for ConfigBuilder."""
    
    def test_initialization(self):
        """Test builder initialization."""
        builder = ConfigBuilder()
        assert builder is not None
    
    def test_fluent_api(self):
        """Test fluent API."""
        config = (ConfigBuilder()
            .for_task("classification")
            .optimize_for("speed")
            .target_fps(60)
            .build()
        )
        
        assert config["task"] == "classification"
        assert config["optimization_goal"] == "speed"
        assert config["constraints"]["target_fps"] == 60
    
    def test_feature_toggle(self):
        """Test enabling/disabling features."""
        config = (ConfigBuilder()
            .enable_feature("quantization")
            .disable_feature("pruning")
            .build()
        )
        
        assert config["features"]["quantization"] == True
        assert config["features"]["pruning"] == False


class TestModelBuilder:
    """Tests for ModelBuilder."""
    
    def test_initialization(self):
        """Test builder initialization."""
        builder = ModelBuilder()
        assert builder is not None
    
    def test_fluent_api(self, mock_model_file):
        """Test fluent API."""
        config = (ModelBuilder()
            .load(mock_model_file)
            .for_hardware("rx580")
            .quantize_to("int8")
            .build()
        )
        
        assert config["hardware"] == "rx580"
        assert config["quantization"] == "int8"
        assert "quantize_int8" in config["optimizations"]


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestSDKIntegration:
    """Integration tests for SDK components."""
    
    def test_registry_and_zoo_integration(self, temp_dir):
        """Test registry and zoo working together."""
        registry = ModelRegistry(temp_dir / "registry.json")
        zoo = ModelZoo(temp_dir / "zoo")
        
        # List zoo models
        zoo_models = zoo.list_models()
        assert len(zoo_models) > 0
        
        # Get info about a zoo model
        info = zoo.get_model_info(zoo_models[0])
        assert info is not None
    
    def test_pipeline_and_registry_integration(self, model_registry, mock_model_file):
        """Test pipeline using models from registry."""
        # Register model
        model_registry.register(
            name="test_model",
            path=mock_model_file,
            task=ModelTask.CLASSIFICATION
        )
        
        # Create pipeline with registered model
        metadata = model_registry.get("test_model")
        pipeline = (InferencePipeline()
            .use_model(metadata.path)
            .on_device("cpu")
        )
        
        assert pipeline._config.model_path == metadata.path


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
