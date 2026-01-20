"""
Tests for Real-World Model Integration (Session 19 - Phase 4)

Tests Llama 2, Stable Diffusion, Whisper, and BERT integrations.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.real_models import (
    ModelConfig,
    RealModelIntegration,
    Llama2Integration,
    StableDiffusionIntegration,
    WhisperIntegration,
    BERTIntegration,
    create_llama2_integration,
    create_stable_diffusion_integration,
    create_whisper_integration,
    create_bert_integration,
)


class TestModelConfig:
    """Tests for ModelConfig dataclass"""
    
    def test_config_creation(self):
        """Test creating model configuration"""
        config = ModelConfig(
            name="test-model",
            framework="pytorch",
            quantization_mode="int8",
            optimization_level=2
        )
        
        assert config.name == "test-model"
        assert config.framework == "pytorch"
        assert config.quantization_mode == "int8"
        assert config.optimization_level == 2
        assert config.device == "amd_gpu"
        assert config.use_cache == True
    
    def test_config_defaults(self):
        """Test default configuration values"""
        config = ModelConfig(name="test", framework="onnx")
        
        assert config.model_path is None
        assert config.quantization_mode == "int8"
        assert config.optimization_level == 2
        assert config.max_batch_size == 1


class TestRealModelIntegration:
    """Tests for base integration class"""
    
    def test_integration_creation(self):
        """Test creating base integration"""
        config = ModelConfig(name="test", framework="pytorch")
        integration = RealModelIntegration(config)
        
        assert integration.config == config
        assert integration.loader is None
        assert integration.optimizer is None
        assert integration.quantizer is None
    
    def test_setup_with_optimization(self):
        """Test setup with optimization enabled"""
        config = ModelConfig(
            name="test",
            framework="pytorch",
            optimization_level=2
        )
        integration = RealModelIntegration(config)
        integration.setup()
        
        # Optimizer should be created
        assert integration.optimizer is not None
    
    def test_setup_without_optimization(self):
        """Test setup with optimization disabled"""
        config = ModelConfig(
            name="test",
            framework="pytorch",
            optimization_level=0
        )
        integration = RealModelIntegration(config)
        integration.setup()
        
        # Optimizer should not be created
        assert integration.optimizer is None
    
    def test_setup_with_quantization(self):
        """Test setup with quantization"""
        config = ModelConfig(
            name="test",
            framework="pytorch",
            quantization_mode="int8"
        )
        integration = RealModelIntegration(config)
        integration.setup()
        
        # Quantizer should be created
        assert integration.quantizer is not None
    
    def test_abstract_methods(self):
        """Test that abstract methods raise NotImplementedError"""
        config = ModelConfig(name="test", framework="pytorch")
        integration = RealModelIntegration(config)
        
        with pytest.raises(NotImplementedError):
            integration.preprocess(None)
        
        with pytest.raises(NotImplementedError):
            integration.postprocess(None)
        
        with pytest.raises(NotImplementedError):
            integration.run(None)


class TestLlama2Integration:
    """Tests for Llama 2 integration"""
    
    def test_llama2_creation(self):
        """Test creating Llama 2 integration"""
        config = ModelConfig(
            name="Llama-2-7b",
            framework="pytorch",
            quantization_mode="int4"
        )
        llama = Llama2Integration(config)
        
        assert llama.tokenizer is None
        assert llama.max_seq_length == 2048
        assert isinstance(llama.kv_cache, dict)
    
    def test_llama2_setup(self):
        """Test Llama 2 setup"""
        config = ModelConfig(
            name="Llama-2-7b",
            framework="pytorch",
            quantization_mode="int4",
            optimization_level=2
        )
        llama = Llama2Integration(config)
        llama.setup()
        
        # Should have optimizer and quantizer
        assert llama.optimizer is not None
        assert llama.quantizer is not None
    
    def test_llama2_preprocess(self):
        """Test Llama 2 preprocessing"""
        config = ModelConfig(name="Llama-2-7b", framework="pytorch")
        llama = Llama2Integration(config)
        
        text = "Hello world"
        inputs = llama.preprocess(text)
        
        assert 'input_ids' in inputs
        assert 'attention_mask' in inputs
        assert isinstance(inputs['input_ids'], np.ndarray)
    
    def test_llama2_postprocess(self):
        """Test Llama 2 postprocessing"""
        config = ModelConfig(name="Llama-2-7b", framework="pytorch")
        llama = Llama2Integration(config)
        
        outputs = np.zeros((1, 100, 32000))
        text = llama.postprocess(outputs)
        
        assert isinstance(text, str)
    
    def test_llama2_generate(self):
        """Test Llama 2 generation"""
        config = ModelConfig(name="Llama-2-7b", framework="pytorch")
        llama = Llama2Integration(config)
        
        response = llama.generate(
            prompt="Test prompt",
            max_length=50,
            temperature=0.7
        )
        
        assert isinstance(response, str)
    
    def test_llama2_factory(self):
        """Test Llama 2 factory function"""
        llama = create_llama2_integration(quantization_mode='int4')
        
        assert isinstance(llama, Llama2Integration)
        assert llama.config.quantization_mode == 'int4'
        assert llama.optimizer is not None


class TestStableDiffusionIntegration:
    """Tests for Stable Diffusion integration"""
    
    def test_sd_creation(self):
        """Test creating Stable Diffusion integration"""
        config = ModelConfig(
            name="stable-diffusion-1.5",
            framework="pytorch",
            quantization_mode="mixed"
        )
        sd = StableDiffusionIntegration(config)
        
        assert sd.text_encoder is None
        assert sd.vae is None
        assert sd.unet is None
        assert sd.image_size == (512, 512)
        assert sd.num_inference_steps == 50
    
    def test_sd_setup(self):
        """Test Stable Diffusion setup"""
        config = ModelConfig(
            name="stable-diffusion-1.5",
            framework="pytorch",
            quantization_mode="mixed",
            optimization_level=2
        )
        sd = StableDiffusionIntegration(config)
        sd.setup()
        
        # Should have optimizer and quantizer
        assert sd.optimizer is not None
        assert sd.quantizer is not None
    
    def test_sd_preprocess(self):
        """Test Stable Diffusion preprocessing"""
        config = ModelConfig(name="stable-diffusion-1.5", framework="pytorch")
        sd = StableDiffusionIntegration(config)
        
        prompt = "A beautiful landscape"
        embeddings = sd.preprocess(prompt)
        
        assert 'text_embeddings' in embeddings
        assert isinstance(embeddings['text_embeddings'], np.ndarray)
    
    def test_sd_postprocess(self):
        """Test Stable Diffusion postprocessing"""
        config = ModelConfig(name="stable-diffusion-1.5", framework="pytorch")
        sd = StableDiffusionIntegration(config)
        
        latents = np.zeros((1, 4, 64, 64))
        image = sd.postprocess(latents)
        
        assert isinstance(image, np.ndarray)
        assert image.shape == (512, 512, 3)
        assert image.dtype == np.uint8
    
    def test_sd_generate(self):
        """Test Stable Diffusion generation"""
        config = ModelConfig(name="stable-diffusion-1.5", framework="pytorch")
        sd = StableDiffusionIntegration(config)
        
        image = sd.generate(
            prompt="A sunset",
            num_inference_steps=50,
            guidance_scale=7.5,
            seed=42
        )
        
        assert isinstance(image, np.ndarray)
        assert image.shape == (512, 512, 3)
    
    def test_sd_factory(self):
        """Test Stable Diffusion factory function"""
        sd = create_stable_diffusion_integration(quantization_mode='mixed')
        
        assert isinstance(sd, StableDiffusionIntegration)
        assert sd.config.quantization_mode == 'mixed'
        assert sd.optimizer is not None


class TestWhisperIntegration:
    """Tests for Whisper integration"""
    
    def test_whisper_creation(self):
        """Test creating Whisper integration"""
        config = ModelConfig(
            name="whisper-base",
            framework="pytorch",
            quantization_mode="int8"
        )
        whisper = WhisperIntegration(config)
        
        assert whisper.sample_rate == 16000
        assert whisper.n_mels == 80
        assert 'en' in whisper.languages
    
    def test_whisper_setup(self):
        """Test Whisper setup"""
        config = ModelConfig(
            name="whisper-base",
            framework="pytorch",
            quantization_mode="int8",
            optimization_level=2
        )
        whisper = WhisperIntegration(config)
        whisper.setup()
        
        # Should have optimizer and quantizer
        assert whisper.optimizer is not None
        assert whisper.quantizer is not None
    
    def test_whisper_preprocess(self):
        """Test Whisper preprocessing"""
        config = ModelConfig(name="whisper-base", framework="pytorch")
        whisper = WhisperIntegration(config)
        
        # 5 seconds of audio
        audio = np.random.randn(5 * 16000).astype(np.float32)
        features = whisper.preprocess(audio)
        
        assert 'mel_spectrogram' in features
        assert isinstance(features['mel_spectrogram'], np.ndarray)
    
    def test_whisper_postprocess(self):
        """Test Whisper postprocessing"""
        config = ModelConfig(name="whisper-base", framework="pytorch")
        whisper = WhisperIntegration(config)
        
        tokens = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)
        text = whisper.postprocess(tokens)
        
        assert isinstance(text, str)
    
    def test_whisper_transcribe(self):
        """Test Whisper transcription"""
        config = ModelConfig(name="whisper-base", framework="pytorch")
        whisper = WhisperIntegration(config)
        
        audio = np.random.randn(5 * 16000).astype(np.float32)
        text = whisper.transcribe(audio, language='en', task='transcribe')
        
        assert isinstance(text, str)
    
    def test_whisper_factory(self):
        """Test Whisper factory function"""
        whisper = create_whisper_integration(quantization_mode='int8')
        
        assert isinstance(whisper, WhisperIntegration)
        assert whisper.config.quantization_mode == 'int8'
        assert whisper.optimizer is not None


class TestBERTIntegration:
    """Tests for BERT integration"""
    
    def test_bert_creation(self):
        """Test creating BERT integration"""
        config = ModelConfig(
            name="bert-base-uncased",
            framework="onnx",
            quantization_mode="int8"
        )
        bert = BERTIntegration(config)
        
        assert bert.tokenizer is None
        assert bert.max_length == 512
        assert bert.hidden_size == 768
    
    def test_bert_setup(self):
        """Test BERT setup"""
        config = ModelConfig(
            name="bert-base-uncased",
            framework="onnx",
            quantization_mode="int8",
            optimization_level=2
        )
        bert = BERTIntegration(config)
        bert.setup()
        
        # Should have optimizer and quantizer
        assert bert.optimizer is not None
        assert bert.quantizer is not None
    
    def test_bert_preprocess(self):
        """Test BERT preprocessing"""
        config = ModelConfig(name="bert-base-uncased", framework="onnx")
        bert = BERTIntegration(config)
        
        text = "This is a test sentence"
        inputs = bert.preprocess(text)
        
        assert 'input_ids' in inputs
        assert 'attention_mask' in inputs
        assert isinstance(inputs['input_ids'], np.ndarray)
    
    def test_bert_postprocess(self):
        """Test BERT postprocessing"""
        config = ModelConfig(name="bert-base-uncased", framework="onnx")
        bert = BERTIntegration(config)
        
        outputs = np.random.randn(1, 10, 768).astype(np.float32)
        embedding = bert.postprocess(outputs)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (768,)
    
    def test_bert_encode(self):
        """Test BERT encoding"""
        config = ModelConfig(name="bert-base-uncased", framework="onnx")
        bert = BERTIntegration(config)
        
        text = "This is a test"
        embedding = bert.encode(text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (768,)
    
    def test_bert_classify(self):
        """Test BERT classification"""
        config = ModelConfig(name="bert-base-uncased", framework="onnx")
        bert = BERTIntegration(config)
        
        text = "This is great!"
        labels = ['positive', 'negative', 'neutral']
        probs = bert.classify(text, labels)
        
        assert isinstance(probs, dict)
        assert len(probs) == 3
        assert all(label in probs for label in labels)
        assert all(0 <= prob <= 1 for prob in probs.values())
        assert abs(sum(probs.values()) - 1.0) < 0.01  # Sum to 1
    
    def test_bert_factory(self):
        """Test BERT factory function"""
        bert = create_bert_integration(quantization_mode='int8')
        
        assert isinstance(bert, BERTIntegration)
        assert bert.config.quantization_mode == 'int8'
        assert bert.optimizer is not None


class TestIntegration:
    """Integration tests"""
    
    def test_all_models_creation(self):
        """Test creating all model integrations"""
        models = [
            create_llama2_integration(),
            create_stable_diffusion_integration(),
            create_whisper_integration(),
            create_bert_integration()
        ]
        
        assert len(models) == 4
        assert all(isinstance(m, RealModelIntegration) for m in models)
    
    def test_quantization_modes(self):
        """Test different quantization modes"""
        modes = ['none', 'int8', 'int4', 'mixed']
        
        for mode in modes:
            config = ModelConfig(
                name="test",
                framework="pytorch",
                quantization_mode=mode
            )
            integration = RealModelIntegration(config)
            integration.setup()
            
            if mode == 'none':
                assert integration.quantizer is None
            else:
                assert integration.quantizer is not None
    
    def test_optimization_levels(self):
        """Test different optimization levels"""
        levels = [0, 1, 2]
        
        for level in levels:
            config = ModelConfig(
                name="test",
                framework="pytorch",
                optimization_level=level
            )
            integration = RealModelIntegration(config)
            integration.setup()
            
            if level == 0:
                assert integration.optimizer is None
            else:
                assert integration.optimizer is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
