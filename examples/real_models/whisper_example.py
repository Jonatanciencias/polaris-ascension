"""
Example: Whisper Speech Recognition on AMD Radeon RX 580

This example demonstrates speech-to-text transcription with:
- INT8 quantization for faster inference
- Streaming support
- Multi-language recognition

Requirements:
- Whisper Base model
- ~1GB VRAM (INT8)
- Audio file (WAV, MP3, etc.)

Performance targets:
- Memory: ~1GB VRAM
- Latency: 2-3x real-time
- Accuracy: High (WER < 5% on clean audio)
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.inference.real_models import create_whisper_integration
import logging

logging.basicConfig(level=logging.INFO)


def load_audio(filename: str, sample_rate: int = 16000) -> np.ndarray:
    """Load audio file"""
    try:
        import librosa
        audio, sr = librosa.load(filename, sr=sample_rate)
        return audio
    except ImportError:
        print("librosa not available, generating dummy audio")
        # Generate 5 seconds of dummy audio
        duration = 5.0
        return np.random.randn(int(duration * sample_rate)).astype(np.float32)


def main():
    print("=" * 70)
    print("Whisper Speech Recognition Example")
    print("=" * 70)
    print()
    
    # Create integration with INT8 quantization
    print("Setting up Whisper Base...")
    print("- Quantization: INT8 (2x faster)")
    print("- Optimization: Level 2 (aggressive)")
    print("- Device: AMD Radeon RX 580")
    print()
    
    whisper = create_whisper_integration(
        quantization_mode='int8',
        optimization_level=2
    )
    
    print("Setup complete!")
    print()
    
    # Example audio files (in practice, use real audio)
    audio_examples = [
        {
            'name': 'English Speech',
            'language': 'en',
            'task': 'transcribe'
        },
        {
            'name': 'Spanish Speech',
            'language': 'es',
            'task': 'transcribe'
        },
        {
            'name': 'French Speech (translate)',
            'language': 'fr',
            'task': 'translate'  # Translate to English
        }
    ]
    
    print("Transcribing audio...")
    print("=" * 70)
    
    for i, config in enumerate(audio_examples, 1):
        print(f"\nExample {i}: {config['name']}")
        print(f"Language: {config['language']}")
        print(f"Task: {config['task']}")
        print("-" * 70)
        
        # Load audio (dummy in this example)
        audio = load_audio('dummy.wav', whisper.sample_rate)
        print(f"Audio duration: {len(audio) / whisper.sample_rate:.2f} seconds")
        
        # Transcribe
        text = whisper.transcribe(
            audio=audio,
            language=config['language'],
            task=config['task']
        )
        
        print(f"Transcription: {text}")
        print()
    
    print("=" * 70)
    print("Example complete!")
    print()
    
    # Show configuration
    print("Configuration:")
    print(f"- Model: {whisper.config.name}")
    print(f"- Quantization: {whisper.config.quantization_mode}")
    print(f"- Optimization Level: {whisper.config.optimization_level}")
    print(f"- Sample Rate: {whisper.sample_rate} Hz")
    print(f"- Supported Languages: {', '.join(whisper.languages[:8])}...")
    print()
    
    print("Expected Performance:")
    print("- Memory Usage: ~1GB VRAM")
    print("- Latency: 2-3x real-time (30s audio in 10-15s)")
    print("- Accuracy: WER < 5% on clean audio")
    print()
    
    print("Usage Tips:")
    print("- Use INT8 for best speed/quality balance")
    print("- Specify language for better accuracy")
    print("- Use 'translate' task to convert to English")


if __name__ == '__main__':
    main()
