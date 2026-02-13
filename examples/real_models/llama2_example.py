"""
Example: Llama 2 7B Text Generation on AMD Radeon RX 580

This example demonstrates how to use Llama 2 for text generation with:
- INT4 quantization for memory efficiency
- Graph optimization for faster inference
- AMD GPU-specific optimizations

Requirements:
- Llama 2 model (download from Hugging Face)
- ~4GB VRAM (with INT4 quantization)

Performance targets:
- Memory: ~3.5GB (INT4 vs ~14GB FP16)
- Latency: 15-20 tokens/sec on RX 580
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.inference.real_models import create_llama2_integration
import logging

logging.basicConfig(level=logging.INFO)


def main():
    print("=" * 70)
    print("Llama 2 7B Text Generation Example")
    print("=" * 70)
    print()

    # Create integration with INT4 quantization
    print("Setting up Llama 2 7B...")
    print("- Quantization: INT4 (saves ~75% memory)")
    print("- Optimization: Level 2 (aggressive)")
    print("- Device: AMD Radeon RX 580")
    print()

    llama = create_llama2_integration(quantization_mode="int4", optimization_level=2)

    print("Setup complete!")
    print()

    # Example prompts
    prompts = [
        "Explain quantum computing in simple terms:",
        "Write a short poem about artificial intelligence:",
        "What are the benefits of GPU computing?",
    ]

    print("Generating responses...")
    print("=" * 70)

    for i, prompt in enumerate(prompts, 1):
        print(f"\nPrompt {i}: {prompt}")
        print("-" * 70)

        # Generate text
        response = llama.generate(prompt=prompt, max_length=150, temperature=0.7, top_p=0.9)

        print(f"Response: {response}")
        print()

    print("=" * 70)
    print("Example complete!")
    print()

    # Show configuration
    print("Configuration:")
    print(f"- Model: {llama.config.name}")
    print(f"- Quantization: {llama.config.quantization_mode}")
    print(f"- Optimization Level: {llama.config.optimization_level}")
    print(f"- Device: {llama.config.device}")
    print()

    print("Expected Performance:")
    print("- Memory Usage: ~3.5GB VRAM (vs ~14GB FP16)")
    print("- Generation Speed: 15-20 tokens/sec")
    print("- Quality: Minimal degradation with INT4")


if __name__ == "__main__":
    main()
