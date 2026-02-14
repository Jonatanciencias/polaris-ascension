"""
Example: Stable Diffusion 1.5 Image Generation on AMD Radeon RX 580

This example demonstrates text-to-image generation with:
- Mixed precision quantization
- Attention optimization
- AMD GPU-specific memory layout

Requirements:
- Stable Diffusion 1.5 model
- ~4GB VRAM (with optimizations)

Performance targets:
- Memory: ~4GB (mixed precision)
- Generation time: 15-20 seconds (50 steps)
- Image quality: High (512x512)
"""

import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging

from src.inference.real_models import create_stable_diffusion_integration

logging.basicConfig(level=logging.INFO)


def save_image(image: np.ndarray, filename: str):
    """Save generated image"""
    try:
        from PIL import Image

        img = Image.fromarray(image)
        img.save(filename)
        print(f"Saved image to: {filename}")
    except ImportError:
        print(f"PIL not available, image array shape: {image.shape}")


def main():
    print("=" * 70)
    print("Stable Diffusion 1.5 Image Generation Example")
    print("=" * 70)
    print()

    # Create integration with mixed precision
    print("Setting up Stable Diffusion 1.5...")
    print("- Quantization: Mixed Precision (FP16 + FP32)")
    print("- Optimization: Level 2 (aggressive)")
    print("- Device: AMD Radeon RX 580")
    print()

    sd = create_stable_diffusion_integration(quantization_mode="mixed", optimization_level=2)

    print("Setup complete!")
    print()

    # Example prompts
    prompts = [
        {
            "prompt": "A beautiful sunset over mountains, oil painting style",
            "negative_prompt": "blurry, low quality, distorted",
            "seed": 42,
        },
        {
            "prompt": "A futuristic city with flying cars, cyberpunk art",
            "negative_prompt": "ugly, deformed, low resolution",
            "seed": 123,
        },
        {
            "prompt": "A cute robot learning to paint, digital art",
            "negative_prompt": "bad anatomy, poorly drawn",
            "seed": 456,
        },
    ]

    print("Generating images...")
    print("=" * 70)

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    for i, config in enumerate(prompts, 1):
        print(f"\nImage {i}:")
        print(f"Prompt: {config['prompt']}")
        print(f"Negative: {config['negative_prompt']}")
        print(f"Seed: {config['seed']}")
        print("-" * 70)

        # Generate image
        image = sd.generate(
            prompt=config["prompt"],
            negative_prompt=config["negative_prompt"],
            num_inference_steps=50,
            guidance_scale=7.5,
            seed=config["seed"],
        )

        # Save image
        filename = output_dir / f"sd_output_{i}.png"
        save_image(image, str(filename))
        print()

    print("=" * 70)
    print("Example complete!")
    print()

    # Show configuration
    print("Configuration:")
    print(f"- Model: {sd.config.name}")
    print(f"- Quantization: {sd.config.quantization_mode}")
    print(f"- Optimization Level: {sd.config.optimization_level}")
    print(f"- Image Size: {sd.image_size}")
    print(f"- Inference Steps: {sd.num_inference_steps}")
    print()

    print("Expected Performance:")
    print("- Memory Usage: ~4GB VRAM")
    print("- Generation Time: 15-20 seconds (50 steps)")
    print("- Image Quality: High (512x512)")
    print(f"- Output Location: {output_dir}")


if __name__ == "__main__":
    main()
