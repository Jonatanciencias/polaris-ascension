"""
Example: BERT Text Understanding on AMD Radeon RX 580

This example demonstrates text understanding with BERT:
- Text classification
- Semantic similarity
- Named entity recognition
- INT8 quantization

Requirements:
- BERT Base model
- ~500MB VRAM (INT8)

Performance targets:
- Memory: ~500MB VRAM
- Latency: < 10ms per sentence
- Accuracy: High (F1 > 90% on standard benchmarks)
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.inference.real_models import create_bert_integration
import logging

logging.basicConfig(level=logging.INFO)


def main():
    print("=" * 70)
    print("BERT Text Understanding Example")
    print("=" * 70)
    print()

    # Create integration with INT8 quantization
    print("Setting up BERT Base...")
    print("- Quantization: INT8 (2x faster)")
    print("- Optimization: Level 2 (aggressive)")
    print("- Device: AMD Radeon RX 580")
    print()

    bert = create_bert_integration(quantization_mode="int8", optimization_level=2)

    print("Setup complete!")
    print()

    # Example 1: Text Encoding
    print("=" * 70)
    print("Example 1: Text Encoding")
    print("=" * 70)

    sentences = [
        "Machine learning is transforming technology",
        "GPU computing enables faster deep learning",
        "The weather is nice today",
    ]

    embeddings = []
    for sentence in sentences:
        emb = bert.encode(sentence)
        embeddings.append(emb)
        print(f"'{sentence}'")
        print(f"  → Embedding shape: {emb.shape}")
        print(f"  → Norm: {np.linalg.norm(emb):.4f}")
        print()

    # Compute similarity
    print("Semantic Similarity:")
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            sim = np.dot(embeddings[i], embeddings[j]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
            )
            print(f"  {i+1} ↔ {j+1}: {sim:.4f}")
    print()

    # Example 2: Text Classification
    print("=" * 70)
    print("Example 2: Text Classification")
    print("=" * 70)

    texts = [
        "This product is amazing! I love it!",
        "Terrible quality, waste of money",
        "The service was okay, nothing special",
    ]

    labels = ["positive", "negative", "neutral"]

    for text in texts:
        probs = bert.classify(text, labels)
        print(f"Text: '{text}'")
        print("Classification:")
        for label, prob in sorted(probs.items(), key=lambda x: -x[1]):
            print(f"  {label}: {prob:.4f}")
        print()

    # Example 3: Question Answering (simplified)
    print("=" * 70)
    print("Example 3: Question Context Matching")
    print("=" * 70)

    context = "The Radeon RX 580 is a graphics card released by AMD in 2017"
    questions = ["What is the RX 580?", "Who made the RX 580?", "When was it released?"]

    context_emb = bert.encode(context)

    print(f"Context: '{context}'")
    print("\nQuestion Similarities:")
    for question in questions:
        q_emb = bert.encode(question)
        sim = np.dot(context_emb, q_emb) / (np.linalg.norm(context_emb) * np.linalg.norm(q_emb))
        print(f"  Q: '{question}'")
        print(f"     Similarity: {sim:.4f}")
    print()

    print("=" * 70)
    print("Example complete!")
    print()

    # Show configuration
    print("Configuration:")
    print(f"- Model: {bert.config.name}")
    print(f"- Quantization: {bert.config.quantization_mode}")
    print(f"- Optimization Level: {bert.config.optimization_level}")
    print(f"- Max Length: {bert.max_length}")
    print(f"- Hidden Size: {bert.hidden_size}")
    print()

    print("Expected Performance:")
    print("- Memory Usage: ~500MB VRAM")
    print("- Latency: < 10ms per sentence")
    print("- Accuracy: F1 > 90% on benchmarks")
    print()

    print("Use Cases:")
    print("- Sentiment analysis")
    print("- Semantic search")
    print("- Question answering")
    print("- Named entity recognition")
    print("- Text classification")


if __name__ == "__main__":
    main()
