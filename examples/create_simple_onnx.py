"""
Create simple ONNX model using onnx library directly

Creates a minimal ONNX model for testing without PyTorch.
"""

import numpy as np
from pathlib import Path

print("Creating test ONNX model using onnx library...")

try:
    import onnx
    from onnx import helper, TensorProto
    
    # Create models directory
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Create a simple linear model: y = Wx + b
    # Input: (batch, 10)
    # Output: (batch, 5)
    
    # Define inputs
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [None, 10])
    
    # Define outputs
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [None, 5])
    
    # Define parameters (weights and bias)
    W = helper.make_tensor(
        'W',
        TensorProto.FLOAT,
        [10, 5],
        np.random.randn(10, 5).flatten().tolist()
    )
    
    b = helper.make_tensor(
        'b',
        TensorProto.FLOAT,
        [5],
        np.random.randn(5).flatten().tolist()
    )
    
    # Create nodes
    # MatMul node: temp = input * W
    matmul_node = helper.make_node(
        'MatMul',
        inputs=['input', 'W'],
        outputs=['temp']
    )
    
    # Add node: output = temp + b
    add_node = helper.make_node(
        'Add',
        inputs=['temp', 'b'],
        outputs=['output']
    )
    
    # Create graph
    graph_def = helper.make_graph(
        [matmul_node, add_node],
        'simple_linear',
        [X],
        [Y],
        [W, b]
    )
    
    # Create model
    model_def = helper.make_model(graph_def, producer_name='radeon_rx_580_test')
    model_def.opset_import[0].version = 13
    
    # Check model
    onnx.checker.check_model(model_def)
    
    # Save model
    output_path = models_dir / "simple_linear.onnx"
    onnx.save(model_def, output_path)
    
    print(f"✅ Created {output_path}")
    print(f"   Size: {output_path.stat().st_size / 1024:.2f} KB")
    print(f"   Input: (batch, 10) -> Output: (batch, 5)")
    print(f"\nYou can now test with:")
    print(f"   python examples/test_model_loaders.py")
    
except ImportError as e:
    print(f"⚠️  ONNX library not installed: {e}")
    print("Install with: pip install onnx")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
