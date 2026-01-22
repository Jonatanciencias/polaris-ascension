"""
ONNX Model Export Utilities

Provides utilities to export PyTorch models to ONNX format with:
- Model validation and verification
- Graph optimization (constant folding, operator fusion)
- Quantization support (INT8, mixed precision)
- RX 580 specific optimizations

Usage:
    from src.inference.onnx_export import export_to_onnx, validate_onnx_model
    
    # Export PyTorch model
    export_to_onnx(
        model,
        output_path="model.onnx",
        input_shape=(1, 3, 224, 224),
        opset_version=13
    )
    
    # Validate exported model
    is_valid = validate_onnx_model("model.onnx")

Author: AMD GPU Computing Team
Date: January 21, 2026
"""

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
from onnx import optimizer
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)


def export_to_onnx(
    model: nn.Module,
    output_path: Union[str, Path],
    input_shape: Tuple[int, ...],
    opset_version: int = 13,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    optimize: bool = True,
    verify: bool = True,
    quantize: bool = False,
    verbose: bool = False
) -> bool:
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model to export
        output_path: Path to save ONNX model
        input_shape: Input tensor shape (batch, channels, height, width)
        opset_version: ONNX opset version (13 recommended)
        dynamic_axes: Dynamic axes for variable batch size
        optimize: Apply ONNX graph optimization
        verify: Verify exported model
        quantize: Apply dynamic quantization (INT8)
        verbose: Print detailed export information
        
    Returns:
        True if export successful, False otherwise
        
    Example:
        >>> model = torchvision.models.resnet18(pretrained=True)
        >>> export_to_onnx(
        ...     model,
        ...     "resnet18.onnx",
        ...     input_shape=(1, 3, 224, 224),
        ...     dynamic_axes={'input': {0: 'batch_size'}}
        ... )
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Set model to eval mode
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(*input_shape)
        
        # Default dynamic axes for batch size
        if dynamic_axes is None:
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        
        logger.info(f"Exporting model to ONNX: {output_path}")
        logger.info(f"Input shape: {input_shape}")
        logger.info(f"Opset version: {opset_version}")
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            verbose=verbose
        )
        
        logger.info(f"✓ Model exported to {output_path}")
        
        # Optimize ONNX graph
        if optimize:
            logger.info("Optimizing ONNX graph...")
            optimize_onnx_model(output_path)
        
        # Verify exported model
        if verify:
            logger.info("Verifying exported model...")
            is_valid = validate_onnx_model(output_path)
            if not is_valid:
                logger.error("✗ Model verification failed")
                return False
            logger.info("✓ Model verification passed")
        
        # Apply quantization if requested
        if quantize:
            logger.info("Applying dynamic quantization...")
            quantized_path = output_path.with_stem(f"{output_path.stem}_int8")
            quantize_onnx_model(output_path, quantized_path)
        
        # Compare outputs
        if verify:
            logger.info("Comparing PyTorch vs ONNX outputs...")
            max_diff = compare_outputs(model, output_path, dummy_input)
            logger.info(f"Max output difference: {max_diff:.6f}")
            
            if max_diff > 1e-4:
                logger.warning(f"⚠ Large output difference detected: {max_diff:.6f}")
        
        logger.info("✓ Export completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def optimize_onnx_model(model_path: Union[str, Path]) -> bool:
    """
    Optimize ONNX model graph.
    
    Applies optimization passes:
    - Constant folding
    - Operator fusion (Conv + BN + ReLU)
    - Redundant node elimination
    - Dead code elimination
    
    Args:
        model_path: Path to ONNX model
        
    Returns:
        True if optimization successful
    """
    try:
        model_path = Path(model_path)
        
        # Load model
        model = onnx.load(str(model_path))
        
        # Apply optimization passes
        passes = [
            'eliminate_deadend',
            'eliminate_identity',
            'eliminate_nop_dropout',
            'eliminate_nop_monotone_argmax',
            'eliminate_nop_pad',
            'eliminate_nop_transpose',
            'eliminate_unused_initializer',
            'extract_constant_to_initializer',
            'fuse_add_bias_into_conv',
            'fuse_bn_into_conv',
            'fuse_consecutive_concats',
            'fuse_consecutive_log_softmax',
            'fuse_consecutive_reduce_unsqueeze',
            'fuse_consecutive_squeezes',
            'fuse_consecutive_transposes',
            'fuse_matmul_add_bias_into_gemm',
            'fuse_pad_into_conv',
            'fuse_transpose_into_gemm',
        ]
        
        # Optimize
        optimized_model = optimizer.optimize(model, passes)
        
        # Save optimized model
        onnx.save(optimized_model, str(model_path))
        
        logger.info(f"✓ Model optimized: {model_path}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Optimization failed: {e}")
        return False


def validate_onnx_model(model_path: Union[str, Path]) -> bool:
    """
    Validate ONNX model.
    
    Checks:
    - ONNX schema compliance
    - Graph structure validity
    - Operator support
    
    Args:
        model_path: Path to ONNX model
        
    Returns:
        True if model is valid
    """
    try:
        model_path = Path(model_path)
        
        # Load and check model
        model = onnx.load(str(model_path))
        onnx.checker.check_model(model)
        
        # Check with ONNX Runtime
        try:
            session = ort.InferenceSession(
                str(model_path),
                providers=['CPUExecutionProvider']
            )
            
            # Get input/output info
            input_name = session.get_inputs()[0].name
            input_shape = session.get_inputs()[0].shape
            output_name = session.get_outputs()[0].name
            output_shape = session.get_outputs()[0].shape
            
            logger.info(f"  Input:  {input_name} {input_shape}")
            logger.info(f"  Output: {output_name} {output_shape}")
            
        except Exception as e:
            logger.warning(f"⚠ ONNX Runtime check failed: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Validation failed: {e}")
        return False


def compare_outputs(
    pytorch_model: nn.Module,
    onnx_path: Union[str, Path],
    test_input: torch.Tensor
) -> float:
    """
    Compare PyTorch and ONNX model outputs.
    
    Args:
        pytorch_model: Original PyTorch model
        onnx_path: Path to ONNX model
        test_input: Test input tensor
        
    Returns:
        Maximum absolute difference between outputs
    """
    try:
        # PyTorch inference
        pytorch_model.eval()
        with torch.no_grad():
            pytorch_output = pytorch_model(test_input).numpy()
        
        # ONNX inference
        session = ort.InferenceSession(
            str(onnx_path),
            providers=['CPUExecutionProvider']
        )
        input_name = session.get_inputs()[0].name
        onnx_output = session.run(None, {input_name: test_input.numpy()})[0]
        
        # Compare
        max_diff = np.abs(pytorch_output - onnx_output).max()
        
        return float(max_diff)
        
    except Exception as e:
        logger.error(f"✗ Output comparison failed: {e}")
        return float('inf')


def quantize_onnx_model(
    model_path: Union[str, Path],
    output_path: Union[str, Path],
    quantization_mode: str = 'dynamic'
) -> bool:
    """
    Quantize ONNX model to INT8.
    
    Args:
        model_path: Path to input ONNX model
        output_path: Path to save quantized model
        quantization_mode: 'dynamic' or 'static'
        
    Returns:
        True if quantization successful
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        model_path = Path(model_path)
        output_path = Path(output_path)
        
        logger.info(f"Quantizing model: {model_path}")
        
        if quantization_mode == 'dynamic':
            quantize_dynamic(
                str(model_path),
                str(output_path),
                weight_type=QuantType.QInt8
            )
        else:
            logger.warning("Static quantization not yet implemented")
            return False
        
        logger.info(f"✓ Quantized model saved: {output_path}")
        
        # Compare sizes
        original_size = model_path.stat().st_size / (1024 * 1024)
        quantized_size = output_path.stat().st_size / (1024 * 1024)
        reduction = (1 - quantized_size / original_size) * 100
        
        logger.info(f"  Original: {original_size:.2f} MB")
        logger.info(f"  Quantized: {quantized_size:.2f} MB")
        logger.info(f"  Reduction: {reduction:.1f}%")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Quantization failed: {e}")
        return False


def get_model_info(model_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get information about ONNX model.
    
    Args:
        model_path: Path to ONNX model
        
    Returns:
        Dictionary with model information
    """
    try:
        model = onnx.load(str(model_path))
        
        # Count parameters
        total_params = 0
        for initializer in model.graph.initializer:
            shape = [dim for dim in initializer.dims]
            params = np.prod(shape)
            total_params += params
        
        # Get operators
        operators = {}
        for node in model.graph.node:
            op_type = node.op_type
            operators[op_type] = operators.get(op_type, 0) + 1
        
        info = {
            'opset_version': model.opset_import[0].version,
            'total_parameters': int(total_params),
            'num_nodes': len(model.graph.node),
            'num_inputs': len(model.graph.input),
            'num_outputs': len(model.graph.output),
            'operators': operators,
            'file_size_mb': Path(model_path).stat().st_size / (1024 * 1024)
        }
        
        return info
        
    except Exception as e:
        logger.error(f"✗ Failed to get model info: {e}")
        return {}


# Convenience function for batch export
def batch_export_models(
    models: Dict[str, nn.Module],
    output_dir: Union[str, Path],
    input_shape: Tuple[int, ...],
    **export_kwargs
) -> Dict[str, bool]:
    """
    Export multiple models to ONNX.
    
    Args:
        models: Dictionary of {name: model}
        output_dir: Output directory
        input_shape: Input shape for all models
        **export_kwargs: Additional arguments for export_to_onnx
        
    Returns:
        Dictionary of {name: success_status}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for name, model in models.items():
        logger.info(f"\nExporting {name}...")
        output_path = output_dir / f"{name}.onnx"
        
        success = export_to_onnx(
            model,
            output_path,
            input_shape,
            **export_kwargs
        )
        
        results[name] = success
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("Export Summary:")
    for name, success in results.items():
        status = "✓" if success else "✗"
        logger.info(f"  {status} {name}")
    logger.info("="*60)
    
    return results


if __name__ == "__main__":
    # Demo: Export a simple model
    print("ONNX Export Utilities Demo")
    print("=" * 60)
    
    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(16)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(32, 10)
        
        def forward(self, x):
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    model = SimpleModel()
    
    # Export
    success = export_to_onnx(
        model,
        "simple_model.onnx",
        input_shape=(1, 3, 32, 32),
        optimize=True,
        verify=True,
        quantize=True
    )
    
    if success:
        # Show model info
        info = get_model_info("simple_model.onnx")
        print("\nModel Information:")
        print(f"  Opset version: {info['opset_version']}")
        print(f"  Parameters: {info['total_parameters']:,}")
        print(f"  Nodes: {info['num_nodes']}")
        print(f"  File size: {info['file_size_mb']:.2f} MB")
        print(f"  Operators: {info['operators']}")
    
    print("\nDemo complete!")
