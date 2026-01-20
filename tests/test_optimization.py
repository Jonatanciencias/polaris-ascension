"""
Tests for Model Optimization Pipeline (Session 19 - Phase 3)

Tests graph optimization, operator fusion, and memory layout optimization.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.optimization import (
    OpType,
    MemoryLayout,
    TensorInfo,
    Operation,
    ComputationGraph,
    GraphOptimizer,
    OperatorFusion,
    MemoryLayoutOptimizer,
    OptimizationPipeline,
    create_optimization_pipeline,
)


class TestComputationGraph:
    """Tests for computation graph data structure"""
    
    def test_graph_creation(self):
        """Test creating a computation graph"""
        graph = ComputationGraph(name="test_graph")
        assert graph.name == "test_graph"
        assert len(graph.operations) == 0
        assert len(graph.tensors) == 0
    
    def test_add_operation(self):
        """Test adding operations to graph"""
        graph = ComputationGraph(name="test")
        op = Operation(name="conv1", op_type=OpType.CONV2D)
        graph.add_operation(op)
        assert len(graph.operations) == 1
    
    def test_add_tensor(self):
        """Test adding tensors to graph"""
        graph = ComputationGraph(name="test")
        tensor = TensorInfo(name="input", shape=(1, 3, 224, 224))
        graph.add_tensor(tensor)
        assert "input" in graph.tensors
    
    def test_get_producers(self):
        """Test finding tensor producers"""
        graph = ComputationGraph(name="test")
        op = Operation(
            name="conv1",
            op_type=OpType.CONV2D,
            inputs=["input"],
            outputs=["conv1_out"]
        )
        graph.add_operation(op)
        
        producers = graph.get_producers("conv1_out")
        assert len(producers) == 1
        assert producers[0].name == "conv1"
    
    def test_get_consumers(self):
        """Test finding tensor consumers"""
        graph = ComputationGraph(name="test")
        op = Operation(
            name="relu1",
            op_type=OpType.RELU,
            inputs=["conv1_out"],
            outputs=["relu1_out"]
        )
        graph.add_operation(op)
        
        consumers = graph.get_consumers("conv1_out")
        assert len(consumers) == 1
        assert consumers[0].name == "relu1"


class TestGraphOptimizer:
    """Tests for graph-level optimizations"""
    
    def test_optimizer_creation(self):
        """Test creating graph optimizer"""
        optimizer = GraphOptimizer(verbose=False)
        assert optimizer is not None
    
    def test_dead_code_elimination(self):
        """Test removing unused operations"""
        graph = ComputationGraph(name="test")
        graph.outputs = ["output"]
        
        # Add used operation
        op1 = Operation(
            name="used",
            op_type=OpType.CONV2D,
            inputs=["input"],
            outputs=["output"]
        )
        graph.add_operation(op1)
        
        # Add unused operation
        op2 = Operation(
            name="unused",
            op_type=OpType.CONV2D,
            inputs=["input2"],
            outputs=["unused_out"]
        )
        graph.add_operation(op2)
        
        optimizer = GraphOptimizer(verbose=False)
        optimized = optimizer.eliminate_dead_code(graph)
        
        assert len(optimized.operations) == 1
        assert optimized.operations[0].name == "used"
    
    def test_constant_folding(self):
        """Test folding constant operations"""
        graph = ComputationGraph(name="test")
        
        # Add constant tensors
        t1 = TensorInfo(name="const1", shape=(2, 2), is_constant=True, data=np.array([[1, 2], [3, 4]]))
        t2 = TensorInfo(name="const2", shape=(2, 2), is_constant=True, data=np.array([[5, 6], [7, 8]]))
        graph.add_tensor(t1)
        graph.add_tensor(t2)
        
        # Add operation on constants
        op = Operation(
            name="add",
            op_type=OpType.ADD,
            inputs=["const1", "const2"],
            outputs=["result"]
        )
        graph.add_operation(op)
        
        optimizer = GraphOptimizer(verbose=False)
        optimized = optimizer.fold_constants(graph)
        
        # Operation should be removed (folded)
        assert len(optimized.operations) == 0
    
    def test_identity_removal(self):
        """Test removing identity operations"""
        graph = ComputationGraph(name="test")
        graph.outputs = ["final"]
        
        # Add identity operation
        op1 = Operation(
            name="identity",
            op_type=OpType.IDENTITY,
            inputs=["input"],
            outputs=["identity_out"]
        )
        
        op2 = Operation(
            name="consumer",
            op_type=OpType.RELU,
            inputs=["identity_out"],
            outputs=["final"]
        )
        
        graph.add_operation(op1)
        graph.add_operation(op2)
        
        optimizer = GraphOptimizer(verbose=False)
        optimized = optimizer.remove_identity_ops(graph)
        
        # Identity should be removed
        assert len(optimized.operations) == 1
        assert optimized.operations[0].name == "consumer"
        # Consumer should now use input directly
        assert "input" in optimized.operations[0].inputs
    
    def test_algebra_simplification(self):
        """Test algebraic simplifications"""
        graph = ComputationGraph(name="test")
        graph.outputs = ["output"]
        
        # Add x + 0 operation
        zero = TensorInfo(name="zero", shape=(2, 2), is_constant=True, data=np.zeros((2, 2)))
        graph.add_tensor(zero)
        
        op = Operation(
            name="add_zero",
            op_type=OpType.ADD,
            inputs=["x", "zero"],
            outputs=["output"]
        )
        graph.add_operation(op)
        
        optimizer = GraphOptimizer(verbose=False)
        optimized = optimizer.simplify_algebra(graph)
        
        # Should be simplified
        assert optimizer.optimization_stats['algebra_simplified'] > 0


class TestOperatorFusion:
    """Tests for operator fusion"""
    
    def test_fusion_creation(self):
        """Test creating operator fusion"""
        fusion = OperatorFusion(verbose=False)
        assert fusion is not None
    
    def test_conv_bn_relu_fusion(self):
        """Test fusing Conv + BatchNorm + ReLU"""
        graph = ComputationGraph(name="test")
        
        # Create Conv -> BN -> ReLU chain
        conv = Operation(
            name="conv",
            op_type=OpType.CONV2D,
            inputs=["input"],
            outputs=["conv_out"]
        )
        
        bn = Operation(
            name="bn",
            op_type=OpType.BATCH_NORM,
            inputs=["conv_out"],
            outputs=["bn_out"]
        )
        
        relu = Operation(
            name="relu",
            op_type=OpType.RELU,
            inputs=["bn_out"],
            outputs=["relu_out"]
        )
        
        graph.add_operation(conv)
        graph.add_operation(bn)
        graph.add_operation(relu)
        
        fusion = OperatorFusion(verbose=False)
        fused = fusion.fuse_conv_bn_relu(graph)
        
        # Should have 1 fused operation
        assert len(fused.operations) == 1
        assert fused.operations[0].attributes.get('fused') == True
        assert fused.operations[0].attributes.get('fusion_pattern') == 'conv_bn_relu'
    
    def test_matmul_add_fusion(self):
        """Test fusing MatMul + Add (bias)"""
        graph = ComputationGraph(name="test")
        
        # Add bias as constant
        bias = TensorInfo(name="bias", shape=(10,), is_constant=True, data=np.zeros(10))
        graph.add_tensor(bias)
        
        # Create MatMul -> Add chain
        matmul = Operation(
            name="matmul",
            op_type=OpType.MATMUL,
            inputs=["input", "weight"],
            outputs=["matmul_out"]
        )
        
        add = Operation(
            name="add",
            op_type=OpType.ADD,
            inputs=["matmul_out", "bias"],
            outputs=["output"]
        )
        
        graph.add_operation(matmul)
        graph.add_operation(add)
        
        fusion = OperatorFusion(verbose=False)
        fused = fusion.fuse_matmul_add(graph)
        
        # Should have 1 fused operation
        assert len(fused.operations) == 1
        assert fused.operations[0].attributes.get('fused') == True
        assert fused.operations[0].attributes.get('has_bias') == True
    
    def test_layernorm_activation_fusion(self):
        """Test fusing LayerNorm + GELU"""
        graph = ComputationGraph(name="test")
        
        # Create LayerNorm -> GELU chain
        ln = Operation(
            name="ln",
            op_type=OpType.LAYERNORM,
            inputs=["input"],
            outputs=["ln_out"]
        )
        
        gelu = Operation(
            name="gelu",
            op_type=OpType.GELU,
            inputs=["ln_out"],
            outputs=["output"]
        )
        
        graph.add_operation(ln)
        graph.add_operation(gelu)
        
        fusion = OperatorFusion(verbose=False)
        fused = fusion.fuse_layernorm_activation(graph)
        
        # Should have 1 fused operation
        assert len(fused.operations) == 1
        assert fused.operations[0].attributes.get('fusion_pattern') == 'layernorm_activation'


class TestMemoryLayoutOptimizer:
    """Tests for memory layout optimization"""
    
    def test_layout_optimizer_creation(self):
        """Test creating layout optimizer"""
        optimizer = MemoryLayoutOptimizer(target_device='amd_gpu', verbose=False)
        assert optimizer.target_device == 'amd_gpu'
    
    def test_layout_conversion(self):
        """Test converting tensor layouts"""
        graph = ComputationGraph(name="test")
        
        # Add tensor with NCHW layout
        tensor = TensorInfo(
            name="conv_weight",
            shape=(64, 3, 3, 3),
            layout=MemoryLayout.NCHW
        )
        graph.add_tensor(tensor)
        
        optimizer = MemoryLayoutOptimizer(target_device='mobile', verbose=False)
        optimized = optimizer.convert_layouts(graph, MemoryLayout.NHWC)
        
        # Layout should be converted
        assert graph.tensors["conv_weight"].layout == MemoryLayout.NHWC
    
    def test_transpose_elimination(self):
        """Test eliminating redundant transposes"""
        graph = ComputationGraph(name="test")
        graph.outputs = ["output"]
        
        # Add transpose pair that cancels out
        t1 = Operation(
            name="transpose1",
            op_type=OpType.TRANSPOSE,
            inputs=["input"],
            outputs=["t1_out"],
            attributes={'perm': [0, 2, 3, 1]}  # NCHW -> NHWC
        )
        
        t2 = Operation(
            name="transpose2",
            op_type=OpType.TRANSPOSE,
            inputs=["t1_out"],
            outputs=["output"],
            attributes={'perm': [0, 3, 1, 2]}  # NHWC -> NCHW
        )
        
        graph.add_operation(t1)
        graph.add_operation(t2)
        
        optimizer = MemoryLayoutOptimizer(verbose=False)
        optimized = optimizer.minimize_transposes(graph)
        
        # Both transposes should be removed
        assert len(optimized.operations) == 0


class TestOptimizationPipeline:
    """Tests for integrated optimization pipeline"""
    
    def test_pipeline_creation(self):
        """Test creating optimization pipeline"""
        pipeline = OptimizationPipeline(target_device='amd_gpu', verbose=False)
        assert pipeline.target_device == 'amd_gpu'
        assert pipeline.graph_optimizer is not None
    
    def test_pipeline_full_optimization(self):
        """Test running complete optimization pipeline"""
        graph = ComputationGraph(name="test_model")
        graph.outputs = ["output"]
        
        # Create a realistic graph: Conv -> BN -> ReLU -> Pool
        conv = Operation(
            name="conv1",
            op_type=OpType.CONV2D,
            inputs=["input"],
            outputs=["conv_out"]
        )
        
        bn = Operation(
            name="bn1",
            op_type=OpType.BATCH_NORM,
            inputs=["conv_out"],
            outputs=["bn_out"]
        )
        
        relu = Operation(
            name="relu1",
            op_type=OpType.RELU,
            inputs=["bn_out"],
            outputs=["relu_out"]
        )
        
        pool = Operation(
            name="pool1",
            op_type=OpType.POOL,
            inputs=["relu_out"],
            outputs=["output"]
        )
        
        # Add unused operation (dead code)
        unused = Operation(
            name="unused",
            op_type=OpType.CONV2D,
            inputs=["input2"],
            outputs=["unused_out"]
        )
        
        graph.add_operation(conv)
        graph.add_operation(bn)
        graph.add_operation(relu)
        graph.add_operation(pool)
        graph.add_operation(unused)
        
        pipeline = OptimizationPipeline(target_device='amd_gpu', verbose=False)
        optimized = pipeline.optimize(graph)
        
        # Should have fewer or equal operations (fusion + DCE)
        # Fusion replaces 3 ops with 1 fused, DCE removes unused
        assert len(optimized.operations) <= len(graph.operations)
        
        # Conv+BN+ReLU should be fused
        fused_ops = [op for op in optimized.operations if op.attributes.get('fused')]
        assert len(fused_ops) > 0
    
    def test_pipeline_optimization_report(self):
        """Test getting optimization report"""
        graph = ComputationGraph(name="test")
        graph.outputs = ["output"]
        
        op = Operation(
            name="conv1",
            op_type=OpType.CONV2D,
            inputs=["input"],
            outputs=["output"]
        )
        graph.add_operation(op)
        
        pipeline = OptimizationPipeline(verbose=False)
        pipeline.optimize(graph)
        
        report = pipeline.get_optimization_report()
        
        assert 'target_device' in report
        assert 'graph_optimizations' in report
        assert 'operator_fusions' in report
        assert 'layout_optimizations' in report
    
    def test_pipeline_with_disabled_stages(self):
        """Test pipeline with some stages disabled"""
        pipeline = OptimizationPipeline(
            enable_fusion=False,
            enable_layout_opt=False,
            verbose=False
        )
        
        assert pipeline.op_fusion is None
        assert pipeline.layout_optimizer is None
    
    def test_create_optimization_pipeline_factory(self):
        """Test factory function"""
        # Level 0: All disabled
        p0 = create_optimization_pipeline(target_device='cpu', optimization_level=0)
        assert p0.enable_fusion == False
        assert p0.enable_layout_opt == False
        
        # Level 1: Basic
        p1 = create_optimization_pipeline(target_device='cpu', optimization_level=1)
        assert p1.enable_fusion == True
        assert p1.enable_layout_opt == False
        
        # Level 2: Aggressive
        p2 = create_optimization_pipeline(target_device='amd_gpu', optimization_level=2)
        assert p2.enable_fusion == True
        assert p2.enable_layout_opt == True


class TestIntegration:
    """Integration tests"""
    
    def test_complex_model_optimization(self):
        """Test optimizing a complex model graph"""
        graph = ComputationGraph(name="resnet_block")
        graph.outputs = ["block_output"]
        
        # Simulate ResNet residual block
        # Main path: Conv -> BN -> ReLU -> Conv -> BN
        conv1 = Operation(name="conv1", op_type=OpType.CONV2D, inputs=["input"], outputs=["c1"])
        bn1 = Operation(name="bn1", op_type=OpType.BATCH_NORM, inputs=["c1"], outputs=["b1"])
        relu1 = Operation(name="relu1", op_type=OpType.RELU, inputs=["b1"], outputs=["r1"])
        conv2 = Operation(name="conv2", op_type=OpType.CONV2D, inputs=["r1"], outputs=["c2"])
        bn2 = Operation(name="bn2", op_type=OpType.BATCH_NORM, inputs=["c2"], outputs=["b2"])
        
        # Skip connection: Identity
        skip = Operation(name="skip", op_type=OpType.IDENTITY, inputs=["input"], outputs=["skip_out"])
        
        # Add: main + skip
        add = Operation(name="add", op_type=OpType.ADD, inputs=["b2", "skip_out"], outputs=["add_out"])
        
        # Final ReLU
        relu2 = Operation(name="relu2", op_type=OpType.RELU, inputs=["add_out"], outputs=["block_output"])
        
        for op in [conv1, bn1, relu1, conv2, bn2, skip, add, relu2]:
            graph.add_operation(op)
        
        # Optimize
        pipeline = OptimizationPipeline(verbose=False)
        optimized = pipeline.optimize(graph)
        
        # Should have fewer or equal operations due to:
        # 1. Identity removal (skip connection)
        # 2. Conv+BN+ReLU fusion (at least one)
        assert len(optimized.operations) <= len(graph.operations)
        
        # Check for fusions
        fused_count = sum(1 for op in optimized.operations if op.attributes.get('fused'))
        assert fused_count >= 1
    
    def test_transformer_block_optimization(self):
        """Test optimizing a transformer block"""
        graph = ComputationGraph(name="transformer_block")
        graph.outputs = ["output"]
        
        # LayerNorm -> Attention -> Add -> LayerNorm -> FFN -> Add
        ln1 = Operation(name="ln1", op_type=OpType.LAYERNORM, inputs=["input"], outputs=["ln1_out"])
        
        # Attention (simplified as matmuls)
        qkv = Operation(name="qkv", op_type=OpType.MATMUL, inputs=["ln1_out", "qkv_w"], outputs=["qkv_out"])
        
        add1 = Operation(name="add1", op_type=OpType.ADD, inputs=["qkv_out", "input"], outputs=["add1_out"])
        
        ln2 = Operation(name="ln2", op_type=OpType.LAYERNORM, inputs=["add1_out"], outputs=["ln2_out"])
        gelu = Operation(name="gelu", op_type=OpType.GELU, inputs=["ln2_out"], outputs=["gelu_out"])
        
        ffn = Operation(name="ffn", op_type=OpType.MATMUL, inputs=["gelu_out", "ffn_w"], outputs=["ffn_out"])
        
        add2 = Operation(name="add2", op_type=OpType.ADD, inputs=["ffn_out", "add1_out"], outputs=["output"])
        
        for op in [ln1, qkv, add1, ln2, gelu, ffn, add2]:
            graph.add_operation(op)
        
        # Optimize
        pipeline = OptimizationPipeline(verbose=False)
        optimized = pipeline.optimize(graph)
        
        # Should have LayerNorm+GELU fusion
        fused_ops = [op for op in optimized.operations 
                    if op.attributes.get('fusion_pattern') == 'layernorm_activation']
        assert len(fused_ops) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
