"""
Model Optimization Pipeline for Inference Acceleration

Session 19 - Phase 3
Implements graph optimization, operator fusion, and memory layout optimization
for maximizing inference performance on AMD GPUs.

Key Features:
1. Graph Optimization - Eliminate redundant operations
2. Operator Fusion - Combine operations for efficiency
3. Memory Layout - Optimize tensor layouts for hardware
4. Custom Rules - Domain-specific optimizations

Academic Foundation:
- TensorRT (NVIDIA, 2016): Graph optimization strategies
- TVM (Chen et al., 2018): Tensor operator compilation
- ONNX Runtime (Microsoft, 2019): Graph transformations
- XLA (Google, 2017): Accelerated linear algebra compiler

Target Performance:
- 10-20% latency reduction through fusion
- 20-30% memory reduction through layout optimization
- Better cache locality and memory access patterns

Author: Radeon RX 580 AI Framework
Version: 0.1.0
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class OpType(Enum):
    """Supported operation types"""
    CONV2D = "conv2d"
    BATCH_NORM = "batch_norm"
    RELU = "relu"
    POOL = "pool"
    MATMUL = "matmul"
    ADD = "add"
    MUL = "mul"
    RESHAPE = "reshape"
    TRANSPOSE = "transpose"
    CONCAT = "concat"
    SPLIT = "split"
    SOFTMAX = "softmax"
    LAYERNORM = "layernorm"
    GELU = "gelu"
    DROPOUT = "dropout"
    IDENTITY = "identity"
    CONSTANT = "constant"


class MemoryLayout(Enum):
    """Tensor memory layouts"""
    NCHW = "nchw"  # Batch, Channels, Height, Width (default)
    NHWC = "nhwc"  # Batch, Height, Width, Channels (mobile-friendly)
    NDHWC = "ndhwc"  # 3D variant
    NC = "nc"  # 2D: Batch, Channels
    UNKNOWN = "unknown"


@dataclass
class TensorInfo:
    """Information about a tensor in the graph"""
    name: str
    shape: Tuple[int, ...]
    dtype: str = "float32"
    layout: MemoryLayout = MemoryLayout.NCHW
    is_constant: bool = False
    data: Optional[np.ndarray] = None


@dataclass
class Operation:
    """Represents an operation in the computation graph"""
    name: str
    op_type: OpType
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.name)


@dataclass
class ComputationGraph:
    """Computation graph representation"""
    name: str
    operations: List[Operation] = field(default_factory=list)
    tensors: Dict[str, TensorInfo] = field(default_factory=dict)
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    
    def add_operation(self, op: Operation):
        """Add operation to graph"""
        self.operations.append(op)
    
    def add_tensor(self, tensor: TensorInfo):
        """Add tensor to graph"""
        self.tensors[tensor.name] = tensor
    
    def get_producers(self, tensor_name: str) -> List[Operation]:
        """Get operations that produce a tensor"""
        return [op for op in self.operations if tensor_name in op.outputs]
    
    def get_consumers(self, tensor_name: str) -> List[Operation]:
        """Get operations that consume a tensor"""
        return [op for op in self.operations if tensor_name in op.inputs]
    
    def remove_operation(self, op: Operation):
        """Remove operation from graph"""
        self.operations = [o for o in self.operations if o.name != op.name]
    
    def replace_operation(self, old_op: Operation, new_op: Operation):
        """Replace an operation with another"""
        idx = self.operations.index(old_op)
        self.operations[idx] = new_op


# =============================================================================
# GRAPH OPTIMIZATION PASSES
# =============================================================================

class GraphOptimizer:
    """
    Graph-level optimizations for computation graphs.
    
    Implements:
    1. Dead code elimination (DCE)
    2. Constant folding
    3. Common subexpression elimination (CSE)
    4. Identity operation removal
    5. Algebraic simplifications
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.optimization_stats = defaultdict(int)
    
    def optimize(self, graph: ComputationGraph) -> ComputationGraph:
        """
        Apply all optimization passes to graph.
        
        Args:
            graph: Input computation graph
            
        Returns:
            Optimized graph
        """
        if self.verbose:
            logger.info(f"[GraphOpt] Optimizing graph: {graph.name}")
            logger.info(f"[GraphOpt] Initial ops: {len(graph.operations)}")
        
        # Apply optimization passes in order
        graph = self.eliminate_dead_code(graph)
        graph = self.fold_constants(graph)
        graph = self.remove_identity_ops(graph)
        graph = self.eliminate_common_subexpressions(graph)
        graph = self.simplify_algebra(graph)
        
        if self.verbose:
            logger.info(f"[GraphOpt] Final ops: {len(graph.operations)}")
            logger.info(f"[GraphOpt] Optimizations applied: {dict(self.optimization_stats)}")
        
        return graph
    
    def eliminate_dead_code(self, graph: ComputationGraph) -> ComputationGraph:
        """
        Remove operations whose outputs are never used.
        
        Algorithm:
        1. Mark all output tensors as live
        2. Backward propagate liveness through the graph
        3. Remove operations that don't contribute to live tensors
        """
        live_tensors = set(graph.outputs)
        live_ops = set()
        
        # Backward pass: mark live operations
        changed = True
        while changed:
            changed = False
            for op in graph.operations:
                if op in live_ops:
                    continue
                
                # If any output is live, operation is live
                if any(out in live_tensors for out in op.outputs):
                    live_ops.add(op)
                    # Mark inputs as live
                    for inp in op.inputs:
                        if inp not in live_tensors:
                            live_tensors.add(inp)
                            changed = True
        
        # Remove dead operations
        dead_ops = [op for op in graph.operations if op not in live_ops]
        for op in dead_ops:
            graph.remove_operation(op)
            self.optimization_stats['dead_code_eliminated'] += 1
            if self.verbose:
                logger.info(f"[DCE] Removed dead operation: {op.name}")
        
        return graph
    
    def fold_constants(self, graph: ComputationGraph) -> ComputationGraph:
        """
        Evaluate constant operations at compile time.
        
        If all inputs to an operation are constants, compute the result
        and replace the operation with a constant tensor.
        """
        folded = []
        
        for op in graph.operations:
            # Check if all inputs are constants
            if not op.inputs:
                continue
            
            all_constant = all(
                graph.tensors.get(inp, TensorInfo(inp, ())).is_constant
                for inp in op.inputs
            )
            
            if all_constant and op.op_type in [OpType.ADD, OpType.MUL, OpType.RESHAPE]:
                # Try to fold
                try:
                    result = self._evaluate_constant_op(op, graph)
                    if result is not None:
                        # Replace operation with constant
                        for out in op.outputs:
                            tensor = graph.tensors.get(out)
                            if tensor:
                                tensor.is_constant = True
                                tensor.data = result
                        
                        folded.append(op)
                        self.optimization_stats['constants_folded'] += 1
                        if self.verbose:
                            logger.info(f"[ConstFold] Folded operation: {op.name}")
                except Exception as e:
                    logger.warning(f"[ConstFold] Failed to fold {op.name}: {e}")
        
        # Remove folded operations
        for op in folded:
            graph.remove_operation(op)
        
        return graph
    
    def _evaluate_constant_op(
        self, op: Operation, graph: ComputationGraph
    ) -> Optional[np.ndarray]:
        """Evaluate a constant operation"""
        # Get input data
        inputs = [graph.tensors[inp].data for inp in op.inputs]
        
        if any(x is None for x in inputs):
            return None
        
        # Evaluate based on operation type
        if op.op_type == OpType.ADD:
            return inputs[0] + inputs[1]
        elif op.op_type == OpType.MUL:
            return inputs[0] * inputs[1]
        elif op.op_type == OpType.RESHAPE:
            shape = op.attributes.get('shape')
            if shape:
                return inputs[0].reshape(shape)
        
        return None
    
    def remove_identity_ops(self, graph: ComputationGraph) -> ComputationGraph:
        """
        Remove identity operations (ops that don't change data).
        
        Examples: Identity, Dropout(training=False), certain Reshapes
        """
        removed = []
        
        for op in graph.operations:
            if op.op_type == OpType.IDENTITY:
                # Bypass identity operation
                if len(op.inputs) == 1 and len(op.outputs) == 1:
                    input_tensor = op.inputs[0]
                    output_tensor = op.outputs[0]
                    
                    # Update consumers to use input directly
                    for consumer in graph.get_consumers(output_tensor):
                        consumer.inputs = [
                            input_tensor if inp == output_tensor else inp
                            for inp in consumer.inputs
                        ]
                    
                    removed.append(op)
                    self.optimization_stats['identity_removed'] += 1
                    if self.verbose:
                        logger.info(f"[Identity] Removed: {op.name}")
            
            elif op.op_type == OpType.DROPOUT:
                # Remove dropout in inference mode
                if not op.attributes.get('training', False):
                    removed.append(op)
                    self.optimization_stats['dropout_removed'] += 1
        
        for op in removed:
            graph.remove_operation(op)
        
        return graph
    
    def eliminate_common_subexpressions(self, graph: ComputationGraph) -> ComputationGraph:
        """
        Eliminate common subexpressions (CSE).
        
        If two operations are identical (same type, inputs, attributes),
        compute once and reuse the result.
        """
        # Group operations by signature
        signatures: Dict[str, List[Operation]] = defaultdict(list)
        
        for op in graph.operations:
            sig = self._operation_signature(op)
            signatures[sig].append(op)
        
        # Find duplicates
        for sig, ops in signatures.items():
            if len(ops) > 1:
                # Keep first, replace others
                canonical = ops[0]
                for dup in ops[1:]:
                    # Redirect consumers to use canonical output
                    for i, out in enumerate(dup.outputs):
                        canonical_out = canonical.outputs[i] if i < len(canonical.outputs) else None
                        if canonical_out:
                            for consumer in graph.get_consumers(out):
                                consumer.inputs = [
                                    canonical_out if inp == out else inp
                                    for inp in consumer.inputs
                                ]
                    
                    graph.remove_operation(dup)
                    self.optimization_stats['cse_eliminated'] += 1
                    if self.verbose:
                        logger.info(f"[CSE] Eliminated duplicate: {dup.name}")
        
        return graph
    
    def _operation_signature(self, op: Operation) -> str:
        """Create signature for operation"""
        return f"{op.op_type.value}:{','.join(sorted(op.inputs))}:{str(sorted(op.attributes.items()))}"
    
    def simplify_algebra(self, graph: ComputationGraph) -> ComputationGraph:
        """
        Apply algebraic simplifications.
        
        Examples:
        - x + 0 = x
        - x * 1 = x
        - x * 0 = 0
        """
        simplified = []
        
        for op in graph.operations:
            if op.op_type == OpType.ADD:
                # Check for x + 0
                for inp in op.inputs:
                    tensor = graph.tensors.get(inp)
                    if tensor and tensor.is_constant and tensor.data is not None:
                        if np.allclose(tensor.data, 0):
                            # x + 0 = x, bypass operation
                            simplified.append(op)
                            self.optimization_stats['algebra_simplified'] += 1
                            break
            
            elif op.op_type == OpType.MUL:
                # Check for x * 1
                for inp in op.inputs:
                    tensor = graph.tensors.get(inp)
                    if tensor and tensor.is_constant and tensor.data is not None:
                        if np.allclose(tensor.data, 1):
                            simplified.append(op)
                            self.optimization_stats['algebra_simplified'] += 1
                            break
        
        for op in simplified:
            # Bypass the operation
            if len(op.inputs) == 2 and len(op.outputs) == 1:
                # Find non-constant input
                non_const = [inp for inp in op.inputs 
                           if not graph.tensors.get(inp, TensorInfo(inp, ())).is_constant]
                if non_const:
                    input_tensor = non_const[0]
                    output_tensor = op.outputs[0]
                    
                    for consumer in graph.get_consumers(output_tensor):
                        consumer.inputs = [
                            input_tensor if inp == output_tensor else inp
                            for inp in consumer.inputs
                        ]
                    
                    graph.remove_operation(op)
        
        return graph


# =============================================================================
# OPERATOR FUSION
# =============================================================================

class OperatorFusion:
    """
    Fuse multiple operations into single optimized kernels.
    
    Common fusion patterns:
    1. Conv + BatchNorm + ReLU → ConvBNReLU
    2. MatMul + Add → FusedLinear
    3. LayerNorm + GELU → FusedLayerNormGELU
    4. Attention blocks (Q, K, V computations)
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.fusion_stats = defaultdict(int)
    
    def fuse(self, graph: ComputationGraph) -> ComputationGraph:
        """
        Apply operator fusion to graph.
        
        Args:
            graph: Input computation graph
            
        Returns:
            Graph with fused operations
        """
        if self.verbose:
            logger.info(f"[OpFusion] Starting operator fusion")
        
        # Apply fusion patterns
        graph = self.fuse_conv_bn_relu(graph)
        graph = self.fuse_matmul_add(graph)
        graph = self.fuse_layernorm_activation(graph)
        
        if self.verbose:
            logger.info(f"[OpFusion] Fusions applied: {dict(self.fusion_stats)}")
        
        return graph
    
    def fuse_conv_bn_relu(self, graph: ComputationGraph) -> ComputationGraph:
        """
        Fuse Conv2D + BatchNorm + ReLU into single operation.
        
        Pattern: Conv → BN → ReLU
        Benefit: 3 kernel launches → 1, better memory locality
        """
        fused = []
        
        for i, op in enumerate(graph.operations):
            if op.op_type != OpType.CONV2D:
                continue
            
            # Check if followed by BatchNorm
            conv_out = op.outputs[0] if op.outputs else None
            if not conv_out:
                continue
            
            bn_ops = [o for o in graph.operations 
                     if o.op_type == OpType.BATCH_NORM and conv_out in o.inputs]
            
            if not bn_ops:
                continue
            
            bn_op = bn_ops[0]
            bn_out = bn_op.outputs[0] if bn_op.outputs else None
            
            if not bn_out:
                continue
            
            # Check if followed by ReLU
            relu_ops = [o for o in graph.operations 
                       if o.op_type == OpType.RELU and bn_out in o.inputs]
            
            if not relu_ops:
                continue
            
            relu_op = relu_ops[0]
            
            # Create fused operation
            fused_op = Operation(
                name=f"fused_conv_bn_relu_{len(fused)}",
                op_type=OpType.CONV2D,  # Keep as conv but mark as fused
                inputs=op.inputs,
                outputs=relu_op.outputs,
                attributes={
                    **op.attributes,
                    'fused': True,
                    'fusion_pattern': 'conv_bn_relu',
                    'bn_params': bn_op.attributes,
                }
            )
            
            # Replace in graph
            graph.add_operation(fused_op)
            graph.remove_operation(op)
            graph.remove_operation(bn_op)
            graph.remove_operation(relu_op)
            
            fused.append((op, bn_op, relu_op))
            self.fusion_stats['conv_bn_relu'] += 1
            
            if self.verbose:
                logger.info(f"[Fusion] Conv+BN+ReLU → {fused_op.name}")
        
        return graph
    
    def fuse_matmul_add(self, graph: ComputationGraph) -> ComputationGraph:
        """
        Fuse MatMul + Add (bias) into single operation.
        
        Pattern: MatMul → Add (with constant bias)
        Benefit: Eliminate separate bias add kernel
        """
        fused = []
        
        for op in graph.operations:
            if op.op_type != OpType.MATMUL:
                continue
            
            matmul_out = op.outputs[0] if op.outputs else None
            if not matmul_out:
                continue
            
            # Check for Add with constant
            add_ops = [o for o in graph.operations 
                      if o.op_type == OpType.ADD and matmul_out in o.inputs]
            
            for add_op in add_ops:
                # Check if other input is constant (bias)
                other_inputs = [inp for inp in add_op.inputs if inp != matmul_out]
                if other_inputs:
                    bias_tensor = graph.tensors.get(other_inputs[0])
                    if bias_tensor and bias_tensor.is_constant:
                        # Fuse MatMul + Add
                        fused_op = Operation(
                            name=f"fused_linear_{len(fused)}",
                            op_type=OpType.MATMUL,
                            inputs=op.inputs + [other_inputs[0]],
                            outputs=add_op.outputs,
                            attributes={
                                **op.attributes,
                                'fused': True,
                                'fusion_pattern': 'matmul_add',
                                'has_bias': True,
                            }
                        )
                        
                        graph.add_operation(fused_op)
                        graph.remove_operation(op)
                        graph.remove_operation(add_op)
                        
                        fused.append((op, add_op))
                        self.fusion_stats['matmul_add'] += 1
                        
                        if self.verbose:
                            logger.info(f"[Fusion] MatMul+Add → {fused_op.name}")
        
        return graph
    
    def fuse_layernorm_activation(self, graph: ComputationGraph) -> ComputationGraph:
        """
        Fuse LayerNorm + GELU/ReLU.
        
        Pattern: LayerNorm → GELU/ReLU
        Benefit: Combined normalization + activation
        """
        fused = []
        
        for op in graph.operations:
            if op.op_type != OpType.LAYERNORM:
                continue
            
            ln_out = op.outputs[0] if op.outputs else None
            if not ln_out:
                continue
            
            # Check for activation
            act_ops = [o for o in graph.operations 
                      if o.op_type in [OpType.GELU, OpType.RELU] and ln_out in o.inputs]
            
            if act_ops:
                act_op = act_ops[0]
                
                fused_op = Operation(
                    name=f"fused_ln_act_{len(fused)}",
                    op_type=OpType.LAYERNORM,
                    inputs=op.inputs,
                    outputs=act_op.outputs,
                    attributes={
                        **op.attributes,
                        'fused': True,
                        'fusion_pattern': 'layernorm_activation',
                        'activation': act_op.op_type.value,
                    }
                )
                
                graph.add_operation(fused_op)
                graph.remove_operation(op)
                graph.remove_operation(act_op)
                
                fused.append((op, act_op))
                self.fusion_stats['layernorm_activation'] += 1
                
                if self.verbose:
                    logger.info(f"[Fusion] LayerNorm+{act_op.op_type.value} → {fused_op.name}")
        
        return graph


# =============================================================================
# MEMORY LAYOUT OPTIMIZATION
# =============================================================================

class MemoryLayoutOptimizer:
    """
    Optimize tensor memory layouts for hardware efficiency.
    
    Strategies:
    1. Convert NCHW ↔ NHWC based on operation characteristics
    2. Minimize layout conversions (transpose operations)
    3. Prefer layouts that maximize memory coalescing
    
    Hardware considerations:
    - AMD GPUs: NHWC often better for compute (better coalescing)
    - Mobile: NHWC preferred (ARM NEON optimization)
    - CPU: NCHW often better (cache locality)
    """
    
    def __init__(self, target_device: str = 'amd_gpu', verbose: bool = True):
        """
        Args:
            target_device: 'amd_gpu', 'cpu', 'mobile'
        """
        self.target_device = target_device
        self.verbose = verbose
        self.layout_stats = defaultdict(int)
    
    def optimize(self, graph: ComputationGraph) -> ComputationGraph:
        """
        Optimize memory layouts in graph.
        
        Args:
            graph: Input computation graph
            
        Returns:
            Graph with optimized layouts
        """
        if self.verbose:
            logger.info(f"[LayoutOpt] Optimizing for: {self.target_device}")
        
        # Determine optimal layout for target
        if self.target_device == 'mobile':
            target_layout = MemoryLayout.NHWC
        elif self.target_device == 'amd_gpu':
            target_layout = MemoryLayout.NHWC  # Better memory coalescing
        else:
            target_layout = MemoryLayout.NCHW  # CPU default
        
        graph = self.convert_layouts(graph, target_layout)
        graph = self.minimize_transposes(graph)
        
        if self.verbose:
            logger.info(f"[LayoutOpt] Conversions: {dict(self.layout_stats)}")
        
        return graph
    
    def convert_layouts(
        self, graph: ComputationGraph, target: MemoryLayout
    ) -> ComputationGraph:
        """Convert tensors to target layout"""
        for tensor_name, tensor in graph.tensors.items():
            if tensor.layout != target and tensor.layout != MemoryLayout.UNKNOWN:
                # Insert transpose operation if needed
                if len(tensor.shape) == 4:  # Conv tensors
                    self.layout_stats[f'{tensor.layout.value}_to_{target.value}'] += 1
                    tensor.layout = target
        
        return graph
    
    def minimize_transposes(self, graph: ComputationGraph) -> ComputationGraph:
        """
        Minimize number of transpose operations.
        
        Strategy: Propagate layouts through graph, insert transposes only
        where absolutely necessary.
        """
        # Remove back-to-back transposes
        removed = []
        
        for i, op in enumerate(graph.operations[:-1]):
            if op.op_type == OpType.TRANSPOSE:
                next_op = graph.operations[i + 1]
                if next_op.op_type == OpType.TRANSPOSE:
                    # Check if they cancel out
                    perm1 = op.attributes.get('perm', [])
                    perm2 = next_op.attributes.get('perm', [])
                    
                    # If permutations are inverses, remove both
                    if self._are_inverse_perms(perm1, perm2):
                        removed.extend([op, next_op])
                        self.layout_stats['transpose_eliminated'] += 2
                        
                        if self.verbose:
                            logger.info(f"[LayoutOpt] Eliminated transpose pair")
        
        for op in removed:
            graph.remove_operation(op)
        
        return graph
    
    def _are_inverse_perms(self, perm1: List[int], perm2: List[int]) -> bool:
        """Check if two permutations are inverses"""
        if len(perm1) != len(perm2):
            return False
        
        # Apply perm1 then perm2, should give identity
        for i in range(len(perm1)):
            if perm2[perm1[i]] != i:
                return False
        
        return True


# =============================================================================
# INTEGRATED OPTIMIZATION PIPELINE
# =============================================================================

class OptimizationPipeline:
    """
    Complete optimization pipeline combining all optimizations.
    
    Pipeline stages:
    1. Graph optimization (DCE, constant folding, CSE)
    2. Operator fusion (Conv+BN+ReLU, MatMul+Add, etc.)
    3. Memory layout optimization (NCHW/NHWC conversion)
    
    Usage:
        >>> pipeline = OptimizationPipeline(target_device='amd_gpu')
        >>> optimized_graph = pipeline.optimize(graph)
    """
    
    def __init__(
        self,
        target_device: str = 'amd_gpu',
        enable_fusion: bool = True,
        enable_layout_opt: bool = True,
        verbose: bool = True
    ):
        """
        Args:
            target_device: Target hardware ('amd_gpu', 'cpu', 'mobile')
            enable_fusion: Enable operator fusion
            enable_layout_opt: Enable memory layout optimization
            verbose: Print optimization progress
        """
        self.target_device = target_device
        self.enable_fusion = enable_fusion
        self.enable_layout_opt = enable_layout_opt
        self.verbose = verbose
        
        # Create optimizers
        self.graph_optimizer = GraphOptimizer(verbose=verbose)
        self.op_fusion = OperatorFusion(verbose=verbose) if enable_fusion else None
        self.layout_optimizer = MemoryLayoutOptimizer(
            target_device=target_device, verbose=verbose
        ) if enable_layout_opt else None
    
    def optimize(self, graph: ComputationGraph) -> ComputationGraph:
        """
        Run complete optimization pipeline.
        
        Args:
            graph: Input computation graph
            
        Returns:
            Fully optimized graph
        """
        if self.verbose:
            logger.info("=" * 70)
            logger.info(f"[Pipeline] Starting optimization pipeline")
            logger.info(f"[Pipeline] Target device: {self.target_device}")
            logger.info(f"[Pipeline] Initial operations: {len(graph.operations)}")
            logger.info("=" * 70)
        
        original_ops = len(graph.operations)
        
        # Stage 1: Graph optimizations
        if self.verbose:
            logger.info("\n[Stage 1] Graph Optimization")
        graph = self.graph_optimizer.optimize(graph)
        
        # Stage 2: Operator fusion
        if self.enable_fusion and self.op_fusion:
            if self.verbose:
                logger.info("\n[Stage 2] Operator Fusion")
            graph = self.op_fusion.fuse(graph)
        
        # Stage 3: Layout optimization
        if self.enable_layout_opt and self.layout_optimizer:
            if self.verbose:
                logger.info("\n[Stage 3] Memory Layout Optimization")
            graph = self.layout_optimizer.optimize(graph)
        
        # Final stats
        optimized_ops = len(graph.operations)
        reduction = (1 - optimized_ops / original_ops) * 100 if original_ops > 0 else 0
        
        if self.verbose:
            logger.info("=" * 70)
            logger.info(f"[Pipeline] Optimization complete!")
            logger.info(f"[Pipeline] Operations: {original_ops} → {optimized_ops} ({reduction:.1f}% reduction)")
            logger.info("=" * 70)
        
        return graph
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get detailed optimization statistics"""
        report = {
            'target_device': self.target_device,
            'graph_optimizations': dict(self.graph_optimizer.optimization_stats),
        }
        
        if self.op_fusion:
            report['operator_fusions'] = dict(self.op_fusion.fusion_stats)
        
        if self.layout_optimizer:
            report['layout_optimizations'] = dict(self.layout_optimizer.layout_stats)
        
        return report


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_optimization_pipeline(
    target_device: str = 'amd_gpu',
    optimization_level: int = 2
) -> OptimizationPipeline:
    """
    Factory to create optimization pipeline with preset configurations.
    
    Args:
        target_device: 'amd_gpu', 'cpu', or 'mobile'
        optimization_level: 0=disabled, 1=basic, 2=aggressive
        
    Returns:
        Configured optimization pipeline
    """
    if optimization_level == 0:
        return OptimizationPipeline(
            target_device=target_device,
            enable_fusion=False,
            enable_layout_opt=False,
            verbose=False
        )
    elif optimization_level == 1:
        return OptimizationPipeline(
            target_device=target_device,
            enable_fusion=True,
            enable_layout_opt=False,
            verbose=True
        )
    else:  # Level 2
        return OptimizationPipeline(
            target_device=target_device,
            enable_fusion=True,
            enable_layout_opt=True,
            verbose=True
        )


__all__ = [
    'OpType',
    'MemoryLayout',
    'TensorInfo',
    'Operation',
    'ComputationGraph',
    'GraphOptimizer',
    'OperatorFusion',
    'MemoryLayoutOptimizer',
    'OptimizationPipeline',
    'create_optimization_pipeline',
]
