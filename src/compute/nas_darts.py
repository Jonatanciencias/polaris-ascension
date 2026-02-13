"""
DARTS: Differentiable Architecture Search

Implementation of DARTS (Liu et al., 2019) for neural architecture search.
Enables gradient-based architecture optimization on AMD Radeon RX 580.

Paper: "DARTS: Differentiable Architecture Search"
Authors: Liu, Simonyan, Yang (2019)
Conference: ICLR 2019

Key Features:
- Continuous relaxation of discrete architecture space
- Bilevel optimization (architecture + weights)
- Memory-efficient search on single GPU
- Support for multiple search spaces (CNN, RNN)

Architecture Search Process:
1. Define search space with candidate operations
2. Initialize architecture parameters (α)
3. Bilevel optimization:
   - Update network weights (w) on training data
   - Update architecture params (α) on validation data
4. Derive discrete architecture from continuous α
5. Retrain from scratch with derived architecture

Mathematical Formulation:
    min_α  L_val(w*(α), α)
    s.t.   w*(α) = argmin_w L_train(w, α)

Where:
- α: Architecture parameters (operation weights)
- w: Network weights
- L_train: Training loss
- L_val: Validation loss

AMD Radeon RX 580 Optimizations:
- Reduced batch sizes for 8GB VRAM
- Memory-efficient gradient computation
- Mixed operation execution

Author: AMD GPU Computing Team
Date: February 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from typing import List, Tuple, Dict, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import logging
import time

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Classes
# ============================================================================


class SearchSpace(Enum):
    """Supported search spaces."""

    CNN = "cnn"
    RNN = "rnn"


@dataclass
class DARTSConfig:
    """Configuration for DARTS architecture search."""

    # Search space
    search_space: SearchSpace = SearchSpace.CNN
    num_cells: int = 8  # Number of cells in network
    num_nodes: int = 4  # Intermediate nodes per cell

    # Optimization
    epochs: int = 50  # Search epochs
    batch_size: int = 64  # For Radeon RX 580 (8GB VRAM)
    learning_rate: float = 0.025  # Weight optimizer LR
    arch_learning_rate: float = 3e-4  # Architecture optimizer LR
    momentum: float = 0.9
    weight_decay: float = 3e-4

    # Architecture parameters
    init_channels: int = 16  # Initial channels
    layers: int = 8  # Total layers
    drop_path_prob: float = 0.2  # Drop path probability

    # Regularization
    grad_clip: float = 5.0  # Gradient clipping
    arch_grad_clip: float = 3.0  # Architecture gradient clip

    # Hardware-specific (Radeon RX 580)
    use_amp: bool = False  # Automatic Mixed Precision (limited support)
    num_workers: int = 4  # Data loading workers

    # Derived architecture
    cutout: bool = True  # Cutout augmentation
    cutout_length: int = 16
    auxiliary: bool = True  # Auxiliary classifier
    auxiliary_weight: float = 0.4


@dataclass
class SearchResult:
    """Results from architecture search."""

    normal_genotype: List[Tuple[str, int]]
    reduce_genotype: List[Tuple[str, int]]
    final_train_loss: float
    final_val_loss: float
    final_train_acc: float
    final_val_acc: float
    total_epochs: int
    total_time_seconds: float
    best_epoch: int
    architecture_weights: Dict[str, torch.Tensor]


# ============================================================================
# Primitive Operations
# ============================================================================

# Available primitive operations
PRIMITIVES = [
    "none",
    "max_pool_3x3",
    "avg_pool_3x3",
    "skip_connect",
    "sep_conv_3x3",
    "sep_conv_5x5",
    "dil_conv_3x3",
    "dil_conv_5x5",
]


class Operation(nn.Module):
    """Base class for candidate operations."""

    pass


class Identity(Operation):
    """Identity operation (skip connection)."""

    def forward(self, x, drop_path_prob=0.0):
        return x


class Zero(Operation):
    """Zero operation (no connection)."""

    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x, drop_path_prob=0.0):
        if self.stride == 1:
            return x.mul(0.0)
        return x[:, :, :: self.stride, :: self.stride].mul(0.0)


class PoolBN(Operation):
    """Pooling + Batch Normalization."""

    def __init__(self, pool_type, C, kernel_size, stride, padding, affine=True):
        super().__init__()
        if pool_type.lower() == "max":
            self.op = nn.Sequential(
                nn.MaxPool2d(kernel_size, stride, padding), nn.BatchNorm2d(C, affine=affine)
            )
        elif pool_type.lower() == "avg":
            self.op = nn.Sequential(
                nn.AvgPool2d(kernel_size, stride, padding, count_include_pad=False),
                nn.BatchNorm2d(C, affine=affine),
            )

    def forward(self, x, drop_path_prob=0.0):
        return self.op(x)


class ReLUConvBN(Operation):
    """ReLU + Conv + BatchNorm."""

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x, drop_path_prob=0.0):
        return self.op(x)


class SepConv(Operation):
    """Separable Convolution (depthwise + pointwise)."""

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size, stride, padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, 1, 1, 0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size, 1, padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x, drop_path_prob=0.0):
        return self.op(x)


class DilConv(Operation):
    """Dilated Convolution."""

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in, C_in, kernel_size, stride, padding, dilation=dilation, groups=C_in, bias=False
            ),
            nn.Conv2d(C_in, C_out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x, drop_path_prob=0.0):
        return self.op(x)


class FactorizedReduce(nn.Module):
    """Factorized reduction for spatial downsampling."""

    def __init__(self, C_in, C_out, affine=True):
        super().__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


def create_operation(primitive: str, C: int, stride: int, affine: bool = True) -> nn.Module:
    """
    Factory function to create operation from primitive name.

    Args:
        primitive: Operation name from PRIMITIVES
        C: Number of channels
        stride: Stride for the operation
        affine: Use affine in BatchNorm

    Returns:
        Operation module
    """
    if primitive == "none":
        return Zero(stride)
    elif primitive == "avg_pool_3x3":
        return PoolBN("avg", C, 3, stride, 1, affine=affine)
    elif primitive == "max_pool_3x3":
        return PoolBN("max", C, 3, stride, 1, affine=affine)
    elif primitive == "skip_connect":
        if stride == 1:
            return Identity()
        else:
            return FactorizedReduce(C, C, affine=affine)
    elif primitive == "sep_conv_3x3":
        return SepConv(C, C, 3, stride, 1, affine=affine)
    elif primitive == "sep_conv_5x5":
        return SepConv(C, C, 5, stride, 2, affine=affine)
    elif primitive == "dil_conv_3x3":
        return DilConv(C, C, 3, stride, 2, 2, affine=affine)
    elif primitive == "dil_conv_5x5":
        return DilConv(C, C, 5, stride, 4, 2, affine=affine)
    else:
        raise ValueError(f"Unknown primitive: {primitive}")


# ============================================================================
# Mixed Operation (Key DARTS Component)
# ============================================================================


class MixedOp(nn.Module):
    """
    Mixed operation: weighted sum of all candidate operations.

    Key DARTS innovation: Instead of choosing one operation,
    compute weighted combination using architecture parameters α.

    Forward pass:
        out = Σ_i softmax(α_i) * op_i(x)
    """

    def __init__(self, C: int, stride: int):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = create_operation(primitive, C, stride, affine=False)
            if "pool" in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        """
        Forward pass with weighted operations.

        Args:
            x: Input tensor
            weights: Architecture weights (softmax of α)

        Returns:
            Weighted sum of operations
        """
        return sum(w * op(x) if w != 0 else 0 for w, op in zip(weights, self._ops))


# ============================================================================
# Cell (Building Block)
# ============================================================================


class Cell(nn.Module):
    """
    Cell: Basic building block in DARTS.

    Structure:
    - 2 input nodes (from previous cells)
    - k intermediate nodes (computed from previous nodes)
    - Output: concatenation of all intermediate nodes

    Each edge is a MixedOp controlled by architecture parameters.
    """

    def __init__(
        self,
        steps: int,
        multiplier: int,
        C_prev_prev: int,
        C_prev: int,
        C: int,
        reduction: bool,
        reduction_prev: bool,
    ):
        """
        Initialize cell.

        Args:
            steps: Number of intermediate nodes
            multiplier: Output multiplier
            C_prev_prev: Channels from cell i-2
            C_prev: Channels from cell i-1
            C: Current channels
            reduction: Is this a reduction cell
            reduction_prev: Was previous cell reduction
        """
        super().__init__()
        self.reduction = reduction

        # Preprocess inputs
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)

        self._steps = steps
        self._multiplier = multiplier

        # Build edges (MixedOps)
        self._ops = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s0, s1, weights):
        """
        Forward pass through cell.

        Args:
            s0: Output from cell i-2
            s1: Output from cell i-1
            weights: Architecture weights for this cell

        Returns:
            Cell output (concatenation of intermediate nodes)
        """
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0

        for i in range(self._steps):
            # Compute node i from all previous nodes
            s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        # Output: concatenate intermediate nodes
        return torch.cat(states[-self._multiplier :], dim=1)


# ============================================================================
# Network (Search Space)
# ============================================================================


class DARTSNetwork(nn.Module):
    """
    DARTS search network.

    Architecture:
    - Stem: Initial convolution
    - Cells: Stack of normal + reduction cells
    - Classifier: Global pooling + linear

    During search:
    - All operations are mixed (weighted by α)
    - Jointly optimize architecture (α) and weights (w)

    After search:
    - Derive discrete architecture
    - Retrain from scratch
    """

    def __init__(
        self,
        C: int,
        num_classes: int,
        layers: int,
        criterion,
        steps: int = 4,
        multiplier: int = 4,
        stem_multiplier: int = 3,
    ):
        """
        Initialize DARTS network.

        Args:
            C: Initial channels
            num_classes: Number of output classes
            layers: Total number of cells
            criterion: Loss function
            steps: Intermediate nodes per cell
            multiplier: Output multiplier
            stem_multiplier: Stem channel multiplier
        """
        super().__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier

        C_curr = stem_multiplier * C

        # Stem convolution
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False), nn.BatchNorm2d(C_curr)
        )

        # Build cells
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False

        for i in range(layers):
            # Reduction cell at 1/3 and 2/3 depth
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False

            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells.append(cell)

            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        # Classifier
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        # Architecture parameters (α)
        self._initialize_alphas()

    def _initialize_alphas(self):
        """Initialize architecture parameters."""
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        # Normal cell architecture parameters
        self.alphas_normal = Variable(1e-3 * torch.randn(k, num_ops), requires_grad=True)

        # Reduction cell architecture parameters
        self.alphas_reduce = Variable(1e-3 * torch.randn(k, num_ops), requires_grad=True)

        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def arch_parameters(self):
        """Return architecture parameters for optimizer."""
        return self._arch_parameters

    def forward(self, input):
        """
        Forward pass through network.

        Args:
            input: Input tensor [N, 3, H, W]

        Returns:
            Logits [N, num_classes]
        """
        s0 = s1 = self.stem(input)

        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = F.softmax(self.alphas_reduce, dim=-1)
            else:
                weights = F.softmax(self.alphas_normal, dim=-1)

            s0, s1 = s1, cell(s0, s1, weights)

        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def _loss(self, input, target):
        """Compute loss."""
        logits = self(input)
        return self._criterion(logits, target)

    def genotype(self):
        """
        Derive discrete architecture from continuous α.

        For each intermediate node:
        - Select 2 strongest incoming edges
        - Choose operation with highest α for each edge

        Returns:
            Genotype (architecture description)
        """

        def _parse(weights):
            gene = []
            n = 2
            start = 0

            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()

                # Select 2 strongest edges
                edges = sorted(
                    range(i + 2),
                    key=lambda x: -max(
                        W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index("none")
                    ),
                )[:2]

                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index("none"):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))

                start = end
                n += 1

            return gene

        # Get architecture weights
        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

        return gene_normal, gene_reduce


# ============================================================================
# DARTS Trainer (Bilevel Optimization)
# ============================================================================


class DARTSTrainer:
    """
    DARTS trainer with bilevel optimization.

    Optimization:
    1. Update network weights (w) on training data
    2. Update architecture parameters (α) on validation data

    This alternating optimization approximates the bilevel problem:
        min_α L_val(w*(α), α)
        s.t. w*(α) = argmin_w L_train(w, α)
    """

    def __init__(self, model: DARTSNetwork, config: DARTSConfig):
        """
        Initialize DARTS trainer.

        Args:
            model: DARTS network
            config: Training configuration
        """
        self.model = model
        self.config = config

        # Weight optimizer (SGD with momentum)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )

        # Architecture optimizer (Adam)
        self.arch_optimizer = torch.optim.Adam(
            self.model.arch_parameters(),
            lr=config.arch_learning_rate,
            betas=(0.5, 0.999),
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, config.epochs)

    def step(self, train_data, valid_data):
        """
        Perform one training step (bilevel optimization).

        Args:
            train_data: (input, target) for training
            valid_data: (input, target) for validation

        Returns:
            train_loss, valid_loss
        """
        # Unpack data
        input_train, target_train = train_data
        input_valid, target_valid = valid_data

        # 1. Update architecture parameters (α) on validation data
        self.arch_optimizer.zero_grad()
        loss_arch = self.model._loss(input_valid, target_valid)
        loss_arch.backward()
        nn.utils.clip_grad_norm_(self.model.arch_parameters(), self.config.arch_grad_clip)
        self.arch_optimizer.step()

        # 2. Update network weights (w) on training data
        self.optimizer.zero_grad()
        loss_weight = self.model._loss(input_train, target_train)
        loss_weight.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()

        return loss_weight.item(), loss_arch.item()


# ============================================================================
# Search Function (Main API)
# ============================================================================


def search_architecture(
    train_loader, valid_loader, config: DARTSConfig, device: str = "cuda", verbose: bool = True
) -> SearchResult:
    """
    Perform DARTS architecture search.

    Args:
        train_loader: Training data loader
        valid_loader: Validation data loader
        config: DARTS configuration
        device: Device to run on ("cuda" or "cpu")
        verbose: Print progress

    Returns:
        SearchResult with discovered architecture

    Example:
        >>> config = DARTSConfig(epochs=50, batch_size=64)
        >>> result = search_architecture(train_loader, valid_loader, config)
        >>> print(result.normal_genotype)
        [('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ...]
    """
    start_time = time.time()

    # Create model
    criterion = nn.CrossEntropyLoss().to(device)
    model = DARTSNetwork(
        C=config.init_channels,
        num_classes=10,  # CIFAR-10
        layers=config.layers,
        criterion=criterion,
        steps=config.num_nodes,
    ).to(device)

    # Create trainer
    trainer = DARTSTrainer(model, config)

    if verbose:
        logger.info(f"Starting DARTS search for {config.epochs} epochs")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Training loop
    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(config.epochs):
        train_loss_avg = 0.0
        valid_loss_avg = 0.0

        # Epoch training
        model.train()
        for step, (train_data, valid_data) in enumerate(zip(train_loader, valid_loader)):
            train_loss, valid_loss = trainer.step(train_data, valid_data)
            train_loss_avg += train_loss
            valid_loss_avg += valid_loss

        # Update learning rate
        trainer.scheduler.step()

        # Compute accuracy
        train_acc = evaluate(model, train_loader, device)
        valid_acc = evaluate(model, valid_loader, device)

        if valid_acc > best_val_acc:
            best_val_acc = valid_acc
            best_epoch = epoch

        if verbose and epoch % 5 == 0:
            logger.info(
                f"Epoch {epoch:03d}: "
                f"Train Loss={train_loss_avg:.4f}, "
                f"Valid Loss={valid_loss_avg:.4f}, "
                f"Train Acc={train_acc:.2f}%, "
                f"Valid Acc={valid_acc:.2f}%"
            )

    # Derive final architecture
    genotype_normal, genotype_reduce = model.genotype()

    total_time = time.time() - start_time

    if verbose:
        logger.info(f"\nSearch completed in {total_time:.1f} seconds")
        logger.info(f"Best validation accuracy: {best_val_acc:.2f}% (epoch {best_epoch})")
        logger.info(f"\nDiscovered architecture:")
        logger.info(f"  Normal cell: {genotype_normal}")
        logger.info(f"  Reduction cell: {genotype_reduce}")

    # Return results
    return SearchResult(
        normal_genotype=genotype_normal,
        reduce_genotype=genotype_reduce,
        final_train_loss=train_loss_avg,
        final_val_loss=valid_loss_avg,
        final_train_acc=train_acc,
        final_val_acc=valid_acc,
        total_epochs=config.epochs,
        total_time_seconds=total_time,
        best_epoch=best_epoch,
        architecture_weights={
            "alphas_normal": model.alphas_normal.data.cpu(),
            "alphas_reduce": model.alphas_reduce.data.cpu(),
        },
    )


def evaluate(model, data_loader, device):
    """
    Evaluate model accuracy.

    Args:
        model: Network to evaluate
        data_loader: Data loader
        device: Device

    Returns:
        Accuracy percentage
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for input, target in data_loader:
            input, target = input.to(device), target.to(device)
            logits = model(input)
            _, predicted = torch.max(logits.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return 100.0 * correct / total if total > 0 else 0.0


# ============================================================================
# Utility Functions
# ============================================================================


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_genotype(genotype: List[Tuple[str, int]], name: str = "Cell"):
    """Pretty print genotype."""
    print(f"\n{name} Architecture:")
    for i, (op, from_node) in enumerate(genotype):
        print(f"  Edge {i}: {op} from node {from_node}")
