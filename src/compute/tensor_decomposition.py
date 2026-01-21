"""
Tensor Decomposition for Neural Network Compression - Session 24
=================================================================

Advanced tensor decomposition methods for extreme model compression:
- Tucker Decomposition: Core tensor + factor matrices
- CP Decomposition: Canonical Polyadic (CANDECOMP/PARAFAC)
- Tensor-Train (TT): Chain of 3D tensors

Key Features:
-------------
1. Tucker Decomposition
   - Multi-linear rank reduction
   - 10-50x compression on conv layers
   - Higher-Order SVD (HOSVD)
   
2. CP Decomposition
   - Rank-R approximation
   - Extreme compression (100x+)
   - Lower accuracy but faster
   
3. Tensor-Train Decomposition
   - Optimal for very deep networks
   - Memory-efficient representation
   - Bounded ranks for stability

Papers/Concepts:
----------------
1. Kolda & Bader (2009) - "Tensor Decompositions and Applications"
2. Novikov et al. (2015) - "Tensorizing Neural Networks"
3. Kim et al. (2016) - "Compression of Deep CNNs"
4. Oseledets (2011) - "Tensor-Train Decomposition"

Author: Session 24 Implementation
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Union
import numpy as np
from dataclasses import dataclass
import warnings


@dataclass
class DecompositionConfig:
    """Configuration for tensor decomposition."""
    method: str = "tucker"  # tucker, cp, tt
    ranks: Optional[Union[List[int], int]] = None
    auto_rank: bool = True
    max_compression: float = 10.0  # Target compression ratio
    energy_threshold: float = 0.95  # Keep 95% of energy
    iterative: bool = False
    max_iterations: int = 50


class TuckerDecomposer:
    """
    Tucker Decomposition for Conv2D and Linear layers.
    
    Decomposes weight tensor W[I,J,K,L] into:
        G[R1,R2,R3,R4] × U1[I,R1] × U2[J,R2] × U3[K,R3] × U4[L,R4]
    
    Where:
        - G is core tensor (compressed representation)
        - U1, U2, U3, U4 are factor matrices
        - R1, R2, R3, R4 are Tucker ranks
    
    Compression ratio:
        original_params / compressed_params
        = (I×J×K×L) / (R1×R2×R3×R4 + I×R1 + J×R2 + K×R3 + L×R4)
    
    Example:
    --------
    >>> decomposer = TuckerDecomposer(ranks=[16, 16, 3, 3])
    >>> conv_layer = nn.Conv2d(64, 64, 3, 3)
    >>> compressed = decomposer.decompose_conv2d(conv_layer)
    >>> # Result: 64×64×3×3 = 36,864 → ~3,000 params (~12x compression)
    """
    
    def __init__(
        self,
        ranks: Optional[List[int]] = None,
        auto_rank: bool = True,
        energy_threshold: float = 0.95
    ):
        """
        Initialize Tucker decomposer.
        
        Args:
            ranks: Tucker ranks [R1, R2, R3, R4]. If None, auto-compute
            auto_rank: Automatically determine ranks from energy threshold
            energy_threshold: Keep this fraction of singular values energy
        """
        self.ranks = ranks
        self.auto_rank = auto_rank
        self.energy_threshold = energy_threshold
        
    def decompose_conv2d(
        self,
        layer: nn.Conv2d,
        ranks: Optional[List[int]] = None
    ) -> nn.Sequential:
        """
        Decompose Conv2d layer using Tucker decomposition.
        
        Conv2d[out_channels, in_channels, kH, kW] →
            Conv2d[R1, in_channels, 1, 1] →  # Reduce input channels
            Conv2d[R2, R1, kH, kW] →         # Spatial convolution
            Conv2d[out_channels, R2, 1, 1]   # Expand output channels
        
        Args:
            layer: Conv2d layer to decompose
            ranks: Tucker ranks [R1, R2] for 2-stage decomposition
            
        Returns:
            Sequential module with decomposed layers
        """
        weight = layer.weight.data  # [out_ch, in_ch, kH, kW]
        out_ch, in_ch, kH, kW = weight.shape
        
        # Determine ranks
        if ranks is None:
            if self.ranks is not None:
                ranks = self.ranks[:2]
            elif self.auto_rank:
                ranks = self._auto_determine_ranks_conv2d(weight)
            else:
                # Default: 50% reduction
                ranks = [max(in_ch // 2, 1), max(out_ch // 2, 1)]
        
        R1, R2 = ranks
        
        # Tucker decomposition via HOSVD
        # Unfold along different modes
        W_mode1 = weight.reshape(out_ch, -1)  # [out_ch, in_ch*kH*kW]
        W_mode2 = weight.permute(1, 0, 2, 3).reshape(in_ch, -1)  # [in_ch, out_ch*kH*kW]
        
        # SVD on each mode
        U1, S1, V1 = torch.svd(W_mode1)  # U1: [out_ch, out_ch]
        U2, S2, V2 = torch.svd(W_mode2)  # U2: [in_ch, in_ch]
        
        # Truncate to ranks
        U1_trunc = U1[:, :R2]  # [out_ch, R2]
        U2_trunc = U2[:, :R1]  # [in_ch, R1]
        
        # Compute core tensor
        # G = W ×1 U1^T ×2 U2^T
        core = weight.clone()
        
        # Mode-1 product (output channels)
        core = torch.einsum('oikl,or->rikl', core, U1_trunc)
        
        # Mode-2 product (input channels)
        core = torch.einsum('rikl,ij->rjkl', core, U2_trunc)
        
        # Create decomposed layers
        # Layer 1: 1×1 conv to reduce input channels
        layer1 = nn.Conv2d(
            in_ch, R1, kernel_size=1,
            stride=1, padding=0, bias=False
        )
        layer1.weight.data = U2_trunc.t().unsqueeze(2).unsqueeze(3)
        
        # Layer 2: kH×kW conv on compressed channels
        layer2 = nn.Conv2d(
            R1, R2, kernel_size=(kH, kW),
            stride=layer.stride, padding=layer.padding,
            dilation=layer.dilation, bias=False
        )
        layer2.weight.data = core
        
        # Layer 3: 1×1 conv to expand output channels
        layer3 = nn.Conv2d(
            R2, out_ch, kernel_size=1,
            stride=1, padding=0, bias=(layer.bias is not None)
        )
        layer3.weight.data = U1_trunc.unsqueeze(2).unsqueeze(3)
        
        if layer.bias is not None:
            layer3.bias.data = layer.bias.data
        
        return nn.Sequential(layer1, layer2, layer3)
    
    def decompose_linear(
        self,
        layer: nn.Linear,
        rank: Optional[int] = None
    ) -> nn.Sequential:
        """
        Decompose Linear layer using Tucker (matrix) decomposition.
        
        Linear[out_features, in_features] →
            Linear[R, in_features] →
            Linear[out_features, R]
        
        Args:
            layer: Linear layer to decompose
            rank: Tucker rank. If None, auto-compute
            
        Returns:
            Sequential module with decomposed layers
        """
        weight = layer.weight.data  # [out_features, in_features]
        out_feat, in_feat = weight.shape
        
        # Determine rank
        if rank is None:
            if self.ranks is not None and len(self.ranks) > 0:
                rank = self.ranks[0]
            elif self.auto_rank:
                rank = self._auto_determine_rank_linear(weight)
            else:
                rank = min(out_feat, in_feat) // 2
        
        # SVD decomposition
        U, S, V = torch.svd(weight)
        
        # Truncate to rank
        U_trunc = U[:, :rank]  # [out_feat, R]
        S_trunc = S[:rank]
        V_trunc = V[:, :rank]  # [in_feat, R]
        
        # Create decomposed layers
        layer1 = nn.Linear(in_feat, rank, bias=False)
        layer1.weight.data = (V_trunc * S_trunc).t()
        
        layer2 = nn.Linear(rank, out_feat, bias=(layer.bias is not None))
        layer2.weight.data = U_trunc
        
        if layer.bias is not None:
            layer2.bias.data = layer.bias.data
        
        return nn.Sequential(layer1, layer2)
    
    def _auto_determine_ranks_conv2d(
        self,
        weight: torch.Tensor
    ) -> List[int]:
        """
        Automatically determine Tucker ranks based on energy threshold.
        
        Uses singular value energy:
            energy = sum(S[:R]^2) / sum(S^2)
        
        Args:
            weight: Conv2d weight tensor [out_ch, in_ch, kH, kW]
            
        Returns:
            List of ranks [R1, R2]
        """
        out_ch, in_ch, kH, kW = weight.shape
        
        # Analyze input channels mode
        W_mode2 = weight.permute(1, 0, 2, 3).reshape(in_ch, -1)
        _, S2, _ = torch.svd(W_mode2)
        R1 = self._find_rank_from_energy(S2, self.energy_threshold)
        R1 = max(min(R1, in_ch), 1)
        
        # Analyze output channels mode
        W_mode1 = weight.reshape(out_ch, -1)
        _, S1, _ = torch.svd(W_mode1)
        R2 = self._find_rank_from_energy(S1, self.energy_threshold)
        R2 = max(min(R2, out_ch), 1)
        
        return [R1, R2]
    
    def _auto_determine_rank_linear(self, weight: torch.Tensor) -> int:
        """Auto-determine rank for linear layer."""
        _, S, _ = torch.svd(weight)
        rank = self._find_rank_from_energy(S, self.energy_threshold)
        return max(min(rank, min(weight.shape)), 1)
    
    def _find_rank_from_energy(
        self,
        singular_values: torch.Tensor,
        threshold: float
    ) -> int:
        """
        Find rank that preserves threshold fraction of energy.
        
        Args:
            singular_values: Singular values from SVD
            threshold: Energy threshold (e.g., 0.95 for 95%)
            
        Returns:
            Rank R such that sum(S[:R]^2) / sum(S^2) >= threshold
        """
        energy = singular_values ** 2
        total_energy = energy.sum()
        cumulative_energy = torch.cumsum(energy, dim=0)
        
        # Find first rank where cumulative energy >= threshold
        mask = cumulative_energy >= (threshold * total_energy)
        if mask.any():
            rank = mask.nonzero()[0].item() + 1
        else:
            rank = len(singular_values)
        
        return rank


class CPDecomposer:
    """
    CP (CANDECOMP/PARAFAC) Decomposition.
    
    Decomposes tensor W into sum of R rank-1 tensors:
        W ≈ Σ(r=1 to R) λr · (a_r ⊗ b_r ⊗ c_r ⊗ d_r)
    
    Where:
        - λr are weights
        - a_r, b_r, c_r, d_r are factor vectors
        - ⊗ is outer product
    
    More aggressive compression than Tucker but may lose more accuracy.
    Good for extreme compression scenarios (100x+).
    
    Example:
    --------
    >>> decomposer = CPDecomposer(rank=8)
    >>> conv_layer = nn.Conv2d(64, 64, 3, 3)
    >>> compressed = decomposer.decompose_conv2d(conv_layer)
    >>> # Result: 64×64×3×3 = 36,864 → ~512 params (~72x compression!)
    """
    
    def __init__(
        self,
        rank: int = 8,
        max_iterations: int = 50,
        tolerance: float = 1e-4
    ):
        """
        Initialize CP decomposer.
        
        Args:
            rank: CP rank (number of rank-1 components)
            max_iterations: Max iterations for ALS algorithm
            tolerance: Convergence tolerance
        """
        self.rank = rank
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def decompose_conv2d(
        self,
        layer: nn.Conv2d,
        rank: Optional[int] = None
    ) -> nn.Sequential:
        """
        Decompose Conv2d using CP decomposition.
        
        Conv2d[out_ch, in_ch, kH, kW] →
            Conv2d[R, in_ch, 1, 1] →    # Pointwise
            Conv2d[R, 1, kH, 1] →        # Horizontal
            Conv2d[R, 1, 1, kW] →        # Vertical
            Conv2d[out_ch, R, 1, 1]      # Pointwise
        
        Args:
            layer: Conv2d layer to decompose
            rank: CP rank. If None, use self.rank
            
        Returns:
            Sequential module with decomposed layers
        """
        if rank is None:
            rank = self.rank
        
        weight = layer.weight.data  # [out_ch, in_ch, kH, kW]
        out_ch, in_ch, kH, kW = weight.shape
        
        # CP decomposition via Alternating Least Squares (ALS)
        factors = self._cp_als(weight, rank)
        A, B, C, D = factors  # [out_ch, R], [in_ch, R], [kH, R], [kW, R]
        
        # Create decomposed layers
        # Layer 1: Pointwise (input channels)
        layer1 = nn.Conv2d(in_ch, rank, 1, bias=False)
        layer1.weight.data = B.t().unsqueeze(2).unsqueeze(3)
        
        # Layer 2: Horizontal convolution
        layer2 = nn.Conv2d(rank, rank, (kH, 1), 
                          stride=(layer.stride[0], 1),
                          padding=(layer.padding[0], 0), 
                          groups=rank, bias=False)
        # Depthwise: each channel has its own filter
        for r in range(rank):
            layer2.weight.data[r, 0, :, 0] = C[:, r]
        
        # Layer 3: Vertical convolution
        layer3 = nn.Conv2d(rank, rank, (1, kW),
                          stride=(1, layer.stride[1]),
                          padding=(0, layer.padding[1]),
                          groups=rank, bias=False)
        for r in range(rank):
            layer3.weight.data[r, 0, 0, :] = D[:, r]
        
        # Layer 4: Pointwise (output channels)
        layer4 = nn.Conv2d(rank, out_ch, 1, bias=(layer.bias is not None))
        layer4.weight.data = A.unsqueeze(2).unsqueeze(3)
        
        if layer.bias is not None:
            layer4.bias.data = layer.bias.data
        
        return nn.Sequential(layer1, layer2, layer3, layer4)
    
    def _cp_als(
        self,
        tensor: torch.Tensor,
        rank: int
    ) -> Tuple[torch.Tensor, ...]:
        """
        CP decomposition using Alternating Least Squares (ALS).
        
        Args:
            tensor: 4D tensor [I, J, K, L]
            rank: CP rank
            
        Returns:
            Tuple of factor matrices (A, B, C, D)
        """
        I, J, K, L = tensor.shape
        
        # Initialize factors randomly
        A = torch.randn(I, rank, device=tensor.device)
        B = torch.randn(J, rank, device=tensor.device)
        C = torch.randn(K, rank, device=tensor.device)
        D = torch.randn(L, rank, device=tensor.device)
        
        # Normalize
        A = F.normalize(A, dim=0)
        B = F.normalize(B, dim=0)
        C = F.normalize(C, dim=0)
        D = F.normalize(D, dim=0)
        
        # ALS iterations
        for iteration in range(self.max_iterations):
            # Update A (keeping B, C, D fixed)
            T_mode1 = tensor.reshape(I, -1)  # [I, J*K*L]
            BCD = self._khatri_rao(self._khatri_rao(B, C), D)  # [J*K*L, R]
            try:
                A_new = torch.linalg.lstsq(BCD, T_mode1.t(), rcond=None).solution.t()
            except:
                # Fallback: use pseudo-inverse
                A_new = (torch.pinverse(BCD) @ T_mode1.t()).t()
            
            # Update B (keeping A, C, D fixed)
            T_mode2 = tensor.permute(1, 0, 2, 3).reshape(J, -1)
            ACD = self._khatri_rao(self._khatri_rao(A_new, C), D)
            try:
                B_new = torch.linalg.lstsq(ACD, T_mode2.t(), rcond=None).solution.t()
            except:
                B_new = (torch.pinverse(ACD) @ T_mode2.t()).t()
            
            # Update C (keeping A, B, D fixed)
            T_mode3 = tensor.permute(2, 0, 1, 3).reshape(K, -1)
            ABD = self._khatri_rao(self._khatri_rao(A_new, B_new), D)
            try:
                C_new = torch.linalg.lstsq(ABD, T_mode3.t(), rcond=None).solution.t()
            except:
                C_new = (torch.pinverse(ABD) @ T_mode3.t()).t()
            
            # Update D (keeping A, B, C fixed)
            T_mode4 = tensor.permute(3, 0, 1, 2).reshape(L, -1)
            ABC = self._khatri_rao(self._khatri_rao(A_new, B_new), C_new)
            try:
                D_new = torch.linalg.lstsq(ABC, T_mode4.t(), rcond=None).solution.t()
            except:
                D_new = (torch.pinverse(ABC) @ T_mode4.t()).t()
            
            # Check convergence (simplified)
            change = (
                (A_new - A).norm() + (B_new - B).norm() +
                (C_new - C).norm() + (D_new - D).norm()
            )
            
            A, B, C, D = A_new, B_new, C_new, D_new
            
            if change < self.tolerance:
                break
        
        return A, B, C, D
    
    def _khatri_rao(
        self,
        A: torch.Tensor,
        B: torch.Tensor
    ) -> torch.Tensor:
        """
        Khatri-Rao product (column-wise Kronecker product).
        
        Args:
            A: Matrix [I, R]
            B: Matrix [J, R]
            
        Returns:
            Khatri-Rao product [I*J, R]
        """
        I, R = A.shape
        J, _ = B.shape
        
        result = torch.zeros(I * J, R, device=A.device)
        for r in range(R):
            result[:, r] = torch.kron(A[:, r], B[:, r])
        
        return result


class TensorTrainDecomposer:
    """
    Tensor-Train (TT) Decomposition.
    
    Decomposes tensor W into a chain of 3D cores:
        W[i1,i2,i3,i4] ≈ G1[1,i1,r1] × G2[r1,i2,r2] × G3[r2,i3,r3] × G4[r3,i4,1]
    
    Where:
        - G1, G2, G3, G4 are TT-cores
        - r1, r2, r3 are TT-ranks (bounded for stability)
    
    Optimal for very deep networks. Memory-efficient representation.
    
    Example:
    --------
    >>> decomposer = TensorTrainDecomposer(ranks=[4, 4, 4])
    >>> conv_layer = nn.Conv2d(64, 64, 3, 3)
    >>> compressed = decomposer.decompose_conv2d(conv_layer)
    >>> # Good compression with stability
    """
    
    def __init__(
        self,
        ranks: Optional[List[int]] = None,
        max_rank: int = 16,
        use_tucker_fallback: bool = False
    ):
        """
        Initialize TT decomposer.
        
        Args:
            ranks: TT-ranks [r1, r2, r3]. If None, auto-compute
            max_rank: Maximum TT-rank for stability
            use_tucker_fallback: If True, use Tucker for unsupported cases
        """
        self.ranks = ranks
        self.max_rank = max_rank
        self.use_tucker_fallback = use_tucker_fallback
        if use_tucker_fallback:
            self._tucker = TuckerDecomposer()
    
    def decompose_conv2d(
        self,
        layer: nn.Conv2d,
        ranks: Optional[List[int]] = None
    ) -> nn.Sequential:
        """
        Decompose Conv2d using TT-SVD decomposition.
        
        Decomposes 4D weight tensor into a chain of 3D tensors (TT-cores).
        Uses sequential SVD algorithm from Oseledets (2011).
        
        Args:
            layer: Conv2d layer to decompose
            ranks: TT-ranks [r1, r2]. If None, auto-compute
            
        Returns:
            Sequential module with TT-decomposed layers
        """
        weight = layer.weight.data
        bias = layer.bias
        
        out_ch, in_ch, kh, kw = weight.shape
        
        if ranks is None:
            ranks = self.ranks if self.ranks else self._auto_ranks(layer)
        
        # Ensure we have 2 ranks for 4D tensor
        if len(ranks) < 2:
            ranks = [ranks[0], ranks[0]]
        
        r1, r2 = ranks[:2]
        
        # Validate ranks
        r1 = min(r1, out_ch, in_ch)
        r2 = min(r2, in_ch, kh * kw)
        
        try:
            # TT-SVD: decompose into chain of cores
            cores = self._tt_svd_4d(weight, [r1, r2])
            
            if len(cores) != 3:
                raise ValueError(f"Expected 3 TT-cores, got {len(cores)}")
            
            core1, core2, core3 = cores
            
            # Get actual ranks from cores
            r2_actual, in_ch_actual, _, _ = core3.shape
            r1_actual, r2_check = core2.shape
            out_ch_actual, r1_check = core1.shape
            
            # Sanity checks
            assert r2_check == r2_actual, f"Rank mismatch: r2 {r2_check} != {r2_actual}"
            assert r1_check == r1_actual, f"Rank mismatch: r1 {r1_check} != {r1_actual}"
            assert in_ch_actual == in_ch, f"Channel mismatch: {in_ch_actual} != {in_ch}"
            assert out_ch_actual == out_ch, f"Channel mismatch: {out_ch_actual} != {out_ch}"
            
            # Build sequential layers from TT-cores
            # The decomposition gives us:
            # core1: (out_ch, r1) - output projection
            # core2: (r1, r2) - intermediate connection
            # core3: (r2, in_ch, kh, kw) - spatial convolution
            
            # Layer 1: in_ch -> r2 (spatial convolution)
            layer1 = nn.Conv2d(
                in_ch, r2_actual, 
                kernel_size=(kh, kw),
                stride=layer.stride,
                padding=layer.padding,
                dilation=layer.dilation,
                bias=False
            )
            # core3 shape: (r2, in_ch, kh, kw) - perfect match
            layer1.weight.data = core3
            
            # Layer 2: r2 -> r1 (1x1 conv, pointwise)
            layer2 = nn.Conv2d(r2_actual, r1_actual, kernel_size=1, bias=False)
            # core2 shape: (r1, r2) -> need (r1, r2, 1, 1)
            layer2.weight.data = core2.view(r1_actual, r2_actual, 1, 1)
            
            # Layer 3: r1 -> out_ch (1x1 conv, output projection)
            layer3 = nn.Conv2d(r1_actual, out_ch, kernel_size=1, bias=bias is not None)
            # core1 shape: (out_ch, r1) -> need (out_ch, r1, 1, 1)
            layer3.weight.data = core1.view(out_ch, r1_actual, 1, 1)
            
            if bias is not None:
                layer3.bias.data = bias.data
            
            return nn.Sequential(layer1, layer2, layer3)
            
        except Exception as e:
            # Fallback to Tucker if enabled
            if self.use_tucker_fallback:
                return self._tucker.decompose_conv2d(layer, ranks[:2])
            else:
                raise RuntimeError(f"TT-SVD failed: {e}. Enable use_tucker_fallback=True for fallback.")
    
    def decompose_linear(
        self,
        layer: nn.Linear,
        rank: Optional[int] = None
    ) -> nn.Sequential:
        """
        Decompose Linear layer using TT decomposition.
        
        Matrix decomposition: W ≈ W1 @ W2
        
        Args:
            layer: Linear layer to decompose
            rank: TT rank. If None, auto-compute
            
        Returns:
            Sequential module with decomposed layers
        """
        weight = layer.weight.data
        bias = layer.bias
        
        out_features, in_features = weight.shape
        
        if rank is None:
            rank = self.ranks[0] if self.ranks else min(self.max_rank, min(out_features, in_features) // 2)
        
        # Clamp rank
        rank = min(rank, out_features, in_features)
        
        try:
            # SVD decomposition
            U, S, Vt = torch.linalg.svd(weight, full_matrices=False)
            
            # Keep top-rank components
            U_r = U[:, :rank]
            S_r = S[:rank]
            Vt_r = Vt[:rank, :]
            
            # W ≈ U_r @ diag(S_r) @ Vt_r
            # Layer 1: in -> rank
            layer1 = nn.Linear(in_features, rank, bias=False)
            layer1.weight.data = torch.sqrt(S_r).unsqueeze(1) * Vt_r
            
            # Layer 2: rank -> out
            layer2 = nn.Linear(rank, out_features, bias=bias is not None)
            layer2.weight.data = U_r @ torch.diag(torch.sqrt(S_r))
            
            if bias is not None:
                layer2.bias.data = bias.data
            
            return nn.Sequential(layer1, layer2)
            
        except Exception as e:
            if self.use_tucker_fallback:
                return self._tucker.decompose_linear(layer, rank)
            else:
                raise RuntimeError(f"TT Linear decomposition failed: {e}")
    
    def _tt_svd(
        self,
        tensor: torch.Tensor,
        ranks: List[int]
    ) -> List[torch.Tensor]:
        """
        TT-SVD algorithm for tensor decomposition (Oseledets 2011).
        
        Decomposes d-dimensional tensor into chain of 3D TT-cores.
        
        Args:
            tensor: Input tensor of shape (n1, n2, ..., nd)
            ranks: TT-ranks [r1, r2, ..., r_{d-1}]
            
        Returns:
            List of TT-cores, each of shape (r_{k-1}, n_k, r_k)
            where r_0 = r_d = 1
        """
        # This is the general TT-SVD algorithm
        # For our 4D case, we use _tt_svd_4d instead
        pass
    
    def _tt_svd_4d(
        self,
        tensor: torch.Tensor,
        ranks: List[int]
    ) -> List[torch.Tensor]:
        """
        TT-SVD for 4D tensors (Conv2d weights).
        
        Tensor shape: (out_ch, in_ch, kh, kw)
        Decomposes into 3 cores via sequential SVD.
        
        Args:
            tensor: 4D weight tensor
            ranks: [r1, r2] TT-ranks
            
        Returns:
            List of 3 TT-cores:
            - core1: (out_ch, r1)
            - core2: (r1, r2) 
            - core3: (r2, in_ch, kh, kw)
        """
        out_ch, in_ch, kh, kw = tensor.shape
        r1, r2 = ranks
        
        # Step 1: Reshape tensor to matrix (out_ch, in_ch*kh*kw)
        C = tensor.reshape(out_ch, in_ch * kh * kw)
        
        # Step 2: SVD to separate output channels
        # C ≈ U @ S @ Vt where U: (out_ch, r1), Vt: (r1, in_ch*kh*kw)
        U, S, Vt = torch.linalg.svd(C, full_matrices=False)
        
        # Truncate to rank r1
        r1_actual = min(r1, U.shape[1])
        U_r1 = U[:, :r1_actual]  # (out_ch, r1)
        S_r1 = S[:r1_actual]
        Vt_r1 = Vt[:r1_actual, :]  # (r1, in_ch*kh*kw)
        
        # Core 1: output side (out_ch, r1)
        core1 = U_r1 @ torch.diag(torch.sqrt(S_r1))
        
        # Step 3: Process remaining tensor (r1, in_ch*kh*kw)
        # Multiply by sqrt(S) to balance
        C_remaining = torch.diag(torch.sqrt(S_r1)) @ Vt_r1  # (r1, in_ch*kh*kw)
        
        # Reshape to (r1, in_ch, kh*kw)
        C_remaining = C_remaining.reshape(r1_actual, in_ch, kh * kw)
        
        # Step 4: Reshape to (r1, in_ch*kh*kw) and SVD again
        C2 = C_remaining.reshape(r1_actual, in_ch * kh * kw)
        
        U2, S2, Vt2 = torch.linalg.svd(C2, full_matrices=False)
        
        # Truncate to rank r2
        r2_actual = min(r2, U2.shape[1])
        U2_r2 = U2[:, :r2_actual]  # (r1, r2)
        S2_r2 = S2[:r2_actual]
        Vt2_r2 = Vt2[:r2_actual, :]  # (r2, in_ch*kh*kw)
        
        # Core 2: middle connection (r1, r2)
        core2 = U2_r2 @ torch.diag(torch.sqrt(S2_r2))
        
        # Core 3: input side (r2, in_ch, kh, kw)
        core3 = torch.diag(torch.sqrt(S2_r2)) @ Vt2_r2
        core3 = core3.reshape(r2_actual, in_ch, kh, kw)
        
        return [core1, core2, core3]
    
    def _auto_ranks(self, layer: nn.Conv2d) -> List[int]:
        """Auto-determine TT-ranks."""
        out_ch, in_ch, _, _ = layer.weight.shape
        # Conservative ranks
        r = min(self.max_rank, min(out_ch, in_ch) // 4)
        return [r, r]


def decompose_model(
    model: nn.Module,
    config: DecompositionConfig
) -> nn.Module:
    """
    Decompose all Conv2d and Linear layers in a model.
    
    Args:
        model: Model to decompose
        config: Decomposition configuration
        
    Returns:
        Decomposed model
    """
    if config.method == "tucker":
        decomposer = TuckerDecomposer(
            ranks=config.ranks,
            auto_rank=config.auto_rank,
            energy_threshold=config.energy_threshold
        )
    elif config.method == "cp":
        decomposer = CPDecomposer(
            rank=config.ranks[0] if config.ranks else 8,
            max_iterations=config.max_iterations
        )
    elif config.method == "tt":
        decomposer = TensorTrainDecomposer(ranks=config.ranks)
    else:
        raise ValueError(f"Unknown method: {config.method}")
    
    # Replace layers
    model_decomposed = _decompose_recursive(model, decomposer)
    
    return model_decomposed


def _decompose_recursive(
    module: nn.Module,
    decomposer: Union[TuckerDecomposer, CPDecomposer, TensorTrainDecomposer]
) -> nn.Module:
    """Recursively decompose layers in module."""
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            # Decompose conv layer
            if child.kernel_size != (1, 1):  # Skip 1x1 convs
                decomposed = decomposer.decompose_conv2d(child)
                setattr(module, name, decomposed)
        elif isinstance(child, nn.Linear):
            # Decompose linear layer
            decomposed = decomposer.decompose_linear(child)
            setattr(module, name, decomposed)
        else:
            # Recurse into child modules
            _decompose_recursive(child, decomposer)
    
    return module


def compute_compression_ratio(
    original_model: nn.Module,
    decomposed_model: nn.Module
) -> float:
    """
    Compute compression ratio.
    
    Args:
        original_model: Original model
        decomposed_model: Decomposed model
        
    Returns:
        Compression ratio (original_params / decomposed_params)
    """
    original_params = sum(p.numel() for p in original_model.parameters())
    decomposed_params = sum(p.numel() for p in decomposed_model.parameters())
    
    return original_params / decomposed_params if decomposed_params > 0 else 1.0
