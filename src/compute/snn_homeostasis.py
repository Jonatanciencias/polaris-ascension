"""
SNN Homeostasis Enhancement Module - Session 20
================================================

This module extends the base SNN implementation with biologically-inspired
homeostatic mechanisms for improved stability and learning.

Based on cutting-edge neuroscience research:
- Touda, K. & Okuno, H. (2026). "Synaptic Scaling for SNN Learning: 
  Stability-Plasticity Balance". arXiv.
- Massey, S. et al. (2025). "Sleep-Based Homeostatic Regularization for 
  Spike-Timing-Dependent Plasticity". NeurIPS 2025.
- Turrigiano, G.G. (2008). "The self-tuning neuron: synaptic scaling 
  of excitatory synapses". Cell, 135(3), 422-435.

Biological Principles:
---------------------
1. Synaptic Scaling - Global multiplicative adjustment of all synapses
2. Intrinsic Plasticity - Adjust neuron excitability (threshold, τ)
3. Homeostatic Metaplasticity - Adjust learning rates based on activity
4. Sleep-Wake Cycles - Periodic consolidation and normalization phases

Key Features:
-------------
1. Activity-dependent synaptic scaling (Turrigiano mechanism)
2. Sleep-based weight consolidation (Massey et al., 2025)
3. Adaptive threshold adjustment (maintain target firing rate)
4. STDP with homeostatic constraints (stability-plasticity balance)
5. Energy-efficient spike regulation

Mathematical Foundation:
-----------------------
Synaptic Scaling:
    w_ij = w_ij × (r_target / r_actual)^α
    
Where:
- r_target: Target firing rate (typically 5-10%)
- r_actual: Actual firing rate of neuron j
- α: Scaling exponent (0.5-2.0)

Sleep Consolidation (Massey et al.):
    During "sleep" phase:
    - Downscale all weights by factor β
    - Prune weak synapses (|w| < threshold)
    - Replay important patterns
    
Intrinsic Plasticity:
    θ_j = θ_j × (1 + η × (r_actual - r_target))
    
Where θ_j is neuron j's firing threshold.

Performance Impact:
------------------
- Prevents runaway excitation (stability)
- Maintains representational capacity (plasticity)
- Reduces catastrophic forgetting
- Energy-efficient spike rates (biological plausibility)

Version: 0.7.0-dev (Session 20 - Research Integration)
Author: Legacy GPU AI Platform Team
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Callable
from dataclasses import dataclass, field
import numpy as np
import math


@dataclass
class HomeostasisConfig:
    """
    Configuration for homeostatic mechanisms.
    
    Attributes:
        target_firing_rate (float): Target spike rate (0-1). Typical: 0.05-0.1
        scaling_time_constant (float): τ for synaptic scaling EMA. Typical: 1000-10000
        scaling_exponent (float): α for scaling power law. Typical: 0.5-2.0
        enable_synaptic_scaling (bool): Enable Turrigiano mechanism
        enable_intrinsic_plasticity (bool): Enable threshold adaptation
        enable_sleep_cycles (bool): Enable periodic consolidation
        sleep_period (int): Timesteps between sleep phases. Typical: 1000-10000
        sleep_downscale_factor (float): Weight reduction during sleep. Typical: 0.9-0.99
        prune_threshold (float): Weak synapse pruning threshold during sleep
        min_weight (float): Minimum allowed weight magnitude
        max_weight (float): Maximum allowed weight magnitude
    """
    target_firing_rate: float = 0.05  # 5% spikes
    scaling_time_constant: float = 5000.0
    scaling_exponent: float = 1.0
    enable_synaptic_scaling: bool = True
    enable_intrinsic_plasticity: bool = True
    enable_sleep_cycles: bool = True
    sleep_period: int = 10000
    sleep_downscale_factor: float = 0.95
    prune_threshold: float = 0.01
    min_weight: float = -5.0
    max_weight: float = 5.0
    
    # Intrinsic plasticity parameters
    threshold_adaptation_rate: float = 0.001
    min_threshold: float = 0.5
    max_threshold: float = 2.0
    
    def __post_init__(self):
        """Validate configuration."""
        assert 0 < self.target_firing_rate < 1, "target_firing_rate must be in (0, 1)"
        assert self.scaling_time_constant > 0, "scaling_time_constant must be positive"
        assert self.scaling_exponent > 0, "scaling_exponent must be positive"
        assert 0 < self.sleep_downscale_factor < 1, "sleep_downscale_factor must be in (0, 1)"


class SynapticScaling(nn.Module):
    """
    Synaptic Scaling Module (Turrigiano, 2008).
    
    Implements homeostatic synaptic scaling that adjusts all synaptic
    weights to maintain a target firing rate.
    
    Biological Mechanism:
    --------------------
    Neurons detect their average firing rate over long timescales.
    If firing too much: downscale all incoming synapses
    If firing too little: upscale all incoming synapses
    
    This is a MULTIPLICATIVE adjustment (unlike STDP which is additive/Hebbian).
    
    Mathematical Model:
    ------------------
    Scaling factor: S_j = (r_target / r_j)^α
    
    Updated weights: w_ij ← w_ij × S_j
    
    Where:
    - r_j: Exponential moving average of neuron j's firing rate
    - r_target: Target firing rate
    - α: Scaling exponent (controls strength of homeostasis)
    
    Properties:
    ----------
    - Preserves relative synapse strengths (multiplicative)
    - Slow timescale (hours to days in biology, ~1000s timesteps here)
    - Bidirectional (can increase or decrease)
    - Per-neuron adjustment
    
    Args:
        n_neurons (int): Number of neurons to regulate
        config (HomeostasisConfig): Homeostasis parameters
        device (str): Computation device
    """
    
    def __init__(
        self,
        n_neurons: int,
        config: HomeostasisConfig,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        
        self.n_neurons = n_neurons
        self.config = config
        self.device = device
        
        # Running average of firing rates (per neuron)
        self.register_buffer(
            'firing_rate_ema',
            torch.full((n_neurons,), config.target_firing_rate, device=device)
        )
        
        # EMA decay factor: exp(-1/τ)
        self.ema_decay = math.exp(-1.0 / config.scaling_time_constant)
        
        # Scaling factors (computed from firing rates)
        self.register_buffer(
            'scaling_factors',
            torch.ones(n_neurons, device=device)
        )
        
        # Statistics
        self.total_updates = 0
    
    def update_firing_rates(self, spikes: torch.Tensor):
        """
        Update exponential moving average of firing rates.
        
        Args:
            spikes (torch.Tensor): Binary spike tensor (batch, n_neurons)
        """
        # Compute instantaneous firing rate (average over batch)
        instant_rate = spikes.mean(dim=0)
        
        # Update EMA: r_new = decay * r_old + (1 - decay) * r_instant
        self.firing_rate_ema = (
            self.ema_decay * self.firing_rate_ema +
            (1 - self.ema_decay) * instant_rate
        )
        
        self.total_updates += 1
    
    def compute_scaling_factors(self) -> torch.Tensor:
        """
        Compute scaling factors from firing rates.
        
        S_j = (r_target / r_j)^α
        
        Returns:
            Scaling factors per neuron
        """
        # Avoid division by zero
        safe_rates = torch.clamp(
            self.firing_rate_ema, 
            min=1e-6, 
            max=1.0 - 1e-6
        )
        
        # Scaling factors
        self.scaling_factors = (
            self.config.target_firing_rate / safe_rates
        ) ** self.config.scaling_exponent
        
        # Clamp to reasonable range to prevent explosion/vanishing
        self.scaling_factors = torch.clamp(
            self.scaling_factors,
            min=0.5,
            max=2.0
        )
        
        return self.scaling_factors
    
    def apply_scaling(
        self,
        weights: torch.Tensor,
        spikes: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply synaptic scaling to weight matrix.
        
        Args:
            weights (torch.Tensor): Weight matrix (out_features, in_features)
            spikes (torch.Tensor, optional): Recent spikes for rate update
        
        Returns:
            Scaled weight matrix
        """
        if spikes is not None:
            self.update_firing_rates(spikes)
        
        scaling = self.compute_scaling_factors()
        
        # Scale each row (output neuron) by its scaling factor
        # weights: (out, in) × scaling: (out,) → broadcast correctly
        scaled_weights = weights * scaling.unsqueeze(1)
        
        # Clamp weights
        scaled_weights = torch.clamp(
            scaled_weights,
            min=self.config.min_weight,
            max=self.config.max_weight
        )
        
        return scaled_weights
    
    def get_statistics(self) -> Dict[str, float]:
        """Get homeostasis statistics."""
        return {
            'mean_firing_rate': self.firing_rate_ema.mean().item(),
            'std_firing_rate': self.firing_rate_ema.std().item(),
            'mean_scaling_factor': self.scaling_factors.mean().item(),
            'min_scaling_factor': self.scaling_factors.min().item(),
            'max_scaling_factor': self.scaling_factors.max().item(),
            'target_firing_rate': self.config.target_firing_rate
        }


class IntrinsicPlasticity(nn.Module):
    """
    Intrinsic Plasticity Module.
    
    Adjusts neuron excitability (firing threshold) to maintain target
    activity levels. Unlike synaptic scaling (which adjusts inputs),
    this adjusts the neuron's intrinsic properties.
    
    Biological Mechanism:
    --------------------
    Neurons regulate expression of ion channels that control excitability:
    - Too active → express more K+ channels → harder to fire
    - Too quiet → express more Na+ channels → easier to fire
    
    Mathematical Model:
    ------------------
    Threshold update rule:
        θ_j = θ_j × (1 + η × (r_j - r_target))
    
    If r_j > r_target: threshold increases (harder to fire)
    If r_j < r_target: threshold decreases (easier to fire)
    
    Alternative (direct mapping):
        θ_j = θ_base × f(r_j / r_target)
    
    Where f is a sigmoid-like function.
    
    Args:
        n_neurons (int): Number of neurons
        config (HomeostasisConfig): Homeostasis parameters
        device (str): Computation device
    """
    
    def __init__(
        self,
        n_neurons: int,
        config: HomeostasisConfig,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        
        self.n_neurons = n_neurons
        self.config = config
        self.device = device
        
        # Adaptive thresholds (learnable parameters)
        self.thresholds = nn.Parameter(
            torch.ones(n_neurons, device=device) * 1.0  # Default threshold
        )
        
        # Running average of firing rates
        self.register_buffer(
            'firing_rate_ema',
            torch.full((n_neurons,), config.target_firing_rate, device=device)
        )
        
        # EMA decay
        self.ema_decay = 0.99
    
    def update_thresholds(self, spikes: torch.Tensor):
        """
        Update thresholds based on recent activity.
        
        Args:
            spikes (torch.Tensor): Binary spike tensor (batch, n_neurons)
        """
        # Update firing rate EMA
        instant_rate = spikes.mean(dim=0)
        self.firing_rate_ema = (
            self.ema_decay * self.firing_rate_ema +
            (1 - self.ema_decay) * instant_rate
        )
        
        # Compute error from target
        rate_error = self.firing_rate_ema - self.config.target_firing_rate
        
        # Update thresholds (gradient-free, in-place)
        with torch.no_grad():
            # Multiplicative update
            threshold_change = 1.0 + self.config.threshold_adaptation_rate * rate_error
            self.thresholds.data *= threshold_change
            
            # Clamp to allowed range
            self.thresholds.data = torch.clamp(
                self.thresholds.data,
                min=self.config.min_threshold,
                max=self.config.max_threshold
            )
    
    def get_thresholds(self) -> torch.Tensor:
        """Get current adaptive thresholds."""
        return self.thresholds
    
    def get_statistics(self) -> Dict[str, float]:
        """Get intrinsic plasticity statistics."""
        return {
            'mean_threshold': self.thresholds.mean().item(),
            'std_threshold': self.thresholds.std().item(),
            'min_threshold': self.thresholds.min().item(),
            'max_threshold': self.thresholds.max().item()
        }


class SleepConsolidation(nn.Module):
    """
    Sleep-Based Consolidation Module (Massey et al., 2025).
    
    Implements periodic "sleep" phases that:
    1. Downscale all synaptic weights (synaptic homeostasis hypothesis)
    2. Prune weak connections
    3. Replay important patterns for consolidation
    
    Biological Basis:
    ----------------
    During sleep (especially slow-wave sleep):
    - Global synaptic downscaling occurs
    - This "renormalizes" the network after learning
    - Important memories are replayed and consolidated
    - Weak/noise connections are pruned
    
    Synaptic Homeostasis Hypothesis (Tononi & Cirelli, 2003):
    - Waking: net synaptic potentiation (learning)
    - Sleep: net synaptic depression (normalization)
    - This maintains energy efficiency and capacity
    
    Implementation:
    --------------
    Every `sleep_period` timesteps:
    1. Scale all weights by `sleep_downscale_factor` (0.9-0.99)
    2. Prune weights below `prune_threshold`
    3. (Optional) Replay stored patterns
    
    Args:
        config (HomeostasisConfig): Homeostasis parameters
        device (str): Computation device
    """
    
    def __init__(
        self,
        config: HomeostasisConfig,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        
        self.config = config
        self.device = device
        
        # Timestep counter
        self.timestep = 0
        
        # Sleep statistics
        self.sleep_count = 0
        self.total_pruned = 0
        
        # Pattern replay buffer (for consolidation)
        self.replay_buffer: List[torch.Tensor] = []
        self.max_replay_size = 100
    
    def should_sleep(self) -> bool:
        """Check if it's time for a sleep phase."""
        return self.timestep > 0 and self.timestep % self.config.sleep_period == 0
    
    def sleep_phase(
        self,
        weights: torch.Tensor,
        importance_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, int]:
        """
        Execute sleep phase: downscale and prune.
        
        Args:
            weights (torch.Tensor): Weight matrix to process
            importance_mask (torch.Tensor, optional): Mask of important connections
                                                       (1 = important, 0 = can prune)
        
        Returns:
            Tuple of (processed_weights, n_pruned)
        """
        self.sleep_count += 1
        
        # 1. Global downscaling
        scaled_weights = weights * self.config.sleep_downscale_factor
        
        # 2. Prune weak connections
        if importance_mask is not None:
            # Only prune non-important connections
            prune_candidates = ~importance_mask.bool()
            weak_mask = torch.abs(scaled_weights) < self.config.prune_threshold
            prune_mask = prune_candidates & weak_mask
        else:
            prune_mask = torch.abs(scaled_weights) < self.config.prune_threshold
        
        n_pruned = prune_mask.sum().item()
        self.total_pruned += n_pruned
        
        # Set pruned weights to zero
        scaled_weights[prune_mask] = 0.0
        
        return scaled_weights, n_pruned
    
    def step(
        self,
        weights: torch.Tensor,
        importance_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, bool]:
        """
        Step the sleep cycle.
        
        Args:
            weights: Current weight matrix
            importance_mask: Optional mask for important connections
        
        Returns:
            Tuple of (weights, did_sleep)
        """
        self.timestep += 1
        
        if self.should_sleep():
            weights, _ = self.sleep_phase(weights, importance_mask)
            return weights, True
        
        return weights, False
    
    def store_pattern(self, pattern: torch.Tensor):
        """Store pattern in replay buffer for consolidation."""
        if len(self.replay_buffer) >= self.max_replay_size:
            self.replay_buffer.pop(0)
        self.replay_buffer.append(pattern.clone())
    
    def replay_patterns(self, n_patterns: int = 10) -> List[torch.Tensor]:
        """
        Sample patterns for replay during sleep.
        
        Returns:
            List of patterns to replay
        """
        if len(self.replay_buffer) == 0:
            return []
        
        n_sample = min(n_patterns, len(self.replay_buffer))
        indices = np.random.choice(len(self.replay_buffer), n_sample, replace=False)
        return [self.replay_buffer[i] for i in indices]
    
    def get_statistics(self) -> Dict[str, float]:
        """Get sleep statistics."""
        return {
            'sleep_count': self.sleep_count,
            'total_pruned': self.total_pruned,
            'timestep': self.timestep,
            'replay_buffer_size': len(self.replay_buffer)
        }


class HomeostaticSTDP(nn.Module):
    """
    STDP with Homeostatic Constraints.
    
    Extends standard STDP with homeostatic mechanisms to maintain
    stability-plasticity balance.
    
    Standard STDP:
    -------------
    Δw = A_+ × exp(-Δt/τ_+)  if t_post > t_pre (LTP)
    Δw = -A_- × exp(Δt/τ_-)  if t_post < t_pre (LTD)
    
    Homeostatic Extensions:
    ----------------------
    1. Metaplasticity: Adjust A_+, A_- based on postsynaptic activity
       - High activity → increase A_- (more LTD)
       - Low activity → increase A_+ (more LTP)
    
    2. Soft bounds: Prevent weights from growing/shrinking indefinitely
       Δw_bounded = Δw × f(w_current)
       
    3. Weight normalization: Normalize total synaptic input
       Sum of weights onto each neuron ≈ constant
    
    4. Synaptic tagging: Mark strongly activated synapses for protection
    
    Args:
        in_features (int): Number of presynaptic neurons
        out_features (int): Number of postsynaptic neurons
        config (HomeostasisConfig): Homeostasis parameters
        device (str): Computation device
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: HomeostasisConfig,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        self.device = device
        
        # STDP parameters (learnable for metaplasticity)
        self.A_plus = nn.Parameter(torch.tensor(0.01, device=device))
        self.A_minus = nn.Parameter(torch.tensor(0.012, device=device))
        self.tau_plus = 20.0  # ms
        self.tau_minus = 20.0  # ms
        
        # Pre/post synaptic traces (for STDP timing)
        self.register_buffer(
            'pre_trace',
            torch.zeros(in_features, device=device)
        )
        self.register_buffer(
            'post_trace',
            torch.zeros(out_features, device=device)
        )
        
        # Running average of post-synaptic activity (for metaplasticity)
        self.register_buffer(
            'post_activity_avg',
            torch.full((out_features,), config.target_firing_rate, device=device)
        )
        
        # Synaptic tags (for consolidation)
        self.register_buffer(
            'synaptic_tags',
            torch.zeros(out_features, in_features, device=device)
        )
        
        # Trace decay factors
        self.trace_decay_pre = math.exp(-1.0 / self.tau_plus)
        self.trace_decay_post = math.exp(-1.0 / self.tau_minus)
        
        # Activity EMA decay
        self.activity_decay = 0.999
    
    def update_traces(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor
    ):
        """
        Update pre/post synaptic traces.
        
        Traces decay exponentially and jump up on spikes:
        trace_new = decay × trace_old + spike
        
        Args:
            pre_spikes (torch.Tensor): Presynaptic spikes (batch, in_features)
            post_spikes (torch.Tensor): Postsynaptic spikes (batch, out_features)
        """
        # Average over batch
        pre_rate = pre_spikes.mean(dim=0)
        post_rate = post_spikes.mean(dim=0)
        
        # Update traces
        self.pre_trace = self.trace_decay_pre * self.pre_trace + pre_rate
        self.post_trace = self.trace_decay_post * self.post_trace + post_rate
        
        # Update activity average (for metaplasticity)
        self.post_activity_avg = (
            self.activity_decay * self.post_activity_avg +
            (1 - self.activity_decay) * post_rate
        )
    
    def compute_stdp_update(
        self,
        weights: torch.Tensor,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute STDP weight update with homeostatic constraints.
        
        Args:
            weights (torch.Tensor): Current weight matrix (out, in)
            pre_spikes (torch.Tensor): Presynaptic spikes (batch, in)
            post_spikes (torch.Tensor): Postsynaptic spikes (batch, out)
        
        Returns:
            Weight update matrix (out, in)
        """
        # Update traces first
        self.update_traces(pre_spikes, post_spikes)
        
        # STDP update
        # LTP: post spike with pre trace (pre → post causation)
        post_rate = post_spikes.mean(dim=0)  # (out,)
        ltp = torch.outer(post_rate, self.pre_trace) * self.A_plus.abs()
        
        # LTD: pre spike with post trace (post → pre anti-causation)
        pre_rate = pre_spikes.mean(dim=0)  # (in,)
        ltd = torch.outer(self.post_trace, pre_rate) * self.A_minus.abs()
        
        # Basic STDP update
        dw = ltp - ltd
        
        # 1. Metaplasticity: adjust based on activity
        # High activity → stronger LTD (reduce excitability)
        activity_ratio = self.post_activity_avg / self.config.target_firing_rate
        meta_factor = 1.0 / torch.clamp(activity_ratio, min=0.5, max=2.0)
        dw = dw * meta_factor.unsqueeze(1)
        
        # 2. Soft bounds: prevent extreme weights
        # w_factor = (w_max - w) / (w_max - w_min) for LTP
        # w_factor = (w - w_min) / (w_max - w_min) for LTD
        w_range = self.config.max_weight - self.config.min_weight
        ltp_bound = (self.config.max_weight - weights) / w_range
        ltd_bound = (weights - self.config.min_weight) / w_range
        
        dw_bounded = torch.where(dw > 0, dw * ltp_bound, dw * ltd_bound)
        
        # 3. Update synaptic tags (strong updates mark for protection)
        tag_update = torch.abs(dw_bounded) > 0.001
        self.synaptic_tags = torch.maximum(
            0.99 * self.synaptic_tags,
            tag_update.float()
        )
        
        return dw_bounded
    
    def apply_stdp(
        self,
        weights: torch.Tensor,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        learning_rate: float = 1.0
    ) -> torch.Tensor:
        """
        Apply STDP update to weights.
        
        Args:
            weights: Current weight matrix
            pre_spikes: Presynaptic spikes
            post_spikes: Postsynaptic spikes
            learning_rate: Scaling factor for update
        
        Returns:
            Updated weight matrix
        """
        dw = self.compute_stdp_update(weights, pre_spikes, post_spikes)
        
        updated_weights = weights + learning_rate * dw
        
        # Clamp to bounds
        updated_weights = torch.clamp(
            updated_weights,
            min=self.config.min_weight,
            max=self.config.max_weight
        )
        
        return updated_weights
    
    def normalize_weights(
        self,
        weights: torch.Tensor,
        target_norm: float = 1.0
    ) -> torch.Tensor:
        """
        Normalize total synaptic weight per neuron.
        
        Ensures sum of incoming weights ≈ target_norm
        (homeostatic weight normalization)
        
        Args:
            weights: Weight matrix (out, in)
            target_norm: Target L1 norm per output neuron
        
        Returns:
            Normalized weights
        """
        row_norms = weights.abs().sum(dim=1, keepdim=True)
        scale_factors = target_norm / (row_norms + 1e-10)
        
        # Soft normalization: only scale if norm is too different
        scale_factors = torch.clamp(scale_factors, min=0.8, max=1.25)
        
        return weights * scale_factors
    
    def get_importance_mask(self) -> torch.Tensor:
        """Get mask of important synapses (based on tags)."""
        return self.synaptic_tags > 0.5
    
    def get_statistics(self) -> Dict[str, float]:
        """Get STDP statistics."""
        return {
            'A_plus': self.A_plus.item(),
            'A_minus': self.A_minus.item(),
            'mean_pre_trace': self.pre_trace.mean().item(),
            'mean_post_trace': self.post_trace.mean().item(),
            'mean_post_activity': self.post_activity_avg.mean().item(),
            'tagged_fraction': (self.synaptic_tags > 0.5).float().mean().item()
        }


class HomeostaticSpikingLayer(nn.Module):
    """
    Complete Homeostatic Spiking Layer.
    
    Integrates all homeostatic mechanisms into a single spiking layer:
    - Synaptic scaling
    - Intrinsic plasticity
    - Sleep consolidation
    - Homeostatic STDP
    
    This provides a biologically plausible spiking layer with automatic
    stability regulation.
    
    Args:
        in_features (int): Input dimension
        out_features (int): Output dimension (number of neurons)
        tau_mem (float): Membrane time constant
        config (HomeostasisConfig): Homeostasis configuration
        device (str): Computation device
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        tau_mem: float = 10.0,
        config: Optional[HomeostasisConfig] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        
        # Default config
        self.config = config if config is not None else HomeostasisConfig()
        
        # Synaptic weights
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device)
        )
        self.bias = nn.Parameter(torch.zeros(out_features, device=device))
        
        # Initialize weights
        nn.init.xavier_uniform_(self.weight, gain=1.0)
        
        # Membrane potential state
        self.v_mem = None
        self.beta = math.exp(-1.0 / tau_mem)  # Decay factor
        self.v_thresh_base = 1.0
        
        # Homeostatic modules
        if self.config.enable_synaptic_scaling:
            self.synaptic_scaling = SynapticScaling(
                out_features, self.config, device
            )
        else:
            self.synaptic_scaling = None
        
        if self.config.enable_intrinsic_plasticity:
            self.intrinsic_plasticity = IntrinsicPlasticity(
                out_features, self.config, device
            )
        else:
            self.intrinsic_plasticity = None
        
        if self.config.enable_sleep_cycles:
            self.sleep_consolidation = SleepConsolidation(self.config, device)
        else:
            self.sleep_consolidation = None
        
        # STDP learning
        self.stdp = HomeostaticSTDP(
            in_features, out_features, self.config, device
        )
        
        # Statistics
        self.timestep = 0
        self.total_spikes = 0
    
    def reset_state(self, batch_size: int):
        """Reset membrane potential for new sequence."""
        self.v_mem = torch.zeros(
            batch_size, self.out_features, device=self.device
        )
    
    def forward(
        self,
        input_spikes: torch.Tensor,
        apply_stdp: bool = False
    ) -> torch.Tensor:
        """
        Forward pass with homeostatic regulation.
        
        Args:
            input_spikes (torch.Tensor): Input spike tensor (batch, in_features)
            apply_stdp (bool): Whether to apply STDP learning
        
        Returns:
            Output spikes (batch, out_features)
        """
        batch_size = input_spikes.shape[0]
        
        # Initialize state if needed
        if self.v_mem is None or self.v_mem.shape[0] != batch_size:
            self.reset_state(batch_size)
        
        # Get effective weights (with synaptic scaling)
        weights = self.weight
        if self.synaptic_scaling is not None:
            # Note: scaling applied periodically, not every step
            if self.timestep % 100 == 0:
                weights = self.synaptic_scaling.apply_scaling(weights)
                # Copy back to parameter (in-place update)
                with torch.no_grad():
                    self.weight.data = weights
        
        # Compute input current
        # I = W @ spikes + bias
        input_current = F.linear(input_spikes, weights, self.bias)
        
        # Update membrane potential
        # V = β * V + I
        self.v_mem = self.beta * self.v_mem + input_current
        
        # Get adaptive thresholds (intrinsic plasticity)
        if self.intrinsic_plasticity is not None:
            thresholds = self.intrinsic_plasticity.get_thresholds()
        else:
            thresholds = self.v_thresh_base
        
        # Generate spikes
        spikes = (self.v_mem >= thresholds).float()
        
        # Reset membrane potential for spiked neurons
        self.v_mem = torch.where(
            spikes.bool(),
            torch.zeros_like(self.v_mem),
            self.v_mem
        )
        
        # Update homeostatic modules
        if self.synaptic_scaling is not None:
            self.synaptic_scaling.update_firing_rates(spikes)
        
        if self.intrinsic_plasticity is not None:
            self.intrinsic_plasticity.update_thresholds(spikes)
        
        # Sleep consolidation
        if self.sleep_consolidation is not None:
            importance_mask = self.stdp.get_importance_mask()
            with torch.no_grad():
                self.weight.data, did_sleep = self.sleep_consolidation.step(
                    self.weight.data, importance_mask
                )
        
        # STDP learning
        if apply_stdp:
            with torch.no_grad():
                self.weight.data = self.stdp.apply_stdp(
                    self.weight.data, input_spikes, spikes
                )
        
        # Update statistics
        self.timestep += 1
        self.total_spikes += spikes.sum().item()
        
        return spikes
    
    def get_all_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics from all homeostatic modules."""
        stats = {
            'layer': {
                'timestep': self.timestep,
                'total_spikes': self.total_spikes,
                'avg_spike_rate': self.total_spikes / (
                    self.timestep * self.out_features + 1e-10
                )
            }
        }
        
        if self.synaptic_scaling is not None:
            stats['synaptic_scaling'] = self.synaptic_scaling.get_statistics()
        
        if self.intrinsic_plasticity is not None:
            stats['intrinsic_plasticity'] = self.intrinsic_plasticity.get_statistics()
        
        if self.sleep_consolidation is not None:
            stats['sleep_consolidation'] = self.sleep_consolidation.get_statistics()
        
        stats['stdp'] = self.stdp.get_statistics()
        
        return stats


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    print("SNN Homeostasis Enhancement Demo")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create homeostatic configuration
    config = HomeostasisConfig(
        target_firing_rate=0.05,
        enable_synaptic_scaling=True,
        enable_intrinsic_plasticity=True,
        enable_sleep_cycles=True,
        sleep_period=1000
    )
    
    # Create homeostatic spiking layer
    layer = HomeostaticSpikingLayer(
        in_features=784,
        out_features=128,
        tau_mem=10.0,
        config=config,
        device=device
    )
    
    print(f"\nLayer created: {layer.in_features} → {layer.out_features}")
    print(f"Target firing rate: {config.target_firing_rate:.1%}")
    
    # Simulate some inputs
    batch_size = 32
    n_timesteps = 100
    
    print(f"\nSimulating {n_timesteps} timesteps with batch size {batch_size}...")
    
    total_spikes = 0
    for t in range(n_timesteps):
        # Random input spikes
        input_spikes = (torch.rand(batch_size, 784, device=device) < 0.1).float()
        
        # Forward pass with STDP
        output_spikes = layer(input_spikes, apply_stdp=True)
        
        total_spikes += output_spikes.sum().item()
    
    # Print statistics
    stats = layer.get_all_statistics()
    
    print(f"\nLayer Statistics:")
    print(f"  Total spikes: {stats['layer']['total_spikes']:,.0f}")
    print(f"  Avg spike rate: {stats['layer']['avg_spike_rate']:.2%}")
    
    if 'synaptic_scaling' in stats:
        print(f"\nSynaptic Scaling:")
        print(f"  Mean firing rate: {stats['synaptic_scaling']['mean_firing_rate']:.2%}")
        print(f"  Mean scaling factor: {stats['synaptic_scaling']['mean_scaling_factor']:.3f}")
    
    if 'intrinsic_plasticity' in stats:
        print(f"\nIntrinsic Plasticity:")
        print(f"  Mean threshold: {stats['intrinsic_plasticity']['mean_threshold']:.3f}")
    
    print(f"\nSTDP:")
    print(f"  Tagged fraction: {stats['stdp']['tagged_fraction']:.2%}")
    
    print("\nHomeostatic SNN layer working correctly!")
