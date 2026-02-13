"""
Agricultural Monitoring with Spiking Neural Networks
=====================================================

This example demonstrates the application of Spiking Neural Networks (SNNs)
with homeostatic mechanisms for agricultural monitoring tasks:

1. Crop Health Classification - From multispectral imagery
2. Pest Detection - Event-driven visual sensors
3. Soil Moisture Prediction - Temporal sensor data
4. Weather-Adaptive Irrigation - Continuous monitoring

Based on research:
- Davies, M. et al. (2018). "Loihi: A Neuromorphic Manycore Processor with
  On-Chip Learning". IEEE Micro.
- Roy, K. et al. (2019). "Towards spike-based machine intelligence with
  neuromorphic computing". Nature.
- Touda, K. & Okuno, H. (2026). "Synaptic Scaling for SNN Learning".

Advantages of SNNs for Agriculture:
----------------------------------
1. Ultra-low power - Suitable for remote sensors (solar-powered)
2. Event-driven - Only process when changes occur
3. Temporal patterns - Natural for time-series data
4. Edge deployment - Run on neuromorphic chips in field
5. Continuous learning - STDP adapts to changing conditions

Biological Inspiration:
----------------------
Agricultural monitoring mimics biological sensory systems:
- Bees detecting flower health (visual + chemical)
- Birds tracking seasonal changes
- Plant roots sensing moisture gradients

Version: 0.7.0-dev (Session 20 - Domain Examples)
Author: Legacy GPU AI Platform Team
License: MIT
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import math

# Import our SNN modules
from src.compute.snn import LIFNeuron, LIFParams, SpikingLayer
from src.compute.snn_homeostasis import (
    HomeostasisConfig,
    HomeostaticSpikingLayer,
    SynapticScaling,
    SleepConsolidation,
)


# ============================================================================
# Data Encoding for Agricultural Sensors
# ============================================================================


class TemporalEncoder:
    """
    Encode continuous sensor data into spike trains.

    Methods:
    --------
    1. Rate Coding - Value → firing rate
    2. Temporal Coding - Value → spike timing
    3. Population Coding - Value → pattern across neurons
    4. Delta Coding - Changes → spikes (event-driven)

    Agricultural sensors:
    - Temperature: 0-50°C → spike rate
    - Humidity: 0-100% → spike rate
    - Light intensity: log scale encoding
    - Soil moisture: threshold-based events
    """

    def __init__(
        self,
        n_neurons_per_feature: int = 10,
        encoding: str = "population",
        dt: float = 1.0,  # Timestep in ms
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.n_neurons = n_neurons_per_feature
        self.encoding = encoding
        self.dt = dt
        self.device = device

        # For population coding: preferred values for each neuron
        self.preferred_values = torch.linspace(0, 1, n_neurons_per_feature)
        self.tuning_width = 1.0 / n_neurons_per_feature

    def rate_encode(self, values: torch.Tensor, n_timesteps: int = 100) -> torch.Tensor:
        """
        Rate coding: Value determines firing probability.

        Higher value → more spikes per time window

        Args:
            values: Normalized sensor values (batch, features) in [0, 1]
            n_timesteps: Number of time steps

        Returns:
            Spike train (batch, n_timesteps, features)
        """
        batch_size, n_features = values.shape

        # Firing probability per timestep
        firing_prob = values.unsqueeze(1).expand(-1, n_timesteps, -1)

        # Generate spikes
        spikes = (torch.rand_like(firing_prob) < firing_prob).float()

        return spikes.to(self.device)

    def temporal_encode(self, values: torch.Tensor, n_timesteps: int = 100) -> torch.Tensor:
        """
        Temporal coding: Value determines spike timing.

        Higher value → earlier spike (time-to-first-spike)

        Args:
            values: Normalized sensor values (batch, features) in [0, 1]
            n_timesteps: Number of time steps

        Returns:
            Spike train (batch, n_timesteps, features)
        """
        batch_size, n_features = values.shape

        # Spike time: high value → early spike
        # spike_time = (1 - value) × n_timesteps
        spike_times = ((1 - values) * n_timesteps).long()
        spike_times = torch.clamp(spike_times, 0, n_timesteps - 1)

        # Create spike train
        spikes = torch.zeros(batch_size, n_timesteps, n_features, device=self.device)

        for b in range(batch_size):
            for f in range(n_features):
                spikes[b, spike_times[b, f], f] = 1.0

        return spikes

    def population_encode(self, values: torch.Tensor, n_timesteps: int = 100) -> torch.Tensor:
        """
        Population coding: Value activates nearby neurons.

        Each neuron has a preferred value; neurons fire based on
        proximity to the input value (like place cells).

        Args:
            values: Normalized sensor values (batch, features) in [0, 1]
            n_timesteps: Number of time steps

        Returns:
            Spike train (batch, n_timesteps, features × n_neurons)
        """
        batch_size, n_features = values.shape

        # Compute activation for each neuron based on distance to preferred value
        # activation = exp(-(value - preferred)² / (2σ²))
        preferred = self.preferred_values.to(self.device)
        values_expanded = values.unsqueeze(-1)  # (batch, features, 1)
        preferred_expanded = preferred.unsqueeze(0).unsqueeze(0)  # (1, 1, n_neurons)

        distances = (values_expanded - preferred_expanded) ** 2
        activations = torch.exp(-distances / (2 * self.tuning_width**2))

        # Reshape to (batch, features × n_neurons)
        activations = activations.view(batch_size, -1)

        # Generate spikes based on activations
        spikes = []
        for t in range(n_timesteps):
            spike_t = (torch.rand_like(activations) < activations * 0.1).float()
            spikes.append(spike_t)

        return torch.stack(spikes, dim=1)  # (batch, time, features × n_neurons)

    def delta_encode(self, values: torch.Tensor, threshold: float = 0.05) -> torch.Tensor:
        """
        Delta coding: Spikes on significant changes only.

        Event-driven encoding - only spike when value changes.
        Ideal for slowly-changing agricultural data.

        Args:
            values: Time series (batch, timesteps, features)
            threshold: Change threshold to trigger spike

        Returns:
            Spike train (batch, timesteps, features × 2) for pos/neg changes
        """
        batch_size, n_timesteps, n_features = values.shape

        # Compute changes between timesteps
        changes = torch.diff(values, dim=1)

        # Positive and negative spikes
        pos_spikes = (changes > threshold).float()
        neg_spikes = (changes < -threshold).float()

        # Pad first timestep (no change info)
        pad = torch.zeros(batch_size, 1, n_features, device=self.device)
        pos_spikes = torch.cat([pad, pos_spikes], dim=1)
        neg_spikes = torch.cat([pad, neg_spikes], dim=1)

        # Combine: (batch, time, features × 2)
        spikes = torch.cat([pos_spikes, neg_spikes], dim=-1)

        return spikes

    def encode(self, values: torch.Tensor, n_timesteps: int = 100) -> torch.Tensor:
        """Encode using configured method."""
        if self.encoding == "rate":
            return self.rate_encode(values, n_timesteps)
        elif self.encoding == "temporal":
            return self.temporal_encode(values, n_timesteps)
        elif self.encoding == "population":
            return self.population_encode(values, n_timesteps)
        else:
            raise ValueError(f"Unknown encoding: {self.encoding}")


# ============================================================================
# Agricultural SNN Models
# ============================================================================


class CropHealthClassifier(nn.Module):
    """
    SNN for Crop Health Classification from Multispectral Data.

    Input: Multispectral indices (NDVI, NDWI, etc.)
    Output: Health categories (healthy, stressed, diseased, dead)

    Architecture:
    - Temporal encoding of spectral indices
    - 2-layer SNN with homeostatic regulation
    - Population output decoding

    Spectral Indices:
    - NDVI: (NIR - Red) / (NIR + Red) - Vegetation vigor
    - NDWI: (NIR - SWIR) / (NIR + SWIR) - Water content
    - GNDVI: (NIR - Green) / (NIR + Green) - Chlorophyll

    Energy efficiency: Processes only when indices change significantly.
    """

    def __init__(
        self,
        n_spectral_features: int = 5,
        n_hidden: int = 64,
        n_classes: int = 4,
        n_timesteps: int = 50,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()

        self.n_features = n_spectral_features
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_timesteps = n_timesteps
        self.device = device

        # Temporal encoder
        self.encoder = TemporalEncoder(
            n_neurons_per_feature=8, encoding="population", device=device
        )

        # Input dimension after population encoding
        self.input_dim = n_spectral_features * 8

        # Homeostatic config for agricultural monitoring
        self.homeostasis_config = HomeostasisConfig(
            target_firing_rate=0.05,  # Low rate for energy efficiency
            enable_synaptic_scaling=True,
            enable_intrinsic_plasticity=True,
            enable_sleep_cycles=True,
            sleep_period=5000,  # Less frequent sleep for real-time monitoring
        )

        # Spiking layers with homeostasis
        self.snn1 = HomeostaticSpikingLayer(
            in_features=self.input_dim,
            out_features=n_hidden,
            tau_mem=20.0,
            config=self.homeostasis_config,
            device=device,
        )

        self.snn2 = HomeostaticSpikingLayer(
            in_features=n_hidden,
            out_features=n_classes,
            tau_mem=20.0,
            config=self.homeostasis_config,
            device=device,
        )

        # Output accumulator (integrate output spikes)
        self.output_accumulator = None

    def reset_state(self, batch_size: int):
        """Reset network state for new sequence."""
        self.snn1.reset_state(batch_size)
        self.snn2.reset_state(batch_size)
        self.output_accumulator = torch.zeros(batch_size, self.n_classes, device=self.device)

    def forward(self, spectral_indices: torch.Tensor, apply_stdp: bool = False) -> torch.Tensor:
        """
        Classify crop health from spectral indices.

        Args:
            spectral_indices: Spectral values (batch, n_features) in [0, 1]
            apply_stdp: Enable online learning

        Returns:
            Class logits (batch, n_classes)
        """
        batch_size = spectral_indices.shape[0]
        self.reset_state(batch_size)

        # Encode to spike trains
        spike_train = self.encoder.encode(spectral_indices, self.n_timesteps)

        # Process through SNN
        for t in range(self.n_timesteps):
            spikes_t = spike_train[:, t, :]

            # Layer 1
            hidden_spikes = self.snn1(spikes_t, apply_stdp=apply_stdp)

            # Layer 2
            output_spikes = self.snn2(hidden_spikes, apply_stdp=apply_stdp)

            # Accumulate output spikes
            self.output_accumulator += output_spikes

        # Normalize by timesteps for rate interpretation
        output_rates = self.output_accumulator / self.n_timesteps

        return output_rates

    def predict(self, spectral_indices: torch.Tensor) -> torch.Tensor:
        """Get class predictions."""
        with torch.no_grad():
            logits = self.forward(spectral_indices)
            return logits.argmax(dim=-1)

    def get_statistics(self) -> Dict[str, Dict]:
        """Get network statistics including energy efficiency."""
        stats = {"layer1": self.snn1.get_all_statistics(), "layer2": self.snn2.get_all_statistics()}

        # Estimate energy efficiency
        total_spikes = (
            stats["layer1"]["layer"]["total_spikes"] + stats["layer2"]["layer"]["total_spikes"]
        )
        max_possible_spikes = self.n_timesteps * (self.n_hidden + self.n_classes)

        stats["energy_efficiency"] = {
            "spike_sparsity": 1 - (total_spikes / (max_possible_spikes + 1e-10)),
            "estimated_power_reduction": f"{100 * (1 - total_spikes / (max_possible_spikes + 1e-10)):.1f}%",
        }

        return stats


class PestDetectionSNN(nn.Module):
    """
    Event-driven SNN for Pest Detection.

    Input: Event camera data (changes only)
    Output: Pest presence / type classification

    Event cameras (like insects' eyes) only report changes:
    - Energy efficient (no redundant data)
    - High temporal resolution
    - Good for detecting motion (pests moving)

    Architecture:
    - Delta encoding of visual events
    - Convolutional-style SNN layer (local receptive fields)
    - Classification output layer

    Biological analog: Insect visual system detecting predators/prey.
    """

    def __init__(
        self,
        input_size: Tuple[int, int] = (32, 32),
        n_classes: int = 5,  # No pest, aphids, caterpillars, beetles, other
        n_timesteps: int = 100,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()

        self.input_size = input_size
        self.n_classes = n_classes
        self.n_timesteps = n_timesteps
        self.device = device

        self.input_dim = input_size[0] * input_size[1]

        # Homeostatic config
        config = HomeostasisConfig(
            target_firing_rate=0.03,  # Very sparse for energy
            enable_sleep_cycles=False,  # Real-time detection
        )

        # Feature extraction layer
        self.feature_layer = HomeostaticSpikingLayer(
            in_features=self.input_dim,
            out_features=128,
            tau_mem=10.0,  # Fast response
            config=config,
            device=device,
        )

        # Classification layer
        self.classifier = HomeostaticSpikingLayer(
            in_features=128, out_features=n_classes, tau_mem=20.0, config=config, device=device
        )

        self.output_accumulator = None

    def reset_state(self, batch_size: int):
        """Reset network state."""
        self.feature_layer.reset_state(batch_size)
        self.classifier.reset_state(batch_size)
        self.output_accumulator = torch.zeros(batch_size, self.n_classes, device=self.device)

    def forward(self, event_stream: torch.Tensor, apply_stdp: bool = False) -> torch.Tensor:
        """
        Process event stream for pest detection.

        Args:
            event_stream: Event data (batch, timesteps, H×W)
            apply_stdp: Enable online learning

        Returns:
            Class logits (batch, n_classes)
        """
        batch_size, n_timesteps, _ = event_stream.shape
        self.reset_state(batch_size)

        # Process event stream
        for t in range(n_timesteps):
            events_t = event_stream[:, t, :]

            # Feature extraction
            features = self.feature_layer(events_t, apply_stdp=apply_stdp)

            # Classification
            output = self.classifier(features, apply_stdp=apply_stdp)

            self.output_accumulator += output

        return self.output_accumulator / n_timesteps


class SoilMoisturePredictorSNN(nn.Module):
    """
    SNN for Soil Moisture Prediction from Temporal Sensor Data.

    Input: Time series of sensor readings
    Output: Future moisture prediction / irrigation recommendation

    Temporal processing is natural for SNNs - the network maintains
    memory through membrane potential dynamics.

    Architecture:
    - Temporal encoding with delta coding (changes only)
    - Recurrent SNN layer (memory)
    - Prediction output

    Sensors:
    - Capacitive moisture sensors
    - Temperature probes
    - Rain gauges
    - Evapotranspiration estimates
    """

    def __init__(
        self,
        n_sensors: int = 4,
        n_hidden: int = 32,
        prediction_horizon: int = 1,  # Hours ahead
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()

        self.n_sensors = n_sensors
        self.n_hidden = n_hidden
        self.device = device

        # Delta encoder for sensor changes
        self.encoder = TemporalEncoder(n_neurons_per_feature=4, encoding="rate", device=device)

        # Homeostatic config for continuous monitoring
        config = HomeostasisConfig(
            target_firing_rate=0.08,
            enable_synaptic_scaling=True,
            sleep_period=10000,  # Longer for continuous operation
        )

        # Hidden layer with temporal memory
        self.hidden = HomeostaticSpikingLayer(
            in_features=n_sensors,
            out_features=n_hidden,
            tau_mem=50.0,  # Long time constant for memory
            config=config,
            device=device,
        )

        # Output layer (regression via rate coding)
        self.output = HomeostaticSpikingLayer(
            in_features=n_hidden, out_features=1, tau_mem=30.0, config=config, device=device
        )

        self.output_accumulator = None

    def forward(self, sensor_history: torch.Tensor, apply_stdp: bool = False) -> torch.Tensor:
        """
        Predict future moisture from sensor history.

        Args:
            sensor_history: Sensor readings (batch, timesteps, n_sensors)
            apply_stdp: Enable online learning

        Returns:
            Moisture prediction (batch, 1)
        """
        batch_size, n_timesteps, _ = sensor_history.shape

        self.hidden.reset_state(batch_size)
        self.output.reset_state(batch_size)
        self.output_accumulator = torch.zeros(batch_size, 1, device=self.device)

        # Process temporal sequence
        for t in range(n_timesteps):
            # Encode sensor values to spikes
            sensor_t = sensor_history[:, t, :]
            spikes_t = (torch.rand_like(sensor_t) < sensor_t).float()

            # Hidden layer
            hidden_spikes = self.hidden(spikes_t, apply_stdp=apply_stdp)

            # Output layer
            output_spikes = self.output(hidden_spikes, apply_stdp=apply_stdp)

            self.output_accumulator += output_spikes

        # Output rate = predicted moisture level
        prediction = self.output_accumulator / n_timesteps

        return prediction


class IrrigationController(nn.Module):
    """
    Complete Irrigation Control System with SNN.

    Combines:
    - Soil moisture prediction
    - Weather integration
    - Crop water requirements
    - Energy-efficient decision making

    Output: Irrigation recommendation (none, light, moderate, heavy)

    Runs continuously on edge device, learning local conditions.
    """

    def __init__(
        self,
        n_sensors: int = 6,  # Moisture, temp, humidity, rain, wind, solar
        n_actions: int = 4,  # Irrigation levels
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()

        self.n_sensors = n_sensors
        self.n_actions = n_actions
        self.device = device

        # Homeostatic config for long-running operation
        config = HomeostasisConfig(
            target_firing_rate=0.04,
            enable_synaptic_scaling=True,
            enable_intrinsic_plasticity=True,
            enable_sleep_cycles=True,
            sleep_period=86400,  # Daily sleep cycle (in timesteps)
        )

        # Sensor processing
        self.sensor_layer = HomeostaticSpikingLayer(
            in_features=n_sensors, out_features=32, tau_mem=30.0, config=config, device=device
        )

        # Decision making
        self.decision_layer = HomeostaticSpikingLayer(
            in_features=32, out_features=n_actions, tau_mem=50.0, config=config, device=device
        )

        # State for continuous operation
        self.accumulated_evidence = None
        self.decision_threshold = 10.0

    def reset_state(self, batch_size: int = 1):
        """Reset for new decision cycle."""
        self.sensor_layer.reset_state(batch_size)
        self.decision_layer.reset_state(batch_size)
        self.accumulated_evidence = torch.zeros(batch_size, self.n_actions, device=self.device)

    def step(self, sensor_reading: torch.Tensor, apply_stdp: bool = True) -> Optional[int]:
        """
        Process single sensor reading.

        Args:
            sensor_reading: Current sensor values (1, n_sensors)
            apply_stdp: Enable continuous learning

        Returns:
            Action (0-3) if decision threshold reached, None otherwise
        """
        if self.accumulated_evidence is None:
            self.reset_state(sensor_reading.shape[0])

        # Encode to spikes
        spikes = (torch.rand_like(sensor_reading) < sensor_reading).float()

        # Process through network
        hidden = self.sensor_layer(spikes, apply_stdp=apply_stdp)
        output = self.decision_layer(hidden, apply_stdp=apply_stdp)

        # Accumulate evidence
        self.accumulated_evidence += output

        # Check if any action reached threshold
        max_evidence = self.accumulated_evidence.max(dim=-1)[0]

        if max_evidence.item() >= self.decision_threshold:
            action = self.accumulated_evidence.argmax(dim=-1).item()
            self.reset_state(sensor_reading.shape[0])
            return action

        return None

    def get_action_name(self, action: int) -> str:
        """Get human-readable action name."""
        names = ["No irrigation", "Light watering", "Moderate watering", "Heavy watering"]
        return names[action] if action < len(names) else "Unknown"


# ============================================================================
# Demo Functions
# ============================================================================


def demo_crop_health():
    """Demo: Crop health classification."""
    print("\n" + "=" * 60)
    print("CROP HEALTH CLASSIFICATION WITH HOMEOSTATIC SNN")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create model
    print("\nCreating crop health classifier...")
    model = CropHealthClassifier(
        n_spectral_features=5,  # NDVI, NDWI, GNDVI, EVI, SAVI
        n_hidden=64,
        n_classes=4,  # Healthy, Stressed, Diseased, Dead
        n_timesteps=50,
        device=device,
    )

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Create synthetic spectral data
    print("\nCreating synthetic multispectral data...")
    batch_size = 16

    # Simulate different crop conditions
    # Healthy: high NDVI (~0.8), moderate NDWI (~0.5)
    healthy = torch.rand(batch_size // 4, 5, device=device) * 0.2 + 0.7

    # Stressed: medium NDVI (~0.5), low NDWI (~0.3)
    stressed = torch.rand(batch_size // 4, 5, device=device) * 0.2 + 0.4

    # Diseased: low NDVI (~0.3), variable
    diseased = torch.rand(batch_size // 4, 5, device=device) * 0.3 + 0.2

    # Dead: very low NDVI (~0.1)
    dead = torch.rand(batch_size // 4, 5, device=device) * 0.15 + 0.05

    spectral_data = torch.cat([healthy, stressed, diseased, dead], dim=0)
    labels = torch.tensor([0] * 4 + [1] * 4 + [2] * 4 + [3] * 4, device=device)

    print(f"Data shape: {spectral_data.shape}")
    print(f"Classes: Healthy(0), Stressed(1), Diseased(2), Dead(3)")

    # Forward pass
    print("\nRunning inference...")
    model.eval()
    with torch.no_grad():
        output = model(spectral_data)
        predictions = output.argmax(dim=-1)

    # Accuracy
    accuracy = (predictions == labels).float().mean().item()
    print(f"Accuracy: {accuracy:.1%}")

    # Get statistics
    stats = model.get_statistics()
    print(f"\nEnergy Efficiency:")
    print(f"  Spike sparsity: {stats['energy_efficiency']['spike_sparsity']:.1%}")
    print(f"  Estimated power reduction: {stats['energy_efficiency']['estimated_power_reduction']}")

    print(f"\nLayer 1 firing rate: {stats['layer1']['layer']['avg_spike_rate']:.2%}")
    print(f"Layer 2 firing rate: {stats['layer2']['layer']['avg_spike_rate']:.2%}")

    return model


def demo_irrigation_controller():
    """Demo: Irrigation control system."""
    print("\n" + "=" * 60)
    print("IRRIGATION CONTROLLER WITH HOMEOSTATIC SNN")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create controller
    print("\nCreating irrigation controller...")
    controller = IrrigationController(
        n_sensors=6,  # Moisture, temp, humidity, rain, wind, solar
        n_actions=4,  # None, light, moderate, heavy
        device=device,
    )

    # Simulate sensor readings over time
    print("\nSimulating field conditions...")

    # Scenario: Dry day, low moisture, high temperature
    print("\nScenario: Hot dry day")
    controller.reset_state()

    for t in range(100):
        # Sensor readings: [moisture, temp, humidity, rain, wind, solar]
        # Low moisture (0.2), high temp (0.8), low humidity (0.3), no rain (0.0)
        reading = torch.tensor([[0.2, 0.8, 0.3, 0.0, 0.4, 0.9]], device=device)

        action = controller.step(reading, apply_stdp=True)

        if action is not None:
            print(f"  Timestep {t}: Decision made - {controller.get_action_name(action)}")
            break
    else:
        print("  No decision reached in 100 timesteps")

    # Scenario: After rain
    print("\nScenario: After rain")
    controller.reset_state()

    for t in range(100):
        # High moisture (0.8), moderate temp (0.5), high humidity (0.7)
        reading = torch.tensor([[0.8, 0.5, 0.7, 0.9, 0.2, 0.4]], device=device)

        action = controller.step(reading, apply_stdp=True)

        if action is not None:
            print(f"  Timestep {t}: Decision made - {controller.get_action_name(action)}")
            break
    else:
        print("  No decision reached (likely: no irrigation needed)")

    return controller


def demo_pest_detection():
    """Demo: Pest detection system."""
    print("\n" + "=" * 60)
    print("PEST DETECTION WITH EVENT-DRIVEN SNN")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create model
    print("\nCreating pest detection SNN...")
    model = PestDetectionSNN(input_size=(32, 32), n_classes=5, n_timesteps=50, device=device)

    # Simulate event camera data
    print("\nSimulating event camera input...")
    batch_size = 8
    n_timesteps = 50

    # Create sparse event stream (mostly zeros, some motion events)
    event_stream = torch.zeros(batch_size, n_timesteps, 32 * 32, device=device)

    # Add some random "motion" events
    for b in range(batch_size):
        # Random moving object trajectory
        n_events = np.random.randint(10, 50)
        for _ in range(n_events):
            t = np.random.randint(0, n_timesteps)
            pixel = np.random.randint(0, 32 * 32)
            event_stream[b, t, pixel] = 1.0

    # Forward pass
    print(f"Event stream shape: {event_stream.shape}")
    print(f"Event sparsity: {1 - event_stream.mean().item():.1%}")

    model.eval()
    with torch.no_grad():
        output = model(event_stream)
        predictions = output.argmax(dim=-1)

    print(f"\nPredictions: {predictions.tolist()}")
    print("Classes: No pest(0), Aphids(1), Caterpillars(2), Beetles(3), Other(4)")

    return model


def main():
    """Run all agricultural demos."""
    print("=" * 60)
    print("AGRICULTURAL MONITORING WITH SPIKING NEURAL NETWORKS")
    print("Session 20 - Research Integration: Domain Examples")
    print("=" * 60)

    print("\nThis example demonstrates SNNs for agriculture:")
    print("  1. Crop Health Classification - Multispectral analysis")
    print("  2. Pest Detection - Event-driven visual sensing")
    print("  3. Irrigation Control - Continuous monitoring")

    print("\nAdvantages of SNNs for agriculture:")
    print("  - Ultra-low power (solar-powered sensors)")
    print("  - Event-driven (process only on changes)")
    print("  - Online learning (adapt to local conditions)")
    print("  - Edge deployment (no cloud required)")

    # Run demos
    try:
        demo_crop_health()
    except Exception as e:
        print(f"Crop health demo error: {e}")

    try:
        demo_irrigation_controller()
    except Exception as e:
        print(f"Irrigation demo error: {e}")

    try:
        demo_pest_detection()
    except Exception as e:
        print(f"Pest detection demo error: {e}")

    print("\n" + "=" * 60)
    print("AGRICULTURAL SNN DEMOS COMPLETE")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  - SNNs provide energy-efficient inference for edge deployment")
    print("  - Homeostatic mechanisms maintain stable operation")
    print("  - STDP enables continuous adaptation to local conditions")
    print("  - Event-driven processing reduces power consumption 10-100×")


if __name__ == "__main__":
    main()
