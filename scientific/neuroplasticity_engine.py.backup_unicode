#!/usr/bin/env python3
"""
Neuroplasticity Engine for Obvivlorum Framework
Computational neuroplasticity with real-time adaptation
"""

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.stats import norm
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
from abc import ABC, abstractmethod
import logging

try:
    import numba
    from numba import jit, prange
    HAS_NUMBA = True
    
    @jit(nopython=True, parallel=True)
    def fast_hebbian_update(weights, pre_synaptic, post_synaptic, learning_rate):
        """Fast Hebbian learning with Numba acceleration"""
        for i in prange(weights.shape[0]):
            for j in prange(weights.shape[1]):
                weights[i, j] += learning_rate * pre_synaptic[j] * post_synaptic[i]
        return weights
    
    @jit(nopython=True, parallel=True) 
    def fast_stdp_update(weights, pre_times, post_times, tau_plus, tau_minus, a_plus, a_minus):
        """Fast STDP update with Numba acceleration"""
        n_pre, n_post = weights.shape
        for i in prange(n_post):
            for j in prange(n_pre):
                if len(pre_times[j]) > 0 and len(post_times[i]) > 0:
                    # Find closest spike pairs
                    min_dt = float('inf')
                    for pre_t in pre_times[j]:
                        for post_t in post_times[i]:
                            dt = post_t - pre_t
                            if abs(dt) < abs(min_dt):
                                min_dt = dt
                    
                    if min_dt != float('inf'):
                        if min_dt > 0:  # Post after pre - potentiation
                            weights[i, j] += a_plus * np.exp(-min_dt / tau_plus)
                        else:  # Pre after post - depression
                            weights[i, j] += a_minus * np.exp(min_dt / tau_minus)
        
        return weights
    
except ImportError:
    HAS_NUMBA = False
    logging.warning("Numba not available. Using standard NumPy implementation.")
    
    def fast_hebbian_update(weights, pre_synaptic, post_synaptic, learning_rate):
        return weights + learning_rate * np.outer(post_synaptic, pre_synaptic)
    
    def fast_stdp_update(weights, pre_times, post_times, tau_plus, tau_minus, a_plus, a_minus):
        # Fallback standard implementation
        return weights

class PlasticityType(Enum):
    """Types of synaptic plasticity"""
    HEBBIAN = "hebbian"
    ANTI_HEBBIAN = "anti_hebbian"
    STDP = "stdp"            # Spike-timing dependent plasticity
    BCM = "bcm"              # Bienenstock-Cooper-Munro rule
    OJA = "oja"              # Oja's rule (normalized Hebbian)
    HOMEOSTATIC = "homeostatic"
    METAPLASTIC = "metaplastic"

@dataclass
class PlasticityRule:
    """Configuration for a plasticity rule"""
    rule_type: PlasticityType
    learning_rate: float = 0.01
    parameters: Dict[str, Any] = field(default_factory=dict)
    weight_bounds: Tuple[float, float] = (-1.0, 1.0)
    enabled: bool = True
    
    def __post_init__(self):
        # Set default parameters based on rule type
        if self.rule_type == PlasticityType.STDP:
            self.parameters.setdefault('tau_plus', 20.0)
            self.parameters.setdefault('tau_minus', 20.0)
            self.parameters.setdefault('a_plus', 0.01)
            self.parameters.setdefault('a_minus', -0.005)
        elif self.rule_type == PlasticityType.BCM:
            self.parameters.setdefault('theta', 1.0)  # Sliding threshold
            self.parameters.setdefault('tau_theta', 1000.0)
        elif self.rule_type == PlasticityType.HOMEOSTATIC:
            self.parameters.setdefault('target_rate', 5.0)  # Target firing rate
            self.parameters.setdefault('tau_homeostatic', 10000.0)

@dataclass
class SynapticConnection:
    """Individual synaptic connection with plasticity"""
    pre_neuron_id: int
    post_neuron_id: int
    weight: float
    delay: float = 1.0  # Synaptic delay in ms
    plasticity_rules: List[PlasticityRule] = field(default_factory=list)
    efficacy: float = 1.0  # Synaptic efficacy
    
    # Plasticity state variables
    eligibility_trace: float = 0.0
    metaplastic_state: float = 0.0
    last_update_time: float = 0.0

@dataclass 
class Neuron:
    """Neuron with adaptive properties"""
    id: int
    membrane_potential: float = -70.0  # mV
    threshold: float = -55.0  # mV
    reset_potential: float = -70.0  # mV
    refractory_period: float = 2.0  # ms
    last_spike_time: float = -np.inf
    
    # Adaptive parameters
    adaptation_current: float = 0.0
    spike_frequency_adaptation: float = 0.0
    intrinsic_excitability: float = 1.0
    
    # Homeostatic variables
    firing_rate: float = 0.0
    target_firing_rate: float = 5.0  # Hz
    homeostatic_scaling: float = 1.0
    
    # Spike history for STDP
    spike_times: List[float] = field(default_factory=list)
    
    def update_firing_rate(self, current_time: float, window: float = 1000.0):
        """Update exponentially weighted firing rate"""
        # Remove old spikes outside window
        cutoff_time = current_time - window
        self.spike_times = [t for t in self.spike_times if t > cutoff_time]
        
        # Calculate instantaneous rate
        if len(self.spike_times) > 1:
            self.firing_rate = len(self.spike_times) / (window / 1000.0)  # Hz
        else:
            self.firing_rate = 0.0

class NeuroplasticNetwork:
    """
    High-performance neuroplastic network with real-time adaptation
    """
    
    def __init__(self, n_neurons: int, connectivity_probability: float = 0.1):
        self.n_neurons = n_neurons
        self.neurons = [Neuron(id=i) for i in range(n_neurons)]
        self.synapses: Dict[Tuple[int, int], SynapticConnection] = {}
        
        # Network matrices for efficient computation
        self.weight_matrix = np.zeros((n_neurons, n_neurons))
        self.delay_matrix = np.ones((n_neurons, n_neurons))  # Default 1ms delay
        
        # Plasticity tracking
        self.plasticity_enabled = True
        self.global_plasticity_rules: List[PlasticityRule] = []
        
        # Performance monitoring
        self.adaptation_history = []
        self.network_activity_history = []
        
        # Initialize random connectivity
        self._initialize_random_connectivity(connectivity_probability)
        
        # Threading for real-time updates
        self._plasticity_thread = None
        self._stop_plasticity = threading.Event()
    
    def _initialize_random_connectivity(self, probability: float):
        """Initialize random synaptic connections"""
        for i in range(self.n_neurons):
            for j in range(self.n_neurons):
                if i != j and np.random.random() < probability:
                    weight = np.random.normal(0, 0.1)  # Small random weights
                    delay = np.random.uniform(0.5, 5.0)  # Random delays
                    
                    connection = SynapticConnection(
                        pre_neuron_id=j,
                        post_neuron_id=i,
                        weight=weight,
                        delay=delay
                    )
                    
                    self.synapses[(j, i)] = connection
                    self.weight_matrix[i, j] = weight
                    self.delay_matrix[i, j] = delay
    
    def add_plasticity_rule(self, rule: PlasticityRule, synapse_filter: Optional[Callable] = None):
        """Add plasticity rule to network or specific synapses"""
        if synapse_filter is None:
            # Apply to all synapses
            self.global_plasticity_rules.append(rule)
        else:
            # Apply to filtered synapses
            for synapse_key, synapse in self.synapses.items():
                if synapse_filter(synapse):
                    synapse.plasticity_rules.append(rule)
    
    def simulate_timestep(self, dt: float, external_input: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Simulate one timestep of network dynamics
        Returns: spike_vector (binary array indicating which neurons spiked)
        """
        current_time = getattr(self, '_current_time', 0.0)
        self._current_time = current_time + dt
        
        if external_input is None:
            external_input = np.zeros(self.n_neurons)
        
        spike_vector = np.zeros(self.n_neurons, dtype=bool)
        
        # Update each neuron
        for neuron in self.neurons:
            # Calculate synaptic input
            synaptic_input = 0.0
            for pre_id in range(self.n_neurons):
                if (pre_id, neuron.id) in self.synapses:
                    synapse = self.synapses[(pre_id, neuron.id)]
                    
                    # Check for delayed spikes from pre-synaptic neuron
                    pre_neuron = self.neurons[pre_id]
                    for spike_time in pre_neuron.spike_times:
                        arrival_time = spike_time + synapse.delay
                        if abs(arrival_time - current_time) < dt/2:  # Spike arrives this timestep
                            synaptic_input += synapse.weight * synapse.efficacy
            
            # Update membrane potential (simple integrate-and-fire)
            neuron.membrane_potential += dt * (
                -(neuron.membrane_potential - neuron.reset_potential) / 10.0 +  # Leak
                synaptic_input * neuron.homeostatic_scaling +  # Synaptic input
                external_input[neuron.id] +  # External input
                -neuron.adaptation_current  # Adaptation current
            )
            
            # Check for spike
            if (neuron.membrane_potential >= neuron.threshold and 
                current_time - neuron.last_spike_time > neuron.refractory_period):
                
                # Spike!
                spike_vector[neuron.id] = True
                neuron.spike_times.append(current_time)
                neuron.last_spike_time = current_time
                neuron.membrane_potential = neuron.reset_potential
                
                # Update adaptation
                neuron.adaptation_current += 2.0
                neuron.spike_frequency_adaptation += 0.1
            
            # Decay adaptation
            neuron.adaptation_current *= np.exp(-dt / 100.0)
            neuron.spike_frequency_adaptation *= np.exp(-dt / 1000.0)
            
            # Update firing rate
            neuron.update_firing_rate(current_time)
        
        # Apply plasticity if enabled
        if self.plasticity_enabled:
            self._apply_plasticity_rules(current_time, dt, spike_vector)
        
        return spike_vector
    
    def _apply_plasticity_rules(self, current_time: float, dt: float, spike_vector: np.ndarray):
        """Apply all active plasticity rules"""
        
        # Pre-compute spike times for efficiency
        pre_spike_times = [[] for _ in range(self.n_neurons)]
        post_spike_times = [[] for _ in range(self.n_neurons)]
        
        for i, neuron in enumerate(self.neurons):
            if spike_vector[i]:
                post_spike_times[i] = [current_time]
            else:
                post_spike_times[i] = []
            
            # Recent pre-synaptic activity (within STDP window)
            recent_spikes = [t for t in neuron.spike_times 
                           if current_time - t < 100.0]  # 100ms window
            pre_spike_times[i] = recent_spikes
        
        # Apply global rules
        for rule in self.global_plasticity_rules:
            self._apply_single_plasticity_rule(rule, current_time, dt, 
                                             pre_spike_times, post_spike_times)
        
        # Apply synapse-specific rules
        for synapse in self.synapses.values():
            for rule in synapse.plasticity_rules:
                self._apply_single_plasticity_rule_to_synapse(
                    rule, synapse, current_time, dt,
                    pre_spike_times, post_spike_times
                )
    
    def _apply_single_plasticity_rule(self, rule: PlasticityRule, current_time: float, 
                                    dt: float, pre_spike_times: List[List[float]], 
                                    post_spike_times: List[List[float]]):
        """Apply a plasticity rule to all eligible synapses"""
        
        if not rule.enabled:
            return
        
        if rule.rule_type == PlasticityType.HEBBIAN:
            self._apply_hebbian_plasticity(rule, pre_spike_times, post_spike_times)
        elif rule.rule_type == PlasticityType.STDP:
            self._apply_stdp_plasticity(rule, pre_spike_times, post_spike_times)
        elif rule.rule_type == PlasticityType.BCM:
            self._apply_bcm_plasticity(rule, current_time, dt, pre_spike_times, post_spike_times)
        elif rule.rule_type == PlasticityType.HOMEOSTATIC:
            self._apply_homeostatic_plasticity(rule, current_time, dt)
    
    def _apply_single_plasticity_rule_to_synapse(self, rule: PlasticityRule, 
                                               synapse: SynapticConnection,
                                               current_time: float, dt: float,
                                               pre_spike_times: List[List[float]],
                                               post_spike_times: List[List[float]]):
        """Apply plasticity rule to specific synapse"""
        
        if not rule.enabled:
            return
        
        pre_id = synapse.pre_neuron_id
        post_id = synapse.post_neuron_id
        
        if rule.rule_type == PlasticityType.HEBBIAN:
            # Simple Hebbian: Δw = η * pre * post
            pre_active = len(pre_spike_times[pre_id]) > 0
            post_active = len(post_spike_times[post_id]) > 0
            
            if pre_active and post_active:
                delta_w = rule.learning_rate
                synapse.weight += delta_w
                synapse.weight = np.clip(synapse.weight, *rule.weight_bounds)
                self.weight_matrix[post_id, pre_id] = synapse.weight
        
        elif rule.rule_type == PlasticityType.STDP:
            # Spike-timing dependent plasticity
            tau_plus = rule.parameters['tau_plus']
            tau_minus = rule.parameters['tau_minus']
            a_plus = rule.parameters['a_plus']
            a_minus = rule.parameters['a_minus']
            
            # Find closest spike pair
            if pre_spike_times[pre_id] and post_spike_times[post_id]:
                min_dt = float('inf')
                for pre_t in pre_spike_times[pre_id]:
                    for post_t in post_spike_times[post_id]:
                        dt_spike = post_t - pre_t
                        if abs(dt_spike) < abs(min_dt):
                            min_dt = dt_spike
                
                if min_dt != float('inf'):
                    if min_dt > 0:  # Post after pre - potentiation
                        delta_w = a_plus * np.exp(-min_dt / tau_plus)
                    else:  # Pre after post - depression
                        delta_w = a_minus * np.exp(min_dt / tau_minus)
                    
                    synapse.weight += delta_w
                    synapse.weight = np.clip(synapse.weight, *rule.weight_bounds)
                    self.weight_matrix[post_id, pre_id] = synapse.weight
    
    def _apply_hebbian_plasticity(self, rule: PlasticityRule, 
                                pre_spike_times: List[List[float]], 
                                post_spike_times: List[List[float]]):
        """Vectorized Hebbian plasticity"""
        # Create activity vectors
        pre_activity = np.array([len(spikes) > 0 for spikes in pre_spike_times], dtype=float)
        post_activity = np.array([len(spikes) > 0 for spikes in post_spike_times], dtype=float)
        
        # Update all weights using fast implementation
        if HAS_NUMBA:
            self.weight_matrix = fast_hebbian_update(
                self.weight_matrix, pre_activity, post_activity, rule.learning_rate
            )
        else:
            self.weight_matrix += rule.learning_rate * np.outer(post_activity, pre_activity)
        
        # Apply bounds and update synapse objects
        self.weight_matrix = np.clip(self.weight_matrix, *rule.weight_bounds)
        
        for synapse_key, synapse in self.synapses.items():
            pre_id, post_id = synapse_key
            synapse.weight = self.weight_matrix[post_id, pre_id]
    
    def _apply_stdp_plasticity(self, rule: PlasticityRule,
                             pre_spike_times: List[List[float]],
                             post_spike_times: List[List[float]]):
        """Vectorized STDP plasticity"""
        tau_plus = rule.parameters['tau_plus']
        tau_minus = rule.parameters['tau_minus']
        a_plus = rule.parameters['a_plus']
        a_minus = rule.parameters['a_minus']
        
        if HAS_NUMBA:
            self.weight_matrix = fast_stdp_update(
                self.weight_matrix, pre_spike_times, post_spike_times,
                tau_plus, tau_minus, a_plus, a_minus
            )
        else:
            # Standard implementation
            for synapse_key, synapse in self.synapses.items():
                pre_id, post_id = synapse_key
                self._apply_single_plasticity_rule_to_synapse(
                    rule, synapse, 0, 0, pre_spike_times, post_spike_times
                )
        
        # Apply bounds
        self.weight_matrix = np.clip(self.weight_matrix, *rule.weight_bounds)
        
        # Update synapse objects
        for synapse_key, synapse in self.synapses.items():
            pre_id, post_id = synapse_key
            synapse.weight = self.weight_matrix[post_id, pre_id]
    
    def _apply_bcm_plasticity(self, rule: PlasticityRule, current_time: float, dt: float,
                            pre_spike_times: List[List[float]], 
                            post_spike_times: List[List[float]]):
        """Bienenstock-Cooper-Munro plasticity with sliding threshold"""
        theta = rule.parameters['theta']
        tau_theta = rule.parameters['tau_theta']
        
        pre_activity = np.array([len(spikes) > 0 for spikes in pre_spike_times], dtype=float)
        post_activity = np.array([len(spikes) > 0 for spikes in post_spike_times], dtype=float)
        
        # Update sliding threshold for each post-synaptic neuron
        for i, neuron in enumerate(self.neurons):
            # Update threshold based on recent activity
            neuron.threshold += dt/tau_theta * (post_activity[i]**2 - theta)
        
        # BCM rule: Δw = η * pre * post * (post - θ)
        for synapse_key, synapse in self.synapses.items():
            pre_id, post_id = synapse_key
            post_neuron = self.neurons[post_id]
            
            if pre_activity[pre_id] and post_activity[post_id]:
                bcm_factor = post_activity[post_id] - post_neuron.threshold
                delta_w = rule.learning_rate * pre_activity[pre_id] * post_activity[post_id] * bcm_factor
                
                synapse.weight += delta_w
                synapse.weight = np.clip(synapse.weight, *rule.weight_bounds)
                self.weight_matrix[post_id, pre_id] = synapse.weight
    
    def _apply_homeostatic_plasticity(self, rule: PlasticityRule, current_time: float, dt: float):
        """Homeostatic synaptic scaling"""
        target_rate = rule.parameters['target_rate']
        tau_homeostatic = rule.parameters['tau_homeostatic']
        
        for neuron in self.neurons:
            # Calculate scaling factor based on firing rate
            rate_error = target_rate - neuron.firing_rate
            scaling_change = dt/tau_homeostatic * rate_error * 0.01  # Small changes
            
            neuron.homeostatic_scaling += scaling_change
            neuron.homeostatic_scaling = np.clip(neuron.homeostatic_scaling, 0.1, 10.0)
    
    def adapt_topology(self, experience: Dict[str, Any]):
        """
        Adapt network topology based on experience
        Add/remove synapses based on activity patterns
        """
        if 'activity_pattern' not in experience:
            return
        
        activity_pattern = experience['activity_pattern']
        adaptation_strength = experience.get('adaptation_strength', 0.01)
        
        # Find highly correlated neuron pairs
        correlations = np.corrcoef(activity_pattern)
        correlation_threshold = 0.7
        
        # Add new synapses for highly correlated neurons
        for i in range(self.n_neurons):
            for j in range(self.n_neurons):
                if i != j and abs(correlations[i, j]) > correlation_threshold:
                    if (j, i) not in self.synapses:
                        # Add new synapse
                        weight = 0.1 * np.sign(correlations[i, j])
                        delay = np.random.uniform(0.5, 3.0)
                        
                        new_synapse = SynapticConnection(
                            pre_neuron_id=j,
                            post_neuron_id=i,
                            weight=weight,
                            delay=delay
                        )
                        
                        self.synapses[(j, i)] = new_synapse
                        self.weight_matrix[i, j] = weight
                        self.delay_matrix[i, j] = delay
        
        # Remove weak synapses
        weak_threshold = 0.01
        synapses_to_remove = []
        
        for synapse_key, synapse in self.synapses.items():
            if abs(synapse.weight) < weak_threshold:
                synapses_to_remove.append(synapse_key)
        
        for synapse_key in synapses_to_remove:
            del self.synapses[synapse_key]
            pre_id, post_id = synapse_key
            self.weight_matrix[post_id, pre_id] = 0.0
    
    def start_real_time_plasticity(self, update_interval: float = 10.0):
        """Start background thread for continuous plasticity updates"""
        if self._plasticity_thread is not None:
            return
        
        def plasticity_worker():
            while not self._stop_plasticity.is_set():
                if self.plasticity_enabled:
                    # Perform background adaptations
                    current_time = getattr(self, '_current_time', 0.0)
                    
                    # Update homeostatic scaling
                    for rule in self.global_plasticity_rules:
                        if rule.rule_type == PlasticityType.HOMEOSTATIC:
                            self._apply_homeostatic_plasticity(rule, current_time, update_interval)
                
                time.sleep(update_interval / 1000.0)  # Convert to seconds
        
        self._plasticity_thread = threading.Thread(target=plasticity_worker, daemon=True)
        self._plasticity_thread.start()
    
    def stop_real_time_plasticity(self):
        """Stop background plasticity thread"""
        if self._plasticity_thread is not None:
            self._stop_plasticity.set()
            self._plasticity_thread.join()
            self._plasticity_thread = None
            self._stop_plasticity.clear()
    
    def get_network_statistics(self) -> Dict[str, Any]:
        """Get comprehensive network statistics"""
        weights = self.weight_matrix[self.weight_matrix != 0]
        
        return {
            "n_neurons": self.n_neurons,
            "n_synapses": len(self.synapses),
            "connectivity": len(self.synapses) / (self.n_neurons * (self.n_neurons - 1)),
            "mean_weight": np.mean(weights) if len(weights) > 0 else 0,
            "weight_std": np.std(weights) if len(weights) > 0 else 0,
            "mean_firing_rate": np.mean([n.firing_rate for n in self.neurons]),
            "mean_homeostatic_scaling": np.mean([n.homeostatic_scaling for n in self.neurons]),
            "plasticity_rules_active": len(self.global_plasticity_rules),
            "mean_intrinsic_excitability": np.mean([n.intrinsic_excitability for n in self.neurons])
        }

# Example usage and demonstration
def demonstrate_neuroplasticity():
    """Demonstrate neuroplasticity capabilities"""
    print("Demonstrating Neuroplastic Network...")
    
    # Create network
    network = NeuroplasticNetwork(n_neurons=50, connectivity_probability=0.15)
    
    # Add plasticity rules
    hebbian_rule = PlasticityRule(
        rule_type=PlasticityType.HEBBIAN,
        learning_rate=0.001,
        weight_bounds=(-0.5, 0.5)
    )
    
    stdp_rule = PlasticityRule(
        rule_type=PlasticityType.STDP,
        learning_rate=1.0,
        parameters={
            'tau_plus': 20.0,
            'tau_minus': 20.0,
            'a_plus': 0.01,
            'a_minus': -0.005
        }
    )
    
    homeostatic_rule = PlasticityRule(
        rule_type=PlasticityType.HOMEOSTATIC,
        learning_rate=0.001,
        parameters={
            'target_rate': 5.0,
            'tau_homeostatic': 5000.0
        }
    )
    
    network.add_plasticity_rule(hebbian_rule)
    network.add_plasticity_rule(stdp_rule)
    network.add_plasticity_rule(homeostatic_rule)
    
    print("\nInitial network statistics:")
    stats = network.get_network_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Simulate network activity
    print("\nSimulating network activity with plasticity...")
    dt = 1.0  # 1ms timesteps
    n_steps = 1000
    
    activity_history = []
    
    for step in range(n_steps):
        # Create structured input pattern
        external_input = np.zeros(network.n_neurons)
        if step % 100 < 20:  # Periodic stimulation
            external_input[:10] = 10.0  # Stimulate first 10 neurons
        
        # Add noise
        external_input += np.random.normal(0, 2.0, network.n_neurons)
        
        # Simulate timestep
        spikes = network.simulate_timestep(dt, external_input)
        activity_history.append(spikes.copy())
    
    print("Simulation complete.")
    
    print("\nFinal network statistics:")
    final_stats = network.get_network_statistics()
    for key, value in final_stats.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Show changes
    print("\nPlasticity-induced changes:")
    print(f"  Connectivity change: {final_stats['connectivity'] - stats['connectivity']:.4f}")
    print(f"  Mean weight change: {final_stats['mean_weight'] - stats['mean_weight']:.4f}")
    print(f"  Firing rate change: {final_stats['mean_firing_rate'] - stats['mean_firing_rate']:.4f}")
    
    # Test topology adaptation
    print("\nTesting topology adaptation...")
    activity_pattern = np.array(activity_history[-200:]).T  # Last 200 steps
    experience = {
        'activity_pattern': activity_pattern,
        'adaptation_strength': 0.05
    }
    
    pre_adaptation_synapses = len(network.synapses)
    network.adapt_topology(experience)
    post_adaptation_synapses = len(network.synapses)
    
    print(f"Synapses before adaptation: {pre_adaptation_synapses}")
    print(f"Synapses after adaptation: {post_adaptation_synapses}")
    print(f"Topology change: {post_adaptation_synapses - pre_adaptation_synapses} synapses")
    
    print("\nNeuroplasticity demonstration complete.")

if __name__ == "__main__":
    demonstrate_neuroplasticity()