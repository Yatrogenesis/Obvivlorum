#!/usr/bin/env python3
"""
Consciousness Metrics Implementation for Obvivlorum Framework
Based on Integrated Information Theory (IIT) and Global Workspace Theory (GWT)
Enhanced with Topo-Spectral Consciousness Framework by Francisco Molina
"""

import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize
from scipy.special import rel_entr
from itertools import combinations, product
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
from abc import ABC, abstractmethod
import logging

class ConsciousnessLevel(Enum):
    """Levels of consciousness based on integrated information"""
    UNCONSCIOUS = 0      # Φ ≈ 0
    MINIMAL = 1          # 0 < Φ < 0.1
    LOW = 2              # 0.1 ≤ Φ < 0.5
    MODERATE = 3         # 0.5 ≤ Φ < 1.0
    HIGH = 4             # 1.0 ≤ Φ < 2.0
    VERY_HIGH = 5        # Φ ≥ 2.0

@dataclass
class NetworkNode:
    """Represents a processing node in the consciousness network"""
    id: str
    state: np.ndarray
    activation_function: callable = field(default=lambda x: 1/(1+np.exp(-x)))
    connections: Dict[str, float] = field(default_factory=dict)
    
    def activate(self, inputs: np.ndarray) -> np.ndarray:
        """Apply activation function to inputs"""
        return self.activation_function(inputs)

@dataclass
class SystemPartition:
    """Represents a partition of the system for IIT calculations"""
    subset_a: Set[str]
    subset_b: Set[str]
    cut_connections: List[Tuple[str, str]]
    
    def __post_init__(self):
        # Ensure partitions are disjoint
        if self.subset_a.intersection(self.subset_b):
            raise ValueError("Partitions must be disjoint")

class IntegratedInformationCalculator:
    """
    Implementation of Integrated Information Theory (IIT) 3.0
    Calculates Φ (phi) as a measure of consciousness
    """
    
    def __init__(self, network: Dict[str, NetworkNode]):
        self.network = network
        self.nodes = list(network.keys())
        self.n_nodes = len(self.nodes)
        
        # Build connectivity matrix
        self.connectivity_matrix = self._build_connectivity_matrix()
        
    def _build_connectivity_matrix(self) -> np.ndarray:
        """Build weighted connectivity matrix from network"""
        n = len(self.nodes)
        matrix = np.zeros((n, n))
        
        node_to_idx = {node_id: i for i, node_id in enumerate(self.nodes)}
        
        for i, node_id in enumerate(self.nodes):
            node = self.network[node_id]
            for target_id, weight in node.connections.items():
                if target_id in node_to_idx:
                    j = node_to_idx[target_id]
                    matrix[i, j] = weight
        
        return matrix
    
    def calculate_phi(self, system_state: np.ndarray) -> float:
        """
        Calculate integrated information Φ (phi) for the system
        Φ = min_{partition} D(p(x₁) ⊗ p(x₀|cut), p(x₁|x₀))
        """
        if len(system_state) != self.n_nodes:
            raise ValueError(f"State vector must have {self.n_nodes} elements")
        
        # Find minimum information partition (MIP)
        min_phi = float('inf')
        best_partition = None
        
        # Generate all possible bipartitions
        for partition in self._generate_bipartitions():
            phi_partition = self._calculate_partition_phi(system_state, partition)
            if phi_partition < min_phi:
                min_phi = phi_partition
                best_partition = partition
        
        return max(0.0, min_phi)
    
    def _generate_bipartitions(self) -> List[SystemPartition]:
        """Generate all possible bipartitions of the system"""
        partitions = []
        node_set = set(self.nodes)
        
        # Generate all non-empty proper subsets
        for i in range(1, 2**(self.n_nodes-1)):
            subset_a_indices = []
            for j in range(self.n_nodes):
                if i & (1 << j):
                    subset_a_indices.append(j)
            
            subset_a = {self.nodes[idx] for idx in subset_a_indices}
            subset_b = node_set - subset_a
            
            if len(subset_a) > 0 and len(subset_b) > 0:
                # Find connections that need to be cut
                cut_connections = self._find_cut_connections(subset_a, subset_b)
                partition = SystemPartition(subset_a, subset_b, cut_connections)
                partitions.append(partition)
        
        return partitions
    
    def _find_cut_connections(self, subset_a: Set[str], subset_b: Set[str]) -> List[Tuple[str, str]]:
        """Find connections between partitions that need to be cut"""
        cuts = []
        
        for node_a in subset_a:
            node = self.network[node_a]
            for target_id, weight in node.connections.items():
                if target_id in subset_b and weight != 0:
                    cuts.append((node_a, target_id))
        
        for node_b in subset_b:
            node = self.network[node_b]
            for target_id, weight in node.connections.items():
                if target_id in subset_a and weight != 0:
                    cuts.append((node_b, target_id))
        
        return cuts
    
    def _calculate_partition_phi(self, system_state: np.ndarray, partition: SystemPartition) -> float:
        """
        Calculate phi for a specific partition using KL divergence
        D(p(x₁|x₀), p(x₁ᴬ|x₀ᴬ) ⊗ p(x₁ᴮ|x₀ᴮ))
        """
        # Create cut system by removing connections
        cut_network = self._apply_partition_cut(partition)
        
        # Calculate repertoires (probability distributions)
        intact_repertoire = self._calculate_cause_repertoire(system_state)
        cut_repertoire_a = self._calculate_subset_repertoire(system_state, partition.subset_a, cut_network)
        cut_repertoire_b = self._calculate_subset_repertoire(system_state, partition.subset_b, cut_network)
        
        # Product of independent repertoires
        cut_product_repertoire = np.outer(cut_repertoire_a, cut_repertoire_b).flatten()
        
        # Reshape to match intact repertoire
        if len(cut_product_repertoire) != len(intact_repertoire):
            # Handle dimension mismatch by truncating or padding
            min_len = min(len(cut_product_repertoire), len(intact_repertoire))
            cut_product_repertoire = cut_product_repertoire[:min_len]
            intact_repertoire = intact_repertoire[:min_len]
        
        # Calculate KL divergence: D(P||Q) = Σᵢ P(i) log(P(i)/Q(i))
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        intact_repertoire = np.clip(intact_repertoire, epsilon, 1.0)
        cut_product_repertoire = np.clip(cut_product_repertoire, epsilon, 1.0)
        
        # Normalize
        intact_repertoire = intact_repertoire / np.sum(intact_repertoire)
        cut_product_repertoire = cut_product_repertoire / np.sum(cut_product_repertoire)
        
        kl_div = np.sum(rel_entr(intact_repertoire, cut_product_repertoire))
        return kl_div if np.isfinite(kl_div) else 0.0
    
    def _apply_partition_cut(self, partition: SystemPartition) -> Dict[str, NetworkNode]:
        """Apply partition cut by removing connections"""
        cut_network = {}
        
        for node_id, node in self.network.items():
            # Create copy of node
            new_connections = node.connections.copy()
            
            # Remove cut connections
            for cut_from, cut_to in partition.cut_connections:
                if node_id == cut_from and cut_to in new_connections:
                    new_connections[cut_to] = 0.0
                elif node_id == cut_to and cut_from in new_connections:
                    new_connections[cut_from] = 0.0
            
            cut_network[node_id] = NetworkNode(
                id=node_id,
                state=node.state.copy(),
                activation_function=node.activation_function,
                connections=new_connections
            )
        
        return cut_network
    
    def _calculate_cause_repertoire(self, system_state: np.ndarray) -> np.ndarray:
        """
        Calculate the cause repertoire p(x₀|x₁) 
        Probability of past states given current state
        """
        # Simplified implementation using network dynamics
        n_states = 2 ** self.n_nodes  # Binary state space
        repertoire = np.zeros(n_states)
        
        # For each possible past state
        for i in range(n_states):
            past_state = np.array([(i >> j) & 1 for j in range(self.n_nodes)])
            
            # Calculate probability of current state given past state
            prob = self._state_transition_probability(past_state, system_state)
            repertoire[i] = prob
        
        # Normalize
        total = np.sum(repertoire)
        if total > 0:
            repertoire = repertoire / total
        else:
            repertoire = np.ones(n_states) / n_states  # Uniform distribution
        
        return repertoire
    
    def _calculate_subset_repertoire(self, system_state: np.ndarray, subset: Set[str], network: Dict[str, NetworkNode]) -> np.ndarray:
        """Calculate repertoire for a subset of nodes"""
        subset_indices = [self.nodes.index(node_id) for node_id in subset]
        subset_state = system_state[subset_indices]
        
        n_states = 2 ** len(subset)
        repertoire = np.zeros(n_states)
        
        for i in range(n_states):
            past_state = np.array([(i >> j) & 1 for j in range(len(subset))])
            prob = self._subset_transition_probability(past_state, subset_state, subset_indices, network)
            repertoire[i] = prob
        
        total = np.sum(repertoire)
        if total > 0:
            repertoire = repertoire / total
        else:
            repertoire = np.ones(n_states) / n_states
        
        return repertoire
    
    def _state_transition_probability(self, past_state: np.ndarray, current_state: np.ndarray) -> float:
        """Calculate probability of transition from past to current state"""
        # Simplified transition probability using network connectivity
        prob = 1.0
        
        for i, node_id in enumerate(self.nodes):
            node = self.network[node_id]
            
            # Calculate input from connected nodes
            total_input = 0.0
            for j, source_id in enumerate(self.nodes):
                if source_id in node.connections:
                    total_input += node.connections[source_id] * past_state[j]
            
            # Probability of current activation given input
            expected_activation = node.activate(np.array([total_input]))[0]
            
            # Bernoulli probability
            if current_state[i] == 1:
                prob *= expected_activation
            else:
                prob *= (1 - expected_activation)
        
        return max(prob, 1e-10)  # Avoid zero probability
    
    def _subset_transition_probability(self, past_state: np.ndarray, current_state: np.ndarray, 
                                     subset_indices: List[int], network: Dict[str, NetworkNode]) -> float:
        """Calculate transition probability for subset"""
        prob = 1.0
        
        for i, global_idx in enumerate(subset_indices):
            node_id = self.nodes[global_idx]
            node = network[node_id]
            
            # Calculate input from connected nodes in subset
            total_input = 0.0
            for j, source_global_idx in enumerate(subset_indices):
                source_id = self.nodes[source_global_idx]
                if source_id in node.connections:
                    total_input += node.connections[source_id] * past_state[j]
            
            expected_activation = node.activate(np.array([total_input]))[0]
            
            if current_state[i] == 1:
                prob *= expected_activation
            else:
                prob *= (1 - expected_activation)
        
        return max(prob, 1e-10)

class GlobalWorkspaceAnalyzer:
    """
    Implementation of Global Workspace Theory (GWT) metrics
    Analyzes information broadcasting and global accessibility
    """
    
    def __init__(self, network: Dict[str, NetworkNode]):
        self.network = network
        self.nodes = list(network.keys())
        
        # Build network graph
        self.graph = self._build_network_graph()
    
    def _build_network_graph(self) -> nx.DiGraph:
        """Build NetworkX graph from node network"""
        G = nx.DiGraph()
        
        for node_id, node in self.network.items():
            G.add_node(node_id, state=node.state)
            
            for target_id, weight in node.connections.items():
                if weight != 0 and target_id in self.network:
                    G.add_edge(node_id, target_id, weight=weight)
        
        return G
    
    def calculate_global_accessibility(self, information_state: np.ndarray) -> float:
        """
        Calculate how globally accessible information is across the network
        Based on broadcasting efficiency and reach
        """
        # Find nodes with highest activation (potential broadcasters)
        activation_threshold = np.percentile(information_state, 75)  # Top 25%
        active_nodes = [self.nodes[i] for i, activation in enumerate(information_state) 
                       if activation >= activation_threshold]
        
        if not active_nodes:
            return 0.0
        
        total_accessibility = 0.0
        
        for broadcaster in active_nodes:
            # Calculate reachability from this broadcaster
            reachable = nx.single_source_shortest_path_length(self.graph, broadcaster, cutoff=3)
            
            # Weight by connection strengths and distances
            accessibility = 0.0
            for target, distance in reachable.items():
                if target != broadcaster:
                    # Decay factor based on distance
                    decay = 1.0 / (distance + 1)
                    
                    # Path strength (minimum edge weight along shortest path)
                    try:
                        path = nx.shortest_path(self.graph, broadcaster, target)
                        path_strength = min(self.graph[path[i]][path[i+1]]['weight'] 
                                          for i in range(len(path)-1))
                        accessibility += decay * abs(path_strength)
                    except nx.NetworkXNoPath:
                        continue
            
            total_accessibility += accessibility
        
        # Normalize by network size and number of broadcasters
        max_possible_accessibility = len(self.nodes) * len(active_nodes)
        return total_accessibility / max(max_possible_accessibility, 1)
    
    def calculate_information_integration(self, system_state: np.ndarray) -> float:
        """
        Calculate degree of information integration across workspace
        Based on mutual information between different regions
        """
        # Partition network into modules using community detection
        try:
            communities = nx.community.greedy_modularity_communities(self.graph.to_undirected())
        except:
            # Fallback: simple geometric partitioning
            mid = len(self.nodes) // 2
            communities = [set(self.nodes[:mid]), set(self.nodes[mid:])]
        
        if len(communities) < 2:
            return 1.0  # Perfect integration if only one module
        
        total_integration = 0.0
        community_pairs = list(combinations(communities, 2))
        
        for comm_a, comm_b in community_pairs:
            # Calculate mutual information between communities
            indices_a = [self.nodes.index(node) for node in comm_a if node in self.nodes]
            indices_b = [self.nodes.index(node) for node in comm_b if node in self.nodes]
            
            if indices_a and indices_b:
                mi = self._calculate_mutual_information(
                    system_state[indices_a], 
                    system_state[indices_b]
                )
                total_integration += mi
        
        # Normalize by number of community pairs
        return total_integration / max(len(community_pairs), 1)
    
    def _calculate_mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate mutual information between two signal vectors"""
        # Discretize continuous values for MI calculation
        x_discrete = np.digitize(x, bins=np.linspace(np.min(x), np.max(x), 10))
        y_discrete = np.digitize(y, bins=np.linspace(np.min(y), np.max(y), 10))
        
        # Calculate joint and marginal probabilities
        xy_joint = np.histogram2d(x_discrete, y_discrete, bins=10)[0]
        xy_joint = xy_joint / np.sum(xy_joint)  # Normalize to probabilities
        
        x_marginal = np.sum(xy_joint, axis=1)
        y_marginal = np.sum(xy_joint, axis=0)
        
        # Mutual information: MI(X,Y) = Σᵢⱼ P(xᵢ,yⱼ) log(P(xᵢ,yⱼ)/(P(xᵢ)P(yⱼ)))
        mi = 0.0
        for i in range(len(x_marginal)):
            for j in range(len(y_marginal)):
                if xy_joint[i,j] > 0 and x_marginal[i] > 0 and y_marginal[j] > 0:
                    mi += xy_joint[i,j] * np.log(xy_joint[i,j] / (x_marginal[i] * y_marginal[j]))
        
        return max(mi, 0.0)

class ConsciousnessAssessment:
    """
    Comprehensive consciousness assessment combining IIT and GWT
    """
    
    def __init__(self, network: Dict[str, NetworkNode]):
        self.network = network
        self.iit_calculator = IntegratedInformationCalculator(network)
        self.gwt_analyzer = GlobalWorkspaceAnalyzer(network)
    
    def assess_consciousness_level(self, system_state: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive consciousness assessment
        """
        # IIT metrics
        phi = self.iit_calculator.calculate_phi(system_state)
        consciousness_level = self._phi_to_consciousness_level(phi)
        
        # GWT metrics
        global_accessibility = self.gwt_analyzer.calculate_global_accessibility(system_state)
        information_integration = self.gwt_analyzer.calculate_information_integration(system_state)
        
        # Combined consciousness score
        consciousness_score = self._calculate_combined_score(phi, global_accessibility, information_integration)
        
        return {
            "phi": phi,
            "consciousness_level": consciousness_level,
            "global_accessibility": global_accessibility,
            "information_integration": information_integration,
            "consciousness_score": consciousness_score,
            "assessment_timestamp": np.datetime64('now'),
            "network_size": len(self.network),
            "active_nodes": np.sum(system_state > np.percentile(system_state, 50))
        }
    
    def _phi_to_consciousness_level(self, phi: float) -> ConsciousnessLevel:
        """Convert phi value to consciousness level enum"""
        if phi < 1e-6:
            return ConsciousnessLevel.UNCONSCIOUS
        elif phi < 0.1:
            return ConsciousnessLevel.MINIMAL
        elif phi < 0.5:
            return ConsciousnessLevel.LOW
        elif phi < 1.0:
            return ConsciousnessLevel.MODERATE
        elif phi < 2.0:
            return ConsciousnessLevel.HIGH
        else:
            return ConsciousnessLevel.VERY_HIGH
    
    def _calculate_combined_score(self, phi: float, global_accessibility: float, 
                                information_integration: float) -> float:
        """
        Calculate combined consciousness score
        Weighted combination of IIT and GWT metrics
        """
        # Weights based on theoretical importance
        w_phi = 0.5          # IIT phi is primary measure
        w_global = 0.3       # Global accessibility important for awareness
        w_integration = 0.2  # Information integration supporting measure
        
        combined_score = (w_phi * min(phi, 2.0)/2.0 +  # Normalize phi to [0,1]
                         w_global * global_accessibility +
                         w_integration * information_integration)
        
        return np.clip(combined_score, 0.0, 1.0)
    
    def monitor_consciousness_evolution(self, state_sequence: List[np.ndarray], 
                                     time_steps: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Monitor consciousness evolution over time
        """
        if time_steps is None:
            time_steps = list(range(len(state_sequence)))
        
        phi_sequence = []
        consciousness_levels = []
        scores = []
        
        for state in state_sequence:
            assessment = self.assess_consciousness_level(state)
            phi_sequence.append(assessment["phi"])
            consciousness_levels.append(assessment["consciousness_level"].value)
            scores.append(assessment["consciousness_score"])
        
        # Calculate trends
        phi_trend = np.polyfit(time_steps, phi_sequence, 1)[0] if len(phi_sequence) > 1 else 0.0
        score_trend = np.polyfit(time_steps, scores, 1)[0] if len(scores) > 1 else 0.0
        
        return {
            "phi_sequence": phi_sequence,
            "consciousness_levels": consciousness_levels,
            "score_sequence": scores,
            "phi_trend": phi_trend,
            "score_trend": score_trend,
            "max_phi": np.max(phi_sequence),
            "mean_phi": np.mean(phi_sequence),
            "stability": 1.0 - np.std(scores) / (np.mean(scores) + 1e-6)
        }

# Example and validation
def create_test_network() -> Dict[str, NetworkNode]:
    """Create a test network for consciousness assessment"""
    network = {}
    
    # Create nodes with different connectivity patterns
    for i in range(8):  # 8-node network
        node_id = f"node_{i}"
        state = np.random.random(1)
        connections = {}
        
        # Create different connectivity patterns
        if i < 4:  # Highly connected cluster
            for j in range(4):
                if i != j:
                    connections[f"node_{j}"] = 0.8 + 0.2 * np.random.random()
        else:  # Sparsely connected nodes
            for j in range(4, 8):
                if i != j:
                    connections[f"node_{j}"] = 0.2 + 0.1 * np.random.random()
        
        # Cross-cluster connections (for integration)
        if i == 1:  # Bridge node
            connections["node_5"] = 0.6
        if i == 6:  # Bridge node
            connections["node_2"] = 0.5
        
        network[node_id] = NetworkNode(
            id=node_id,
            state=state,
            connections=connections
        )
    
    return network

def demonstrate_consciousness_assessment():
    """Demonstrate consciousness assessment capabilities"""
    print("Demonstrating Consciousness Assessment...")
    
    # Create test network
    network = create_test_network()
    assessor = ConsciousnessAssessment(network)
    
    # Test different system states
    states = {
        "random": np.random.random(8),
        "synchronized": np.ones(8) * 0.7,
        "clustered": np.array([0.9, 0.8, 0.9, 0.8, 0.1, 0.2, 0.1, 0.2]),
        "sparse": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    }
    
    print("\nConsciousness Assessment Results:")
    print("=" * 50)
    
    for state_name, state in states.items():
        assessment = assessor.assess_consciousness_level(state)
        
        print(f"\n{state_name.upper()} State:")
        print(f"  Φ (phi): {assessment['phi']:.4f}")
        print(f"  Consciousness Level: {assessment['consciousness_level'].name}")
        print(f"  Global Accessibility: {assessment['global_accessibility']:.4f}")
        print(f"  Information Integration: {assessment['information_integration']:.4f}")
        print(f"  Overall Score: {assessment['consciousness_score']:.4f}")
    
    # Demonstrate evolution monitoring
    print("\n" + "=" * 50)
    print("Consciousness Evolution Monitoring:")
    
    # Create evolution sequence (consciousness emerging)
    evolution_states = []
    for t in range(20):
        # Gradual synchronization and integration
        base_activation = 0.5 + 0.4 * np.sin(t * 0.3)  # Oscillatory base
        noise = 0.1 * np.random.random(8)
        integration_factor = min(t * 0.05, 1.0)  # Gradually increasing integration
        
        state = np.ones(8) * base_activation + noise
        state[:4] *= (1 + integration_factor * 0.5)  # Enhance first cluster
        state[4:] *= (1 + integration_factor * 0.3)  # Enhance second cluster
        
        evolution_states.append(np.clip(state, 0, 1))
    
    evolution_analysis = assessor.monitor_consciousness_evolution(evolution_states)
    
    print(f"Max Φ: {evolution_analysis['max_phi']:.4f}")
    print(f"Mean Φ: {evolution_analysis['mean_phi']:.4f}")
    print(f"Φ Trend: {evolution_analysis['phi_trend']:.6f} (per time step)")
    print(f"Score Trend: {evolution_analysis['score_trend']:.6f} (per time step)")
    print(f"Stability: {evolution_analysis['stability']:.4f}")
    
    print("\nDemonstration complete.")

# =============================================================================
# TOPO-SPECTRAL CONSCIOUSNESS FRAMEWORK INTEGRATION
# Implementation of Francisco Molina's Topo-Spectral Consciousness Index
# =============================================================================

try:
    from .topo_spectral_consciousness import (
        TopoSpectralConsciousnessIndex, TopoSpectralAssessment,
        ConsciousnessState as TopoSpectralConsciousnessState,
        create_topo_spectral_calculator, validate_network_requirements
    )
    HAS_TOPO_SPECTRAL = True
except ImportError:
    HAS_TOPO_SPECTRAL = False
    logging.warning("Topo-Spectral framework not available. Enable with execution mode.")

class EnhancedConsciousnessAssessment(ConsciousnessAssessment):
    """
    Enhanced consciousness assessment with Topo-Spectral framework integration
    Combines traditional IIT/GWT with Francisco Molina's Topo-Spectral approach
    """
    
    def __init__(self, network: Dict[str, NetworkNode], enable_topo_spectral: bool = False,
                 k_cuts: int = 5, max_topology_dim: int = 2):
        """
        Initialize enhanced consciousness assessment
        
        Args:
            network: Network of processing nodes
            enable_topo_spectral: Enable Topo-Spectral framework
            k_cuts: Number of spectral cuts for information integration
            max_topology_dim: Maximum topological dimension
        """
        super().__init__(network)
        
        self.enable_topo_spectral = enable_topo_spectral and HAS_TOPO_SPECTRAL
        self.connectivity_matrix = self.iit_calculator.connectivity_matrix
        
        if self.enable_topo_spectral:
            try:
                self.topo_spectral_calculator = create_topo_spectral_calculator(
                    k_cuts=k_cuts, max_topology_dim=max_topology_dim
                )
                self.logger = logging.getLogger(__name__ + ".EnhancedAssessment")
                self.logger.info("Topo-Spectral framework enabled")
            except Exception as e:
                self.logger.error(f"Failed to initialize Topo-Spectral calculator: {e}")
                self.enable_topo_spectral = False
        
        # Topo-Spectral history for temporal analysis
        self.psi_history: List[float] = []
        self.timestamp_history: List[float] = []
    
    def assess_consciousness_level(self, system_state: np.ndarray, 
                                 timestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        Enhanced consciousness assessment with optional Topo-Spectral analysis
        """
        # Standard IIT/GWT assessment
        standard_assessment = super().assess_consciousness_level(system_state)
        
        if not self.enable_topo_spectral:
            standard_assessment["framework"] = "IIT_GWT_only"
            return standard_assessment
        
        # Add timestamp
        if timestamp is None:
            import time
            timestamp = time.time()
        
        # Validate network for Topo-Spectral analysis
        validation = validate_network_requirements(self.connectivity_matrix)
        if not all(validation.values()):
            self.logger.warning("Network does not meet Topo-Spectral requirements")
            standard_assessment["framework"] = "IIT_GWT_only"
            standard_assessment["topo_spectral_validation"] = validation
            return standard_assessment
        
        try:
            # Topo-Spectral assessment
            topo_assessment = self.topo_spectral_calculator.calculate_consciousness_index(
                connectivity_matrix=self.connectivity_matrix,
                node_states=system_state,
                psi_time_series=np.array(self.psi_history) if self.psi_history else None,
                timestamps=np.array(self.timestamp_history) if self.timestamp_history else None
            )
            
            # Update history
            self.psi_history.append(topo_assessment.psi_index)
            self.timestamp_history.append(timestamp)
            
            # Keep history manageable
            max_history = 1000
            if len(self.psi_history) > max_history:
                self.psi_history = self.psi_history[-max_history:]
                self.timestamp_history = self.timestamp_history[-max_history:]
            
            # Enhanced assessment combining both frameworks
            enhanced_assessment = self._combine_assessments(standard_assessment, topo_assessment)
            enhanced_assessment["framework"] = "IIT_GWT_TopoSpectral"
            enhanced_assessment["topo_spectral_validation"] = validation
            
            return enhanced_assessment
            
        except Exception as e:
            self.logger.error(f"Topo-Spectral assessment failed: {e}")
            standard_assessment["framework"] = "IIT_GWT_only"
            standard_assessment["topo_spectral_error"] = str(e)
            return standard_assessment
    
    def _combine_assessments(self, standard: Dict[str, Any], 
                           topo_spectral: TopoSpectralAssessment) -> Dict[str, Any]:
        """
        Combine standard IIT/GWT assessment with Topo-Spectral results
        """
        # Map Topo-Spectral consciousness states to standard levels
        topo_to_standard_mapping = {
            TopoSpectralConsciousnessState.DEEP_SLEEP: ConsciousnessLevel.UNCONSCIOUS,
            TopoSpectralConsciousnessState.LIGHT_SLEEP: ConsciousnessLevel.MINIMAL,
            TopoSpectralConsciousnessState.AWAKE: ConsciousnessLevel.MODERATE,
            TopoSpectralConsciousnessState.ALERT: ConsciousnessLevel.HIGH,
            TopoSpectralConsciousnessState.PSYCHEDELIC: ConsciousnessLevel.VERY_HIGH
        }
        
        # Combined assessment
        combined_assessment = {
            # Standard IIT/GWT metrics
            "phi": standard["phi"],
            "consciousness_level": standard["consciousness_level"],
            "global_accessibility": standard["global_accessibility"],
            "information_integration": standard["information_integration"],
            "consciousness_score": standard["consciousness_score"],
            
            # Topo-Spectral metrics
            "psi_index": topo_spectral.psi_index,
            "phi_spectral": topo_spectral.phi_spectral,
            "topological_resilience": topo_spectral.topological_resilience,
            "synchronization_factor": topo_spectral.synchronization_factor,
            "topo_spectral_state": topo_spectral.consciousness_state.value,
            "topo_spectral_state_mapped": topo_to_standard_mapping[topo_spectral.consciousness_state],
            
            # Combined metrics
            "combined_consciousness_score": self._calculate_combined_consciousness_score(
                standard["consciousness_score"], topo_spectral.psi_index
            ),
            "framework_agreement": self._assess_framework_agreement(
                standard["consciousness_level"], topo_to_standard_mapping[topo_spectral.consciousness_state]
            ),
            
            # Detailed analysis
            "spectral_cuts_count": len(topo_spectral.spectral_cuts),
            "topology_features_count": len(topo_spectral.topology_features),
            "temporal_variance": topo_spectral.temporal_variance,
            "classification_confidence": topo_spectral.classification_confidence,
            
            # Metadata
            "assessment_timestamp": standard["assessment_timestamp"],
            "topo_spectral_timestamp": topo_spectral.timestamp,
            "network_size": standard["network_size"],
            "active_nodes": standard["active_nodes"]
        }
        
        return combined_assessment
    
    def _calculate_combined_consciousness_score(self, standard_score: float, psi_index: float) -> float:
        """
        Calculate combined consciousness score from both frameworks
        Uses weighted geometric mean to ensure both contribute
        """
        # Weights based on framework reliability and validation
        w_standard = 0.4  # IIT/GWT traditional approach
        w_topo_spectral = 0.6  # Topo-Spectral with higher validation accuracy
        
        # Geometric mean to ensure both frameworks contribute
        if standard_score > 0 and psi_index > 0:
            combined_score = (standard_score ** w_standard) * (psi_index ** w_topo_spectral)
        else:
            # Fallback to weighted arithmetic mean
            combined_score = w_standard * standard_score + w_topo_spectral * psi_index
        
        return np.clip(combined_score, 0.0, 1.0)
    
    def _assess_framework_agreement(self, standard_level: ConsciousnessLevel, 
                                  topo_level: ConsciousnessLevel) -> float:
        """
        Assess agreement between frameworks (0=complete disagreement, 1=perfect agreement)
        """
        level_values = {
            ConsciousnessLevel.UNCONSCIOUS: 0,
            ConsciousnessLevel.MINIMAL: 1,
            ConsciousnessLevel.LOW: 2,
            ConsciousnessLevel.MODERATE: 3,
            ConsciousnessLevel.HIGH: 4,
            ConsciousnessLevel.VERY_HIGH: 5
        }
        
        standard_val = level_values[standard_level]
        topo_val = level_values[topo_level]
        
        # Agreement as inverse of normalized difference
        max_diff = max(level_values.values()) - min(level_values.values())
        agreement = 1.0 - abs(standard_val - topo_val) / max_diff
        
        return agreement
    
    def monitor_consciousness_evolution(self, state_sequence: List[np.ndarray], 
                                     time_steps: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Enhanced consciousness evolution monitoring with Topo-Spectral analysis
        """
        if time_steps is None:
            time_steps = list(range(len(state_sequence)))
        
        # Standard evolution monitoring
        standard_evolution = super().monitor_consciousness_evolution(state_sequence, time_steps)
        
        if not self.enable_topo_spectral:
            standard_evolution["framework"] = "IIT_GWT_only"
            return standard_evolution
        
        # Build connectivity matrices sequence (assuming static connectivity)
        connectivity_matrices = [self.connectivity_matrix] * len(state_sequence)
        
        try:
            # Topo-Spectral evolution monitoring
            topo_evolution = self.topo_spectral_calculator.monitor_consciousness_evolution(
                connectivity_matrices=connectivity_matrices,
                node_states_sequence=state_sequence,
                timestamps=np.array(time_steps)
            )
            
            # Combine evolution analyses
            combined_evolution = {
                # Standard metrics
                "phi_sequence": standard_evolution["phi_sequence"],
                "consciousness_levels": standard_evolution["consciousness_levels"],
                "score_sequence": standard_evolution["score_sequence"],
                "phi_trend": standard_evolution["phi_trend"],
                "score_trend": standard_evolution["score_trend"],
                "max_phi": standard_evolution["max_phi"],
                "mean_phi": standard_evolution["mean_phi"],
                "stability": standard_evolution["stability"],
                
                # Topo-Spectral metrics
                "psi_sequence": topo_evolution["psi_sequence"],
                "topo_spectral_states": [state.value for state in topo_evolution["consciousness_states"]],
                "phi_spectral_sequence": topo_evolution["phi_sequence"],
                "topo_resilience_sequence": topo_evolution["topo_sequence"],
                "sync_sequence": topo_evolution["sync_sequence"],
                "psi_trend": topo_evolution["psi_trend"],
                "mean_psi": topo_evolution["mean_psi"],
                "max_psi": topo_evolution["max_psi"],
                "psi_stability": topo_evolution["psi_stability"],
                "state_transitions": topo_evolution["state_transitions"],
                
                # Framework metadata
                "framework": "IIT_GWT_TopoSpectral",
                "timestamps": time_steps,
                "evolution_length": len(state_sequence)
            }
            
            return combined_evolution
            
        except Exception as e:
            self.logger.error(f"Topo-Spectral evolution monitoring failed: {e}")
            standard_evolution["framework"] = "IIT_GWT_only"
            standard_evolution["topo_spectral_error"] = str(e)
            return standard_evolution

def create_enhanced_consciousness_assessor(network: Dict[str, NetworkNode],
                                         enable_topo_spectral: bool = None) -> EnhancedConsciousnessAssessment:
    """
    Factory function to create enhanced consciousness assessor
    
    Args:
        network: Network of processing nodes
        enable_topo_spectral: Override for Topo-Spectral (None=auto-detect)
    """
    # Auto-detect Topo-Spectral availability if not specified
    if enable_topo_spectral is None:
        from config.execution_mode_manager import is_topo_spectral_enabled
        enable_topo_spectral = is_topo_spectral_enabled()
    
    return EnhancedConsciousnessAssessment(
        network=network,
        enable_topo_spectral=enable_topo_spectral
    )

def demonstrate_enhanced_consciousness_assessment():
    """Demonstrate enhanced consciousness assessment with Topo-Spectral framework"""
    print("Demonstrating Enhanced Consciousness Assessment (IIT/GWT + Topo-Spectral)...")
    
    # Create test network
    network = create_test_network()
    
    # Create enhanced assessor (auto-detect Topo-Spectral availability)
    assessor = create_enhanced_consciousness_assessor(network)
    
    print(f"Topo-Spectral enabled: {assessor.enable_topo_spectral}")
    print(f"Framework: {assessor.__class__.__name__}")
    
    # Test different system states
    states = {
        "random": np.random.random(8),
        "synchronized": np.ones(8) * 0.7,
        "clustered": np.array([0.9, 0.8, 0.9, 0.8, 0.1, 0.2, 0.1, 0.2]),
        "sparse": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    }
    
    print("\nEnhanced Consciousness Assessment Results:")
    print("=" * 60)
    
    for state_name, state in states.items():
        assessment = assessor.assess_consciousness_level(state)
        
        print(f"\n{state_name.upper()} State ({assessment['framework']}):")
        print(f"  Standard Φ (phi): {assessment['phi']:.4f}")
        print(f"  Consciousness Level: {assessment['consciousness_level'].name}")
        
        if assessor.enable_topo_spectral:
            print(f"  Topo-Spectral Ψ: {assessment['psi_index']:.4f}")
            print(f"  Φ̂ Spectral: {assessment['phi_spectral']:.4f}")
            print(f"  T̂ Topological: {assessment['topological_resilience']:.4f}")
            print(f"  Sync Factor: {assessment['synchronization_factor']:.4f}")
            print(f"  Topo-Spectral State: {assessment['topo_spectral_state']}")
            print(f"  Combined Score: {assessment['combined_consciousness_score']:.4f}")
            print(f"  Framework Agreement: {assessment['framework_agreement']:.4f}")
        
        print(f"  Overall Score: {assessment['consciousness_score']:.4f}")
    
    # Demonstrate enhanced evolution monitoring
    if assessor.enable_topo_spectral:
        print("\n" + "=" * 60)
        print("Enhanced Consciousness Evolution Monitoring:")
        
        # Create evolution sequence
        evolution_states = []
        for t in range(15):
            base_activation = 0.5 + 0.3 * np.sin(t * 0.4)
            integration_factor = min(t * 0.07, 1.0)
            noise = 0.1 * np.random.random(8)
            
            state = np.ones(8) * base_activation + noise
            state[:4] *= (1 + integration_factor * 0.6)
            state[4:] *= (1 + integration_factor * 0.4)
            
            evolution_states.append(np.clip(state, 0, 1))
        
        evolution_analysis = assessor.monitor_consciousness_evolution(evolution_states)
        
        print(f"Standard - Max Φ: {evolution_analysis['max_phi']:.4f}")
        print(f"Standard - Mean Φ: {evolution_analysis['mean_phi']:.4f}")
        print(f"Standard - Trend: {evolution_analysis['phi_trend']:.6f}")
        
        if 'psi_sequence' in evolution_analysis:
            print(f"Topo-Spectral - Max Ψ: {evolution_analysis['max_psi']:.4f}")
            print(f"Topo-Spectral - Mean Ψ: {evolution_analysis['mean_psi']:.4f}")
            print(f"Topo-Spectral - Trend: {evolution_analysis['psi_trend']:.6f}")
            print(f"Topo-Spectral - Stability: {evolution_analysis['psi_stability']:.4f}")
            print(f"State Transitions: {evolution_analysis['state_transitions']['transition_count']}")
    
    print("\nEnhanced demonstration complete.")

if __name__ == "__main__":
    # Run both demonstrations
    demonstrate_consciousness_assessment()
    print("\n" + "=" * 80 + "\n")
    demonstrate_enhanced_consciousness_assessment()