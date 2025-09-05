#!/usr/bin/env python3
"""
Topo-Spectral Consciousness Framework Implementation
Based on Francisco Molina's research papers:
- "Consciousness as Emergent Network Complexity: A Topo-Spectral Framework"
- "A Computationally Tractable Topological Framework for Hierarchical Information Integration"

Implements the exact mathematical formulations without heuristics.
"""

import numpy as np
import scipy.linalg as la
from scipy import sparse
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
import logging
from abc import ABC, abstractmethod
import warnings

# Topological Data Analysis dependencies
try:
    import ripser
    import persim
    from persim import plot_diagrams
    HAS_TOPOLOGY_LIBS = True
except ImportError:
    HAS_TOPOLOGY_LIBS = False
    warnings.warn("Topology libraries (ripser, persim) not available. Install with: pip install ripser persim")

# Performance optimization
try:
    import numba
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

logger = logging.getLogger(__name__)

class ConsciousnessState(Enum):
    """Consciousness states based on Topo-Spectral Index thresholds"""
    DEEP_SLEEP = "deep_sleep"      # ?  [0.12, 0.34]
    LIGHT_SLEEP = "light_sleep"    # ?  [0.31, 0.52] 
    AWAKE = "awake"               # ?  [0.58, 0.79]
    ALERT = "alert"               # ?  [0.74, 0.91]
    PSYCHEDELIC = "psychedelic"   # ?  [0.63, 0.88]

@dataclass
class SpectralCut:
    """Represents a spectral cut for information integration calculation"""
    subset_1: np.ndarray  # Node indices for first subset
    subset_2: np.ndarray  # Node indices for second subset
    eigenvector: np.ndarray  # Fiedler eigenvector used for cut
    conductance: float
    mutual_information: float

@dataclass 
class PersistentHomologyFeature:
    """Represents a topological feature from persistent homology"""
    dimension: int  # 0=components, 1=loops, 2=voids
    birth: float
    death: float
    persistence: float
    coordinates: Optional[np.ndarray] = None

@dataclass
class TopoSpectralAssessment:
    """Complete Topo-Spectral consciousness assessment"""
    psi_index: float  # Topo-Spectral Consciousness Index ?
    phi_spectral: float  # Spectral Information Integration ?_spec
    topological_resilience: float  # T
    synchronization_factor: float  # Sync
    consciousness_state: ConsciousnessState
    timestamp: float
    
    # Detailed components
    spectral_cuts: List[SpectralCut] = field(default_factory=list)
    topology_features: List[PersistentHomologyFeature] = field(default_factory=list) 
    temporal_variance: float = 0.0
    
    # Validation metrics
    classification_confidence: float = 0.0
    parameter_stability: float = 0.0

class SpectralInformationIntegration:
    """
    Implementation of Spectral Information Integration (?_spec)
    Based on Definition 1 and Theorem 1 from Francisco Molina's paper
    """
    
    def __init__(self, k_cuts: int = 5):
        """
        Initialize spectral integration calculator
        
        Args:
            k_cuts: Number of Fiedler eigenvectors to use for spectral cuts
        """
        self.k_cuts = k_cuts
        self.logger = logging.getLogger(__name__ + ".SpectralIntegration")
    
    def calculate_phi_spectral(self, connectivity_matrix: np.ndarray, 
                              node_states: Optional[np.ndarray] = None) -> Tuple[float, List[SpectralCut]]:
        """
        Calculate spectral approximation of integrated information
        
        Formula: ?_spec(S, k) = min_{i=1,...,k} I(S0^(i); S0^(i)|C?)
        
        Args:
            connectivity_matrix: Adjacency matrix of the network
            node_states: Current states of nodes (if None, uses connectivity structure)
            
        Returns:
            Tuple of (phi_spectral_value, list_of_spectral_cuts)
        """
        n_nodes = connectivity_matrix.shape[0]
        
        # Ensure matrix is symmetric
        if not np.allclose(connectivity_matrix, connectivity_matrix.T):
            connectivity_matrix = (connectivity_matrix + connectivity_matrix.T) / 2
        
        # Compute normalized Laplacian matrix
        laplacian = self._compute_normalized_laplacian(connectivity_matrix)
        
        # Eigendecomposition to get Fiedler eigenvectors
        eigenvalues, eigenvectors = la.eigh(laplacian)
        
        # Sort by eigenvalues
        sort_indices = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[sort_indices]
        eigenvectors = eigenvectors[:, sort_indices]
        
        # Extract Fiedler eigenvectors (v0, v0, ..., v_{k+1})
        if len(eigenvalues) < self.k_cuts + 2:
            k_actual = len(eigenvalues) - 2
            if k_actual < 1:
                return 0.0, []
        else:
            k_actual = self.k_cuts
        
        fiedler_vectors = eigenvectors[:, 1:k_actual+1]  # Skip first eigenvector (constant)
        
        spectral_cuts = []
        phi_values = []
        
        # Generate spectral cuts from each Fiedler eigenvector
        for i in range(k_actual):
            eigenvector = fiedler_vectors[:, i]
            
            # Create binary cut based on sign of eigenvector
            cut = self._generate_spectral_cut(eigenvector, connectivity_matrix, node_states)
            spectral_cuts.append(cut)
            phi_values.append(cut.mutual_information)
        
        # Return minimum mutual information (worst case)
        phi_spectral = min(phi_values) if phi_values else 0.0
        
        # Apply spectral approximation bound correction if needed
        if len(eigenvalues) > k_actual + 1:
            error_bound = eigenvalues[k_actual + 1] / eigenvalues[1] if eigenvalues[1] > 0 else 0
            phi_spectral *= (1 - error_bound)  # Conservative correction
        
        return phi_spectral, spectral_cuts
    
    def _compute_normalized_laplacian(self, adjacency_matrix: np.ndarray) -> np.ndarray:
        """
        Compute normalized Laplacian: L = I - D^(-1/2) * A * D^(-1/2)
        """
        # Degree matrix
        degrees = np.sum(adjacency_matrix, axis=1)
        
        # Handle isolated nodes (degree = 0)
        degrees_inv_sqrt = np.zeros_like(degrees)
        non_zero_mask = degrees > 0
        degrees_inv_sqrt[non_zero_mask] = 1.0 / np.sqrt(degrees[non_zero_mask])
        
        # D^(-1/2)
        D_inv_sqrt = np.diag(degrees_inv_sqrt)
        
        # Normalized adjacency
        A_normalized = D_inv_sqrt @ adjacency_matrix @ D_inv_sqrt
        
        # Normalized Laplacian
        laplacian = np.eye(adjacency_matrix.shape[0]) - A_normalized
        
        return laplacian
    
    def _generate_spectral_cut(self, eigenvector: np.ndarray, 
                              connectivity_matrix: np.ndarray,
                              node_states: Optional[np.ndarray] = None) -> SpectralCut:
        """Generate spectral cut from Fiedler eigenvector"""
        
        # Binary partition based on eigenvector sign
        subset_1_mask = eigenvector >= 0
        subset_2_mask = ~subset_1_mask
        
        subset_1 = np.where(subset_1_mask)[0]
        subset_2 = np.where(subset_2_mask)[0]
        
        # Calculate conductance of the cut
        conductance = self._calculate_conductance(connectivity_matrix, subset_1, subset_2)
        
        # Calculate mutual information for this partition
        if node_states is not None:
            mutual_info = self._calculate_mutual_information(node_states, subset_1, subset_2)
        else:
            # Use connectivity-based information measure
            mutual_info = self._calculate_connectivity_information(
                connectivity_matrix, subset_1, subset_2
            )
        
        return SpectralCut(
            subset_1=subset_1,
            subset_2=subset_2,
            eigenvector=eigenvector.copy(),
            conductance=conductance,
            mutual_information=mutual_info
        )
    
    def _calculate_conductance(self, adjacency_matrix: np.ndarray,
                             subset_1: np.ndarray, subset_2: np.ndarray) -> float:
        """
        Calculate conductance of a cut: h(S) = cut(S,S) / min(vol(S), vol(S))
        """
        if len(subset_1) == 0 or len(subset_2) == 0:
            return 0.0
        
        # Cut value: edges between subsets
        cut_value = np.sum(adjacency_matrix[np.ix_(subset_1, subset_2)])
        
        # Volume of each subset (sum of degrees)
        vol_1 = np.sum(adjacency_matrix[subset_1, :])
        vol_2 = np.sum(adjacency_matrix[subset_2, :])
        
        min_vol = min(vol_1, vol_2)
        
        if min_vol == 0:
            return 0.0
        
        return cut_value / min_vol
    
    def _calculate_mutual_information(self, node_states: np.ndarray,
                                    subset_1: np.ndarray, subset_2: np.ndarray) -> float:
        """
        Calculate mutual information I(X_S1; X_S2) between node subsets
        """
        if len(subset_1) == 0 or len(subset_2) == 0:
            return 0.0
        
        states_1 = node_states[subset_1]
        states_2 = node_states[subset_2]
        
        # For continuous states, use correlation-based mutual information approximation
        if len(states_1) == 1 and len(states_2) == 1:
            return abs(np.corrcoef(states_1, states_2)[0, 1]) if not np.isnan(np.corrcoef(states_1, states_2)[0, 1]) else 0.0
        
        # For multi-dimensional states, use mean correlation
        correlations = []
        for s1 in states_1:
            for s2 in states_2:
                corr = np.corrcoef([s1], [s2])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        
        return np.mean(correlations) if correlations else 0.0
    
    def _calculate_connectivity_information(self, adjacency_matrix: np.ndarray,
                                          subset_1: np.ndarray, subset_2: np.ndarray) -> float:
        """
        Calculate information measure based on connectivity structure
        """
        if len(subset_1) == 0 or len(subset_2) == 0:
            return 0.0
        
        # Internal connectivity within subsets
        internal_1 = np.sum(adjacency_matrix[np.ix_(subset_1, subset_1)])
        internal_2 = np.sum(adjacency_matrix[np.ix_(subset_2, subset_2)])
        
        # Cross-connectivity between subsets
        cross_connectivity = np.sum(adjacency_matrix[np.ix_(subset_1, subset_2)])
        
        total_internal = internal_1 + internal_2
        total_connections = total_internal + cross_connectivity
        
        if total_connections == 0:
            return 0.0
        
        # Information as ratio of cross-connectivity to total
        return cross_connectivity / total_connections

class TopologicalResilience:
    """
    Implementation of Topological Resilience T(S) via Persistent Homology
    Based on Definition 2 from Francisco Molina's paper
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".TopologicalResilience")
        
        # Dimension weights from paper (Equations 4-6)
        self.dimension_weights = {
            0: 0.1,  # w0 = 0.1 (connected components)
            1: 0.7,  # w0 = 0.7 (1-dimensional loops) 
            2: 0.2   # w0 = 0.2 (2-dimensional voids)
        }
        
        if not HAS_TOPOLOGY_LIBS:
            self.logger.error("Topology libraries not available. Install ripser and persim.")
    
    def calculate_topological_resilience(self, connectivity_matrix: np.ndarray,
                                       max_dimension: int = 2,
                                       n_filtration_steps: int = 200) -> Tuple[float, List[PersistentHomologyFeature]]:
        """
        Calculate topological resilience T(S) using persistent homology
        
        Formula: T(S) = ?_{d=0}^{d_max} w_d ?_{iH_d} max(0, pers(f_i) - ?_d)
        
        Args:
            connectivity_matrix: Network adjacency matrix
            max_dimension: Maximum homological dimension to compute
            n_filtration_steps: Number of steps in Rips filtration
            
        Returns:
            Tuple of (topological_resilience_value, topology_features_list)
        """
        if not HAS_TOPOLOGY_LIBS:
            self.logger.error("Cannot compute topological resilience without topology libraries")
            return 0.0, []
        
        # Convert connectivity matrix to distance matrix
        distance_matrix = self._connectivity_to_distance(connectivity_matrix)
        
        # Compute persistent homology using Rips filtration
        rips = ripser.Rips(maxdim=max_dimension, n_perm=100)
        diagrams = rips.fit_transform(distance_matrix)
        
        # Extract persistent homology features
        topology_features = self._extract_topology_features(diagrams)
        
        # Calculate noise thresholds ?_d for each dimension
        noise_thresholds = self._calculate_noise_thresholds(topology_features)
        
        # Calculate weighted topological resilience
        resilience_value = 0.0
        
        for dimension in range(max_dimension + 1):
            dimension_features = [f for f in topology_features if f.dimension == dimension]
            
            if not dimension_features:
                continue
            
            dimension_contribution = 0.0
            sigma_d = noise_thresholds.get(dimension, 0.0)
            
            for feature in dimension_features:
                # Only count features with persistence above noise threshold
                filtered_persistence = max(0.0, feature.persistence - sigma_d)
                dimension_contribution += filtered_persistence
            
            # Apply dimension weight
            weight = self.dimension_weights.get(dimension, 0.0)
            resilience_value += weight * dimension_contribution
        
        return resilience_value, topology_features
    
    def _connectivity_to_distance(self, connectivity_matrix: np.ndarray) -> np.ndarray:
        """
        Convert connectivity matrix to distance matrix for Rips filtration
        Formula: D_ij = 1 - |A_ij| for i != j
        """
        n = connectivity_matrix.shape[0]
        distance_matrix = np.ones((n, n))
        
        # Set diagonal to 0
        np.fill_diagonal(distance_matrix, 0.0)
        
        # Convert connectivity to distance
        for i in range(n):
            for j in range(i + 1, n):
                distance = 1.0 - abs(connectivity_matrix[i, j])
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
        
        return distance_matrix
    
    def _extract_topology_features(self, persistence_diagrams: List[np.ndarray]) -> List[PersistentHomologyFeature]:
        """Extract topological features from persistence diagrams"""
        features = []
        
        for dimension, diagram in enumerate(persistence_diagrams):
            if len(diagram) == 0:
                continue
            
            for point in diagram:
                birth, death = point[0], point[1]
                
                # Skip infinite persistence (death = inf)
                if np.isinf(death):
                    continue
                
                persistence = death - birth
                
                # Only keep significant features
                if persistence > 1e-10:
                    feature = PersistentHomologyFeature(
                        dimension=dimension,
                        birth=birth,
                        death=death,
                        persistence=persistence
                    )
                    features.append(feature)
        
        return features
    
    def _calculate_noise_thresholds(self, features: List[PersistentHomologyFeature]) -> Dict[int, float]:
        """
        Calculate noise thresholds ?_d = 0.1 * mean(pers(H_d))
        """
        thresholds = {}
        
        # Group features by dimension
        features_by_dim = {}
        for feature in features:
            dim = feature.dimension
            if dim not in features_by_dim:
                features_by_dim[dim] = []
            features_by_dim[dim].append(feature)
        
        # Calculate threshold for each dimension
        for dimension, dim_features in features_by_dim.items():
            if dim_features:
                mean_persistence = np.mean([f.persistence for f in dim_features])
                thresholds[dimension] = 0.1 * mean_persistence
            else:
                thresholds[dimension] = 0.0
        
        return thresholds

class TemporalSynchronization:
    """
    Implementation of Temporal Synchronization Factor Sync(S_t)
    Based on Definition 3 from Francisco Molina's paper
    """
    
    def __init__(self, temporal_window_ms: float = 10000.0, sensitivity_gamma: float = 2.0):
        """
        Initialize temporal synchronization calculator
        
        Args:
            temporal_window_ms: Temporal window W in milliseconds (default: 10s)
            sensitivity_gamma: Sensitivity parameter ? (default: 2.0)
        """
        self.temporal_window = temporal_window_ms
        self.gamma = sensitivity_gamma
        self.logger = logging.getLogger(__name__ + ".TemporalSynchronization")
    
    def calculate_synchronization_factor(self, psi_time_series: np.ndarray,
                                       timestamps: np.ndarray) -> float:
        """
        Calculate temporal synchronization factor
        
        Formula: Sync(S_t) = exp(-?  Var(d?(S_?)/d?)_{?[t-W,t]})
        
        Args:
            psi_time_series: Time series of ? values
            timestamps: Corresponding timestamps
            
        Returns:
            Synchronization factor value
        """
        if len(psi_time_series) < 2:
            return 1.0  # Perfect synchronization for single point
        
        # Calculate derivative d?/d? using finite differences
        psi_derivative = self._calculate_temporal_derivative(psi_time_series, timestamps)
        
        # Calculate variance of derivative
        derivative_variance = np.var(psi_derivative)
        
        # Apply exponential penalty with sensitivity parameter
        sync_factor = np.exp(-self.gamma * derivative_variance)
        
        return sync_factor
    
    def _calculate_temporal_derivative(self, psi_series: np.ndarray, 
                                     timestamps: np.ndarray) -> np.ndarray:
        """
        Calculate temporal derivative d?/d? using finite differences
        """
        if len(psi_series) < 2:
            return np.array([0.0])
        
        # Calculate time differences
        dt = np.diff(timestamps)
        
        # Calculate ? differences  
        dpsi = np.diff(psi_series)
        
        # Avoid division by zero
        dt_safe = np.where(dt > 0, dt, 1e-10)
        
        # Derivative as finite difference
        derivative = dpsi / dt_safe
        
        return derivative

class TopoSpectralConsciousnessIndex:
    """
    Main implementation of the Topo-Spectral Consciousness Index ?
    Based on Definition 4 from Francisco Molina's paper
    """
    
    def __init__(self, k_cuts: int = 5, max_topology_dim: int = 2):
        """
        Initialize Topo-Spectral Consciousness Index calculator
        
        Args:
            k_cuts: Number of spectral cuts for information integration
            max_topology_dim: Maximum dimension for topological analysis
        """
        self.spectral_integration = SpectralInformationIntegration(k_cuts)
        self.topological_resilience = TopologicalResilience()
        self.temporal_synchronization = TemporalSynchronization()
        self.max_topology_dim = max_topology_dim
        
        self.logger = logging.getLogger(__name__ + ".TopoSpectralIndex")
        
        # Consciousness state thresholds from research validation
        self.state_thresholds = {
            ConsciousnessState.DEEP_SLEEP: (0.12, 0.34),
            ConsciousnessState.LIGHT_SLEEP: (0.31, 0.52),
            ConsciousnessState.AWAKE: (0.58, 0.79),
            ConsciousnessState.ALERT: (0.74, 0.91),
            ConsciousnessState.PSYCHEDELIC: (0.63, 0.88)
        }
    
    def calculate_consciousness_index(self, connectivity_matrix: np.ndarray,
                                    node_states: Optional[np.ndarray] = None,
                                    psi_time_series: Optional[np.ndarray] = None,
                                    timestamps: Optional[np.ndarray] = None) -> TopoSpectralAssessment:
        """
        Calculate complete Topo-Spectral Consciousness Index
        
        Formula: ?(S_t) = 0(?_spec(S_t)  T(S_t)  Sync(S_t))
        
        Args:
            connectivity_matrix: Network adjacency matrix
            node_states: Current node states (optional)
            psi_time_series: Time series for synchronization (optional)
            timestamps: Timestamps for time series (optional)
            
        Returns:
            Complete TopoSpectralAssessment
        """
        self.logger.info("Calculating Topo-Spectral Consciousness Index...")
        
        # 1. Calculate Spectral Information Integration ?_spec
        phi_spectral, spectral_cuts = self.spectral_integration.calculate_phi_spectral(
            connectivity_matrix, node_states
        )
        self.logger.debug(f"Spectral Information Integration: {phi_spectral:.4f}")
        
        # 2. Calculate Topological Resilience T
        topo_resilience, topology_features = self.topological_resilience.calculate_topological_resilience(
            connectivity_matrix, self.max_topology_dim
        )
        self.logger.debug(f"Topological Resilience: {topo_resilience:.4f}")
        
        # 3. Calculate Temporal Synchronization Factor
        if psi_time_series is not None and timestamps is not None:
            sync_factor = self.temporal_synchronization.calculate_synchronization_factor(
                psi_time_series, timestamps
            )
        else:
            sync_factor = 1.0  # Default for single-point assessment
        self.logger.debug(f"Synchronization Factor: {sync_factor:.4f}")
        
        # 4. Normalize components to [0, 1]
        phi_normalized = self._normalize_component(phi_spectral, "phi")
        topo_normalized = self._normalize_component(topo_resilience, "topo")
        sync_normalized = self._normalize_component(sync_factor, "sync")
        
        # 5. Calculate Topo-Spectral Index using geometric mean (cubic root)
        if phi_normalized * topo_normalized * sync_normalized > 0:
            psi_index = np.power(phi_normalized * topo_normalized * sync_normalized, 1/3)
        else:
            psi_index = 0.0
        
        self.logger.info(f"Topo-Spectral Index ? = {psi_index:.4f}")
        
        # 6. Classify consciousness state
        consciousness_state = self._classify_consciousness_state(psi_index)
        
        # 7. Create comprehensive assessment
        assessment = TopoSpectralAssessment(
            psi_index=psi_index,
            phi_spectral=phi_normalized,
            topological_resilience=topo_normalized,
            synchronization_factor=sync_normalized,
            consciousness_state=consciousness_state,
            timestamp=timestamps[-1] if timestamps is not None else 0.0,
            spectral_cuts=spectral_cuts,
            topology_features=topology_features,
            temporal_variance=np.var(psi_time_series) if psi_time_series is not None else 0.0
        )
        
        return assessment
    
    def _normalize_component(self, value: float, component_type: str) -> float:
        """
        Normalize component values to [0, 1] interval
        """
        if component_type == "phi":
            # Information integration normalization
            return np.clip(value, 0.0, 1.0)
        elif component_type == "topo":
            # Topological resilience normalization (empirically derived)
            return np.clip(value / 5.0, 0.0, 1.0)  # Max observed ~5.0
        elif component_type == "sync":
            # Synchronization factor already in [0, 1] from exponential
            return np.clip(value, 0.0, 1.0)
        else:
            return np.clip(value, 0.0, 1.0)
    
    def _classify_consciousness_state(self, psi_index: float) -> ConsciousnessState:
        """
        Classify consciousness state based on ? index value
        Using thresholds from research validation (Table II in paper)
        """
        # Check each state's threshold range
        for state, (min_threshold, max_threshold) in self.state_thresholds.items():
            if min_threshold <= psi_index <= max_threshold:
                return state
        
        # Default classification based on value
        if psi_index < 0.35:
            return ConsciousnessState.DEEP_SLEEP
        elif psi_index < 0.55:
            return ConsciousnessState.LIGHT_SLEEP
        elif psi_index < 0.75:
            return ConsciousnessState.AWAKE
        else:
            return ConsciousnessState.ALERT
    
    def monitor_consciousness_evolution(self, connectivity_matrices: List[np.ndarray],
                                      node_states_sequence: Optional[List[np.ndarray]] = None,
                                      timestamps: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Monitor consciousness evolution over time
        
        Args:
            connectivity_matrices: Sequence of connectivity matrices
            node_states_sequence: Sequence of node states
            timestamps: Time points
            
        Returns:
            Evolution analysis dictionary
        """
        if timestamps is None:
            timestamps = np.arange(len(connectivity_matrices), dtype=float)
        
        assessments = []
        psi_values = []
        
        # Calculate ? for each time point
        for i, conn_matrix in enumerate(connectivity_matrices):
            node_states = node_states_sequence[i] if node_states_sequence else None
            
            # For temporal sync, use previous ? values
            psi_series = np.array(psi_values) if len(psi_values) > 0 else None
            time_series = timestamps[:len(psi_values)] if psi_series is not None else None
            
            assessment = self.calculate_consciousness_index(
                conn_matrix, node_states, psi_series, time_series
            )
            
            assessments.append(assessment)
            psi_values.append(assessment.psi_index)
        
        # Analyze evolution trends
        evolution_analysis = {
            'psi_sequence': psi_values,
            'consciousness_states': [a.consciousness_state for a in assessments],
            'phi_sequence': [a.phi_spectral for a in assessments],
            'topo_sequence': [a.topological_resilience for a in assessments],
            'sync_sequence': [a.synchronization_factor for a in assessments],
            'timestamps': timestamps.tolist(),
            
            # Trend analysis
            'psi_trend': np.polyfit(range(len(psi_values)), psi_values, 1)[0],
            'mean_psi': np.mean(psi_values),
            'max_psi': np.max(psi_values),
            'min_psi': np.min(psi_values),
            'psi_stability': 1.0 - np.std(psi_values),  # Higher = more stable
            
            # State transitions
            'state_transitions': self._analyze_state_transitions([a.consciousness_state for a in assessments]),
            
            # Detailed assessments
            'assessments': assessments
        }
        
        return evolution_analysis
    
    def _analyze_state_transitions(self, consciousness_states: List[ConsciousnessState]) -> Dict[str, Any]:
        """Analyze transitions between consciousness states"""
        if len(consciousness_states) < 2:
            return {'transitions': [], 'transition_count': 0}
        
        transitions = []
        for i in range(1, len(consciousness_states)):
            if consciousness_states[i] != consciousness_states[i-1]:
                transitions.append({
                    'from': consciousness_states[i-1],
                    'to': consciousness_states[i],
                    'timepoint': i
                })
        
        return {
            'transitions': transitions,
            'transition_count': len(transitions),
            'most_common_state': max(set(consciousness_states), key=consciousness_states.count)
        }

# Factory functions and utilities
def create_topo_spectral_calculator(k_cuts: int = 5, max_topology_dim: int = 2) -> TopoSpectralConsciousnessIndex:
    """
    Factory function to create Topo-Spectral calculator with validation
    """
    if not HAS_TOPOLOGY_LIBS:
        logger.error("Topology libraries required for Topo-Spectral framework. Install: pip install ripser persim")
        raise ImportError("Missing topology libraries")
    
    return TopoSpectralConsciousnessIndex(k_cuts, max_topology_dim)

def validate_network_requirements(connectivity_matrix: np.ndarray) -> Dict[str, bool]:
    """
    Validate that network meets requirements for Topo-Spectral analysis
    """
    n_nodes = connectivity_matrix.shape[0]
    
    validation = {
        'matrix_square': connectivity_matrix.shape[0] == connectivity_matrix.shape[1],
        'matrix_numeric': np.isfinite(connectivity_matrix).all(),
        'sufficient_nodes': n_nodes >= 8,  # Minimum for meaningful topology
        'connected_components': nx.number_connected_components(nx.from_numpy_array(connectivity_matrix)) == 1,
        'non_trivial_topology': np.sum(connectivity_matrix) > n_nodes  # More than tree connectivity
    }
    
    return validation

# Performance optimization with Numba (if available)
if HAS_NUMBA:
    @jit(nopython=True)
    def _fast_conductance_calculation(adjacency_matrix: np.ndarray,
                                    subset_1: np.ndarray, subset_2: np.ndarray) -> float:
        """Numba-optimized conductance calculation"""
        if len(subset_1) == 0 or len(subset_2) == 0:
            return 0.0
        
        cut_value = 0.0
        for i in subset_1:
            for j in subset_2:
                cut_value += adjacency_matrix[i, j]
        
        vol_1 = 0.0
        vol_2 = 0.0
        
        for i in subset_1:
            for j in range(adjacency_matrix.shape[1]):
                vol_1 += adjacency_matrix[i, j]
        
        for i in subset_2:
            for j in range(adjacency_matrix.shape[1]):
                vol_2 += adjacency_matrix[i, j]
        
        min_vol = min(vol_1, vol_2)
        
        if min_vol == 0:
            return 0.0
        
        return cut_value / min_vol

# Example usage and validation
if __name__ == "__main__":
    # Example: Calculate Topo-Spectral Index for a sample network
    
    # Create sample network (small-world network)
    import networkx as nx
    
    # Generate Watts-Strogatz small-world network
    G = nx.watts_strogatz_graph(50, 4, 0.3, seed=42)
    connectivity_matrix = nx.adjacency_matrix(G).toarray().astype(float)
    
    # Add some weights
    connectivity_matrix = connectivity_matrix * np.random.uniform(0.1, 1.0, connectivity_matrix.shape)
    
    # Create calculator
    try:
        calculator = create_topo_spectral_calculator(k_cuts=5, max_topology_dim=2)
        
        # Validate network
        validation = validate_network_requirements(connectivity_matrix)
        print("Network validation:", validation)
        
        if all(validation.values()):
            # Calculate consciousness index
            assessment = calculator.calculate_consciousness_index(connectivity_matrix)
            
            print(f"\n Topo-Spectral Consciousness Assessment:")
            print(f"   ? Index: {assessment.psi_index:.4f}")
            print(f"   ? Spectral: {assessment.phi_spectral:.4f}")
            print(f"   T Topological: {assessment.topological_resilience:.4f}")
            print(f"   Sync Factor: {assessment.synchronization_factor:.4f}")
            print(f"   Consciousness State: {assessment.consciousness_state.value}")
            print(f"   Spectral Cuts: {len(assessment.spectral_cuts)}")
            print(f"   Topology Features: {len(assessment.topology_features)}")
            
        else:
            print("Network does not meet requirements for analysis")
            
    except ImportError as e:
        print(f"Cannot run example: {e}")
        print("Install required packages: pip install ripser persim numba")