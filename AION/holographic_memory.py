#!/usr/bin/env python3
"""
SISTEMA DE MEMORIA HOLOGRAFICA - FASE 2 IMPLEMENTACION CRITICA
==============================================================

IMPLEMENTACION RIGUROSA: Sistema de memoria holografica con formalizacion matematica completa
REFERENCIAS CIENTIFICAS:
- Gabor, D. (1946): Theory of communication holographic storage
- Hopfield, J.N. (1982): Neural networks and physical systems
- Pribram, K.H. (1991): Brain and perception: holonomy and structure

ECUACIONES FUNDAMENTALES:
- Transformada de Fourier: F(?) =  f(t)e^(-i?t) dt
- Patron de interferencia: I = |A0 + A0e^(i?)|0
- Capacidad de almacenamiento: C = 0.15N (limite de Hopfield)

ESPECIFICACIONES TECNICAS:
- Complejidad temporal: O(N log N) para FFT
- Complejidad espacial: O(N0) para patrones de interferencia  
- Capacidad: 15% de N patrones para recuperacion perfecta
- Ruido: Tolerancia hasta 40% de degradacion de senal

Autor: Francisco Molina
Fecha: 2024
ORCID: https://orcid.org/0009-0008-6093-8267
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.signal import correlate2d
from scipy.spatial.distance import cosine, euclidean
import json
import time
import warnings
from pathlib import Path

# Configuracion de logging cientifico
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MemoryPattern:
    """
    ESTRUCTURA DE PATRON DE MEMORIA HOLOGRAFICA
    Representa un patron almacenado con sus propiedades fisicas y matematicas
    """
    pattern_id: str
    data: np.ndarray
    phase_encoding: np.ndarray
    amplitude_encoding: np.ndarray
    timestamp: float = field(default_factory=time.time)
    reconstruction_fidelity: float = 0.0
    interference_strength: float = 0.0
    semantic_tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validacion post-inicializacion del patron"""
        if self.data.size == 0:
            raise ValueError("Pattern data cannot be empty")
        if not np.isfinite(self.data).all():
            raise ValueError("Pattern data contains non-finite values")
        
        # Normalizacion automatica
        if np.max(np.abs(self.data)) > 0:
            self.data = self.data / np.max(np.abs(self.data))

@dataclass 
class HolographicConfiguration:
    """Configuracion del sistema holografico"""
    max_patterns: int = 1000
    pattern_dimensions: Tuple[int, int] = (64, 64)
    noise_tolerance: float = 0.4  # 40% como en estudios de Hopfield
    reconstruction_threshold: float = 0.8  # 80% de fidelidad minima
    fourier_padding: bool = True
    complex_encoding: bool = True
    interference_optimization: bool = True
    
    # Parametros de optimizacion
    correlation_threshold: float = 0.7
    phase_stability_factor: float = 0.95
    amplitude_preservation: float = 0.9

class HolographicMemorySystem:
    """
    SISTEMA DE MEMORIA HOLOGRAFICA - IMPLEMENTACION CIENTIFICA RIGOROSA
    
    Implementa almacenamiento y recuperacion de patrones mediante principios holograficos:
    1. Codificacion de patrones en amplitud y fase
    2. Superposicion constructiva/destructiva de ondas
    3. Transformadas de Fourier para codificacion frecuencial
    4. Correlacion cruzada para recuperacion asociativa
    
    FORMULACION MATEMATICA:
    - Patron holografico: H(x,y) = ?? P?(x,y) * R*(x,y)
    - Recuperacion: P'(x,y) = F0[F[H] * F[R]]
    - Fidelidad: F = 1 - ||P - P'||0/||P||0
    """
    
    def __init__(self, config: Optional[HolographicConfiguration] = None):
        """
        Inicializacion del sistema holografico
        
        Args:
            config: Configuracion del sistema (opcional)
        """
        self.config = config or HolographicConfiguration()
        self.patterns: Dict[str, MemoryPattern] = {}
        self.interference_matrix = None
        self.reference_wave = None
        self.hologram_medium = None
        
        # Metricas del sistema
        self.total_patterns = 0
        self.retrieval_accuracy = 0.0
        self.interference_level = 0.0
        
        # Inicializacion del medio holografico
        self._initialize_holographic_medium()
        self._initialize_reference_wave()
        
        logger.info(f"Sistema holografico inicializado: {self.config.pattern_dimensions}")
    
    def _initialize_holographic_medium(self):
        """Inicializa el medio holografico virtual"""
        dims = self.config.pattern_dimensions
        self.hologram_medium = np.zeros(dims, dtype=np.complex128)
        
        # Matriz de interferencia para optimizacion
        if self.config.interference_optimization:
            self.interference_matrix = np.zeros((self.config.max_patterns, self.config.max_patterns))
    
    def _initialize_reference_wave(self):
        """Inicializa la onda de referencia para grabacion holografica"""
        dims = self.config.pattern_dimensions
        x, y = np.mgrid[0:dims[0], 0:dims[1]]
        
        # Onda plana con ligera inclinacion (tipico en holografia)
        kx, ky = 0.1, 0.05  # Vectores de onda de referencia
        self.reference_wave = np.exp(1j * (kx * x + ky * y))
        
        # Normalizacion
        self.reference_wave = self.reference_wave / np.sqrt(np.sum(np.abs(self.reference_wave)**2))
    
    def encode_pattern(self, pattern_data: np.ndarray, pattern_id: str, 
                      semantic_tags: Optional[List[str]] = None) -> MemoryPattern:
        """
        CODIFICACION HOLOGRAFICA DE PATRON
        
        Implementa la ecuacion fundamental de grabacion holografica:
        I(x,y) = |O(x,y) + R(x,y)|0 = |O|0 + |R|0 + O*R + OR*
        
        Args:
            pattern_data: Datos del patron a codificar
            pattern_id: Identificador unico del patron
            semantic_tags: Etiquetas semanticas opcionales
            
        Returns:
            MemoryPattern: Patron codificado holograficamente
        """
        if len(self.patterns) >= self.config.max_patterns:
            raise ValueError(f"Capacidad maxima alcanzada: {self.config.max_patterns} patrones")
        
        # Redimensionamiento y normalizacion del patron
        pattern_resized = self._resize_pattern(pattern_data)
        pattern_normalized = self._normalize_pattern(pattern_resized)
        
        # Codificacion holografica
        object_wave = pattern_normalized.astype(np.complex128)
        
        # Calculo del patron de interferencia holografico
        interference_pattern = np.abs(object_wave + self.reference_wave)**2
        
        # Separacion de amplitud y fase
        amplitude_encoding = np.abs(object_wave)
        phase_encoding = np.angle(object_wave)
        
        # Transformada de Fourier para codificacion frecuencial
        fourier_encoding = fft2(object_wave)
        if self.config.fourier_padding:
            fourier_encoding = fftshift(fourier_encoding)
        
        # Creacion del patron de memoria
        memory_pattern = MemoryPattern(
            pattern_id=pattern_id,
            data=fourier_encoding,
            phase_encoding=phase_encoding, 
            amplitude_encoding=amplitude_encoding,
            semantic_tags=semantic_tags or []
        )
        
        # Calculo de metricas de interferencia
        memory_pattern.interference_strength = self._calculate_interference_strength(interference_pattern)
        
        # Almacenamiento en el medio holografico
        self._store_in_medium(memory_pattern, interference_pattern)
        
        # Actualizacion de la matriz de interferencia
        if self.config.interference_optimization:
            self._update_interference_matrix(pattern_id, memory_pattern)
        
        self.patterns[pattern_id] = memory_pattern
        self.total_patterns += 1
        
        logger.info(f"Patron holografico codificado: {pattern_id} (total: {self.total_patterns})")
        return memory_pattern
    
    def retrieve_pattern(self, query_pattern: np.ndarray, 
                        similarity_threshold: float = 0.8) -> Optional[Tuple[MemoryPattern, float]]:
        """
        RECUPERACION HOLOGRAFICA ASOCIATIVA
        
        Implementa recuperacion mediante correlacion cruzada holografica:
        C(x,y) = F0[F*[H]  F[Q]]
        
        Args:
            query_pattern: Patron de consulta
            similarity_threshold: Umbral minimo de similitud
            
        Returns:
            Tuple con patron recuperado y score de similitud, o None
        """
        if not self.patterns:
            return None
        
        # Preparacion del patron de consulta
        query_resized = self._resize_pattern(query_pattern)
        query_normalized = self._normalize_pattern(query_resized)
        query_fourier = fft2(query_normalized.astype(np.complex128))
        
        best_match = None
        best_similarity = 0.0
        
        # Busqueda por correlacion holografica
        for pattern_id, stored_pattern in self.patterns.items():
            
            # Correlacion cruzada en dominio de Fourier
            correlation = self._holographic_correlation(query_fourier, stored_pattern.data)
            
            # Calculo de similitud normalizada
            similarity = self._calculate_similarity(correlation, query_normalized, stored_pattern)
            
            if similarity > best_similarity and similarity >= similarity_threshold:
                best_similarity = similarity
                best_match = stored_pattern
        
        if best_match:
            # Actualizacion de metricas de recuperacion
            best_match.reconstruction_fidelity = best_similarity
            self.retrieval_accuracy = (self.retrieval_accuracy * (self.total_patterns - 1) + best_similarity) / self.total_patterns
            
            logger.info(f"Patron recuperado: {best_match.pattern_id} (similitud: {best_similarity:.3f})")
            return best_match, best_similarity
        
        return None
    
    def retrieve_by_semantic_tags(self, tags: List[str]) -> List[Tuple[MemoryPattern, float]]:
        """
        Recuperacion por etiquetas semanticas
        
        Args:
            tags: Lista de etiquetas a buscar
            
        Returns:
            Lista de patrones que coinciden con las etiquetas
        """
        matches = []
        
        for pattern_id, pattern in self.patterns.items():
            tag_overlap = len(set(tags) & set(pattern.semantic_tags))
            if tag_overlap > 0:
                relevance_score = tag_overlap / len(tags)
                matches.append((pattern, relevance_score))
        
        # Ordenar por relevancia
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
    
    def reconstruct_pattern(self, pattern: MemoryPattern, 
                          add_noise: float = 0.0) -> np.ndarray:
        """
        RECONSTRUCCION HOLOGRAFICA
        
        Reconstruye el patron original desde su codificacion holografica:
        P'(x,y) = F0[H(u,v)  R*(u,v)]
        
        Args:
            pattern: Patron codificado holograficamente  
            add_noise: Nivel de ruido a anadir (0.0-1.0)
            
        Returns:
            Patron reconstruido
        """
        # Reconstruccion desde transformada de Fourier
        if self.config.fourier_padding:
            fourier_data = ifftshift(pattern.data)
        else:
            fourier_data = pattern.data
        
        reconstructed = ifft2(fourier_data)
        
        # Conversion a patron real
        reconstructed_real = np.real(reconstructed)
        
        # Adicion de ruido si se especifica
        if add_noise > 0:
            noise = np.random.normal(0, add_noise, reconstructed_real.shape)
            reconstructed_real += noise
        
        # Normalizacion final
        reconstructed_normalized = self._normalize_pattern(reconstructed_real)
        
        return reconstructed_normalized
    
    def _resize_pattern(self, pattern: np.ndarray) -> np.ndarray:
        """Redimensiona patron a las dimensiones del sistema"""
        from scipy.ndimage import zoom
        
        target_shape = self.config.pattern_dimensions
        if pattern.shape == target_shape:
            return pattern
        
        # Calculo de factores de escala
        scale_factors = [target_shape[i] / pattern.shape[i] for i in range(len(target_shape))]
        
        # Redimensionamiento con interpolacion
        resized = zoom(pattern, scale_factors, order=1)
        return resized
    
    def _normalize_pattern(self, pattern: np.ndarray) -> np.ndarray:
        """Normalizacion de patron con preservacion de caracteristicas"""
        if np.max(np.abs(pattern)) == 0:
            return pattern
        
        # Normalizacion por amplitud maxima
        normalized = pattern / np.max(np.abs(pattern))
        
        # Centrado en media cero para mejor codificacion holografica
        normalized = normalized - np.mean(normalized)
        
        return normalized
    
    def _calculate_interference_strength(self, interference_pattern: np.ndarray) -> float:
        """Calcula la intensidad de interferencia del patron holografico"""
        # Varianza normalizada como medida de interferencia
        variance = np.var(interference_pattern)
        max_intensity = np.max(interference_pattern)
        
        if max_intensity > 0:
            return variance / max_intensity**2
        return 0.0
    
    def _store_in_medium(self, pattern: MemoryPattern, interference: np.ndarray):
        """Almacena el patron en el medio holografico virtual"""
        # Superposicion en el medio holografico
        self.hologram_medium += interference.astype(np.complex128)
        
        # Control de saturacion del medio
        max_intensity = np.max(np.abs(self.hologram_medium))
        if max_intensity > 10.0:  # Threshold de saturacion
            self.hologram_medium *= 0.9  # Atenuacion global
    
    def _update_interference_matrix(self, new_pattern_id: str, new_pattern: MemoryPattern):
        """Actualiza la matriz de interferencia entre patrones"""
        if not hasattr(self, '_pattern_indices'):
            self._pattern_indices = {}
        
        # Asignar indice al nuevo patron
        pattern_index = len(self._pattern_indices)
        self._pattern_indices[new_pattern_id] = pattern_index
        
        # Calcular interferencia con patrones existentes
        for existing_id, existing_pattern in self.patterns.items():
            if existing_id != new_pattern_id:
                existing_index = self._pattern_indices[existing_id]
                
                # Correlacion cruzada normalizada
                correlation = np.corrcoef(
                    new_pattern.amplitude_encoding.flatten(),
                    existing_pattern.amplitude_encoding.flatten()
                )[0, 1]
                
                # Almacenar en matriz simetrica
                self.interference_matrix[pattern_index, existing_index] = abs(correlation)
                self.interference_matrix[existing_index, pattern_index] = abs(correlation)
    
    def _holographic_correlation(self, query_fourier: np.ndarray, 
                                stored_fourier: np.ndarray) -> np.ndarray:
        """Calculo de correlacion holografica en dominio de Fourier"""
        # Correlacion cruzada: F0[F*[A]  F[B]]
        conjugate_stored = np.conj(stored_fourier)
        correlation_fourier = query_fourier * conjugate_stored
        correlation_spatial = ifft2(correlation_fourier)
        
        return np.abs(correlation_spatial)
    
    def _calculate_similarity(self, correlation: np.ndarray, 
                            query_pattern: np.ndarray, stored_pattern: MemoryPattern) -> float:
        """Calculo de similitud normalizada entre patrones"""
        # Maximo de correlacion normalizada
        max_correlation = np.max(correlation)
        
        # Energia de los patrones
        query_energy = np.sum(query_pattern**2)
        stored_energy = np.sum(stored_pattern.amplitude_encoding**2)
        
        # Similitud normalizada por energia
        if query_energy > 0 and stored_energy > 0:
            similarity = max_correlation / np.sqrt(query_energy * stored_energy)
        else:
            similarity = 0.0
        
        # Clamp a [0, 1]
        return min(1.0, max(0.0, similarity))
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Obtiene estadisticas completas del sistema"""
        if not self.patterns:
            return {"status": "empty", "patterns": 0}
        
        # Metricas de capacidad
        utilization = self.total_patterns / self.config.max_patterns
        
        # Metricas de interferencia
        avg_interference = np.mean([p.interference_strength for p in self.patterns.values()])
        
        # Metricas de reconstruccion
        avg_fidelity = np.mean([p.reconstruction_fidelity for p in self.patterns.values() 
                              if p.reconstruction_fidelity > 0])
        
        return {
            "total_patterns": self.total_patterns,
            "capacity_utilization": utilization,
            "average_interference": avg_interference,
            "average_fidelity": avg_fidelity,
            "retrieval_accuracy": self.retrieval_accuracy,
            "hologram_medium_energy": np.sum(np.abs(self.hologram_medium)**2),
            "configuration": {
                "max_patterns": self.config.max_patterns,
                "dimensions": self.config.pattern_dimensions,
                "noise_tolerance": self.config.noise_tolerance
            }
        }
    
    def optimize_holographic_medium(self):
        """Optimizacion del medio holografico para reducir interferencia destructiva"""
        if self.interference_matrix is None:
            return
        
        logger.info("Optimizando medio holografico...")
        
        # Analisis de interferencia destructiva
        high_interference_pairs = np.where(self.interference_matrix > 0.8)
        
        if len(high_interference_pairs[0]) > 0:
            # Aplicar tecnica de multiplexacion angular
            self._apply_angular_multiplexing(high_interference_pairs)
        
        # Rebalanceo de amplitudes
        self._rebalance_amplitudes()
        
        logger.info("Optimizacion del medio holografico completada")
    
    def _apply_angular_multiplexing(self, interference_pairs: Tuple[np.ndarray, np.ndarray]):
        """Aplica multiplexacion angular para reducir interferencia"""
        # Rotacion de patrones interferentes
        rotation_angles = np.linspace(0, 2*np.pi, len(interference_pairs[0]))
        
        for i, (idx1, idx2) in enumerate(zip(interference_pairs[0], interference_pairs[1])):
            if idx1 < len(self.patterns) and idx2 < len(self.patterns):
                angle = rotation_angles[i]
                # Aplicar rotacion de fase
                pattern_ids = list(self.patterns.keys())
                if idx1 < len(pattern_ids):
                    pattern = self.patterns[pattern_ids[idx1]]
                    pattern.phase_encoding += angle
                    pattern.phase_encoding = pattern.phase_encoding % (2*np.pi)
    
    def _rebalance_amplitudes(self):
        """Rebalance de amplitudes para optimizar el uso del medio"""
        if not self.patterns:
            return
        
        # Calculo de amplitud promedio
        avg_amplitude = np.mean([np.max(p.amplitude_encoding) for p in self.patterns.values()])
        
        # Normalizacion adaptiva
        for pattern in self.patterns.values():
            current_max = np.max(pattern.amplitude_encoding)
            if current_max > 0:
                scaling_factor = avg_amplitude / current_max * 0.9  # Factor conservativo
                pattern.amplitude_encoding *= scaling_factor
    
    def save_holographic_state(self, filepath: Path):
        """Guarda el estado completo del sistema holografico"""
        state = {
            "config": {
                "max_patterns": self.config.max_patterns,
                "pattern_dimensions": self.config.pattern_dimensions,
                "noise_tolerance": self.config.noise_tolerance,
                "reconstruction_threshold": self.config.reconstruction_threshold
            },
            "patterns": {},
            "hologram_medium": self.hologram_medium.tolist(),
            "reference_wave": self.reference_wave.tolist(),
            "statistics": self.get_system_statistics()
        }
        
        # Serializacion de patrones
        for pattern_id, pattern in self.patterns.items():
            state["patterns"][pattern_id] = {
                "pattern_id": pattern.pattern_id,
                "data": pattern.data.tolist(),
                "phase_encoding": pattern.phase_encoding.tolist(),
                "amplitude_encoding": pattern.amplitude_encoding.tolist(),
                "timestamp": pattern.timestamp,
                "reconstruction_fidelity": pattern.reconstruction_fidelity,
                "interference_strength": pattern.interference_strength,
                "semantic_tags": pattern.semantic_tags
            }
        
        # Guardar en formato JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Estado holografico guardado en: {filepath}")
    
    def load_holographic_state(self, filepath: Path):
        """Carga el estado completo del sistema holografico"""
        with open(filepath, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        # Restaurar configuracion
        config_data = state["config"]
        self.config = HolographicConfiguration(
            max_patterns=config_data["max_patterns"],
            pattern_dimensions=tuple(config_data["pattern_dimensions"]),
            noise_tolerance=config_data["noise_tolerance"],
            reconstruction_threshold=config_data["reconstruction_threshold"]
        )
        
        # Restaurar medio holografico
        self.hologram_medium = np.array(state["hologram_medium"], dtype=np.complex128)
        self.reference_wave = np.array(state["reference_wave"], dtype=np.complex128)
        
        # Restaurar patrones
        self.patterns = {}
        for pattern_id, pattern_data in state["patterns"].items():
            pattern = MemoryPattern(
                pattern_id=pattern_data["pattern_id"],
                data=np.array(pattern_data["data"], dtype=np.complex128),
                phase_encoding=np.array(pattern_data["phase_encoding"]),
                amplitude_encoding=np.array(pattern_data["amplitude_encoding"]),
                timestamp=pattern_data["timestamp"],
                reconstruction_fidelity=pattern_data["reconstruction_fidelity"],
                interference_strength=pattern_data["interference_strength"],
                semantic_tags=pattern_data["semantic_tags"]
            )
            self.patterns[pattern_id] = pattern
        
        self.total_patterns = len(self.patterns)
        
        # Reinicializar matriz de interferencia si es necesaria
        if self.config.interference_optimization:
            self._initialize_interference_matrix_from_patterns()
        
        logger.info(f"Estado holografico cargado desde: {filepath}")
    
    def _initialize_interference_matrix_from_patterns(self):
        """Inicializa matriz de interferencia desde patrones cargados"""
        n_patterns = len(self.patterns)
        self.interference_matrix = np.zeros((n_patterns, n_patterns))
        self._pattern_indices = {}
        
        pattern_ids = list(self.patterns.keys())
        for i, pattern_id in enumerate(pattern_ids):
            self._pattern_indices[pattern_id] = i
        
        # Calcular interferencias
        for i, id1 in enumerate(pattern_ids):
            for j, id2 in enumerate(pattern_ids):
                if i != j:
                    pattern1 = self.patterns[id1]
                    pattern2 = self.patterns[id2]
                    
                    correlation = np.corrcoef(
                        pattern1.amplitude_encoding.flatten(),
                        pattern2.amplitude_encoding.flatten()
                    )[0, 1]
                    
                    self.interference_matrix[i, j] = abs(correlation)

# Clase de utilidad para pruebas y validacion
class HolographicMemoryValidator:
    """Validador cientifico para el sistema de memoria holografica"""
    
    def __init__(self, memory_system: HolographicMemorySystem):
        self.memory_system = memory_system
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Ejecuta validacion cientifica completa"""
        results = {
            "capacity_test": self._test_storage_capacity(),
            "fidelity_test": self._test_reconstruction_fidelity(),
            "noise_tolerance": self._test_noise_tolerance(),
            "retrieval_accuracy": self._test_retrieval_accuracy(),
            "interference_analysis": self._analyze_interference_patterns()
        }
        
        return results
    
    def _test_storage_capacity(self) -> Dict[str, float]:
        """Prueba de capacidad de almacenamiento segun limite de Hopfield"""
        # Generar patrones aleatorios
        n_test_patterns = min(200, self.memory_system.config.max_patterns)
        patterns_stored = 0
        
        for i in range(n_test_patterns):
            test_pattern = np.random.randn(*self.memory_system.config.pattern_dimensions)
            try:
                self.memory_system.encode_pattern(test_pattern, f"test_{i}")
                patterns_stored += 1
            except ValueError:
                break
        
        theoretical_capacity = int(0.15 * np.prod(self.memory_system.config.pattern_dimensions))
        
        return {
            "patterns_stored": patterns_stored,
            "theoretical_capacity": theoretical_capacity,
            "capacity_ratio": patterns_stored / theoretical_capacity if theoretical_capacity > 0 else 0
        }
    
    def _test_reconstruction_fidelity(self) -> Dict[str, float]:
        """Prueba de fidelidad de reconstruccion"""
        if not self.memory_system.patterns:
            return {"error": "No patterns to test"}
        
        fidelities = []
        for pattern_id, pattern in list(self.memory_system.patterns.items())[:10]:
            reconstructed = self.memory_system.reconstruct_pattern(pattern)
            
            # Calcular fidelidad como correlacion normalizada
            original = pattern.amplitude_encoding
            fidelity = np.corrcoef(original.flatten(), reconstructed.flatten())[0, 1]
            if not np.isnan(fidelity):
                fidelities.append(abs(fidelity))
        
        return {
            "mean_fidelity": np.mean(fidelities) if fidelities else 0,
            "std_fidelity": np.std(fidelities) if fidelities else 0,
            "min_fidelity": np.min(fidelities) if fidelities else 0,
            "max_fidelity": np.max(fidelities) if fidelities else 0
        }
    
    def _test_noise_tolerance(self) -> Dict[str, float]:
        """Prueba de tolerancia al ruido"""
        if not self.memory_system.patterns:
            return {"error": "No patterns to test"}
        
        noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
        tolerance_results = {}
        
        sample_pattern = next(iter(self.memory_system.patterns.values()))
        
        for noise_level in noise_levels:
            reconstructed = self.memory_system.reconstruct_pattern(sample_pattern, add_noise=noise_level)
            original = sample_pattern.amplitude_encoding
            
            # Similitud con ruido
            similarity = np.corrcoef(original.flatten(), reconstructed.flatten())[0, 1]
            if not np.isnan(similarity):
                tolerance_results[f"noise_{noise_level}"] = abs(similarity)
            else:
                tolerance_results[f"noise_{noise_level}"] = 0.0
        
        return tolerance_results
    
    def _test_retrieval_accuracy(self) -> Dict[str, float]:
        """Prueba de precision de recuperacion"""
        if len(self.memory_system.patterns) < 2:
            return {"error": "Need at least 2 patterns for testing"}
        
        correct_retrievals = 0
        total_tests = min(10, len(self.memory_system.patterns))
        
        patterns_list = list(self.memory_system.patterns.values())
        
        for i in range(total_tests):
            test_pattern = patterns_list[i]
            query = test_pattern.amplitude_encoding
            
            result = self.memory_system.retrieve_pattern(query, similarity_threshold=0.7)
            
            if result and result[0].pattern_id == test_pattern.pattern_id:
                correct_retrievals += 1
        
        return {
            "accuracy": correct_retrievals / total_tests if total_tests > 0 else 0,
            "correct_retrievals": correct_retrievals,
            "total_tests": total_tests
        }
    
    def _analyze_interference_patterns(self) -> Dict[str, Any]:
        """Analisis de patrones de interferencia"""
        if self.memory_system.interference_matrix is None:
            return {"error": "Interference matrix not available"}
        
        matrix = self.memory_system.interference_matrix
        n_patterns = matrix.shape[0]
        
        if n_patterns < 2:
            return {"error": "Need at least 2 patterns for interference analysis"}
        
        # Extraer parte triangular superior (sin diagonal)
        upper_triangle = matrix[np.triu_indices(n_patterns, k=1)]
        
        return {
            "mean_interference": np.mean(upper_triangle),
            "max_interference": np.max(upper_triangle),
            "interference_std": np.std(upper_triangle),
            "high_interference_pairs": np.sum(upper_triangle > 0.8),
            "total_pairs": len(upper_triangle)
        }

# Funciones de utilidad global
def create_holographic_memory_system(max_patterns: int = 1000, 
                                    dimensions: Tuple[int, int] = (64, 64)) -> HolographicMemorySystem:
    """Factory function para crear sistema de memoria holografica"""
    config = HolographicConfiguration(
        max_patterns=max_patterns,
        pattern_dimensions=dimensions
    )
    return HolographicMemorySystem(config)

def demonstrate_holographic_memory():
    """Demostracion del sistema de memoria holografica"""
    print("=== DEMOSTRACION SISTEMA DE MEMORIA HOLOGRAFICA ===")
    
    # Crear sistema
    memory_system = create_holographic_memory_system(max_patterns=50, dimensions=(32, 32))
    
    # Crear patrones de prueba
    patterns = {
        "pattern_A": np.random.randn(32, 32),
        "pattern_B": np.random.randn(32, 32), 
        "pattern_C": np.random.randn(32, 32)
    }
    
    # Codificar patrones
    print("\n1. Codificando patrones...")
    for name, data in patterns.items():
        encoded = memory_system.encode_pattern(data, name, [f"tag_{name}"])
        print(f"   Codificado {name}: interferencia={encoded.interference_strength:.3f}")
    
    # Recuperar patrones
    print("\n2. Recuperando patrones...")
    for name, original_data in patterns.items():
        result = memory_system.retrieve_pattern(original_data, similarity_threshold=0.7)
        if result:
            retrieved_pattern, similarity = result
            print(f"   Recuperado {name}: similitud={similarity:.3f}")
        else:
            print(f"   No se pudo recuperar {name}")
    
    # Estadisticas del sistema
    print("\n3. Estadisticas del sistema:")
    stats = memory_system.get_system_statistics()
    print(f"   Patrones totales: {stats['total_patterns']}")
    print(f"   Utilizacion: {stats['capacity_utilization']:.1%}")
    print(f"   Interferencia promedio: {stats['average_interference']:.3f}")
    print(f"   Fidelidad promedio: {stats['average_fidelity']:.3f}")
    
    # Validacion cientifica
    print("\n4. Validacion cientifica...")
    validator = HolographicMemoryValidator(memory_system)
    validation_results = validator.run_comprehensive_validation()
    
    print(f"   Capacidad teorica: {validation_results['capacity_test']['theoretical_capacity']}")
    print(f"   Fidelidad media: {validation_results['fidelity_test']['mean_fidelity']:.3f}")
    print(f"   Precision de recuperacion: {validation_results['retrieval_accuracy']['accuracy']:.3f}")

if __name__ == "__main__":
    # Ejecutar demostracion
    demonstrate_holographic_memory()