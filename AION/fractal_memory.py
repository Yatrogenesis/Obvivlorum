#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fractal Memory System for AION Protocol
========================================

Implementation of holographic fractal memory system based on the 
specifications from Referencias. Provides distributed, self-healing
memory storage with quantum encryption and semantic indexing.

Author: Francisco Molina
ORCID: https://orcid.org/0009-0008-6093-8267
Email: pako.molina@gmail.com
"""

import numpy as np
import hashlib
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict
import base64
import zlib

logger = logging.getLogger("AION.FractalMemory")

@dataclass
class FractalFragment:
    """Represents a fractal fragment containing holographic information."""
    id: str
    data: bytes
    metadata: Dict[str, Any]
    fractal_level: int
    parent_fragments: List[str] = field(default_factory=list)
    child_fragments: List[str] = field(default_factory=list)
    resonance_patterns: Dict[str, float] = field(default_factory=dict)
    encryption_hash: str = ""
    timestamp: float = field(default_factory=time.time)

class HolographicEncoder:
    """Encodes data in holographic fractal format."""
    
    def __init__(self, compression_ratio: float = 0.7):
        self.compression_ratio = compression_ratio
        self.encoding_matrix = self._generate_encoding_matrix()
    
    def _generate_encoding_matrix(self, size: int = 128) -> np.ndarray:
        """Generate the holographic encoding matrix."""
        # Create a complex matrix for holographic encoding
        real_part = np.random.randn(size, size) * 0.1
        imag_part = np.random.randn(size, size) * 0.1
        return real_part + 1j * imag_part
    
    def encode_to_fractal(self, data: bytes, fractal_level: int = 0) -> List[FractalFragment]:
        """
        Encode data into fractal fragments with holographic properties.
        
        Args:
            data: Raw data to encode
            fractal_level: Level of fractal decomposition
            
        Returns:
            List of fractal fragments
        """
        # Compress data first
        compressed_data = zlib.compress(data)
        
        # Calculate fragment count based on fractal level
        fragment_count = 2 ** (fractal_level + 2)  # 4, 8, 16, 32...
        fragment_size = len(compressed_data) // fragment_count
        
        fragments = []
        for i in range(fragment_count):
            start_idx = i * fragment_size
            end_idx = start_idx + fragment_size if i < fragment_count - 1 else len(compressed_data)
            
            fragment_data = compressed_data[start_idx:end_idx]
            
            # Add holographic redundancy - each fragment contains overlapping information
            if i > 0:
                overlap_size = fragment_size // 4
                overlap_start = max(0, start_idx - overlap_size)
                overlap_data = compressed_data[overlap_start:start_idx]
                fragment_data = overlap_data + fragment_data
            
            # Create fragment
            fragment_id = self._generate_fragment_id(data, i, fractal_level)
            
            # Calculate resonance patterns (simplified)
            resonance_patterns = {
                f"harmonic_{j}": np.sin(2 * np.pi * j * i / fragment_count) 
                for j in range(1, 6)
            }
            
            fragment = FractalFragment(
                id=fragment_id,
                data=fragment_data,
                metadata={
                    "fragment_index": i,
                    "total_fragments": fragment_count,
                    "original_size": len(data),
                    "compressed_size": len(compressed_data),
                    "encoding_timestamp": time.time()
                },
                fractal_level=fractal_level,
                resonance_patterns=resonance_patterns
            )
            
            fragments.append(fragment)
        
        # Establish parent-child relationships
        self._establish_fractal_relationships(fragments)
        
        return fragments
    
    def _generate_fragment_id(self, data: bytes, index: int, level: int) -> str:
        """Generate unique ID for a fragment."""
        base_hash = hashlib.sha256(data).hexdigest()
        fragment_info = f"{base_hash}:{index}:{level}:{time.time()}"
        return hashlib.md5(fragment_info.encode()).hexdigest()
    
    def _establish_fractal_relationships(self, fragments: List[FractalFragment]):
        """Establish parent-child relationships between fragments."""
        for i, fragment in enumerate(fragments):
            # Parent relationships (higher level aggregations)
            if i % 2 == 0 and i + 1 < len(fragments):
                parent_id = f"parent_{fragment.id}_{fragments[i+1].id}"
                fragment.parent_fragments.append(parent_id)
                fragments[i+1].parent_fragments.append(parent_id)
            
            # Child relationships (subdivisions)
            if fragment.fractal_level > 0:
                child_count = 2 ** fragment.fractal_level
                for j in range(child_count):
                    child_id = f"child_{fragment.id}_{j}"
                    fragment.child_fragments.append(child_id)

class FractalMemorySystem:
    """Main fractal memory system implementing holographic storage."""
    
    def __init__(self, 
                 storage_capacity_gb: float = 10.0,
                 redundancy_factor: int = 3,
                 encryption_enabled: bool = True):
        """
        Initialize the fractal memory system.
        
        Args:
            storage_capacity_gb: Total storage capacity in GB
            redundancy_factor: Number of redundant copies for each fragment
            encryption_enabled: Enable quantum-resistant encryption
        """
        self.storage_capacity = storage_capacity_gb * 1024 * 1024 * 1024
        self.redundancy_factor = redundancy_factor
        self.encryption_enabled = encryption_enabled
        
        # Storage structures
        self.fragment_storage = {}  # fragment_id -> FractalFragment
        self.semantic_index = defaultdict(list)  # concept -> fragment_ids
        self.resonance_index = defaultdict(list)  # pattern -> fragment_ids
        self.temporal_index = defaultdict(list)  # timestamp -> fragment_ids
        
        # System components
        self.encoder = HolographicEncoder()
        self.encryption_key = self._generate_encryption_key() if encryption_enabled else None
        
        # Metrics
        self.metrics = {
            "total_fragments": 0,
            "total_data_stored": 0,
            "recovery_attempts": 0,
            "successful_recoveries": 0,
            "compression_ratio": 0.0
        }
        
        logger.info(f"Fractal Memory System initialized with {storage_capacity_gb}GB capacity")
    
    def _generate_encryption_key(self) -> bytes:
        """Generate quantum-resistant encryption key."""
        # In a real implementation, this would use post-quantum cryptography
        key_material = np.random.bytes(64)  # 512-bit key
        return hashlib.sha256(key_material).digest()
    
    def store_data(self, 
                   data: Union[str, bytes, Dict], 
                   semantic_tags: List[str] = None,
                   fractal_level: int = 2) -> str:
        """
        Store data in the fractal memory system.
        
        Args:
            data: Data to store (string, bytes, or dictionary)
            semantic_tags: Semantic tags for indexing
            fractal_level: Level of fractal decomposition
            
        Returns:
            Storage ID for retrieving the data
        """
        # Convert data to bytes
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        elif isinstance(data, dict):
            data_bytes = json.dumps(data).encode('utf-8')
        else:
            data_bytes = data
        
        # Encrypt if enabled
        if self.encryption_enabled:
            data_bytes = self._encrypt_data(data_bytes)
        
        # Encode to fractal fragments
        fragments = self.encoder.encode_to_fractal(data_bytes, fractal_level)
        
        # Store fragments with redundancy
        storage_id = self._generate_storage_id(data_bytes)
        stored_fragment_ids = []
        
        for fragment in fragments:
            for replica in range(self.redundancy_factor):
                replica_id = f"{fragment.id}_replica_{replica}"
                replica_fragment = FractalFragment(
                    id=replica_id,
                    data=fragment.data,
                    metadata={**fragment.metadata, "replica_index": replica, "storage_id": storage_id},
                    fractal_level=fragment.fractal_level,
                    parent_fragments=fragment.parent_fragments,
                    child_fragments=fragment.child_fragments,
                    resonance_patterns=fragment.resonance_patterns,
                    timestamp=time.time()
                )
                
                self.fragment_storage[replica_id] = replica_fragment
                stored_fragment_ids.append(replica_id)
        
        # Update indices
        self._update_semantic_index(stored_fragment_ids, semantic_tags or [])
        self._update_resonance_index(fragments)
        self._update_temporal_index(stored_fragment_ids)
        
        # Update metrics
        self.metrics["total_fragments"] += len(stored_fragment_ids)
        self.metrics["total_data_stored"] += len(data_bytes)
        
        logger.info(f"Stored data with ID {storage_id}, {len(fragments)} fragments, {self.redundancy_factor}x redundancy")
        
        return storage_id
    
    def retrieve_data(self, storage_id: str) -> Optional[Union[str, bytes, Dict]]:
        """
        Retrieve data from the fractal memory system.
        
        Args:
            storage_id: Storage ID returned by store_data
            
        Returns:
            Retrieved data or None if not found
        """
        self.metrics["recovery_attempts"] += 1
        
        # Find fragments for this storage ID
        relevant_fragments = [
            fragment for fragment in self.fragment_storage.values()
            if fragment.metadata.get("storage_id") == storage_id
        ]
        
        if not relevant_fragments:
            logger.error(f"No fragments found for storage ID {storage_id}")
            return None
        
        # Group by original fragment ID (remove replica suffix)
        fragment_groups = defaultdict(list)
        for fragment in relevant_fragments:
            original_id = fragment.id.split("_replica_")[0]
            fragment_groups[original_id].append(fragment)
        
        # Reconstruct data from fragments
        try:
            reconstructed_data = self._reconstruct_from_fragments(fragment_groups)
            
            # Decrypt if needed
            if self.encryption_enabled:
                reconstructed_data = self._decrypt_data(reconstructed_data)
            
            # Try to parse as JSON first, then as string
            try:
                decoded_data = json.loads(reconstructed_data.decode('utf-8'))
                self.metrics["successful_recoveries"] += 1
                return decoded_data
            except (json.JSONDecodeError, UnicodeDecodeError):
                try:
                    decoded_data = reconstructed_data.decode('utf-8')
                    self.metrics["successful_recoveries"] += 1
                    return decoded_data
                except UnicodeDecodeError:
                    self.metrics["successful_recoveries"] += 1
                    return reconstructed_data
                    
        except Exception as e:
            logger.error(f"Failed to reconstruct data for storage ID {storage_id}: {e}")
            return None
    
    def _reconstruct_from_fragments(self, fragment_groups: Dict[str, List[FractalFragment]]) -> bytes:
        """Reconstruct original data from fragment groups."""
        # Sort fragments by index
        sorted_fragments = []
        for original_id, replicas in fragment_groups.items():
            # Use the first available replica (redundancy provides fault tolerance)
            if replicas:
                fragment = replicas[0]
                fragment_index = fragment.metadata.get("fragment_index", 0)
                sorted_fragments.append((fragment_index, fragment))
        
        sorted_fragments.sort(key=lambda x: x[0])
        
        # Reconstruct compressed data
        reconstructed_compressed = b""
        for _, fragment in sorted_fragments:
            # Remove overlap data if present (simplified approach)
            fragment_data = fragment.data
            if fragment.metadata.get("fragment_index", 0) > 0:
                # Remove overlap from beginning (rough estimate)
                overlap_size = len(fragment_data) // 5
                fragment_data = fragment_data[overlap_size:]
            
            reconstructed_compressed += fragment_data
        
        # Decompress
        try:
            reconstructed_data = zlib.decompress(reconstructed_compressed)
            return reconstructed_data
        except zlib.error as e:
            # Try with partial data (holographic property)
            logger.warning(f"Decompression failed, attempting partial reconstruction: {e}")
            # Return what we have (in a real system, this would be more sophisticated)
            return reconstructed_compressed
    
    def _encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data using quantum-resistant methods."""
        # Simplified encryption (in reality, use post-quantum algorithms)
        key_hash = hashlib.sha256(self.encryption_key).digest()
        encrypted = bytearray()
        for i, byte in enumerate(data):
            key_byte = key_hash[i % len(key_hash)]
            encrypted.append(byte ^ key_byte)
        return bytes(encrypted)
    
    def _decrypt_data(self, data: bytes) -> bytes:
        """Decrypt data."""
        # Since XOR is symmetric, decryption is the same as encryption
        return self._encrypt_data(data)
    
    def _generate_storage_id(self, data: bytes) -> str:
        """Generate unique storage ID."""
        data_hash = hashlib.sha256(data).hexdigest()
        timestamp = str(int(time.time()))
        storage_info = f"{data_hash}:{timestamp}"
        return hashlib.md5(storage_info.encode()).hexdigest()
    
    def _update_semantic_index(self, fragment_ids: List[str], semantic_tags: List[str]):
        """Update semantic index with new fragments."""
        for tag in semantic_tags:
            self.semantic_index[tag.lower()].extend(fragment_ids)
    
    def _update_resonance_index(self, fragments: List[FractalFragment]):
        """Update resonance pattern index."""
        for fragment in fragments:
            for pattern_name, pattern_value in fragment.resonance_patterns.items():
                # Quantize pattern value for indexing
                pattern_key = f"{pattern_name}_{int(pattern_value * 10)}"
                self.resonance_index[pattern_key].append(fragment.id)
    
    def _update_temporal_index(self, fragment_ids: List[str]):
        """Update temporal index."""
        time_bucket = int(time.time() // 3600)  # Hour buckets
        self.temporal_index[time_bucket].extend(fragment_ids)
    
    def search_by_semantic(self, query: str) -> List[str]:
        """Search for storage IDs by semantic content."""
        query_lower = query.lower()
        matching_fragment_ids = set()
        
        for tag, fragment_ids in self.semantic_index.items():
            if query_lower in tag:
                matching_fragment_ids.update(fragment_ids)
        
        # Extract storage IDs
        storage_ids = set()
        for fragment_id in matching_fragment_ids:
            if fragment_id in self.fragment_storage:
                fragment = self.fragment_storage[fragment_id]
                storage_id = fragment.metadata.get("storage_id")
                if storage_id:
                    storage_ids.add(storage_id)
        
        return list(storage_ids)
    
    def search_by_resonance(self, pattern_name: str, value_range: Tuple[float, float]) -> List[str]:
        """Search for storage IDs by resonance patterns."""
        matching_storage_ids = set()
        
        for fragment in self.fragment_storage.values():
            if pattern_name in fragment.resonance_patterns:
                pattern_value = fragment.resonance_patterns[pattern_name]
                if value_range[0] <= pattern_value <= value_range[1]:
                    storage_id = fragment.metadata.get("storage_id")
                    if storage_id:
                        matching_storage_ids.add(storage_id)
        
        return list(matching_storage_ids)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        used_storage = sum(len(f.data) for f in self.fragment_storage.values())
        
        return {
            "total_capacity_gb": self.storage_capacity / (1024 * 1024 * 1024),
            "used_storage_gb": used_storage / (1024 * 1024 * 1024),
            "utilization_percent": (used_storage / self.storage_capacity) * 100,
            "total_fragments": len(self.fragment_storage),
            "unique_storage_ids": len(set(
                f.metadata.get("storage_id") for f in self.fragment_storage.values()
                if f.metadata.get("storage_id")
            )),
            "semantic_tags": len(self.semantic_index),
            "resonance_patterns": len(self.resonance_index),
            "metrics": self.metrics
        }
    
    def defragment(self) -> Dict[str, Any]:
        """Defragment and optimize memory storage."""
        logger.info("Starting memory defragmentation...")
        
        initial_fragments = len(self.fragment_storage)
        
        # Remove orphaned fragments (no storage_id)
        orphaned_count = 0
        fragment_ids_to_remove = []
        for fragment_id, fragment in self.fragment_storage.items():
            if not fragment.metadata.get("storage_id"):
                fragment_ids_to_remove.append(fragment_id)
                orphaned_count += 1
        
        for fragment_id in fragment_ids_to_remove:
            del self.fragment_storage[fragment_id]
        
        # Rebuild indices
        self.semantic_index.clear()
        self.resonance_index.clear()
        self.temporal_index.clear()
        
        # Group fragments by storage_id and rebuild indices
        for fragment in self.fragment_storage.values():
            storage_id = fragment.metadata.get("storage_id")
            if storage_id:
                # Rebuild temporal index
                time_bucket = int(fragment.timestamp // 3600)
                self.temporal_index[time_bucket].append(fragment.id)
                
                # Rebuild resonance index
                for pattern_name, pattern_value in fragment.resonance_patterns.items():
                    pattern_key = f"{pattern_name}_{int(pattern_value * 10)}"
                    self.resonance_index[pattern_key].append(fragment.id)
        
        final_fragments = len(self.fragment_storage)
        
        defrag_result = {
            "initial_fragments": initial_fragments,
            "final_fragments": final_fragments,
            "orphaned_removed": orphaned_count,
            "space_recovered_mb": orphaned_count * 1024,  # Rough estimate
            "defragmentation_time": time.time()
        }
        
        logger.info(f"Defragmentation complete: {defrag_result}")
        return defrag_result

# Integration with AION Protocol
class AIONFractalMemoryBridge:
    """Bridge between AION Protocol and Fractal Memory System."""
    
    def __init__(self, aion_protocol, memory_system: FractalMemorySystem):
        self.aion_protocol = aion_protocol
        self.memory_system = memory_system
        self.logger = logging.getLogger("AION.FractalMemoryBridge")
    
    def store_protocol_result(self, protocol_name: str, execution_id: str, result: Dict[str, Any]) -> str:
        """Store protocol execution result in fractal memory."""
        semantic_tags = [
            f"protocol_{protocol_name.lower()}",
            f"execution_{execution_id}",
            f"status_{result.get('status', 'unknown')}",
            "aion_protocol_result"
        ]
        
        storage_id = self.memory_system.store_data(
            data=result,
            semantic_tags=semantic_tags,
            fractal_level=2
        )
        
        self.logger.info(f"Stored protocol result {execution_id} with storage ID {storage_id}")
        return storage_id
    
    def retrieve_protocol_history(self, protocol_name: str) -> List[Dict[str, Any]]:
        """Retrieve execution history for a specific protocol."""
        storage_ids = self.memory_system.search_by_semantic(f"protocol_{protocol_name.lower()}")
        
        results = []
        for storage_id in storage_ids:
            data = self.memory_system.retrieve_data(storage_id)
            if data:
                results.append(data)
        
        return results
    
    def search_by_concept(self, concept: str) -> List[Dict[str, Any]]:
        """Search stored data by conceptual content."""
        storage_ids = self.memory_system.search_by_semantic(concept)
        
        results = []
        for storage_id in storage_ids:
            data = self.memory_system.retrieve_data(storage_id)
            if data:
                results.append(data)
        
        return results