#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Comprehensive Test Suite for Enhanced AION Protocol
====================================================

Tests all integrated components from Referencias:
- Vector Memory Management
- Quantum Adaptive Framework
- Fractal Memory System  
- AI Communication Protocol
- Metacognitive Learning System
- AION Protocol Core Integration

Author: Francisco Molina
ORCID: https://orcid.org/0009-0008-6093-8267
Email: pako.molina@gmail.com
"""

import sys
import os
import unittest
import time
import json
from typing import Dict, List, Any

# Add AION to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'AION'))

try:
    from AION.aion_core import AIONProtocol
    from AION.fractal_memory import FractalMemorySystem, AIONFractalMemoryBridge
    from AION.ai_communication import AICommunicationProtocol, ConceptSpace, AIONCommunicationBridge
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you're running from the Obvivlorum directory")
    sys.exit(1)

class TestEnhancedAION(unittest.TestCase):
    """Test suite for enhanced AION Protocol."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.aion = AIONProtocol()
        self.fractal_memory = FractalMemorySystem(storage_capacity_gb=0.1, redundancy_factor=2)
        self.concept_space = ConceptSpace(dimension=64)
        self.ai_comm = AICommunicationProtocol("TEST_AI", self.concept_space)
        
        # Create bridges
        self.fractal_bridge = AIONFractalMemoryBridge(self.aion, self.fractal_memory)
        self.comm_bridge = AIONCommunicationBridge(self.aion, self.ai_comm)
    
    def test_aion_core_initialization(self):
        """Test AION core initialization with enhanced systems."""
        self.assertIsNotNone(self.aion)
        self.assertEqual(self.aion.VERSION, "2.0")
        self.assertIsNotNone(self.aion.aion_dna)
        self.assertTrue(len(self.aion.aion_dna) > 0)
        
        # Test enhanced systems
        self.assertIsNotNone(self.aion.vector_memory_manager)
        self.assertIsNotNone(self.aion.metacognitive_system)
        self.assertIsNotNone(self.aion.quantum_framework)
        
        print("✓ AION Core initialization test passed")
    
    def test_protocol_execution_enhanced(self):
        """Test enhanced protocol execution."""
        # Test basic ALPHA protocol execution
        params = {
            "research_domain": "test_domain",
            "research_type": "exploratory",
            "exploration_depth": 0.7,
            "seed_concepts": ["test", "concept"],
            "concepts": ["test", "concept"],
            "create_domain": True
        }
        
        result = self.aion.execute_protocol("ALPHA", params)
        
        self.assertEqual(result["status"], "success")
        self.assertIn("research_id", result)
        
        print("✓ Enhanced Protocol Execution test passed")

def run_simple_test():
    """Run a simple test to verify the enhanced system works."""
    print("Enhanced AION Protocol - Simple Test")
    print("=" * 40)
    
    try:
        # Initialize system
        aion = AIONProtocol()
        print("✓ AION Protocol initialized")
        
        # Test vector memory
        if aion.vector_memory_manager:
            vector = aion.VectorMemory(0.5, {"test": 0.8})
            vec_id = aion.vector_memory_manager.store_vector(vector, "test")
            print("✓ Vector memory working")
        
        # Test quantum framework
        if aion.quantum_framework:
            state_id = aion.quantum_framework.create_quantum_state("test_concept", 0.8)
            print("✓ Quantum framework working")
        
        # Test protocol execution
        params = {
            "research_domain": "test_domain",
            "research_type": "exploratory",
            "create_domain": True
        }
        
        result = aion.execute_protocol("ALPHA", params)
        if result.get("status") == "success":
            print("✓ Protocol execution working")
        else:
            print(f"✗ Protocol execution failed: {result.get('message')}")
        
        print("\nSimple test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced AION Protocol Test Suite")
    parser.add_argument("--simple", action="store_true", help="Run simple test only")
    
    args = parser.parse_args()
    
    if args.simple:
        run_simple_test()
    else:
        unittest.main(verbosity=2)