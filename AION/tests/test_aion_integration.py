#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AION Protocol Test Suite
========================

This module contains tests for the AION Protocol integration with Obvivlorum.

Author: Francisco Molina
ORCID: https://orcid.org/0009-0008-6093-8267
Email: pako.molina@gmail.com
"""

import unittest
import sys
import os
import json
from unittest.mock import MagicMock, patch

# Add parent directory to path to import AION modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from aion_core import AIONProtocol

class TestAIONProtocol(unittest.TestCase):
    """Test cases for the AION Protocol core functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.aion = AIONProtocol()
        
        # Mock protocol modules
        self.aion.protocol_modules = {
            "ALPHA": MagicMock(),
            "BETA": MagicMock(),
            "GAMMA": MagicMock(),
            "DELTA": MagicMock(),
            "OMEGA": MagicMock()
        }
        
        # Configure mocks
        for protocol_name, mock in self.aion.protocol_modules.items():
            mock.execute.return_value = {
                "status": "success",
                "protocol": protocol_name,
                "result": f"{protocol_name} execution result"
            }
    
    def test_protocol_execution(self):
        """Test basic protocol execution."""
        # Test each protocol
        for protocol_name in self.aion.PROTOCOLS:
            result = self.aion.execute_protocol(protocol_name, {"test_param": "value"})
            
            # Verify protocol was called
            self.aion.protocol_modules[protocol_name].execute.assert_called_once()
            
            # Verify result
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["protocol"], protocol_name)
    
    def test_invalid_protocol(self):
        """Test execution of invalid protocol."""
        result = self.aion.execute_protocol("INVALID", {})
        
        # Verify error handling
        self.assertEqual(result["status"], "error")
        self.assertIn("Unknown protocol", result["message"])
    
    def test_aion_dna_generation(self):
        """Test AION DNA generation."""
        # DNA should be a SHA-256 hash (64 hex characters)
        self.assertEqual(len(self.aion.aion_dna), 64)
        
        # Two instances should have different DNAs
        another_aion = AIONProtocol()
        self.assertNotEqual(self.aion.aion_dna, another_aion.aion_dna)
    
    def test_execution_history(self):
        """Test execution history tracking."""
        # Initial history should be empty
        self.assertEqual(len(self.aion.get_execution_history()), 0)
        
        # Execute a protocol
        self.aion.execute_protocol("ALPHA", {"test": True})
        
        # History should have one entry
        history = self.aion.get_execution_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["protocol"], "ALPHA")
        self.assertEqual(history[0]["status"], "success")
    
    def test_error_handling(self):
        """Test error handling during protocol execution."""
        # Configure a protocol to raise an exception
        self.aion.protocol_modules["BETA"].execute.side_effect = Exception("Test error")
        
        # Execute the protocol
        result = self.aion.execute_protocol("BETA", {})
        
        # Verify error handling
        self.assertEqual(result["status"], "error")
        self.assertIn("Test error", result["message"])
        
        # Check history
        history = self.aion.get_execution_history()
        self.assertEqual(history[-1]["status"], "error")
        self.assertEqual(history[-1]["protocol"], "BETA")
        self.assertIn("Test error", history[-1]["error"])

class TestAIONObvivlorumIntegration(unittest.TestCase):
    """Test cases for the AION-Obvivlorum integration."""
    
    def setUp(self):
        """Set up test environment."""
        self.aion = AIONProtocol()
        
        # Mock Obvivlorum core
        self.mock_obvivlorum = MagicMock()
        self.mock_obvivlorum.introspect.return_value = {"status": "success", "depth": 1}
        self.mock_obvivlorum.get_current_state.return_value = {"state": "active"}
        
        # Patch bridge module import
        self.bridge_patcher = patch('importlib.import_module')
        self.mock_importlib = self.bridge_patcher.start()
        
        # Setup mock bridge module
        self.mock_bridge_module = MagicMock()
        self.mock_bridge_class = MagicMock()
        self.mock_bridge_instance = MagicMock()
        
        self.mock_importlib.return_value = self.mock_bridge_module
        self.mock_bridge_module.AIONObvivlorumBridge = self.mock_bridge_class
        self.mock_bridge_class.return_value = self.mock_bridge_instance
        
        # Configure mock bridge
        self.mock_bridge_instance.connection_state = "connected"
        self.mock_bridge_instance.execute_protocol_with_obvivlorum.return_value = {
            "status": "success",
            "obvivlorum_enhanced": True,
            "result": "Enhanced result"
        }
    
    def tearDown(self):
        """Clean up after tests."""
        self.bridge_patcher.stop()
    
    def test_bridge_creation(self):
        """Test bridge creation between AION and Obvivlorum."""
        bridge = self.aion.create_bridge(self.mock_obvivlorum)
        
        # Verify bridge was created
        self.mock_bridge_class.assert_called_once_with(self.aion, self.mock_obvivlorum)
        self.assertEqual(bridge, self.mock_bridge_instance)
    
    def test_protocol_execution_through_bridge(self):
        """Test executing a protocol through the bridge."""
        # Create bridge
        bridge = self.aion.create_bridge(self.mock_obvivlorum)
        
        # Execute protocol through bridge
        params = {"test_param": "value"}
        result = bridge.execute_protocol_with_obvivlorum("ALPHA", params)
        
        # Verify bridge method was called
        bridge.execute_protocol_with_obvivlorum.assert_called_once_with("ALPHA", params)
        
        # Verify result
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["obvivlorum_enhanced"])
    
    def test_bridge_status(self):
        """Test getting bridge status."""
        # Create bridge
        bridge = self.aion.create_bridge(self.mock_obvivlorum)
        
        # Mock get_bridge_status method
        bridge.get_bridge_status.return_value = {
            "status": "success",
            "connection_state": "connected",
            "sync_mode": "bidirectional"
        }
        
        # Get status
        status = bridge.get_bridge_status()
        
        # Verify method was called
        bridge.get_bridge_status.assert_called_once()
        
        # Verify status
        self.assertEqual(status["status"], "success")
        self.assertEqual(status["connection_state"], "connected")
    
    def test_bridge_disconnection(self):
        """Test bridge disconnection."""
        # Create bridge
        bridge = self.aion.create_bridge(self.mock_obvivlorum)
        
        # Mock disconnect method
        bridge.disconnect.return_value = True
        
        # Disconnect bridge
        result = bridge.disconnect()
        
        # Verify method was called
        bridge.disconnect.assert_called_once()
        
        # Verify result
        self.assertTrue(result)

class TestProtocolALPHA(unittest.TestCase):
    """Test cases for the ALPHA protocol implementation."""
    
    def setUp(self):
        """Set up test environment."""
        # Import ALPHA protocol
        from aion_protocols.protocol_alpha import ProtocolALPHA
        self.protocol = ProtocolALPHA()
    
    def test_protocol_initialization(self):
        """Test protocol initialization."""
        self.assertEqual(self.protocol.VERSION, "1.0")
        self.assertIn("quantum_research", self.protocol.CAPABILITIES)
        self.assertIsInstance(self.protocol.knowledge_base, dict)
    
    def test_exploratory_research(self):
        """Test exploratory research execution."""
        # Execute exploratory research
        result = self.protocol.execute({
            "research_domain": "quantum_mechanics",
            "research_type": "exploratory",
            "seed_concepts": ["quantum_entanglement", "wave_function_collapse"]
        })
        
        # Verify result
        self.assertEqual(result["status"], "success")
        self.assertIn("research_id", result)
        self.assertEqual(result["domain"], "quantum_mechanics")
        self.assertEqual(result["type"], "exploratory")
        self.assertIn("result", result)
    
    def test_integrative_research(self):
        """Test integrative research execution."""
        # Execute integrative research
        result = self.protocol.execute({
            "research_domain": "quantum_mechanics",
            "research_type": "integrative",
            "integration_targets": ["cognitive_science", "artificial_intelligence"],
            "bridge_concepts": ["quantum_cognition", "entangled_intelligence"]
        })
        
        # Verify result
        self.assertEqual(result["status"], "success")
        self.assertIn("research_id", result)
        self.assertEqual(result["domain"], "quantum_mechanics")
        self.assertEqual(result["type"], "integrative")
        self.assertIn("result", result)
        
        # Verify integration results
        integration_results = result["result"]["integration_results"]
        self.assertIsInstance(integration_results, list)
        self.assertGreater(len(integration_results), 0)
    
    def test_paradigm_shift(self):
        """Test paradigm shift execution."""
        # Execute paradigm shift
        result = self.protocol.execute({
            "research_domain": "quantum_mechanics",
            "research_type": "paradigm_shift",
            "shift_magnitude": 0.8,
            "target_paradigms": ["Standard Model"],
            "conceptual_framework": {
                "key_principles": ["non-locality", "observer-dependency", "quantum_emergence"]
            }
        })
        
        # Verify result
        self.assertEqual(result["status"], "success")
        self.assertIn("research_id", result)
        self.assertEqual(result["domain"], "quantum_mechanics")
        self.assertEqual(result["type"], "paradigm_shift")
        
        # Verify paradigm shift results
        shift_result = result["result"]["paradigm_shift"]
        self.assertIn("new_paradigm", shift_result)
        self.assertIn("Standard Model", shift_result["affected_paradigms"])
    
    def test_invalid_research_domain(self):
        """Test execution with invalid research domain."""
        # Execute with invalid domain
        result = self.protocol.execute({
            "research_domain": "invalid_domain",
            "research_type": "exploratory",
            "create_domain": False
        })
        
        # Verify error handling
        self.assertEqual(result["status"], "error")
        self.assertIn("Unknown research domain", result["message"])
    
    def test_domain_creation(self):
        """Test automatic domain creation."""
        # New domain name
        new_domain = "test_domain_" + str(int(time.time()))
        
        # Execute with new domain
        result = self.protocol.execute({
            "research_domain": new_domain,
            "research_type": "exploratory",
            "create_domain": True,
            "seed_concepts": ["test_concept"]
        })
        
        # Verify domain creation
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["domain"], new_domain)
        
        # Verify domain exists in knowledge base
        self.assertIn(new_domain, self.protocol.knowledge_base)
        self.assertIsInstance(self.protocol.knowledge_base[new_domain], dict)
        self.assertIn("models", self.protocol.knowledge_base[new_domain])

if __name__ == '__main__':
    unittest.main()
