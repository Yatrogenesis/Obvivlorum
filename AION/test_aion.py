#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AION Protocol - Initialize and Test
==================================

This script initializes the AION Protocol system and tests its basic functionality.

Author: Francisco Molina
ORCID: https://orcid.org/0009-0008-6093-8267
Email: pako.molina@gmail.com
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from aion_core import AIONProtocol

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AION.Test")

def initialize_aion() -> AIONProtocol:
    """Initialize the AION Protocol system."""
    logger.info("Initializing AION Protocol...")
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    
    # Initialize AION Protocol
    aion = AIONProtocol(config_path)
    
    logger.info(f"AION Protocol v{aion.VERSION} initialized")
    logger.info(f"DNA: {aion.aion_dna[:8]}...")
    
    return aion

def test_protocols(aion: AIONProtocol):
    """Test all protocols with basic parameters."""
    logger.info("Testing all protocols...")
    
    # Test ALPHA protocol
    logger.info("Testing ALPHA protocol...")
    alpha_params = {
        "research_domain": "artificial_intelligence",
        "research_type": "exploratory",
        "seed_concepts": ["quantum_consciousness", "neuroplastic_systems"]
    }
    
    alpha_result = aion.execute_protocol("ALPHA", alpha_params)
    logger.info(f"ALPHA result status: {alpha_result.get('status')}")
    
    # Test BETA protocol
    logger.info("Testing BETA protocol...")
    beta_params = {
        "app_name": "QuantumMind",
        "architecture": "flutter",
        "platforms": ["Android", "iOS"]
    }
    
    beta_result = aion.execute_protocol("BETA", beta_params)
    logger.info(f"BETA result status: {beta_result.get('status')}")
    
    # Test GAMMA protocol
    logger.info("Testing GAMMA protocol...")
    gamma_params = {
        "project_name": "EnterpriseAI",
        "architecture": "microservices"
    }
    
    gamma_result = aion.execute_protocol("GAMMA", gamma_params)
    logger.info(f"GAMMA result status: {gamma_result.get('status')}")
    
    # Test DELTA protocol
    logger.info("Testing DELTA protocol...")
    delta_params = {
        "app_name": "AI Dashboard",
        "tech_stack": "next_js"
    }
    
    delta_result = aion.execute_protocol("DELTA", delta_params)
    logger.info(f"DELTA result status: {delta_result.get('status')}")
    
    # Test OMEGA protocol
    logger.info("Testing OMEGA protocol...")
    omega_params = {
        "project_name": "AI License Manager",
        "action": "generate_license",
        "license_type": "COMMERCIAL_BASIC",
        "software_name": "QuantumMind Pro"
    }
    
    omega_result = aion.execute_protocol("OMEGA", omega_params)
    logger.info(f"OMEGA result status: {omega_result.get('status')}")
    
    # Get execution history
    history = aion.get_execution_history()
    logger.info(f"Executed {len(history)} protocols")

def save_state(aion: AIONProtocol):
    """Save AION Protocol state."""
    state_path = os.path.join(os.path.dirname(__file__), "aion_state.json")
    
    result = aion.save_state(state_path)
    logger.info(f"State saved: {result}")
    
    return state_path

def main():
    """Main function."""
    print("AION Protocol Test\n")
    
    # Initialize AION Protocol
    aion = initialize_aion()
    
    # Test protocols
    test_protocols(aion)
    
    # Save state
    state_path = save_state(aion)
    
    print("\nTest completed successfully!")
    print(f"State saved to: {state_path}")

if __name__ == "__main__":
    main()
