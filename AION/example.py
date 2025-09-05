#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AION Protocol - Example Usage
=============================

This example demonstrates how to use the AION Protocol with Obvivlorum.

Author: Francisco Molina
ORCID: https://orcid.org/0009-0008-6093-8267
Email: pako.molina@gmail.com
"""

import os
import sys
import json
import logging

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from aion_core import AIONProtocol

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AION.Example")

def main():
    """Main example function."""
    print("AION Protocol Example\n")
    
    # Initialize AION Protocol
    aion = AIONProtocol()
    print(f"AION Protocol v{aion.VERSION} initialized")
    print(f"DNA: {aion.aion_dna[:8]}...\n")
    
    # Example: Execute Protocol ALPHA (Scientific-Disruptive Development)
    print("Executing Protocol ALPHA...")
    
    alpha_params = {
        "research_domain": "artificial_intelligence",
        "research_type": "exploratory",
        "exploration_depth": 0.8,
        "novelty_bias": 0.7,
        "seed_concepts": [
            "quantum_neural_networks",
            "neuroplastic_operating_systems",
            "consciousness_emergence"
        ]
    }
    
    alpha_result = aion.execute_protocol("ALPHA", alpha_params)
    print(f"Result: {json.dumps(alpha_result, indent=2)}\n")
    
    # Example: Execute Protocol BETA (Mobile Applications)
    print("Executing Protocol BETA...")
    
    beta_params = {
        "app_name": "QuantumMind",
        "platforms": ["Android", "iOS", "HarmonyOS"],
        "features": [
            "AI-powered analysis",
            "Quantum data visualization",
            "Neural-symbolic reasoning"
        ],
        "architecture": "flutter",
        "performance_targets": {
            "response_time": "1ms",
            "memory_usage": "<50MB"
        }
    }
    
    beta_result = aion.execute_protocol("BETA", beta_params)
    print(f"Result: {json.dumps(beta_result, indent=2)}\n")
    
    # Show execution history
    print("Execution History:")
    history = aion.get_execution_history()
    print(f"- Executed {len(history)} protocols")
    for entry in history:
        print(f"  - {entry['protocol']}: {entry['status']}")
    
    print("\nExample completed successfully")

if __name__ == "__main__":
    main()
