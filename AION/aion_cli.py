#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AION Protocol Command Line Interface
====================================

This module provides a command-line interface for the AION Protocol system.

Author: Francisco Molina
ORCID: https://orcid.org/0009-0008-6093-8267
Email: pako.molina@gmail.com
"""

import argparse
import json
import sys
import os
import logging
from typing import Dict, Any, Optional

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from aion_core import AIONProtocol

def setup_logging():
    """Configure logging for the CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="AION Protocol CLI")
    
    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to configuration file')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Initialize command
    init_parser = subparsers.add_parser('init', help='Initialize AION Protocol')
    
    # Execute protocol command
    exec_parser = subparsers.add_parser('execute', help='Execute an AION protocol')
    exec_parser.add_argument('--protocol', type=str, required=True,
                            help='Protocol to execute (ALPHA, BETA, GAMMA, DELTA, OMEGA)')
    exec_parser.add_argument('--params', type=str, required=True,
                            help='JSON string or file path with protocol parameters')
    
    # Bridge command
    bridge_parser = subparsers.add_parser('bridge', help='Create bridge with Obvivlorum')
    bridge_parser.add_argument('--action', type=str, required=True,
                              choices=['create', 'status', 'disconnect'],
                              help='Bridge action to perform')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Get AION Protocol status')
    
    # History command
    history_parser = subparsers.add_parser('history', help='View execution history')
    history_parser.add_argument('--limit', type=int, default=10,
                               help='Maximum number of history entries to show')
    
    # Performance command
    perf_parser = subparsers.add_parser('performance', help='Analyze performance metrics')
    
    # Help command
    help_parser = subparsers.add_parser('help', help='Show help information')
    help_parser.add_argument('--topic', type=str,
                            help='Specific topic to get help on')
    
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration settings
    """
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Configuration file not found: {config_path}")
        print("Using default configuration.")
        return {}
    except json.JSONDecodeError:
        print(f"Error parsing configuration file: {config_path}")
        print("Using default configuration.")
        return {}

def parse_params(params_arg: str) -> Dict[str, Any]:
    """
    Parse protocol parameters from string or file.
    
    Args:
        params_arg: JSON string or file path
        
    Returns:
        Dictionary containing parameters
    """
    # Check if it's a file path
    if os.path.exists(params_arg) and os.path.isfile(params_arg):
        try:
            with open(params_arg, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Error parsing parameters file: {params_arg}")
            sys.exit(1)
    
    # Try parsing as JSON string
    try:
        return json.loads(params_arg)
    except json.JSONDecodeError:
        print(f"Error parsing parameters JSON: {params_arg}")
        sys.exit(1)

def print_json(data: Dict[str, Any]):
    """Print data as formatted JSON."""
    print(json.dumps(data, indent=2))

def main():
    """Main CLI function."""
    setup_logging()
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize AION Protocol
    aion = AIONProtocol(args.config if os.path.exists(args.config) else None)
    
    # Process command
    if args.command == 'init':
        print(f"AION Protocol v{aion.VERSION} initialized")
        print(f"DNA: {aion.aion_dna}")
        
    elif args.command == 'execute':
        protocol = args.protocol.upper()
        params = parse_params(args.params)
        
        print(f"Executing {protocol} protocol...")
        result = aion.execute_protocol(protocol, params)
        print_json(result)
        
    elif args.command == 'bridge':
        if args.action == 'create':
            print("Creating bridge with Obvivlorum...")
            print("Error: Direct Obvivlorum integration requires programmatic access.")
            print("Use the AIONProtocol.create_bridge() method in your code.")
            
        elif args.action == 'status':
            print("Bridge status:")
            print("Error: Direct Obvivlorum integration requires programmatic access.")
            print("Use the bridge.get_bridge_status() method in your code.")
            
        elif args.action == 'disconnect':
            print("Disconnecting bridge...")
            print("Error: Direct Obvivlorum integration requires programmatic access.")
            print("Use the bridge.disconnect() method in your code.")
        
    elif args.command == 'status':
        print("AION Protocol Status:")
        status = {
            "version": aion.VERSION,
            "dna": aion.aion_dna,
            "active_protocol": aion.active_protocol,
            "loaded_protocols": list(aion.protocol_modules.keys()),
            "execution_count": len(aion.execution_history)
        }
        print_json(status)
        
    elif args.command == 'history':
        history = aion.get_execution_history(args.limit)
        print(f"Execution History (last {len(history)} entries):")
        print_json(history)
        
    elif args.command == 'performance':
        performance = aion.analyze_performance()
        print("Performance Analysis:")
        print_json(performance)
        
    elif args.command == 'help':
        if args.topic:
            if args.topic.upper() in aion.PROTOCOLS:
                protocol = args.topic.upper()
                info = aion.get_protocol_info(protocol)
                print(f"Help for {protocol} Protocol:")
                print_json(info)
            else:
                print(f"Unknown topic: {args.topic}")
        else:
            print("AION Protocol Help:")
            print("------------------")
            print("Available protocols:")
            for protocol in aion.PROTOCOLS:
                print(f"  - {protocol}")
            print("\nUse 'help --topic PROTOCOL' for specific protocol information.")
    
    else:
        print("No command specified. Use --help for available commands.")

if __name__ == "__main__":
    main()
