#!/usr/bin/env python3
"""
Basic test to verify system imports and functionality.
This test ensures the CI/CD pipeline has something to validate.
"""

import sys
import os

def test_basic_imports():
    """Test that basic modules can be imported."""
    try:
        import json
        import logging
        import pathlib
        return True
    except ImportError as e:
        print(f"Import error: {e}")
        return False

def test_aion_directory():
    """Test that AION directory exists."""
    aion_path = os.path.join(os.path.dirname(__file__), "AION")
    return os.path.exists(aion_path)

def test_config_files():
    """Test that essential config files exist."""
    config_files = [
        "config_optimized.json",
        "AION/config.json"
    ]
    
    base_dir = os.path.dirname(__file__)
    for config_file in config_files:
        config_path = os.path.join(base_dir, config_file)
        if not os.path.exists(config_path):
            print(f"Missing config file: {config_file}")
            return False
    return True

def test_core_files():
    """Test that core system files exist."""
    core_files = [
        "ai_symbiote.py",
        "README.md",
        "AION/aion_core.py"
    ]
    
    base_dir = os.path.dirname(__file__)
    for core_file in core_files:
        file_path = os.path.join(base_dir, core_file)
        if not os.path.exists(file_path):
            print(f"Missing core file: {core_file}")
            return False
    return True

def main():
    """Run all basic tests."""
    tests = [
        ("Basic Imports", test_basic_imports),
        ("AION Directory", test_aion_directory),
        ("Config Files", test_config_files),
        ("Core Files", test_core_files)
    ]
    
    passed = 0
    total = len(tests)
    
    print("Running basic system tests...")
    print("-" * 40)
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            status = "PASS" if result else "FAIL"
            print(f"{test_name}: {status}")
            if result:
                passed += 1
        except Exception as e:
            print(f"{test_name}: ERROR - {e}")
    
    print("-" * 40)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("All basic tests passed!")
        return 0
    else:
        print("Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())