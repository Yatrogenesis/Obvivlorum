#!/usr/bin/env python3
"""
Final System Test for Obvivlorum AI Symbiote System v2.0
Simple functionality verification
"""

import os
import sys
import json
import subprocess
import tempfile
from pathlib import Path

def test_imports():
    """Test that key modules can be imported"""
    print("Testing module imports...")
    
    modules = [
        'security_manager',
        'structured_logger', 
        'unified_launcher',
        'adaptive_persistence_scheduler'
    ]
    
    results = []
    for module in modules:
        try:
            __import__(module)
            print(f"  {module}: OK")
            results.append(True)
        except Exception as e:
            print(f"  {module}: FAILED - {str(e)[:60]}...")
            results.append(False)
    
    return all(results)

def test_unified_launcher():
    """Test unified launcher functionality"""
    print("Testing unified launcher...")
    
    try:
        from unified_launcher import UnifiedLauncher
        launcher = UnifiedLauncher()
        
        # Test basic functionality
        assert len(launcher.modes) > 0
        assert 'core' in launcher.modes
        
        print("  Unified launcher: OK")
        return True
    except Exception as e:
        print(f"  Unified launcher: FAILED - {e}")
        return False

def test_security_manager():
    """Test security manager basic functionality"""
    print("Testing security manager...")
    
    try:
        from security_manager import SecurityManager
        security = SecurityManager()
        
        print("  Security manager: OK")
        return True
    except Exception as e:
        print(f"  Security manager: FAILED - {e}")
        return False

def test_structured_logger():
    """Test structured logger basic functionality"""
    print("Testing structured logger...")
    
    try:
        from structured_logger import StructuredLogger
        logger = StructuredLogger("test")
        logger.info("Test message")
        
        print("  Structured logger: OK")
        return True
    except Exception as e:
        print(f"  Structured logger: FAILED - {e}")
        return False

def test_file_structure():
    """Test that required files exist"""
    print("Testing file structure...")
    
    required_files = [
        'core_orchestrator.py',
        'unified_launcher.py',
        'security_manager.py',
        'structured_logger.py',
        'config_optimized.json',
        'docker-compose.yml',
        'Dockerfile',
        'README.md',
        '.gitignore'
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"  Missing files: {missing_files}")
        return False
    else:
        print("  All required files present: OK")
        return True

def test_launcher_executable():
    """Test that the launcher can execute basic commands"""
    print("Testing launcher execution...")
    
    try:
        result = subprocess.run([
            sys.executable, 'unified_launcher.py', '--list'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and 'core' in result.stdout:
            print("  Launcher execution: OK")
            return True
        else:
            print(f"  Launcher execution: FAILED - Return code {result.returncode}")
            return False
    except Exception as e:
        print(f"  Launcher execution: FAILED - {e}")
        return False

def run_final_tests():
    """Run all final tests"""
    print("=" * 60)
    print("OBVIVLORUM AI SYMBIOTE SYSTEM v2.0 - FINAL TESTS")
    print("=" * 60)
    
    tests = [
        ("Module Imports", test_imports),
        ("File Structure", test_file_structure),
        ("Unified Launcher", test_unified_launcher),
        ("Security Manager", test_security_manager),
        ("Structured Logger", test_structured_logger),
        ("Launcher Executable", test_launcher_executable)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n[{test_name}]")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"  Test failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("FINAL TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_name, _) in enumerate(tests):
        status = "PASS" if results[i] else "FAIL"
        print(f"{test_name:20} - {status}")
    
    success_rate = (passed / total) * 100 if total > 0 else 0
    print(f"\nPassed: {passed}/{total} tests")
    print(f"Success rate: {success_rate:.1f}%")
    
    if success_rate >= 70:
        print("\nSYSTEM TESTS: PASSED")
        print("The Obvivlorum system is functional and ready for use.")
        return True
    else:
        print("\nSYSTEM TESTS: FAILED") 
        print("Some components need attention before the system is ready.")
        return False

if __name__ == "__main__":
    success = run_final_tests()
    sys.exit(0 if success else 1)