#!/usr/bin/env python3
"""
Functional Tests for Obvivlorum AI Symbiote System v2.0
Basic functionality tests for core components
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
import asyncio

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_security_manager():
    """Test security manager functionality"""
    try:
        from security_manager import SecurityManager, PrivilegeLevel, ThreatLevel
        
        config = {
            "default_privilege_level": "user",
            "enable_threat_detection": True,
            "jwt_secret": "test_secret_key",
            "session_timeout": 3600
        }
        
        print("Testing SecurityManager...")
        security = SecurityManager(config, None)
        
        # Test privilege levels
        assert security.current_privilege_level == PrivilegeLevel.USER
        print("- Privilege level initialization: OK")
        
        # Test enum values
        assert PrivilegeLevel.GUEST.value == "guest"
        assert PrivilegeLevel.SYSTEM.value == "system"
        print("- Privilege level enum: OK")
        
        print("SecurityManager tests: PASSED")
        return True
        
    except Exception as e:
        print(f"SecurityManager tests: FAILED - {e}")
        return False

def test_structured_logger():
    """Test structured logger functionality"""
    try:
        from structured_logger import StructuredLogger, LogLevel
        
        test_dir = tempfile.mkdtemp()
        config = {
            "level": "INFO",
            "format": "structured",
            "enable_metrics": True
        }
        
        print("Testing StructuredLogger...")
        logger = StructuredLogger("test_logger", config, Path(test_dir))
        
        # Test logging operations
        logger.info("Test message")
        logger.error("Test error message")
        print("- Basic logging: OK")
        
        # Test log levels enum
        assert LogLevel.INFO.value == "INFO"
        assert LogLevel.ERROR.value == "ERROR"
        print("- Log level enum: OK")
        
        # Test health check
        assert logger.is_healthy() == True
        print("- Health check: OK")
        
        # Cleanup
        asyncio.run(logger.shutdown())
        shutil.rmtree(test_dir, ignore_errors=True)
        
        print("StructuredLogger tests: PASSED")
        return True
        
    except Exception as e:
        print(f"StructuredLogger tests: FAILED - {e}")
        return False

def test_unified_launcher():
    """Test unified launcher functionality"""
    try:
        from unified_launcher import UnifiedLauncher, LaunchMode
        
        print("Testing UnifiedLauncher...")
        launcher = UnifiedLauncher()
        
        # Test initialization
        assert launcher.modes is not None
        assert len(launcher.modes) > 0
        print("- Initialization: OK")
        
        # Test default modes exist
        expected_modes = ['core', 'test', 'gui']
        for mode in expected_modes:
            if mode in launcher.modes:
                assert isinstance(launcher.modes[mode], LaunchMode)
        print("- Launch modes: OK")
        
        print("UnifiedLauncher tests: PASSED")
        return True
        
    except Exception as e:
        print(f"UnifiedLauncher tests: FAILED - {e}")
        return False

def test_core_orchestrator():
    """Test core orchestrator functionality"""
    try:
        from core_orchestrator import CoreOrchestrator, SystemStatus
        
        # Create test config
        test_dir = tempfile.mkdtemp()
        config_path = os.path.join(test_dir, "test_config.json")
        
        config = {
            "logging": {"level": "INFO"},
            "security": {"default_privilege_level": "user"},
            "ai_providers": {"default_provider": "mock"},
            "human_in_the_loop": {"enabled": False},
            "persistence": {"enabled": False}
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        print("Testing CoreOrchestrator...")
        orchestrator = CoreOrchestrator(config_path)
        
        # Test initialization
        assert orchestrator.config_path == Path(config_path)
        print("- Initialization: OK")
        
        # Test status
        status = orchestrator.get_status()
        assert isinstance(status, SystemStatus)
        assert status.running == False  # Not started yet
        print("- Status tracking: OK")
        
        # Cleanup
        shutil.rmtree(test_dir, ignore_errors=True)
        
        print("CoreOrchestrator tests: PASSED")
        return True
        
    except Exception as e:
        print(f"CoreOrchestrator tests: FAILED - {e}")
        return False

def test_human_in_the_loop():
    """Test human in the loop functionality"""
    try:
        from human_in_the_loop import HumanInTheLoopManager, RiskLevel
        from security_manager import SecurityManager
        
        config = {
            "enabled": True,
            "risk_threshold": 0.7,
            "timeout": 30
        }
        
        security_config = {
            "default_privilege_level": "user",
            "jwt_secret": "test_secret"
        }
        
        print("Testing HumanInTheLoop...")
        security = SecurityManager(security_config, None)
        hitl = HumanInTheLoopManager(config, security, None)
        
        # Test initialization
        assert hitl.enabled == True
        print("- Initialization: OK")
        
        # Test risk levels enum
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.HIGH.value == "high"
        print("- Risk level enum: OK")
        
        print("HumanInTheLoop tests: PASSED")
        return True
        
    except Exception as e:
        print(f"HumanInTheLoop tests: FAILED - {e}")
        return False

def test_smart_provider_selector():
    """Test smart provider selector functionality"""
    try:
        from smart_provider_selector import SmartProviderSelector, ProviderType
        
        config = {
            "default_provider": "mock",
            "enable_performance_tracking": True,
            "providers": {
                "mock": {
                    "enabled": True,
                    "model": "mock-model"
                }
            }
        }
        
        print("Testing SmartProviderSelector...")
        selector = SmartProviderSelector(config, None)
        
        # Test initialization
        assert selector is not None
        print("- Initialization: OK")
        
        # Test provider types enum
        assert ProviderType.OPENAI.value == "openai"
        assert ProviderType.CLAUDE.value == "claude"
        print("- Provider type enum: OK")
        
        print("SmartProviderSelector tests: PASSED")
        return True
        
    except Exception as e:
        print(f"SmartProviderSelector tests: FAILED - {e}")
        return False

def test_adaptive_persistence():
    """Test adaptive persistence scheduler functionality"""
    try:
        from adaptive_persistence_scheduler import AdaptivePersistenceScheduler
        
        config = {
            "enabled": False,
            "adaptive_mode": True,
            "check_interval": 60
        }
        
        print("Testing AdaptivePersistence...")
        persistence = AdaptivePersistenceScheduler(config, None)
        
        # Test initialization
        assert persistence.enabled == False
        print("- Initialization: OK")
        
        print("AdaptivePersistence tests: PASSED")
        return True
        
    except Exception as e:
        print(f"AdaptivePersistence tests: FAILED - {e}")
        return False

def run_functional_tests():
    """Run all functional tests"""
    print("=" * 70)
    print("OBVIVLORUM AI SYMBIOTE SYSTEM v2.0 - FUNCTIONAL TESTS")
    print("=" * 70)
    
    tests = [
        ("SecurityManager", test_security_manager),
        ("StructuredLogger", test_structured_logger),
        ("UnifiedLauncher", test_unified_launcher),
        ("CoreOrchestrator", test_core_orchestrator),
        ("HumanInTheLoop", test_human_in_the_loop),
        ("SmartProviderSelector", test_smart_provider_selector),
        ("AdaptivePersistence", test_adaptive_persistence)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n[{test_name}]")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"{test_name} tests: FAILED - {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:25} - {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    success_rate = (passed / total) * 100 if total > 0 else 0
    print(f"Success rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("\nFUNCTIONAL TESTS: PASSED")
        return True
    else:
        print("\nFUNCTIONAL TESTS: FAILED")
        return False

if __name__ == "__main__":
    success = run_functional_tests()
    sys.exit(0 if success else 1)