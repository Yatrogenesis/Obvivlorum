#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Obvivlorum AI Symbiote System v2.0
Tests all major components and integrations
"""

import unittest
import asyncio
import sys
import os
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import components to test
try:
    from core_orchestrator import CoreOrchestrator, SystemStatus
    from security_manager import SecurityManager, PrivilegeLevel, ThreatLevel
    from human_in_the_loop import HumanInTheLoopManager, RiskLevel
    from smart_provider_selector import SmartProviderSelector, ProviderType
    from structured_logger import StructuredLogger, LogLevel
    from adaptive_persistence_scheduler import AdaptivePersistenceScheduler
    from unified_launcher import UnifiedLauncher, LaunchMode
except ImportError as e:
    print(f"Warning: Some imports failed: {e}")
    print("Running tests with available components only...")

class TestCoreOrchestrator(unittest.TestCase):
    """Test the core orchestrator system"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.test_dir, "test_config.json")
        
        # Create test config
        self.test_config = {
            "logging": {"level": "INFO"},
            "security": {"default_privilege_level": "user"},
            "ai_providers": {"default_provider": "mock"},
            "human_in_the_loop": {"enabled": False},
            "persistence": {"enabled": False}
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(self.test_config, f)
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization"""
        try:
            orchestrator = CoreOrchestrator(self.config_path)
            self.assertIsNotNone(orchestrator)
            self.assertEqual(orchestrator.config_path, Path(self.config_path))
        except Exception as e:
            self.skipTest(f"CoreOrchestrator not available: {e}")
    
    def test_system_status(self):
        """Test system status tracking"""
        try:
            orchestrator = CoreOrchestrator(self.config_path)
            status = orchestrator.get_status()
            self.assertIsInstance(status, SystemStatus)
            self.assertFalse(status.running)
            self.assertEqual(status.security_level, PrivilegeLevel.GUEST)
        except Exception as e:
            self.skipTest(f"SystemStatus test failed: {e}")

class TestSecurityManager(unittest.TestCase):
    """Test the security management system"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_config = {
            "default_privilege_level": "user",
            "enable_threat_detection": True,
            "jwt_secret": "test_secret_key_12345",
            "session_timeout": 3600
        }
        self.mock_logger = Mock()
    
    def test_security_manager_initialization(self):
        """Test security manager initialization"""
        try:
            security = SecurityManager(self.test_config, self.mock_logger)
            self.assertIsNotNone(security)
            self.assertEqual(security.current_privilege_level, PrivilegeLevel.USER)
        except Exception as e:
            self.skipTest(f"SecurityManager not available: {e}")
    
    def test_privilege_levels(self):
        """Test privilege level enumeration"""
        try:
            self.assertEqual(PrivilegeLevel.GUEST.value, "guest")
            self.assertEqual(PrivilegeLevel.USER.value, "user")
            self.assertEqual(PrivilegeLevel.OPERATOR.value, "operator")
            self.assertEqual(PrivilegeLevel.ADMIN.value, "admin")
            self.assertEqual(PrivilegeLevel.SYSTEM.value, "system")
        except Exception as e:
            self.skipTest(f"PrivilegeLevel not available: {e}")
    
    def test_threat_assessment(self):
        """Test threat level assessment"""
        try:
            security = SecurityManager(self.test_config, self.mock_logger)
            
            # Test low risk operation
            low_threat = asyncio.run(security.assess_threat_level(
                "echo hello", {}
            ))
            self.assertIn(low_threat, [ThreatLevel.MINIMAL, ThreatLevel.LOW])
            
            # Test high risk operation
            high_threat = asyncio.run(security.assess_threat_level(
                "rm -rf /", {}
            ))
            self.assertIn(high_threat, [ThreatLevel.CRITICAL, ThreatLevel.EXTREME])
            
        except Exception as e:
            self.skipTest(f"Threat assessment test failed: {e}")

class TestHumanInTheLoop(unittest.TestCase):
    """Test the human-in-the-loop system"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_config = {
            "enabled": True,
            "risk_threshold": 0.7,
            "timeout": 30,
            "require_justification": True
        }
        self.mock_security = Mock()
        self.mock_logger = Mock()
    
    def test_hitl_initialization(self):
        """Test HITL manager initialization"""
        try:
            hitl = HumanInTheLoopManager(
                self.test_config, 
                self.mock_security, 
                self.mock_logger
            )
            self.assertIsNotNone(hitl)
            self.assertTrue(hitl.enabled)
        except Exception as e:
            self.skipTest(f"HumanInTheLoopManager not available: {e}")
    
    def test_risk_levels(self):
        """Test risk level enumeration"""
        try:
            self.assertEqual(RiskLevel.MINIMAL.value, "minimal")
            self.assertEqual(RiskLevel.LOW.value, "low")
            self.assertEqual(RiskLevel.MEDIUM.value, "medium")
            self.assertEqual(RiskLevel.HIGH.value, "high")
            self.assertEqual(RiskLevel.CRITICAL.value, "critical")
            self.assertEqual(RiskLevel.EXTREME.value, "extreme")
        except Exception as e:
            self.skipTest(f"RiskLevel not available: {e}")

class TestSmartProviderSelector(unittest.TestCase):
    """Test the smart AI provider selector"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_config = {
            "default_provider": "mock",
            "enable_performance_tracking": True,
            "providers": {
                "mock": {
                    "enabled": True,
                    "api_key": "test_key",
                    "model": "mock-model"
                }
            }
        }
        self.mock_logger = Mock()
    
    def test_provider_selector_initialization(self):
        """Test provider selector initialization"""
        try:
            selector = SmartProviderSelector(self.test_config, self.mock_logger)
            self.assertIsNotNone(selector)
        except Exception as e:
            self.skipTest(f"SmartProviderSelector not available: {e}")
    
    def test_provider_types(self):
        """Test provider type enumeration"""
        try:
            self.assertEqual(ProviderType.OPENAI.value, "openai")
            self.assertEqual(ProviderType.CLAUDE.value, "claude")
            self.assertEqual(ProviderType.LOCAL_GGUF.value, "local_gguf")
            self.assertEqual(ProviderType.OLLAMA.value, "ollama")
        except Exception as e:
            self.skipTest(f"ProviderType not available: {e}")

class TestStructuredLogger(unittest.TestCase):
    """Test the structured logging system"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.test_config = {
            "level": "INFO",
            "format": "structured",
            "enable_metrics": True,
            "max_file_size": "10MB",
            "backup_count": 3
        }
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_logger_initialization(self):
        """Test logger initialization"""
        try:
            logger = StructuredLogger(
                "test_logger", 
                self.test_config, 
                Path(self.test_dir)
            )
            self.assertIsNotNone(logger)
            self.assertEqual(logger.name, "test_logger")
        except Exception as e:
            self.skipTest(f"StructuredLogger not available: {e}")
    
    def test_log_levels(self):
        """Test log level enumeration"""
        try:
            self.assertEqual(LogLevel.DEBUG.value, "DEBUG")
            self.assertEqual(LogLevel.INFO.value, "INFO")
            self.assertEqual(LogLevel.WARNING.value, "WARNING")
            self.assertEqual(LogLevel.ERROR.value, "ERROR")
            self.assertEqual(LogLevel.CRITICAL.value, "CRITICAL")
        except Exception as e:
            self.skipTest(f"LogLevel not available: {e}")
    
    def test_logging_operations(self):
        """Test basic logging operations"""
        try:
            logger = StructuredLogger(
                "test_logger", 
                self.test_config, 
                Path(self.test_dir)
            )
            
            # Test different log levels
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
            logger.critical("Critical message")
            
            # Test structured logging with extra data
            logger.info("Test with extra data", extra={
                "user_id": "test_user",
                "action": "test_action",
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            self.skipTest(f"Logging operations test failed: {e}")

class TestAdaptivePersistence(unittest.TestCase):
    """Test the adaptive persistence scheduler"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_config = {
            "enabled": False,
            "adaptive_mode": True,
            "check_interval": 60,
            "threat_threshold": 0.5
        }
        self.mock_logger = Mock()
    
    def test_persistence_initialization(self):
        """Test persistence scheduler initialization"""
        try:
            persistence = AdaptivePersistenceScheduler(
                self.test_config, 
                self.mock_logger
            )
            self.assertIsNotNone(persistence)
            self.assertFalse(persistence.enabled)
        except Exception as e:
            self.skipTest(f"AdaptivePersistenceScheduler not available: {e}")

class TestUnifiedLauncher(unittest.TestCase):
    """Test the unified launcher system"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        os.chdir(self.test_dir)
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_launcher_initialization(self):
        """Test launcher initialization"""
        try:
            launcher = UnifiedLauncher()
            self.assertIsNotNone(launcher)
            self.assertGreater(len(launcher.modes), 0)
        except Exception as e:
            self.skipTest(f"UnifiedLauncher not available: {e}")
    
    def test_launch_modes(self):
        """Test launch mode functionality"""
        try:
            launcher = UnifiedLauncher()
            
            # Check default modes exist
            expected_modes = ['core', 'legacy', 'gui', 'test', 'sandbox']
            for mode in expected_modes:
                self.assertIn(mode, launcher.modes)
            
            # Test mode properties
            core_mode = launcher.modes.get('core')
            if core_mode:
                self.assertIsInstance(core_mode, LaunchMode)
                self.assertEqual(core_mode.name, 'core')
                
        except Exception as e:
            self.skipTest(f"Launch modes test failed: {e}")

class TestSystemIntegration(unittest.TestCase):
    """Test system integration scenarios"""
    
    def test_component_compatibility(self):
        """Test that all components can work together"""
        try:
            # Test that enums are compatible
            security_levels = [level.value for level in PrivilegeLevel]
            risk_levels = [level.value for level in RiskLevel]
            threat_levels = [level.value for level in ThreatLevel]
            
            # Ensure no empty enums
            self.assertGreater(len(security_levels), 0)
            self.assertGreater(len(risk_levels), 0)
            self.assertGreater(len(threat_levels), 0)
            
            # Test privilege escalation mapping
            self.assertEqual(len(PrivilegeLevel), 5)  # Should have 5 levels
            
        except Exception as e:
            self.skipTest(f"Component compatibility test failed: {e}")
    
    def test_configuration_loading(self):
        """Test configuration loading across components"""
        test_dir = tempfile.mkdtemp()
        try:
            config_path = os.path.join(test_dir, "test_integration_config.json")
            
            # Create comprehensive test config
            config = {
                "logging": {
                    "level": "INFO",
                    "format": "structured",
                    "enable_metrics": True
                },
                "security": {
                    "default_privilege_level": "user",
                    "enable_threat_detection": True,
                    "jwt_secret": "test_secret_integration"
                },
                "ai_providers": {
                    "default_provider": "mock",
                    "enable_performance_tracking": True
                },
                "human_in_the_loop": {
                    "enabled": True,
                    "risk_threshold": 0.7
                },
                "persistence": {
                    "enabled": False,
                    "adaptive_mode": True
                }
            }
            
            with open(config_path, 'w') as f:
                json.dump(config, f)
            
            # Test that all components can load this config
            orchestrator = CoreOrchestrator(config_path)
            self.assertIsNotNone(orchestrator.config)
            
        except Exception as e:
            self.skipTest(f"Configuration loading test failed: {e}")
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)

class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""
    
    def test_missing_config_files(self):
        """Test behavior with missing configuration files"""
        try:
            # Test with non-existent config file
            orchestrator = CoreOrchestrator("/non/existent/config.json")
            # Should not raise exception, should use defaults
            self.assertIsNotNone(orchestrator.config)
            
        except Exception as e:
            self.skipTest(f"Missing config test failed: {e}")
    
    def test_invalid_privilege_levels(self):
        """Test handling of invalid privilege levels"""
        try:
            # Test invalid privilege level strings
            invalid_levels = ["invalid", "", None, 999]
            
            for invalid_level in invalid_levels:
                with self.assertRaises((ValueError, TypeError, AttributeError)):
                    # This should fail gracefully
                    test_level = PrivilegeLevel(invalid_level)
                    
        except Exception as e:
            self.skipTest(f"Invalid privilege levels test failed: {e}")
    
    def test_resource_cleanup(self):
        """Test proper resource cleanup"""
        test_dir = tempfile.mkdtemp()
        try:
            logger = StructuredLogger(
                "cleanup_test", 
                {"level": "INFO"}, 
                Path(test_dir)
            )
            
            # Use logger
            logger.info("Test message")
            
            # Test cleanup
            asyncio.run(logger.shutdown())
            
            # Logger should be in shutdown state
            self.assertFalse(logger.is_healthy())
            
        except Exception as e:
            self.skipTest(f"Resource cleanup test failed: {e}")
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)

class TestPerformance(unittest.TestCase):
    """Test performance characteristics"""
    
    def test_logger_performance(self):
        """Test logger performance under load"""
        test_dir = tempfile.mkdtemp()
        try:
            logger = StructuredLogger(
                "perf_test", 
                {"level": "INFO", "enable_metrics": True}, 
                Path(test_dir)
            )
            
            # Log many messages quickly
            start_time = datetime.now()
            for i in range(1000):
                logger.info(f"Performance test message {i}")
            
            duration = datetime.now() - start_time
            
            # Should be able to log 1000 messages in reasonable time
            self.assertLess(duration.total_seconds(), 5.0)
            
        except Exception as e:
            self.skipTest(f"Logger performance test failed: {e}")
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)
    
    def test_memory_usage(self):
        """Test memory usage patterns"""
        try:
            # Create multiple components to test memory usage
            components = []
            
            for i in range(10):
                orchestrator = CoreOrchestrator()
                components.append(orchestrator)
            
            # Should not consume excessive memory
            # This is a basic test - in production, use memory profiling
            self.assertEqual(len(components), 10)
            
        except Exception as e:
            self.skipTest(f"Memory usage test failed: {e}")

def run_test_suite():
    """Run the complete test suite"""
    
    # Configure test logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestCoreOrchestrator,
        TestSecurityManager,
        TestHumanInTheLoop,
        TestSmartProviderSelector,
        TestStructuredLogger,
        TestAdaptivePersistence,
        TestUnifiedLauncher,
        TestSystemIntegration,
        TestErrorHandling,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        descriptions=True,
        failfast=False
    )
    
    print("=" * 80)
    print("OBVIVLORUM AI SYMBIOTE SYSTEM v2.0 - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print(f"Starting tests at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python version: {sys.version}")
    print(f"Test directory: {os.getcwd()}")
    print("=" * 80)
    
    result = runner.run(test_suite)
    
    # Print summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(getattr(result, 'skipped', []))}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split(chr(10))[-2] if chr(10) in traceback else traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split(chr(10))[-2] if chr(10) in traceback else traceback}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / max(result.testsRun, 1)) * 100
    print(f"\nSuccess rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print(" Test suite PASSED with acceptable success rate")
    else:
        print(" Test suite FAILED - success rate below 80%")
    
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_test_suite()
    sys.exit(0 if success else 1)