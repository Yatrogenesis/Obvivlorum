#!/usr/bin/env python3
"""
Unit tests for core Obvivlorum components
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
import tempfile
import os

# Import core components
from security_manager import SecurityManager
from structured_logger import StructuredLogger
from unified_launcher import UnifiedLauncher


class TestSecurityManager:
    """Test suite for SecurityManager component"""
    
    @pytest.fixture
    def security_manager(self):
        """Create SecurityManager instance for testing"""
        return SecurityManager()
    
    def test_initialization(self, security_manager):
        """Test SecurityManager initialization"""
        assert security_manager is not None
        
    @pytest.mark.asyncio
    async def test_threat_assessment_basic(self, security_manager):
        """Test basic threat assessment functionality"""
        # Test safe command
        safe_threat = await security_manager.assess_threat_level("echo hello", {})
        assert safe_threat is not None
        
        # Test dangerous command
        dangerous_threat = await security_manager.assess_threat_level("rm -rf /", {})
        assert dangerous_threat is not None


class TestStructuredLogger:
    """Test suite for StructuredLogger component"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def logger(self, temp_dir):
        """Create StructuredLogger instance for testing"""
        config = {"level": "DEBUG"}
        return StructuredLogger("test_logger", config)
    
    def test_initialization(self, logger):
        """Test logger initialization"""
        assert logger is not None
        assert logger.name == "test_logger"
    
    def test_basic_logging(self, logger):
        """Test basic logging functionality"""
        logger.info("Test message")
        logger.warning("Test warning", extra={"key": "value"})
        logger.error("Test error")
        
        # Verify logger is healthy
        assert logger.is_healthy()
    
    def test_structured_logging(self, logger):
        """Test structured logging with extra data"""
        extra_data = {
            "user_id": "test_user",
            "action": "test_action",
            "timestamp": "2024-01-01T00:00:00Z"
        }
        logger.info("Structured test message", extra=extra_data)
    
    @pytest.mark.asyncio
    async def test_logger_shutdown(self, logger):
        """Test logger shutdown process"""
        await logger.shutdown()
        assert not logger.is_healthy()


class TestUnifiedLauncher:
    """Test suite for UnifiedLauncher component"""
    
    @pytest.fixture
    def launcher(self):
        """Create UnifiedLauncher instance for testing"""
        return UnifiedLauncher()
    
    def test_initialization(self, launcher):
        """Test launcher initialization"""
        assert launcher is not None
        assert len(launcher.modes) > 0
    
    def test_default_modes_exist(self, launcher):
        """Test that expected default modes exist"""
        expected_modes = ['core', 'test', 'gui']
        for mode in expected_modes:
            assert mode in launcher.modes
    
    def test_launch_mode_validation(self, launcher):
        """Test launch mode validation"""
        # Test valid mode
        assert 'core' in launcher.modes
        
        # Test invalid mode
        assert 'invalid_mode_123' not in launcher.modes


class TestSystemIntegration:
    """Integration tests between components"""
    
    @pytest.mark.asyncio
    async def test_logger_security_integration(self):
        """Test integration between logger and security manager"""
        logger = StructuredLogger("security_test")
        security = SecurityManager()
        
        # Test that security manager can use logger
        threat_level = await security.assess_threat_level("test command", {})
        logger.info("Threat assessment completed", extra={
            "threat_level": str(threat_level),
            "command": "test command"
        })
        
        # Cleanup
        await logger.shutdown()


@pytest.mark.benchmark
class TestPerformance:
    """Performance benchmarks for core components"""
    
    def test_security_manager_performance(self, benchmark):
        """Benchmark security manager threat assessment"""
        security = SecurityManager()
        
        async def assess_threat():
            return await security.assess_threat_level("echo test", {})
        
        # Note: This is a simplified benchmark
        # In practice, you'd use pytest-benchmark with async support
        result = benchmark(lambda: asyncio.run(assess_threat()))
        assert result is not None
    
    def test_logger_performance(self, benchmark):
        """Benchmark logger performance"""
        logger = StructuredLogger("perf_test")
        
        def log_messages():
            for i in range(100):
                logger.info(f"Performance test message {i}")
        
        benchmark(log_messages)
        
        # Cleanup
        asyncio.run(logger.shutdown())


# Test fixtures and utilities
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Test configuration
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "benchmark: mark test as a benchmark")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")