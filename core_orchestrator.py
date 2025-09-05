#!/usr/bin/env python3
"""
Obvivlorum Core Orchestrator - Simplified Architecture
Unified system manager that integrates all components cleanly
"""

import asyncio
import sys
import os
import json
import signal
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import threading
import queue

# Import all our specialized components
from structured_logger import StructuredLogger
from security_manager import SecurityManager, PrivilegeLevel
from smart_provider_selector import SmartProviderSelector
from human_in_the_loop import HumanInTheLoop
from adaptive_persistence_scheduler import AdaptivePersistenceScheduler

@dataclass
class SystemStatus:
    """System status tracking"""
    running: bool = False
    sandbox_active: bool = False
    persistence_enabled: bool = False
    security_level: PrivilegeLevel = PrivilegeLevel.GUEST
    active_components: List[str] = None
    last_activity: datetime = None
    
    def __post_init__(self):
        if self.active_components is None:
            self.active_components = []
        if self.last_activity is None:
            self.last_activity = datetime.now()

class CoreOrchestrator:
    """
    Unified core orchestrator that manages all system components
    Simplified architecture with clean separation of concerns
    """
    
    def __init__(self, config_path: str = "config_optimized.json"):
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.status = SystemStatus()
        
        # Core components
        self.logger: Optional[StructuredLogger] = None
        self.security: Optional[SecurityManager] = None
        self.ai_selector: Optional[SmartProviderSelector] = None
        self.hitl: Optional[HumanInTheLoop] = None
        self.persistence: Optional[AdaptivePersistenceScheduler] = None
        
        # System control
        self._shutdown_event = asyncio.Event()
        self._command_queue = asyncio.Queue()
        self._component_tasks: Dict[str, asyncio.Task] = {}
        
        # Thread-safe status updates
        self._status_lock = threading.Lock()
        
    async def initialize(self) -> bool:
        """Initialize all system components"""
        try:
            # Load configuration
            await self._load_config()
            
            # Initialize logger first (needed by other components)
            self.logger = StructuredLogger(
                name="obvivlorum_core",
                config=self.config.get("logging", {}),
                base_dir=Path(".")
            )
            
            self.logger.info("Initializing Obvivlorum Core Orchestrator")
            
            # Initialize security manager
            self.security = SecurityManager(
                config=self.config.get("security", {}),
                logger=self.logger
            )
            
            # Initialize AI provider selector
            self.ai_selector = SmartProviderSelector(
                config=self.config.get("ai_providers", {}),
                logger=self.logger
            )
            
            # Initialize Human-in-the-Loop
            self.hitl = HumanInTheLoop(
                config=self.config.get("human_in_the_loop", {}),
                security_manager=self.security,
                logger=self.logger
            )
            
            # Initialize persistence scheduler
            self.persistence = AdaptivePersistenceScheduler(
                config=self.config.get("persistence", {}),
                logger=self.logger
            )
            
            # Update status
            with self._status_lock:
                self.status.running = True
                self.status.active_components = [
                    "logger", "security", "ai_selector", "hitl", "persistence"
                ]
            
            self.logger.info("Core orchestrator initialized successfully", extra={
                "components": self.status.active_components,
                "security_level": self.status.security_level.value
            })
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to initialize core orchestrator: {e}")
            else:
                print(f"CRITICAL: Failed to initialize core orchestrator: {e}")
            return False
    
    async def _load_config(self) -> None:
        """Load system configuration"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
            else:
                # Default configuration
                self.config = {
                    "logging": {
                        "level": "INFO",
                        "format": "structured",
                        "enable_metrics": True
                    },
                    "security": {
                        "default_privilege_level": "user",
                        "enable_threat_detection": True
                    },
                    "ai_providers": {
                        "default_provider": "smart_selection",
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
                
                # Save default config
                await self._save_config()
                
        except Exception as e:
            raise Exception(f"Failed to load configuration: {e}")
    
    async def _save_config(self) -> None:
        """Save current configuration"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to save configuration: {e}")
    
    async def start_system(self) -> None:
        """Start the complete system"""
        try:
            if not await self.initialize():
                raise Exception("System initialization failed")
            
            # Start component tasks
            self._component_tasks["command_processor"] = asyncio.create_task(
                self._command_processor()
            )
            self._component_tasks["status_monitor"] = asyncio.create_task(
                self._status_monitor()
            )
            self._component_tasks["persistence_manager"] = asyncio.create_task(
                self._persistence_manager()
            )
            
            self.logger.info("Obvivlorum system started successfully")
            
            # Setup signal handlers
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            # Wait for shutdown
            await self._shutdown_event.wait()
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"System startup failed: {e}")
            raise
    
    def _signal_handler(self, signum, frame):
        """Handle system signals"""
        if self.logger:
            self.logger.info(f"Received signal {signum}, initiating shutdown")
        asyncio.create_task(self.shutdown())
    
    async def shutdown(self) -> None:
        """Graceful system shutdown"""
        try:
            if self.logger:
                self.logger.info("Initiating system shutdown")
            
            # Set shutdown event
            self._shutdown_event.set()
            
            # Cancel component tasks
            for name, task in self._component_tasks.items():
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        if self.logger:
                            self.logger.debug(f"Component task {name} cancelled")
            
            # Shutdown components
            if self.persistence:
                await self.persistence.shutdown()
            
            if self.hitl:
                await self.hitl.shutdown()
                
            if self.ai_selector:
                await self.ai_selector.shutdown()
            
            if self.security:
                await self.security.shutdown()
            
            # Final log
            if self.logger:
                self.logger.info("System shutdown completed")
                await self.logger.shutdown()
            
            # Update status
            with self._status_lock:
                self.status.running = False
                self.status.active_components = []
                
        except Exception as e:
            print(f"Error during shutdown: {e}")
    
    async def _command_processor(self) -> None:
        """Process queued commands"""
        while not self._shutdown_event.is_set():
            try:
                # Get command with timeout
                command = await asyncio.wait_for(
                    self._command_queue.get(), timeout=1.0
                )
                
                # Process command
                await self._process_command(command)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Command processing error: {e}")
    
    async def _process_command(self, command: Dict[str, Any]) -> None:
        """Process a single command"""
        try:
            cmd_type = command.get("type")
            cmd_data = command.get("data", {})
            
            if cmd_type == "execute":
                await self._execute_command(cmd_data)
            elif cmd_type == "query_ai":
                await self._query_ai(cmd_data)
            elif cmd_type == "security_check":
                await self._security_check(cmd_data)
            elif cmd_type == "update_config":
                await self._update_config(cmd_data)
            else:
                if self.logger:
                    self.logger.warning(f"Unknown command type: {cmd_type}")
                    
        except Exception as e:
            if self.logger:
                self.logger.error(f"Command processing failed: {e}")
    
    async def _execute_command(self, cmd_data: Dict[str, Any]) -> None:
        """Execute system command through HITL"""
        if self.hitl and self.security:
            result = await self.hitl.request_approval(
                command=cmd_data.get("command", ""),
                context=cmd_data.get("context", {}),
                requester=cmd_data.get("user", "system")
            )
            
            if result.approved:
                # Execute through security manager
                await self.security.execute_privileged_operation(
                    operation=cmd_data.get("command"),
                    required_level=result.required_privilege_level,
                    context=cmd_data.get("context", {})
                )
    
    async def _query_ai(self, cmd_data: Dict[str, Any]) -> None:
        """Query AI through smart provider selector"""
        if self.ai_selector:
            response = await self.ai_selector.get_completion(
                prompt=cmd_data.get("prompt", ""),
                task_type=cmd_data.get("task_type", "general"),
                context=cmd_data.get("context", {})
            )
            
            # Log AI interaction
            if self.logger:
                self.logger.info("AI query processed", extra={
                    "provider": response.get("provider"),
                    "task_type": cmd_data.get("task_type"),
                    "response_length": len(str(response.get("content", "")))
                })
    
    async def _security_check(self, cmd_data: Dict[str, Any]) -> None:
        """Perform security check"""
        if self.security:
            threat_level = await self.security.assess_threat_level(
                operation=cmd_data.get("operation", ""),
                context=cmd_data.get("context", {})
            )
            
            if self.logger:
                self.logger.info("Security check completed", extra={
                    "operation": cmd_data.get("operation"),
                    "threat_level": threat_level
                })
    
    async def _update_config(self, cmd_data: Dict[str, Any]) -> None:
        """Update system configuration"""
        try:
            # Update config
            for key, value in cmd_data.items():
                if key in self.config:
                    self.config[key].update(value)
                else:
                    self.config[key] = value
            
            # Save config
            await self._save_config()
            
            if self.logger:
                self.logger.info("Configuration updated", extra={
                    "updated_keys": list(cmd_data.keys())
                })
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Config update failed: {e}")
    
    async def _status_monitor(self) -> None:
        """Monitor system status"""
        while not self._shutdown_event.is_set():
            try:
                # Update status
                with self._status_lock:
                    self.status.last_activity = datetime.now()
                
                # Health check components
                await self._health_check()
                
                # Wait before next check
                await asyncio.sleep(30)
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Status monitoring error: {e}")
    
    async def _health_check(self) -> None:
        """Perform health check on all components"""
        healthy_components = []
        
        try:
            # Check each component
            if self.logger and self.logger.is_healthy():
                healthy_components.append("logger")
                
            if self.security and await self.security.health_check():
                healthy_components.append("security")
                
            if self.ai_selector and await self.ai_selector.health_check():
                healthy_components.append("ai_selector")
                
            if self.hitl and await self.hitl.health_check():
                healthy_components.append("hitl")
                
            if self.persistence and await self.persistence.health_check():
                healthy_components.append("persistence")
            
            # Update status
            with self._status_lock:
                self.status.active_components = healthy_components
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Health check failed: {e}")
    
    async def _persistence_manager(self) -> None:
        """Manage persistence operations"""
        while not self._shutdown_event.is_set():
            try:
                if self.persistence:
                    # Check if persistence should be active
                    should_persist = await self.persistence.should_activate()
                    
                    with self._status_lock:
                        self.status.persistence_enabled = should_persist
                    
                    if should_persist:
                        await self.persistence.activate_persistence()
                    else:
                        await self.persistence.deactivate_persistence()
                
                # Wait before next check
                await asyncio.sleep(60)
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Persistence management error: {e}")
    
    # Public API methods
    async def queue_command(self, command: Dict[str, Any]) -> None:
        """Queue a command for processing"""
        await self._command_queue.put(command)
    
    def get_status(self) -> SystemStatus:
        """Get current system status"""
        with self._status_lock:
            return SystemStatus(
                running=self.status.running,
                sandbox_active=self.status.sandbox_active,
                persistence_enabled=self.status.persistence_enabled,
                security_level=self.status.security_level,
                active_components=self.status.active_components.copy(),
                last_activity=self.status.last_activity
            )
    
    async def set_security_level(self, level: PrivilegeLevel) -> bool:
        """Set system security level"""
        if self.security:
            success = await self.security.set_privilege_level(level)
            if success:
                with self._status_lock:
                    self.status.security_level = level
                    
                if self.logger:
                    self.logger.info(f"Security level updated to {level.value}")
            return success
        return False

# Main execution
async def main():
    """Main entry point"""
    orchestrator = CoreOrchestrator()
    
    try:
        await orchestrator.start_system()
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"System error: {e}")
    finally:
        await orchestrator.shutdown()

if __name__ == "__main__":
    asyncio.run(main())