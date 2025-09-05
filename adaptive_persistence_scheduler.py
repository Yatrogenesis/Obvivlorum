#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Adaptive Persistence Scheduler for AI Symbiote
===============================================

Scheduler adaptativo que habilita/deshabilita persistencia basado en:
1. Estabilidad del sistema (logs de arranque)
2. Entorno de operación (lab/producción) 
3. Nivel de amenaza detectado
4. Interruptores de control manual

Author: Francisco Molina
ORCID: https://orcid.org/0009-0008-6093-8267
Email: pako.molina@gmail.com
"""

import os
import sys
import json
import time
import logging
import hashlib
import psutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger("AdaptivePersistenceScheduler")

class EnvironmentType(Enum):
    LABORATORY = "laboratory"
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    UNKNOWN = "unknown"

class SystemStability(Enum):
    STABLE = "stable"
    UNSTABLE = "unstable"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

class PersistenceMode(Enum):
    DISABLED = "disabled"
    SAFE = "safe"
    ADAPTIVE = "adaptive"
    AGGRESSIVE = "aggressive"
    FORCE_ENABLED = "force_enabled"

class AdaptivePersistenceScheduler:
    """
    Scheduler inteligente para control de persistencia adaptativa.
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the Adaptive Persistence Scheduler."""
        self.config_path = config_path or "adaptive_persistence_config.json"
        self.config = self._load_or_create_config()
        self.is_active = False
        self.last_stability_check = 0
        self.stability_check_interval = 300  # 5 minutes
        
        # Control flags
        self.force_disable_flag = Path("DISABLE_PERSISTENCE.flag")
        self.force_enable_flag = Path("FORCE_ENABLE_PERSISTENCE.flag")
        self.lab_mode_flag = Path("LAB_MODE.flag")
        
        # Stability metrics
        self.boot_stability_score = 0.0
        self.runtime_stability_score = 0.0
        self.threat_level_impact = 0.0
        
        logger.info("Adaptive Persistence Scheduler initialized")
    
    def _load_or_create_config(self) -> Dict[str, Any]:
        """Load or create scheduler configuration."""
        default_config = {
            "scheduler_enabled": True,
            "stability_thresholds": {
                "boot_success_rate": 0.85,  # 85% boot success
                "runtime_uptime": 300,      # 5 minutes stable runtime
                "max_crashes_per_hour": 2   # Max 2 crashes per hour
            },
            "persistence_rules": {
                "laboratory": {
                    "allow_aggressive": True,
                    "force_enable_override": True,
                    "stability_requirement": 0.5
                },
                "development": {
                    "allow_aggressive": False,
                    "force_enable_override": True,
                    "stability_requirement": 0.7
                },
                "production": {
                    "allow_aggressive": False,
                    "force_enable_override": False,
                    "stability_requirement": 0.9
                }
            },
            "threat_level_rules": {
                "SAFE": {"persistence_multiplier": 1.0},
                "LOW": {"persistence_multiplier": 0.8},
                "MEDIUM": {"persistence_multiplier": 0.5},
                "HIGH": {"persistence_multiplier": 0.2},
                "CRITICAL": {"persistence_multiplier": 0.0}
            }
        }
        
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, indent=2)
                return default_config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return default_config
    
    def detect_environment(self) -> EnvironmentType:
        """Detect the current operating environment."""
        try:
            # Check for environment flags
            if self.lab_mode_flag.exists():
                return EnvironmentType.LABORATORY
            
            # Check for development indicators
            dev_indicators = [
                Path("DEBUG_MODE.flag"),
                Path("DEVELOPMENT.flag"),
                Path("dev_config.json")
            ]
            
            if any(flag.exists() for flag in dev_indicators):
                return EnvironmentType.DEVELOPMENT
            
            # Check for production indicators
            prod_indicators = [
                Path("PRODUCTION.flag"),
                Path("production_config.json")
            ]
            
            if any(flag.exists() for flag in prod_indicators):
                return EnvironmentType.PRODUCTION
            
            # Check system characteristics
            if os.name == 'nt':  # Windows
                # Check if running in virtual machine
                try:
                    import wmi
                    c = wmi.WMI()
                    for system in c.Win32_ComputerSystem():
                        if 'virtual' in system.Model.lower():
                            return EnvironmentType.LABORATORY
                except ImportError:
                    pass
            
            # Default based on common development paths
            current_path = str(Path.cwd()).lower()
            if any(path in current_path for path in ['dev', 'test', 'debug', 'lab']):
                return EnvironmentType.DEVELOPMENT
            
            return EnvironmentType.UNKNOWN
            
        except Exception as e:
            logger.error(f"Error detecting environment: {e}")
            return EnvironmentType.UNKNOWN
    
    def assess_system_stability(self) -> SystemStability:
        """Assess current system stability."""
        try:
            stability_score = 0.0
            max_score = 0.0
            
            # Boot stability (check logs for successful starts)
            boot_score = self._assess_boot_stability()
            stability_score += boot_score * 0.4
            max_score += 0.4
            
            # Runtime stability (uptime, memory usage, crashes)
            runtime_score = self._assess_runtime_stability()
            stability_score += runtime_score * 0.4
            max_score += 0.4
            
            # System health (CPU, memory, disk)
            health_score = self._assess_system_health()
            stability_score += health_score * 0.2
            max_score += 0.2
            
            # Calculate final stability score
            final_score = stability_score / max_score if max_score > 0 else 0.0
            self.boot_stability_score = boot_score
            self.runtime_stability_score = runtime_score
            
            # Determine stability level
            if final_score >= 0.8:
                return SystemStability.STABLE
            elif final_score >= 0.6:
                return SystemStability.UNSTABLE
            else:
                return SystemStability.CRITICAL
                
        except Exception as e:
            logger.error(f"Error assessing stability: {e}")
            return SystemStability.UNKNOWN
    
    def _assess_boot_stability(self) -> float:
        """Assess boot stability from logs."""
        try:
            log_files = [
                "ai_symbiote.log",
                "aion_protocol.log"
            ]
            
            successful_boots = 0
            total_boot_attempts = 0
            
            for log_file in log_files:
                if Path(log_file).exists():
                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        
                        # Count successful initializations
                        successful_boots += content.count("system initialized successfully")
                        successful_boots += content.count("AI Symbiote main loop started")
                        
                        # Count total boot attempts
                        total_boot_attempts += content.count("Initializing AI Symbiote system")
            
            if total_boot_attempts == 0:
                return 0.5  # No data, assume neutral
                
            return min(1.0, successful_boots / total_boot_attempts)
            
        except Exception as e:
            logger.error(f"Error assessing boot stability: {e}")
            return 0.0
    
    def _assess_runtime_stability(self) -> float:
        """Assess runtime stability."""
        try:
            stability_factors = []
            
            # System uptime
            try:
                uptime_seconds = time.time() - psutil.boot_time()
                uptime_score = min(1.0, uptime_seconds / 3600)  # Max score at 1 hour
                stability_factors.append(uptime_score)
            except:
                stability_factors.append(0.5)
            
            # Memory usage stability
            try:
                memory = psutil.virtual_memory()
                memory_score = 1.0 - (memory.percent / 100)  # Lower usage = more stable
                stability_factors.append(memory_score)
            except:
                stability_factors.append(0.5)
            
            # CPU usage stability
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                cpu_score = 1.0 - (cpu_percent / 100)
                stability_factors.append(cpu_score)
            except:
                stability_factors.append(0.5)
            
            return sum(stability_factors) / len(stability_factors) if stability_factors else 0.0
            
        except Exception as e:
            logger.error(f"Error assessing runtime stability: {e}")
            return 0.0
    
    def _assess_system_health(self) -> float:
        """Assess overall system health."""
        try:
            health_factors = []
            
            # Disk usage
            try:
                disk = psutil.disk_usage('/')
                disk_score = 1.0 - (disk.percent / 100)
                health_factors.append(disk_score)
            except:
                health_factors.append(0.5)
            
            # Process count (too many processes = less stable)
            try:
                process_count = len(psutil.pids())
                process_score = max(0.0, 1.0 - (process_count / 500))  # Assume 500 as high
                health_factors.append(process_score)
            except:
                health_factors.append(0.5)
            
            return sum(health_factors) / len(health_factors) if health_factors else 0.0
            
        except Exception as e:
            logger.error(f"Error assessing system health: {e}")
            return 0.0
    
    def get_threat_level(self) -> str:
        """Get current threat level from security module."""
        try:
            # Try to get from adaptive persistence controller
            from adaptive_persistence_controller import AdaptivePersistenceController
            controller = AdaptivePersistenceController("dummy_path")
            return controller.current_threat_level.name
        except:
            try:
                # Try to get from security protection module
                from security_protection_module import SecurityProtectionModule
                security = SecurityProtectionModule()
                return security.threat_level if hasattr(security, 'threat_level') else "SAFE"
            except:
                return "SAFE"  # Default to safe if no security module
    
    def calculate_persistence_recommendation(self) -> Dict[str, Any]:
        """Calculate persistence recommendation based on all factors."""
        try:
            # Check force flags first
            if self.force_disable_flag.exists():
                return {
                    "mode": PersistenceMode.DISABLED,
                    "reason": "Force disable flag detected",
                    "confidence": 1.0,
                    "safe_to_enable": False
                }
            
            if self.force_enable_flag.exists():
                return {
                    "mode": PersistenceMode.FORCE_ENABLED,
                    "reason": "Force enable flag detected",
                    "confidence": 1.0,
                    "safe_to_enable": True
                }
            
            # Get environmental factors
            environment = self.detect_environment()
            stability = self.assess_system_stability()
            threat_level = self.get_threat_level()
            
            # Calculate scores
            env_rules = self.config["persistence_rules"].get(environment.value, 
                                                           self.config["persistence_rules"]["production"])
            stability_requirement = env_rules["stability_requirement"]
            threat_multiplier = self.config["threat_level_rules"].get(threat_level, 
                                                                     {"persistence_multiplier": 0.5})["persistence_multiplier"]
            
            # Calculate final score
            stability_score = min(1.0, (self.boot_stability_score + self.runtime_stability_score) / 2)
            final_score = stability_score * threat_multiplier
            
            # Determine mode
            if final_score >= stability_requirement:
                if environment == EnvironmentType.LABORATORY and env_rules["allow_aggressive"]:
                    mode = PersistenceMode.AGGRESSIVE
                else:
                    mode = PersistenceMode.ADAPTIVE
            elif final_score >= 0.5:
                mode = PersistenceMode.SAFE
            else:
                mode = PersistenceMode.DISABLED
            
            return {
                "mode": mode,
                "reason": f"Environment: {environment.value}, Stability: {stability.value}, Threat: {threat_level}",
                "confidence": final_score,
                "safe_to_enable": final_score >= 0.5,
                "factors": {
                    "environment": environment.value,
                    "stability": stability.value,
                    "threat_level": threat_level,
                    "stability_score": stability_score,
                    "threat_multiplier": threat_multiplier,
                    "final_score": final_score
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating persistence recommendation: {e}")
            return {
                "mode": PersistenceMode.DISABLED,
                "reason": f"Error in calculation: {e}",
                "confidence": 0.0,
                "safe_to_enable": False
            }
    
    def apply_persistence_configuration(self, recommendation: Dict[str, Any]) -> bool:
        """Apply the persistence configuration based on recommendation."""
        try:
            mode = recommendation["mode"]
            
            # Import persistence manager
            from windows_persistence import WindowsPersistenceManager
            persistence_manager = WindowsPersistenceManager("dummy_path")
            
            if mode == PersistenceMode.DISABLED:
                # Disable all persistence methods
                self._disable_persistence(persistence_manager)
                logger.info("Persistence disabled by scheduler")
                return True
                
            elif mode in [PersistenceMode.SAFE, PersistenceMode.ADAPTIVE]:
                # Enable safe persistence methods only
                self._enable_safe_persistence(persistence_manager)
                logger.info(f"Safe persistence enabled - Mode: {mode.value}")
                return True
                
            elif mode == PersistenceMode.AGGRESSIVE:
                # Enable all persistence methods
                self._enable_aggressive_persistence(persistence_manager)
                logger.info("Aggressive persistence enabled")
                return True
                
            elif mode == PersistenceMode.FORCE_ENABLED:
                # Force enable regardless of safety
                self._enable_aggressive_persistence(persistence_manager)
                logger.info("Persistence force enabled")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error applying persistence configuration: {e}")
            return False
    
    def _disable_persistence(self, persistence_manager):
        """Disable all persistence methods."""
        try:
            persistence_manager.uninstall_persistence()
        except Exception as e:
            logger.error(f"Error disabling persistence: {e}")
    
    def _enable_safe_persistence(self, persistence_manager):
        """Enable only safe persistence methods."""
        try:
            safe_config = {
                "registry_persistence": True,  # Safe method
                "startup_folder": False,       # Can be detected
                "scheduled_task": False,       # Requires admin
                "service_mode": False,         # Too aggressive
                "stealth_mode": False,         # Not needed
                "self_healing": False          # Might cause loops
            }
            persistence_manager.config.update(safe_config)
            persistence_manager.install_persistence()
        except Exception as e:
            logger.error(f"Error enabling safe persistence: {e}")
    
    def _enable_aggressive_persistence(self, persistence_manager):
        """Enable all persistence methods."""
        try:
            aggressive_config = {
                "registry_persistence": True,
                "startup_folder": True,
                "scheduled_task": True,
                "service_mode": False,  # Still avoid service mode
                "stealth_mode": True,
                "self_healing": True
            }
            persistence_manager.config.update(aggressive_config)
            persistence_manager.install_persistence()
        except Exception as e:
            logger.error(f"Error enabling aggressive persistence: {e}")
    
    def run_assessment(self) -> Dict[str, Any]:
        """Run complete assessment and return recommendation."""
        logger.info("Running adaptive persistence assessment...")
        
        recommendation = self.calculate_persistence_recommendation()
        
        # Save assessment results
        assessment_result = {
            "timestamp": datetime.now().isoformat(),
            "recommendation": {
                "mode": recommendation["mode"].value,
                "reason": recommendation["reason"],
                "confidence": recommendation["confidence"],
                "safe_to_enable": recommendation["safe_to_enable"]
            },
            "factors": recommendation.get("factors", {})
        }
        
        # Save to file
        with open("last_persistence_assessment.json", 'w') as f:
            json.dump(assessment_result, f, indent=2)
        
        logger.info(f"Assessment complete - Recommendation: {recommendation['mode'].value}")
        return recommendation

if __name__ == "__main__":
    scheduler = AdaptivePersistenceScheduler()
    recommendation = scheduler.run_assessment()
    
    print(f"\\n=== ADAPTIVE PERSISTENCE ASSESSMENT ===")
    print(f"Mode: {recommendation['mode'].value}")
    print(f"Reason: {recommendation['reason']}")
    print(f"Confidence: {recommendation['confidence']:.2f}")
    print(f"Safe to Enable: {recommendation['safe_to_enable']}")
    
    if recommendation.get('factors'):
        print("\\nDetailed Factors:")
        for key, value in recommendation['factors'].items():
            print(f"  {key}: {value}")