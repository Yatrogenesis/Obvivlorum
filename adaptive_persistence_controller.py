#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Adaptive Persistence Controller for AI Symbiote
===============================================

Intelligent persistence system with threat level assessment and adaptive response.
Implements escalating defense mechanisms based on detected threat levels.

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
import threading
import subprocess
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
import psutil

logger = logging.getLogger("AdaptivePersistence")

class ThreatLevel(Enum):
    """Threat level enumeration with response parameters."""
    SAFE = {
        "level": 0,
        "name": "Safe",
        "color": "green",
        "check_interval": 300,  # 5 minutes
        "restart_threshold": 5,  # High tolerance
        "auto_restart": False,
        "aggressive_mode": False,
        "description": "Normal operation, no threats detected"
    }
    
    LOW = {
        "level": 1,
        "name": "Low",
        "color": "yellow",
        "check_interval": 180,  # 3 minutes
        "restart_threshold": 3,
        "auto_restart": False,
        "aggressive_mode": False,
        "description": "Minor anomalies detected, monitoring increased"
    }
    
    MEDIUM = {
        "level": 2,
        "name": "Medium",
        "color": "orange",
        "check_interval": 60,  # 1 minute
        "restart_threshold": 2,
        "auto_restart": True,
        "aggressive_mode": False,
        "description": "Suspicious activity detected, defensive measures active"
    }
    
    HIGH = {
        "level": 3,
        "name": "High",
        "color": "red",
        "check_interval": 30,  # 30 seconds
        "restart_threshold": 1,
        "auto_restart": True,
        "aggressive_mode": True,
        "description": "Active threat detected, aggressive defense enabled"
    }
    
    CRITICAL = {
        "level": 4,
        "name": "Critical",
        "color": "darkred",
        "check_interval": 5,  # 5 seconds
        "restart_threshold": 0,  # Instant response
        "auto_restart": True,
        "aggressive_mode": True,
        "instant_response": True,
        "description": "System under attack, maximum defense activated"
    }


class AdaptivePersistenceController:
    """
    Adaptive persistence controller with intelligent threat assessment.
    """
    
    def __init__(self, symbiote_path: str, config: Dict[str, Any] = None):
        """
        Initialize the Adaptive Persistence Controller.
        
        Args:
            symbiote_path: Path to the AI Symbiote executable
            config: Configuration dictionary
        """
        self.symbiote_path = Path(symbiote_path).resolve()
        self.config = config or self._default_config()
        
        # State variables
        self.current_threat_level = ThreatLevel.SAFE
        self.is_active = False
        self.authorized_mode = False  # Requires elevation for aggressive modes
        self.emergency_override = False
        
        # Threat detection
        self.threat_indicators = deque(maxlen=100)
        self.attack_patterns = {}
        self.false_positive_cache = set()
        
        # Response timing
        self.last_check = time.time()
        self.last_restart = 0
        self.restart_count = 0
        self.consecutive_failures = 0
        
        # Monitoring threads
        self.assessment_thread = None
        self.response_thread = None
        
        # Authorization
        self.auth_token = None
        self.auth_expires = None
        
        logger.info(f"Adaptive Persistence Controller initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "enable_adaptive": True,
            "require_authorization": True,  # For aggressive modes
            "max_restart_rate": 10,  # Max restarts per hour
            "threat_decay_time": 600,  # 10 minutes to decay threat level
            "learning_enabled": True,
            "attack_patterns_file": "attack_patterns.json",
            "whitelist_processes": ["explorer.exe", "svchost.exe", "System"],
            "suspicious_ports": [22, 23, 445, 3389, 5900],  # SSH, Telnet, SMB, RDP, VNC
            "critical_files": [
                "D:/Obvivlorum/ai_symbiote.py",
                "D:/Obvivlorum/AION/aion_core.py",
                "D:/Obvivlorum/security_protection_module.py"
            ]
        }
    
    def request_authorization(self, admin_password: str = None) -> bool:
        """
        Request authorization for elevated operations.
        
        Args:
            admin_password: Admin password or token
            
        Returns:
            True if authorized, False otherwise
        """
        if not self.config.get("require_authorization", True):
            self.authorized_mode = True
            return True
        
        try:
            # In production, this would verify against secure credentials
            # For now, check for admin privileges or specific token
            if admin_password == "EMERGENCY_OVERRIDE_2025":
                self.authorized_mode = True
                self.auth_token = hashlib.sha256(admin_password.encode()).hexdigest()
                self.auth_expires = datetime.now() + timedelta(hours=1)
                logger.warning("EMERGENCY AUTHORIZATION GRANTED - Aggressive modes enabled")
                return True
            
            # Check if running as admin
            if os.name == 'nt':
                import ctypes
                if ctypes.windll.shell32.IsUserAnAdmin():
                    self.authorized_mode = True
                    self.auth_expires = datetime.now() + timedelta(hours=1)
                    logger.info("Admin privileges detected - Authorization granted")
                    return True
            
        except Exception as e:
            logger.error(f"Authorization error: {e}")
        
        logger.warning("Authorization denied - Aggressive modes disabled")
        return False
    
    def assess_threat_level(self) -> ThreatLevel:
        """
        Assess current threat level based on multiple indicators.
        
        Returns:
            Calculated threat level
        """
        threat_score = 0
        indicators = []
        
        try:
            # 1. Check process anomalies
            process_threat = self._check_process_threats()
            threat_score += process_threat["score"]
            if process_threat["indicators"]:
                indicators.extend(process_threat["indicators"])
            
            # 2. Check network anomalies
            network_threat = self._check_network_threats()
            threat_score += network_threat["score"]
            if network_threat["indicators"]:
                indicators.extend(network_threat["indicators"])
            
            # 3. Check file integrity
            file_threat = self._check_file_integrity()
            threat_score += file_threat["score"]
            if file_threat["indicators"]:
                indicators.extend(file_threat["indicators"])
            
            # 4. Check system resources
            resource_threat = self._check_resource_abuse()
            threat_score += resource_threat["score"]
            if resource_threat["indicators"]:
                indicators.extend(resource_threat["indicators"])
            
            # 5. Check for known attack patterns
            pattern_threat = self._check_attack_patterns()
            threat_score += pattern_threat["score"]
            if pattern_threat["indicators"]:
                indicators.extend(pattern_threat["indicators"])
            
            # Store indicators
            if indicators:
                self.threat_indicators.append({
                    "timestamp": datetime.now().isoformat(),
                    "indicators": indicators,
                    "score": threat_score
                })
            
            # Determine threat level based on score
            if threat_score >= 80:
                return ThreatLevel.CRITICAL
            elif threat_score >= 60:
                return ThreatLevel.HIGH
            elif threat_score >= 40:
                return ThreatLevel.MEDIUM
            elif threat_score >= 20:
                return ThreatLevel.LOW
            else:
                return ThreatLevel.SAFE
            
        except Exception as e:
            logger.error(f"Error assessing threat level: {e}")
            return self.current_threat_level
    
    def _check_process_threats(self) -> Dict[str, Any]:
        """Check for process-based threats."""
        score = 0
        indicators = []
        
        try:
            suspicious_processes = []
            whitelist = set(self.config.get("whitelist_processes", []))
            
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'connections']):
                try:
                    name = proc.info['name']
                    
                    # Check for suspicious process names
                    suspicious_names = ['nc.exe', 'ncat.exe', 'mimikatz', 'pwdump', 
                                      'procdump', 'lazagne', 'wce.exe']
                    if any(susp in name.lower() for susp in suspicious_names):
                        suspicious_processes.append(name)
                        score += 30
                        indicators.append(f"Suspicious process: {name}")
                    
                    # Check for unsigned or modified system processes
                    if name in whitelist:
                        # Verify process integrity (simplified check)
                        cmdline = proc.info.get('cmdline', [])
                        if cmdline and any('powershell' in str(arg).lower() and 
                                         ('-enc' in str(arg) or '-e' in str(arg)) 
                                         for arg in cmdline):
                            score += 20
                            indicators.append(f"Encoded PowerShell execution detected")
                    
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Check for process injection indicators
            if len(suspicious_processes) > 2:
                score += 20
                indicators.append(f"Multiple suspicious processes: {suspicious_processes}")
            
        except Exception as e:
            logger.debug(f"Process threat check error: {e}")
        
        return {"score": score, "indicators": indicators}
    
    def _check_network_threats(self) -> Dict[str, Any]:
        """Check for network-based threats."""
        score = 0
        indicators = []
        
        try:
            suspicious_ports = set(self.config.get("suspicious_ports", []))
            connections = psutil.net_connections()
            
            # Count connections by state and port
            listening_suspicious = 0
            established_suspicious = 0
            external_connections = 0
            
            for conn in connections:
                if conn.status == 'LISTEN' and conn.laddr.port in suspicious_ports:
                    listening_suspicious += 1
                
                if conn.status == 'ESTABLISHED':
                    if conn.laddr.port in suspicious_ports or \
                       (conn.raddr and conn.raddr.port in suspicious_ports):
                        established_suspicious += 1
                    
                    # Check for external connections
                    if conn.raddr and not conn.raddr.ip.startswith(('127.', '192.168.', '10.', '172.')):
                        external_connections += 1
            
            # Score based on suspicious activity
            if listening_suspicious > 0:
                score += listening_suspicious * 15
                indicators.append(f"Suspicious ports listening: {listening_suspicious}")
            
            if established_suspicious > 0:
                score += established_suspicious * 20
                indicators.append(f"Suspicious connections established: {established_suspicious}")
            
            if external_connections > 50:  # Threshold for unusual external activity
                score += 25
                indicators.append(f"High external connection count: {external_connections}")
            
        except Exception as e:
            logger.debug(f"Network threat check error: {e}")
        
        return {"score": score, "indicators": indicators}
    
    def _check_file_integrity(self) -> Dict[str, Any]:
        """Check critical file integrity."""
        score = 0
        indicators = []
        
        try:
            critical_files = self.config.get("critical_files", [])
            
            for file_path in critical_files:
                file_path = Path(file_path)
                if not file_path.exists():
                    score += 25
                    indicators.append(f"Critical file missing: {file_path}")
                else:
                    # Check for recent modifications (within last minute)
                    mtime = file_path.stat().st_mtime
                    if time.time() - mtime < 60:
                        score += 20
                        indicators.append(f"Critical file recently modified: {file_path}")
            
        except Exception as e:
            logger.debug(f"File integrity check error: {e}")
        
        return {"score": score, "indicators": indicators}
    
    def _check_resource_abuse(self) -> Dict[str, Any]:
        """Check for resource abuse indicators."""
        score = 0
        indicators = []
        
        try:
            # CPU abuse
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 90:
                score += 15
                indicators.append(f"High CPU usage: {cpu_percent}%")
            
            # Memory abuse
            memory = psutil.virtual_memory()
            if memory.percent > 95:
                score += 15
                indicators.append(f"Critical memory usage: {memory.percent}%")
            
            # Disk I/O abuse
            disk_io = psutil.disk_io_counters()
            if disk_io and disk_io.read_bytes > 1024 * 1024 * 1024:  # 1GB/sec
                score += 10
                indicators.append("High disk I/O detected")
            
        except Exception as e:
            logger.debug(f"Resource abuse check error: {e}")
        
        return {"score": score, "indicators": indicators}
    
    def _check_attack_patterns(self) -> Dict[str, Any]:
        """Check for known attack patterns."""
        score = 0
        indicators = []
        
        try:
            # Load known attack patterns
            if not self.attack_patterns and self.config.get("learning_enabled", True):
                patterns_file = Path(self.config.get("attack_patterns_file", "attack_patterns.json"))
                if patterns_file.exists():
                    with open(patterns_file, 'r') as f:
                        self.attack_patterns = json.load(f)
            
            # Check recent indicators against patterns
            if len(self.threat_indicators) >= 3:
                recent = list(self.threat_indicators)[-3:]
                
                # Rapid escalation pattern
                if all(ind["score"] > 20 for ind in recent):
                    score += 30
                    indicators.append("Rapid threat escalation detected")
                
                # Persistence attempt pattern
                indicator_texts = [str(ind["indicators"]) for ind in recent]
                if any("registry" in text.lower() or "startup" in text.lower() 
                      for text in indicator_texts):
                    score += 20
                    indicators.append("Persistence attempt pattern detected")
            
        except Exception as e:
            logger.debug(f"Attack pattern check error: {e}")
        
        return {"score": score, "indicators": indicators}
    
    def apply_threat_response(self, threat_level: ThreatLevel) -> None:
        """
        Apply appropriate response based on threat level.
        
        Args:
            threat_level: Current threat level
        """
        level_config = threat_level.value
        
        logger.info(f"Applying {level_config['name']} threat response: {level_config['description']}")
        
        # Check authorization for aggressive modes
        if level_config.get("aggressive_mode", False):
            if not self.authorized_mode:
                logger.warning("Aggressive mode requires authorization - degrading to MEDIUM response")
                threat_level = ThreatLevel.MEDIUM
                level_config = threat_level.value
        
        # Update check interval
        self.config["check_interval"] = level_config["check_interval"]
        
        # Handle restart policy
        if level_config.get("auto_restart", False):
            if self.consecutive_failures >= level_config.get("restart_threshold", 3):
                self._perform_defensive_restart()
        
        # Handle instant response for CRITICAL level
        if level_config.get("instant_response", False) and self.authorized_mode:
            self._perform_emergency_response()
    
    def _perform_defensive_restart(self) -> None:
        """Perform defensive restart of the symbiote."""
        try:
            # Check restart rate limit
            current_time = time.time()
            if current_time - self.last_restart < 30:  # Minimum 30 seconds between restarts
                logger.info("Restart rate limit enforced")
                return
            
            logger.warning("Performing defensive restart of AI Symbiote")
            
            # Kill existing processes
            for proc in psutil.process_iter(['pid', 'cmdline']):
                try:
                    cmdline = proc.info.get('cmdline', [])
                    if cmdline and 'ai_symbiote' in str(cmdline):
                        proc.terminate()
                        logger.info(f"Terminated existing symbiote process: {proc.pid}")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            time.sleep(2)  # Wait for termination
            
            # Restart symbiote with elevated priority
            cmd = [
                sys.executable,
                str(self.symbiote_path),
                "--background",
                "--persistent",
                "--threat-level", self.current_threat_level.name
            ]
            
            subprocess.Popen(
                cmd,
                cwd=str(self.symbiote_path.parent),
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            
            self.last_restart = current_time
            self.restart_count += 1
            self.consecutive_failures = 0
            
            logger.info("Defensive restart completed")
            
        except Exception as e:
            logger.error(f"Error during defensive restart: {e}")
    
    def _perform_emergency_response(self) -> None:
        """Perform emergency response for critical threats."""
        if not self.authorized_mode:
            logger.error("Emergency response requires authorization!")
            return
        
        logger.critical("EMERGENCY RESPONSE ACTIVATED")
        
        try:
            # 1. Isolate network (optional - requires admin)
            # This is commented out for safety but shows the capability
            # subprocess.run(["netsh", "advfirewall", "set", "allprofiles", "state", "on"], 
            #               capture_output=True, check=False)
            
            # 2. Kill all suspicious processes
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    name = proc.info['name'].lower()
                    suspicious = ['nc.exe', 'ncat.exe', 'mimikatz', 'pwdump', 'lazagne']
                    if any(s in name for s in suspicious):
                        proc.kill()
                        logger.warning(f"Killed suspicious process: {name} (PID: {proc.pid})")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # 3. Restart critical components immediately
            self._perform_defensive_restart()
            
            # 4. Enable maximum monitoring
            self.config["check_interval"] = 1  # Check every second
            
            # 5. Log emergency event
            emergency_log = {
                "timestamp": datetime.now().isoformat(),
                "threat_level": "CRITICAL",
                "indicators": list(self.threat_indicators)[-10:],
                "response": "EMERGENCY_RESPONSE_EXECUTED"
            }
            
            log_file = Path("D:/Obvivlorum/emergency_response.json")
            with open(log_file, 'a') as f:
                json.dump(emergency_log, f)
                f.write('\n')
            
        except Exception as e:
            logger.error(f"Emergency response error: {e}")
    
    def start(self) -> None:
        """Start the adaptive persistence controller."""
        if self.is_active:
            return
        
        self.is_active = True
        
        # Start threat assessment thread
        self.assessment_thread = threading.Thread(
            target=self._assessment_loop,
            daemon=True,
            name="ThreatAssessment"
        )
        self.assessment_thread.start()
        
        # Start response thread
        self.response_thread = threading.Thread(
            target=self._response_loop,
            daemon=True,
            name="ThreatResponse"
        )
        self.response_thread.start()
        
        logger.info("Adaptive persistence controller started")
    
    def _assessment_loop(self) -> None:
        """Continuous threat assessment loop."""
        while self.is_active:
            try:
                # Assess current threat level
                new_threat_level = self.assess_threat_level()
                
                # Check for threat level change
                if new_threat_level != self.current_threat_level:
                    old_level = self.current_threat_level
                    self.current_threat_level = new_threat_level
                    
                    logger.warning(f"THREAT LEVEL CHANGED: {old_level.value['name']} -> "
                                 f"{new_threat_level.value['name']}")
                    
                    # Apply new threat response
                    self.apply_threat_response(new_threat_level)
                
                # Sleep based on current threat level
                sleep_time = self.current_threat_level.value["check_interval"]
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Assessment loop error: {e}")
                time.sleep(30)
    
    def _response_loop(self) -> None:
        """Response execution loop."""
        while self.is_active:
            try:
                # Check symbiote health
                if not self._check_symbiote_health():
                    self.consecutive_failures += 1
                    
                    # Apply response based on threat level
                    level_config = self.current_threat_level.value
                    if self.consecutive_failures >= level_config.get("restart_threshold", 3):
                        if level_config.get("auto_restart", False):
                            if self.authorized_mode or not level_config.get("aggressive_mode", False):
                                self._perform_defensive_restart()
                else:
                    self.consecutive_failures = 0
                
                # Sleep based on threat level
                sleep_time = self.current_threat_level.value["check_interval"]
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Response loop error: {e}")
                time.sleep(30)
    
    def _check_symbiote_health(self) -> bool:
        """Check if symbiote is healthy."""
        try:
            for proc in psutil.process_iter(['pid', 'cmdline']):
                try:
                    cmdline = proc.info.get('cmdline', [])
                    if cmdline and 'ai_symbiote' in str(cmdline):
                        return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            return False
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return True  # Assume healthy on error
    
    def stop(self) -> None:
        """Stop the adaptive persistence controller."""
        logger.info("Stopping adaptive persistence controller")
        self.is_active = False
        
        # Wait for threads
        if self.assessment_thread and self.assessment_thread.is_alive():
            self.assessment_thread.join(timeout=5)
        
        if self.response_thread and self.response_thread.is_alive():
            self.response_thread.join(timeout=5)
        
        logger.info("Adaptive persistence controller stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current controller status."""
        return {
            "active": self.is_active,
            "current_threat_level": self.current_threat_level.value["name"],
            "authorized_mode": self.authorized_mode,
            "auth_expires": self.auth_expires.isoformat() if self.auth_expires else None,
            "consecutive_failures": self.consecutive_failures,
            "restart_count": self.restart_count,
            "last_restart": datetime.fromtimestamp(self.last_restart).isoformat() if self.last_restart else None,
            "recent_indicators": list(self.threat_indicators)[-5:] if self.threat_indicators else [],
            "check_interval": self.current_threat_level.value["check_interval"]
        }


# CLI for testing and management
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Adaptive Persistence Controller")
    parser.add_argument("--authorize", action="store_true", help="Request authorization")
    parser.add_argument("--status", action="store_true", help="Get status")
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    parser.add_argument("--threat-level", choices=["SAFE", "LOW", "MEDIUM", "HIGH", "CRITICAL"],
                       help="Set threat level manually")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    controller = AdaptivePersistenceController("D:/Obvivlorum/ai_symbiote.py")
    
    if args.authorize:
        password = input("Enter admin password: ")
        if controller.request_authorization(password):
            print("Authorization granted")
        else:
            print("Authorization denied")
    
    if args.threat_level:
        controller.current_threat_level = ThreatLevel[args.threat_level]
        print(f"Threat level set to: {args.threat_level}")
    
    if args.test:
        print("Starting Adaptive Persistence Controller in test mode...")
        controller.start()
        print("Controller active. Press Ctrl+C to stop.")
        
        try:
            while True:
                time.sleep(30)
                status = controller.get_status()
                print(f"\n{'='*50}")
                print(f"Status: {json.dumps(status, indent=2)}")
                print(f"{'='*50}")
        except KeyboardInterrupt:
            print("\nStopping controller...")
            controller.stop()
    
    elif args.status:
        controller.start()
        time.sleep(2)
        status = controller.get_status()
        print(json.dumps(status, indent=2))
        controller.stop()