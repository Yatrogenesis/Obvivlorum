#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Security Protection Module for AI Symbiote
==========================================

Integrates SecurityAI-Assistant capabilities for system protection and monitoring.
Provides silent background monitoring with adaptive learning without aggressive restarts.

Author: Francisco Molina
ORCID: https://orcid.org/0009-0008-6093-8267
Email: pako.molina@gmail.com
"""

import os
import sys
import json
import time
import logging
import threading
import subprocess
import hashlib
import psutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger("SecurityProtection")

class SecurityProtectionModule:
    """
    Advanced security protection module with integrated monitoring capabilities.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Security Protection Module.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._default_config()
        self.is_active = False
        self.monitoring_thread = None
        self.learning_thread = None
        
        # Security metrics
        self.security_events = deque(maxlen=1000)
        self.threat_patterns = {}
        self.system_baseline = {}
        self.anomaly_threshold = 0.85
        
        # Health monitoring (non-intrusive)
        self.last_health_check = time.time()
        self.health_check_interval = 300  # 5 minutes
        self.consecutive_failures = 0
        self.max_failures_before_alert = 3  # Only alert, don't restart
        
        # Learning system
        self.learning_data = {
            "process_patterns": {},
            "network_patterns": {},
            "file_access_patterns": {},
            "user_behavior": {}
        }
        
        # Integration with SecurityAI-Assistant if available
        self.security_ai_path = Path("D:/SecurityAI-Assistant")
        self.has_security_ai = self.security_ai_path.exists()
        
        logger.info("Security Protection Module initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "monitoring_enabled": True,
            "learning_enabled": True,
            "silent_mode": True,  # No aggressive actions
            "protection_level": "adaptive",  # adaptive, passive, active
            "health_check_interval": 300,  # 5 minutes
            "auto_restart": False,  # DISABLED - no automatic restarts
            "alert_only": True,  # Only alert on issues, don't take action
            "log_security_events": True,
            "anomaly_detection": True,
            "threat_response": "log_only",  # log_only, alert, block
            "wsl_integration": True,
            "parrot_os_tools": ["nmap", "nikto", "dirb"],
            "max_cpu_percent": 20,  # Keep CPU usage low
            "max_memory_mb": 100,  # Keep memory usage low
        }
    
    def start_protection(self):
        """Start the protection and monitoring system."""
        if self.is_active:
            logger.info("Protection already active")
            return
        
        self.is_active = True
        
        # Initialize system baseline
        self._establish_baseline()
        
        # Start monitoring thread
        if self.config.get("monitoring_enabled", True):
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True,
                name="SecurityMonitor"
            )
            self.monitoring_thread.start()
            logger.info("Security monitoring started")
        
        # Start learning thread
        if self.config.get("learning_enabled", True):
            self.learning_thread = threading.Thread(
                target=self._learning_loop,
                daemon=True,
                name="SecurityLearning"
            )
            self.learning_thread.start()
            logger.info("Security learning started")
    
    def _establish_baseline(self):
        """Establish system baseline for anomaly detection."""
        try:
            # Process baseline
            self.system_baseline["processes"] = [
                p.info for p in psutil.process_iter(['pid', 'name', 'username'])
            ]
            
            # Network baseline
            self.system_baseline["network"] = psutil.net_connections()
            
            # CPU and memory baseline
            self.system_baseline["cpu_percent"] = psutil.cpu_percent(interval=1)
            self.system_baseline["memory"] = psutil.virtual_memory().percent
            
            # Disk usage baseline
            self.system_baseline["disk"] = {
                part.mountpoint: psutil.disk_usage(part.mountpoint).percent
                for part in psutil.disk_partitions()
                if not part.opts.startswith('cd')
            }
            
            logger.info("System baseline established")
            
        except Exception as e:
            logger.error(f"Error establishing baseline: {e}")
    
    def _monitoring_loop(self):
        """Main monitoring loop - non-intrusive."""
        logger.info("Security monitoring loop started")
        
        while self.is_active:
            try:
                # Perform health check
                self._perform_health_check()
                
                # Monitor for anomalies
                if self.config.get("anomaly_detection", True):
                    self._detect_anomalies()
                
                # Check resource usage (stay within limits)
                self._check_resource_usage()
                
                # Sleep based on monitoring intensity
                sleep_time = self.config.get("health_check_interval", 300)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait before retry
    
    def _perform_health_check(self):
        """
        Perform non-intrusive health check of the symbiote system.
        No automatic restarts - only logging and alerts.
        """
        try:
            current_time = time.time()
            
            # Check if symbiote process is running
            symbiote_running = self._check_symbiote_process()
            
            if not symbiote_running:
                self.consecutive_failures += 1
                logger.warning(f"Symbiote process not detected (check {self.consecutive_failures})")
                
                # Only alert, don't restart
                if self.consecutive_failures >= self.max_failures_before_alert:
                    self._send_alert("Symbiote process appears to be stopped")
                    self.consecutive_failures = 0  # Reset counter after alert
            else:
                if self.consecutive_failures > 0:
                    logger.info("Symbiote process recovered")
                self.consecutive_failures = 0
            
            self.last_health_check = current_time
            
        except Exception as e:
            logger.error(f"Error in health check: {e}")
    
    def _check_symbiote_process(self) -> bool:
        """Check if the symbiote process is running."""
        try:
            for process in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = process.info.get('cmdline', [])
                    if cmdline and any('ai_symbiote' in str(arg) for arg in cmdline):
                        return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            return False
        except Exception as e:
            logger.error(f"Error checking symbiote process: {e}")
            return True  # Assume running on error to avoid false positives
    
    def _detect_anomalies(self):
        """Detect system anomalies without taking aggressive action."""
        try:
            anomalies = []
            
            # Check for suspicious processes
            current_processes = [
                p.info for p in psutil.process_iter(['pid', 'name', 'username'])
            ]
            
            # Compare with baseline (simplified check)
            baseline_names = {p['name'] for p in self.system_baseline.get('processes', [])}
            current_names = {p['name'] for p in current_processes}
            
            new_processes = current_names - baseline_names
            if len(new_processes) > 10:  # Threshold for suspicious activity
                anomalies.append({
                    "type": "process_anomaly",
                    "detail": f"Unusual number of new processes: {len(new_processes)}",
                    "severity": "medium"
                })
            
            # Check network connections
            current_connections = len(psutil.net_connections())
            baseline_connections = len(self.system_baseline.get('network', []))
            
            if current_connections > baseline_connections * 2:
                anomalies.append({
                    "type": "network_anomaly",
                    "detail": f"Unusual network activity: {current_connections} connections",
                    "severity": "medium"
                })
            
            # Log anomalies
            if anomalies:
                for anomaly in anomalies:
                    self.security_events.append({
                        "timestamp": datetime.now().isoformat(),
                        "event": anomaly
                    })
                    logger.warning(f"Security anomaly detected: {anomaly}")
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
    
    def _check_resource_usage(self):
        """Monitor and limit resource usage to stay non-intrusive."""
        try:
            # Get current resource usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Check against limits
            max_cpu = self.config.get("max_cpu_percent", 20)
            max_memory = self.config.get("max_memory_mb", 100)
            
            if cpu_percent > max_cpu:
                logger.info(f"CPU usage high ({cpu_percent}%), throttling operations")
                time.sleep(1)  # Brief pause to reduce CPU usage
            
            if memory_mb > max_memory:
                logger.info(f"Memory usage high ({memory_mb:.1f}MB), optimizing")
                # Clear some caches if needed
                if len(self.security_events) > 500:
                    # Keep only recent events
                    self.security_events = deque(
                        list(self.security_events)[-500:],
                        maxlen=1000
                    )
            
        except Exception as e:
            logger.debug(f"Resource check error: {e}")
    
    def _learning_loop(self):
        """Background learning loop for adaptive behavior."""
        logger.info("Security learning loop started")
        
        while self.is_active:
            try:
                # Learn process patterns
                self._learn_process_patterns()
                
                # Learn network patterns
                self._learn_network_patterns()
                
                # Learn user behavior
                self._learn_user_behavior()
                
                # Save learning data periodically
                if len(self.security_events) % 100 == 0:
                    self._save_learning_data()
                
                # Long sleep for background learning
                time.sleep(600)  # 10 minutes
                
            except Exception as e:
                logger.error(f"Error in learning loop: {e}")
                time.sleep(300)
    
    def _learn_process_patterns(self):
        """Learn normal process patterns."""
        try:
            current_processes = {}
            for p in psutil.process_iter(['name', 'cpu_percent', 'memory_percent']):
                name = p.info['name']
                if name not in self.learning_data["process_patterns"]:
                    self.learning_data["process_patterns"][name] = {
                        "count": 0,
                        "avg_cpu": 0,
                        "avg_memory": 0
                    }
                
                pattern = self.learning_data["process_patterns"][name]
                pattern["count"] += 1
                pattern["avg_cpu"] = (
                    pattern["avg_cpu"] * (pattern["count"] - 1) + p.info['cpu_percent']
                ) / pattern["count"]
                pattern["avg_memory"] = (
                    pattern["avg_memory"] * (pattern["count"] - 1) + p.info['memory_percent']
                ) / pattern["count"]
            
        except Exception as e:
            logger.debug(f"Process pattern learning error: {e}")
    
    def _learn_network_patterns(self):
        """Learn normal network patterns."""
        try:
            connections = psutil.net_connections()
            
            # Count connections by state
            state_counts = {}
            for conn in connections:
                state = conn.status
                state_counts[state] = state_counts.get(state, 0) + 1
            
            # Update learning data
            if "connection_states" not in self.learning_data["network_patterns"]:
                self.learning_data["network_patterns"]["connection_states"] = {}
            
            for state, count in state_counts.items():
                if state not in self.learning_data["network_patterns"]["connection_states"]:
                    self.learning_data["network_patterns"]["connection_states"][state] = []
                
                self.learning_data["network_patterns"]["connection_states"][state].append({
                    "timestamp": datetime.now().isoformat(),
                    "count": count
                })
                
                # Keep only recent data (last 100 samples)
                if len(self.learning_data["network_patterns"]["connection_states"][state]) > 100:
                    self.learning_data["network_patterns"]["connection_states"][state] = \
                        self.learning_data["network_patterns"]["connection_states"][state][-100:]
            
        except Exception as e:
            logger.debug(f"Network pattern learning error: {e}")
    
    def _learn_user_behavior(self):
        """Learn user behavior patterns."""
        try:
            # Simple time-based activity pattern
            current_hour = datetime.now().hour
            
            if "hourly_activity" not in self.learning_data["user_behavior"]:
                self.learning_data["user_behavior"]["hourly_activity"] = [0] * 24
            
            self.learning_data["user_behavior"]["hourly_activity"][current_hour] += 1
            
        except Exception as e:
            logger.debug(f"User behavior learning error: {e}")
    
    def _send_alert(self, message: str, severity: str = "info"):
        """Send security alert without taking aggressive action."""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "severity": severity
        }
        
        # Log the alert
        logger.info(f"SECURITY ALERT [{severity}]: {message}")
        
        # Add to security events
        self.security_events.append(alert)
        
        # Save alert to file for review
        try:
            alert_file = Path("D:/Obvivlorum/security_alerts.json")
            alerts = []
            if alert_file.exists():
                with open(alert_file, 'r') as f:
                    alerts = json.load(f)
            
            alerts.append(alert)
            
            # Keep only recent alerts (last 100)
            if len(alerts) > 100:
                alerts = alerts[-100:]
            
            with open(alert_file, 'w') as f:
                json.dump(alerts, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving alert: {e}")
    
    def _save_learning_data(self):
        """Save learning data to file."""
        try:
            learning_file = Path("D:/Obvivlorum/security_learning.json")
            with open(learning_file, 'w') as f:
                json.dump(self.learning_data, f, indent=2, default=str)
            logger.debug("Learning data saved")
        except Exception as e:
            logger.error(f"Error saving learning data: {e}")
    
    def integrate_security_ai(self) -> bool:
        """
        Integrate with SecurityAI-Assistant if available.
        
        Returns:
            True if integration successful, False otherwise
        """
        if not self.has_security_ai:
            logger.info("SecurityAI-Assistant not found, skipping integration")
            return False
        
        try:
            # Check if SecurityAI modules are available
            security_ai_src = self.security_ai_path / "src"
            if security_ai_src.exists():
                # Add to Python path
                sys.path.insert(0, str(self.security_ai_path))
                logger.info("SecurityAI-Assistant integration enabled")
                return True
            
        except Exception as e:
            logger.error(f"Error integrating SecurityAI-Assistant: {e}")
        
        return False
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get current security status."""
        return {
            "active": self.is_active,
            "protection_level": self.config.get("protection_level", "adaptive"),
            "monitoring_enabled": self.config.get("monitoring_enabled", True),
            "learning_enabled": self.config.get("learning_enabled", True),
            "consecutive_failures": self.consecutive_failures,
            "last_health_check": datetime.fromtimestamp(self.last_health_check).isoformat(),
            "security_events_count": len(self.security_events),
            "recent_alerts": list(self.security_events)[-5:] if self.security_events else [],
            "has_security_ai": self.has_security_ai,
            "resource_usage": {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_mb": psutil.Process().memory_info().rss / 1024 / 1024
            }
        }
    
    def stop_protection(self):
        """Stop the protection system gracefully."""
        logger.info("Stopping security protection")
        self.is_active = False
        
        # Save final learning data
        self._save_learning_data()
        
        # Wait for threads to finish
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        if self.learning_thread and self.learning_thread.is_alive():
            self.learning_thread.join(timeout=5)
        
        logger.info("Security protection stopped")


# Standalone execution for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Security Protection Module")
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    parser.add_argument("--status", action="store_true", help="Get security status")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create protection module
    protection = SecurityProtectionModule()
    
    if args.test:
        print("Starting Security Protection Module in test mode...")
        protection.start_protection()
        print("Protection active. Press Ctrl+C to stop.")
        
        try:
            while True:
                time.sleep(30)
                status = protection.get_security_status()
                print(f"\nSecurity Status: {json.dumps(status, indent=2)}")
        except KeyboardInterrupt:
            print("\nStopping protection...")
            protection.stop_protection()
    
    elif args.status:
        protection.start_protection()
        time.sleep(2)  # Let it initialize
        status = protection.get_security_status()
        print(json.dumps(status, indent=2))
        protection.stop_protection()
    
    else:
        print("Use --test for test mode or --status to check security status")