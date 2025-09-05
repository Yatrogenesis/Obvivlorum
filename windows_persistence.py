#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Windows Persistence Manager for AI Symbiote
===========================================

This module implements Windows-specific persistence mechanisms for the AI Symbiote system.
It ensures the system remains active and operational across Windows sessions.

Author: Francisco Molina
ORCID: https://orcid.org/0009-0008-6093-8267
Email: pako.molina@gmail.com
"""

import os
import sys
import winreg
import logging
import json
import time
import threading
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger("WindowsPersistence")

class WindowsPersistenceManager:
    """
    Manages Windows-specific persistence mechanisms for the AI Symbiote.
    """
    
    def __init__(self, symbiote_path: str, config: Dict[str, Any] = None):
        """
        Initialize the Windows Persistence Manager.
        
        Args:
            symbiote_path: Path to the main symbiote executable/script
            config: Configuration dictionary
        """
        self.symbiote_path = Path(symbiote_path).resolve()
        self.config = config or self._default_config()
        self.is_active = False
        self.monitoring_thread = None
        self.persistence_methods = []
        
        # Anti-spam: cooldown times for repair attempts
        self.last_scheduled_task_attempt = 0
        self.scheduled_task_cooldown = 300  # 5 minutes between attempts
        
        # Windows-specific paths
        self.startup_folder = Path(os.path.expanduser("~")) / "AppData" / "Roaming" / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
        self.temp_folder = Path(os.environ.get("TEMP", "C:\\Temp"))
        self.appdata_folder = Path(os.path.expanduser("~")) / "AppData" / "Local" / "AISymbiote"
        
        # Ensure directories exist
        self.appdata_folder.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Windows Persistence Manager initialized for: {self.symbiote_path}")
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "auto_start": False,  # DISABLED FOR SAFETY
            "registry_persistence": False,  # DISABLED FOR SAFETY
            "scheduled_task": False,  # DISABLED FOR SAFETY
            "startup_folder": False,  # DISABLED FOR SAFETY
            "service_mode": False,  # Advanced: requires admin privileges
            "stealth_mode": False,  # DISABLED FOR SAFETY
            "self_healing": False,  # DISABLED FOR SAFETY
            "update_check_interval": 3600,  # 1 hour
            "heartbeat_interval": 300,  # 5 minutes
            "max_restart_attempts": 5,
            "restart_delay": 30  # seconds
        }
    
    def install_persistence(self) -> Dict[str, Any]:
        """
        Install all configured persistence mechanisms.
        
        Returns:
            Dictionary containing installation results
        """
        results = {
            "status": "success",
            "installed_methods": [],
            "failed_methods": [],
            "errors": []
        }
        
        try:
            # Method 1: Registry Run key (most common) - SECURITY DISABLED
            if self.config.get("registry_persistence", False):
                if self._install_registry_persistence():
                    results["installed_methods"].append("registry_run_key")
                else:
                    results["failed_methods"].append("registry_run_key")
            
            # Method 2: Startup folder - SECURITY DISABLED
            if self.config.get("startup_folder", False):
                if self._install_startup_folder():
                    results["installed_methods"].append("startup_folder")
                else:
                    results["failed_methods"].append("startup_folder")
            
            # Method 3: Scheduled Task - SECURITY DISABLED
            if self.config.get("scheduled_task", False):
                if self._install_scheduled_task():
                    results["installed_methods"].append("scheduled_task")
                else:
                    results["failed_methods"].append("scheduled_task")
            
            # Method 4: Windows Service (requires admin)
            if self.config.get("service_mode", False):
                if self._install_windows_service():
                    results["installed_methods"].append("windows_service")
                else:
                    results["failed_methods"].append("windows_service")
            
            # Create configuration file
            self._save_persistence_config(results)
            
            # Start monitoring if any method succeeded
            if results["installed_methods"]:
                self.start_monitoring()
                results["monitoring_started"] = True
            
            logger.info(f"Persistence installation completed: {len(results['installed_methods'])} methods installed")
            
        except Exception as e:
            logger.error(f"Error during persistence installation: {e}")
            results["status"] = "error"
            results["errors"].append(str(e))
        
        return results
    
    def _install_registry_persistence(self) -> bool:
        """Install persistence via Windows Registry Run key."""
        try:
            # Create a batch file to run the symbiote
            batch_file = self.appdata_folder / "ai_symbiote_launcher.bat"
            
            with open(batch_file, 'w') as f:
                f.write(f'@echo off\n')
                f.write(f'cd /d "{self.symbiote_path.parent}"\n')
                f.write(f'python "{self.symbiote_path}" --background --persistent\n')
            
            # Add to registry
            key_path = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Run"
            
            try:
                # Try HKEY_CURRENT_USER first (doesn't require admin)
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE) as key:
                    winreg.SetValueEx(key, "AISymbiote", 0, winreg.REG_SZ, str(batch_file))
                    logger.info("Registry persistence installed (HKCU)")
                    return True
            except Exception as e:
                logger.warning(f"Could not install HKCU registry persistence: {e}")
                
                # Try HKEY_LOCAL_MACHINE (requires admin)
                try:
                    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path, 0, winreg.KEY_SET_VALUE) as key:
                        winreg.SetValueEx(key, "AISymbiote", 0, winreg.REG_SZ, str(batch_file))
                        logger.info("Registry persistence installed (HKLM)")
                        return True
                except Exception as e2:
                    logger.error(f"Could not install HKLM registry persistence: {e2}")
                    return False
                    
        except Exception as e:
            logger.error(f"Registry persistence installation failed: {e}")
            return False
    
    def _install_startup_folder(self) -> bool:
        """Install persistence via Windows Startup folder."""
        try:
            if not self.startup_folder.exists():
                logger.warning(f"Startup folder not found: {self.startup_folder}")
                return False
            
            # Create shortcut in startup folder
            shortcut_path = self.startup_folder / "AI Symbiote.lnk"
            
            # Create batch file to run
            batch_file = self.appdata_folder / "startup_launcher.bat"
            with open(batch_file, 'w') as f:
                f.write(f'@echo off\n')
                f.write(f'cd /d "{self.symbiote_path.parent}"\n')
                f.write(f'python "{self.symbiote_path}" --background --persistent\n')
            
            # Copy batch file to startup folder (simpler than creating .lnk)
            startup_batch = self.startup_folder / "AI_Symbiote_Startup.bat"
            shutil.copy2(batch_file, startup_batch)
            
            logger.info(f"Startup folder persistence installed: {startup_batch}")
            return True
            
        except Exception as e:
            logger.error(f"Startup folder persistence installation failed: {e}")
            return False
    
    def _install_scheduled_task(self) -> bool:
        """Install persistence via Windows Task Scheduler."""
        try:
            task_name = "AI_Symbiote_Maintenance"
            
            # Create XML for scheduled task
            task_xml = f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Date>{datetime.now().isoformat()}</Date>
    <Author>AI Symbiote</Author>
    <Description>Maintains AI Symbiote system</Description>
  </RegistrationInfo>
  <Triggers>
    <LogonTrigger>
      <Enabled>true</Enabled>
    </LogonTrigger>
    <TimeTrigger>
      <Enabled>true</Enabled>
      <Repetition>
        <Interval>PT1H</Interval>
        <StopAtDurationEnd>false</StopAtDurationEnd>
      </Repetition>
      <StartBoundary>{(datetime.now() + timedelta(minutes=5)).isoformat()}</StartBoundary>
    </TimeTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <IdleSettings>
      <StopOnIdleEnd>false</StopOnIdleEnd>
      <RestartOnIdle>false</RestartOnIdle>
    </IdleSettings>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>true</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <Priority>7</Priority>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>python</Command>
      <Arguments>"{self.symbiote_path}" --background --scheduled</Arguments>
      <WorkingDirectory>{self.symbiote_path.parent}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>"""
            
            # Save task XML
            task_xml_file = self.appdata_folder / "symbiote_task.xml"
            with open(task_xml_file, 'w', encoding='utf-16') as f:
                f.write(task_xml)
            
            # Create scheduled task using schtasks command
            cmd = [
                "schtasks", "/create",
                "/tn", task_name,
                "/xml", str(task_xml_file),
                "/f"  # Force overwrite if exists
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
            
            if result.returncode == 0:
                logger.info(f"Scheduled task '{task_name}' created successfully")
                return True
            else:
                # Check if it's an access denied error
                if "Access is denied" in result.stderr or "0x80070005" in result.stderr:
                    logger.warning(f"Access denied when creating scheduled task. Trying alternative method...")
                    # Try creating a simpler task without XML
                    return self._install_simple_scheduled_task()
                else:
                    logger.error(f"Failed to create scheduled task: {result.stderr}")
                    return False
                
        except Exception as e:
            logger.error(f"Scheduled task installation failed: {e}")
            return False
    
    def _install_simple_scheduled_task(self) -> bool:
        """Install a simpler scheduled task without XML (for non-admin users)."""
        try:
            task_name = "AI_Symbiote_User_Task"
            
            # Create a batch file to run
            batch_file = self.appdata_folder / "scheduled_launcher.bat"
            with open(batch_file, 'w') as f:
                f.write(f'@echo off\n')
                f.write(f'cd /d "{self.symbiote_path.parent}"\n')
                f.write(f'python "{self.symbiote_path}" --background --scheduled\n')
            
            # Create a simpler scheduled task without elevation
            cmd = [
                "schtasks", "/create",
                "/tn", task_name,
                "/tr", f'"{batch_file}"',
                "/sc", "onlogon",  # Run at logon
                "/f",  # Force overwrite
                "/rl", "limited"  # Run with limited privileges
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
            
            if result.returncode == 0:
                logger.info(f"Simple scheduled task '{task_name}' created successfully")
                return True
            else:
                logger.warning(f"Simple scheduled task also failed: {result.stderr}")
                # If even the simple task fails, we'll rely on other persistence methods
                return False
                
        except Exception as e:
            logger.error(f"Simple scheduled task installation failed: {e}")
            return False
    
    def _install_windows_service(self) -> bool:
        """Install persistence as Windows Service (requires admin privileges)."""
        try:
            # This would require admin privileges and additional service framework
            # For now, return False as it's not implemented
            logger.info("Windows Service installation not implemented (requires admin privileges)")
            return False
            
        except Exception as e:
            logger.error(f"Windows service installation failed: {e}")
            return False
    
    def start_monitoring(self) -> bool:
        """Start the persistence monitoring thread."""
        if self.is_active:
            logger.warning("Monitoring already active")
            return True
        
        try:
            self.is_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            
            logger.info("Persistence monitoring started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
            self.is_active = False
            return False
    
    def stop_monitoring(self) -> bool:
        """Stop the persistence monitoring."""
        try:
            self.is_active = False
            
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5)
            
            logger.info("Persistence monitoring stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop monitoring: {e}")
            return False
    
    def _monitoring_loop(self):
        """Main monitoring loop for persistence health checks."""
        logger.info("Persistence monitoring loop started")
        
        last_heartbeat = time.time()
        last_update_check = time.time()
        restart_attempts = 0
        
        while self.is_active:
            try:
                current_time = time.time()
                
                # Heartbeat check
                if current_time - last_heartbeat >= self.config["heartbeat_interval"]:
                    if self._check_symbiote_health():
                        restart_attempts = 0  # Reset counter on successful health check
                    else:
                        restart_attempts += 1
                        logger.warning(f"Symbiote health check failed (attempt {restart_attempts})")
                        
                        if restart_attempts <= self.config["max_restart_attempts"]:
                            if self._restart_symbiote():
                                logger.info("Symbiote restart successful")
                                restart_attempts = 0
                            else:
                                logger.error("Symbiote restart failed")
                        else:
                            logger.error("Maximum restart attempts exceeded")
                    
                    last_heartbeat = current_time
                
                # Update check
                if current_time - last_update_check >= self.config["update_check_interval"]:
                    self._check_for_updates()
                    last_update_check = current_time
                
                # Self-healing check
                if self.config.get("self_healing", True):
                    self._self_healing_check()
                
                # Sleep for a short interval
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _check_symbiote_health(self) -> bool:
        """Check if the main symbiote process is healthy."""
        try:
            # Check if the main process is running
            # This is simplified - in a real implementation, you'd check specific process indicators
            
            # Check for lock file or status file
            status_file = self.appdata_folder / "symbiote_status.json"
            
            if status_file.exists():
                with open(status_file, 'r') as f:
                    status = json.load(f)
                
                last_update = status.get("last_update", 0)
                if time.time() - last_update < 600:  # 10 minutes
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def _restart_symbiote(self) -> bool:
        """Attempt to restart the symbiote system."""
        try:
            logger.info("Attempting to restart symbiote system...")
            
            # Wait a bit before restart
            time.sleep(self.config["restart_delay"])
            
            # Launch symbiote in background mode
            cmd = ["python", str(self.symbiote_path), "--background", "--auto-restart"]
            
            subprocess.Popen(
                cmd,
                cwd=str(self.symbiote_path.parent),
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )
            
            logger.info("Symbiote restart initiated")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restart symbiote: {e}")
            return False
    
    def _check_for_updates(self):
        """Check for symbiote system updates."""
        try:
            # This would connect to an update server or check version files
            # For now, just log that we're checking
            logger.debug("Checking for updates...")
            
            # In a real implementation, this would:
            # 1. Check remote version
            # 2. Download updates if available
            # 3. Apply updates safely
            # 4. Restart system if needed
            
        except Exception as e:
            logger.error(f"Update check failed: {e}")
    
    def _self_healing_check(self):
        """Perform self-healing checks and repairs."""
        try:
            # Check registry entries
            if self.config.get("registry_persistence", True):
                if not self._verify_registry_persistence():
                    logger.warning("Registry persistence corrupted, attempting repair...")
                    self._install_registry_persistence()
            
            # Check startup folder
            if self.config.get("startup_folder", True):
                startup_batch = self.startup_folder / "AI_Symbiote_Startup.bat"
                if not startup_batch.exists():
                    logger.warning("Startup folder entry missing, attempting repair...")
                    self._install_startup_folder()
            
            # Check scheduled task (with cooldown to prevent spam)
            if self.config.get("scheduled_task", True):
                current_time = time.time()
                if not self._verify_scheduled_task():
                    # Only attempt repair if cooldown period has passed
                    if current_time - self.last_scheduled_task_attempt >= self.scheduled_task_cooldown:
                        logger.warning("Scheduled task missing, attempting repair...")
                        self.last_scheduled_task_attempt = current_time
                        
                        # Don't try to repair if we don't have permissions
                        try:
                            self._install_scheduled_task()
                        except Exception as e:
                            logger.warning(f"Could not repair scheduled task: {e}")
                    # else: Skip repair attempt during cooldown
            
        except Exception as e:
            logger.error(f"Self-healing check failed: {e}")
    
    def _verify_registry_persistence(self) -> bool:
        """Verify registry persistence is intact."""
        try:
            key_path = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Run"
            
            # Check HKEY_CURRENT_USER
            try:
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_READ) as key:
                    value, _ = winreg.QueryValueEx(key, "AISymbiote")
                    return bool(value)
            except FileNotFoundError:
                pass
            
            # Check HKEY_LOCAL_MACHINE
            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path, 0, winreg.KEY_READ) as key:
                    value, _ = winreg.QueryValueEx(key, "AISymbiote")
                    return bool(value)
            except FileNotFoundError:
                pass
            
            return False
            
        except Exception as e:
            logger.error(f"Registry verification failed: {e}")
            return False
    
    def _verify_scheduled_task(self) -> bool:
        """Verify scheduled task exists."""
        try:
            # Check for the main task
            result = subprocess.run(
                ["schtasks", "/query", "/tn", "AI_Symbiote_Maintenance"],
                capture_output=True,
                text=True,
                shell=True
            )
            if result.returncode == 0:
                return True
            
            # Check for the simple user task as fallback
            result = subprocess.run(
                ["schtasks", "/query", "/tn", "AI_Symbiote_User_Task"],
                capture_output=True,
                text=True,
                shell=True
            )
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"Scheduled task verification failed: {e}")
            return False
    
    def _save_persistence_config(self, results: Dict[str, Any]):
        """Save persistence configuration and results."""
        try:
            config_file = self.appdata_folder / "persistence_config.json"
            
            config_data = {
                "installation_date": datetime.now().isoformat(),
                "symbiote_path": str(self.symbiote_path),
                "config": self.config,
                "installation_results": results,
                "version": "1.0"
            }
            
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"Persistence configuration saved: {config_file}")
            
        except Exception as e:
            logger.error(f"Failed to save persistence config: {e}")
    
    def uninstall_persistence(self) -> Dict[str, Any]:
        """Remove all persistence mechanisms."""
        results = {
            "status": "success",
            "removed_methods": [],
            "failed_methods": [],
            "errors": []
        }
        
        try:
            # Stop monitoring
            self.stop_monitoring()
            
            # Remove registry entries
            try:
                key_path = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Run"
                
                # Try HKEY_CURRENT_USER
                try:
                    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE) as key:
                        winreg.DeleteValue(key, "AISymbiote")
                        results["removed_methods"].append("registry_hkcu")
                except FileNotFoundError:
                    pass
                
                # Try HKEY_LOCAL_MACHINE
                try:
                    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path, 0, winreg.KEY_SET_VALUE) as key:
                        winreg.DeleteValue(key, "AISymbiote")
                        results["removed_methods"].append("registry_hklm")
                except FileNotFoundError:
                    pass
                    
            except Exception as e:
                logger.error(f"Failed to remove registry entries: {e}")
                results["failed_methods"].append("registry")
                results["errors"].append(str(e))
            
            # Remove startup folder entries
            try:
                startup_batch = self.startup_folder / "AI_Symbiote_Startup.bat"
                if startup_batch.exists():
                    startup_batch.unlink()
                    results["removed_methods"].append("startup_folder")
            except Exception as e:
                logger.error(f"Failed to remove startup folder entry: {e}")
                results["failed_methods"].append("startup_folder")
                results["errors"].append(str(e))
            
            # Remove scheduled task
            try:
                result = subprocess.run(
                    ["schtasks", "/delete", "/tn", "AI_Symbiote_Maintenance", "/f"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    results["removed_methods"].append("scheduled_task")
                else:
                    results["failed_methods"].append("scheduled_task")
                    results["errors"].append(result.stderr)
            except Exception as e:
                logger.error(f"Failed to remove scheduled task: {e}")
                results["failed_methods"].append("scheduled_task")
                results["errors"].append(str(e))
            
            # Clean up files
            try:
                if self.appdata_folder.exists():
                    shutil.rmtree(self.appdata_folder)
                    results["removed_methods"].append("appdata_folder")
            except Exception as e:
                logger.error(f"Failed to remove appdata folder: {e}")
                results["failed_methods"].append("appdata_folder")
                results["errors"].append(str(e))
            
            logger.info(f"Persistence uninstallation completed: {len(results['removed_methods'])} methods removed")
            
        except Exception as e:
            logger.error(f"Error during persistence uninstallation: {e}")
            results["status"] = "error"
            results["errors"].append(str(e))
        
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """Get current persistence status."""
        registry_active = self._verify_registry_persistence()
        scheduled_active = self._verify_scheduled_task()
        startup_active = (self.startup_folder / "AI_Symbiote_Startup.bat").exists()
        
        # Determine overall status
        active_methods = sum([registry_active, scheduled_active, startup_active])
        if active_methods >= 2:
            overall_status = "fully_active"
        elif active_methods >= 1:
            overall_status = "partially_active"
        else:
            overall_status = "inactive"
        
        return {
            "status": overall_status,  # Add status field that tests expect
            "is_active": self.is_active,
            "monitoring_active": self.monitoring_thread and self.monitoring_thread.is_alive(),
            "symbiote_path": str(self.symbiote_path),
            "config": self.config,
            "appdata_folder": str(self.appdata_folder),
            "registry_active": registry_active,
            "scheduled_task_active": scheduled_active,
            "startup_folder_active": startup_active
        }


def main():
    """Test the Windows Persistence Manager."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Windows Persistence Manager")
    parser.add_argument("--install", action="store_true", help="Install persistence")
    parser.add_argument("--uninstall", action="store_true", help="Uninstall persistence")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--symbiote-path", default=__file__, help="Path to symbiote script")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create persistence manager
    manager = WindowsPersistenceManager(args.symbiote_path)
    
    if args.install:
        print("Installing Windows persistence...")
        result = manager.install_persistence()
        print(f"Installation result: {result}")
        
    elif args.uninstall:
        print("Uninstalling Windows persistence...")
        result = manager.uninstall_persistence()
        print(f"Uninstallation result: {result}")
        
    elif args.status:
        print("Windows persistence status:")
        status = manager.get_status()
        print(json.dumps(status, indent=2))
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()