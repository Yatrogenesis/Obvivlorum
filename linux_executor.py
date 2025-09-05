#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Linux Execution Engine for AI Symbiote
======================================

This module provides Linux execution capabilities for command execution,
system integration, and cross-platform operations through WSL or direct Linux access.

Author: Francisco Molina
ORCID: https://orcid.org/0009-0008-6093-8267
Email: pako.molina@gmail.com
"""

import os
import sys
import subprocess
import logging
import json
import time
import threading
import shutil
from pathlib import Path, PurePosixPath
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import tempfile

logger = logging.getLogger("LinuxExecutor")

class LinuxExecutionEngine:
    """
    Provides Linux execution capabilities, supporting both native Linux
    and Windows Subsystem for Linux (WSL) environments.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Linux Execution Engine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._default_config()
        self.is_wsl = self._detect_wsl_environment()
        self.wsl_distros = self._detect_wsl_distros()
        self.default_distro = self._get_default_distro()
        self.execution_history = []
        self.active_sessions = {}
        
        logger.info(f"Linux Execution Engine initialized")
        logger.info(f"WSL Environment: {self.is_wsl}")
        logger.info(f"Available WSL Distros: {self.wsl_distros}")
        logger.info(f"Default Distro: {self.default_distro}")
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "timeout": 300,  # 5 minutes default timeout
            "max_concurrent_processes": 50,
            "log_commands": True,
            "safe_mode": False,  # UNRESTRICTED MODE FOR USER COMMANDS
            "unrestricted_commands": True,
            "auto_install_deps": True,
            "preferred_shell": "/bin/bash",
            "working_directory": "/tmp/ai_symbiote",
            "environment_isolation": False,
            "parrot_security_tools": True,
            "bypass_restrictions": True,
            "user_command_override": True,
            "resource_limits": {
                "max_memory": "8G",
                "max_cpu": "90%",
                "max_disk": "50G"
            }
        }
    
    def _detect_wsl_environment(self) -> bool:
        """Detect if running in WSL environment."""
        try:
            # Method 1: Check for WSL-specific files
            if os.path.exists("/proc/version"):
                with open("/proc/version", "r") as f:
                    version_info = f.read().lower()
                    if "microsoft" in version_info or "wsl" in version_info:
                        return True
            
            # Method 2: Check if we're on Windows but can run wsl command
            if sys.platform == "win32":
                try:
                    result = subprocess.run(["wsl", "--status"], 
                                          capture_output=True, text=True, timeout=5)
                    return result.returncode == 0
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    pass
            
            # Method 3: Check for Linux kernel but Windows-style paths
            if os.name == "posix" and os.path.exists("/mnt/c"):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error detecting WSL environment: {e}")
            return False
    
    def _detect_wsl_distros(self) -> List[str]:
        """Detect available WSL distributions."""
        distros = []
        
        try:
            if sys.platform == "win32":
                # Use wsl --list command
                result = subprocess.run(["wsl", "--list", "--quiet"], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith('Windows Subsystem'):
                            # Remove BOM and special characters
                            clean_line = line.encode('utf-8', errors='ignore').decode('utf-8')
                            clean_line = ''.join(c for c in clean_line if c.isprintable())
                            if clean_line:
                                distros.append(clean_line)
            
            # If we're inside WSL, we can only use the current distro
            elif self.is_wsl:
                # Try to get distro name from environment
                distro_name = os.environ.get("WSL_DISTRO_NAME", "Unknown")
                if distro_name != "Unknown":
                    distros.append(distro_name)
                else:
                    distros.append("current")
                    
        except Exception as e:
            logger.error(f"Error detecting WSL distros: {e}")
        
        return distros
    
    def _get_default_distro(self) -> Optional[str]:
        """Get the default WSL distribution."""
        if not self.wsl_distros:
            return None
        
        # If we have distros, use the first one as default
        # In WSL2 this is usually the default distro
        return self.wsl_distros[0]
    
    def execute_command(
        self, 
        command: Union[str, List[str]], 
        distro: Optional[str] = None,
        timeout: Optional[int] = None,
        capture_output: bool = True,
        working_dir: Optional[str] = None,
        env_vars: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Execute a Linux command.
        
        Args:
            command: Command to execute (string or list)
            distro: WSL distribution to use (None for default)
            timeout: Command timeout in seconds
            capture_output: Whether to capture stdout/stderr
            working_dir: Working directory
            env_vars: Environment variables to set
            
        Returns:
            Dictionary containing execution results
        """
        start_time = time.time()
        execution_id = f"exec_{int(start_time)}_{len(self.execution_history)}"
        
        # Prepare command
        if isinstance(command, list):
            cmd_str = ' '.join(command)
            full_command = command
        else:
            cmd_str = command
            full_command = ["/bin/bash", "-c", command]
        
        # Safety check
        if self.config.get("safe_mode", True):
            if self._is_dangerous_command(cmd_str):
                return {
                    "status": "error",
                    "message": f"Dangerous command blocked: {cmd_str}",
                    "execution_id": execution_id
                }
        
        # Log command if enabled
        if self.config.get("log_commands", True):
            logger.info(f"Executing Linux command: {cmd_str}")
        
        try:
            # Build final command based on environment
            if sys.platform == "win32" and self.wsl_distros:
                # Running on Windows with WSL
                target_distro = distro or self.default_distro
                if target_distro:
                    wsl_command = ["wsl", "-d", target_distro]
                else:
                    wsl_command = ["wsl"]
                
                # Add working directory
                if working_dir:
                    wsl_command.extend(["--cd", working_dir])
                
                # Add environment variables
                if env_vars:
                    for key, value in env_vars.items():
                        wsl_command.extend(["--exec", "/bin/bash", "-c", f"export {key}='{value}' && {cmd_str}"])
                        break  # WSL doesn't support multiple env vars easily
                else:
                    wsl_command.extend(full_command)
                
                final_command = wsl_command
                
            else:
                # Native Linux or inside WSL
                final_command = full_command
                
                # Set working directory
                if working_dir and not os.path.exists(working_dir):
                    os.makedirs(working_dir, exist_ok=True)
            
            # Execute command
            process_kwargs = {
                "timeout": timeout or self.config["timeout"],
                "capture_output": capture_output,
                "text": True
            }
            
            # Set working directory for native Linux
            if sys.platform != "win32" and working_dir:
                process_kwargs["cwd"] = working_dir
            
            # Set environment variables for native Linux
            if sys.platform != "win32" and env_vars:
                env = os.environ.copy()
                env.update(env_vars)
                process_kwargs["env"] = env
            
            result = subprocess.run(final_command, **process_kwargs)
            
            execution_time = time.time() - start_time
            
            # Prepare result
            execution_result = {
                "status": "success" if result.returncode == 0 else "error",
                "execution_id": execution_id,
                "command": cmd_str,
                "return_code": result.returncode,
                "execution_time": execution_time,
                "distro": distro or self.default_distro,
                "timestamp": datetime.now().isoformat()
            }
            
            if capture_output:
                execution_result.update({
                    "stdout": result.stdout,
                    "stderr": result.stderr
                })
            
            # Add to history
            self.execution_history.append(execution_result)
            
            logger.info(f"Command executed in {execution_time:.2f}s with return code {result.returncode}")
            
            return execution_result
            
        except subprocess.TimeoutExpired as e:
            execution_time = time.time() - start_time
            error_result = {
                "status": "timeout",
                "execution_id": execution_id,
                "command": cmd_str,
                "execution_time": execution_time,
                "error": "Command timed out",
                "timeout": timeout or self.config["timeout"],
                "timestamp": datetime.now().isoformat()
            }
            
            self.execution_history.append(error_result)
            logger.error(f"Command timed out after {execution_time:.2f}s")
            
            return error_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_result = {
                "status": "error",
                "execution_id": execution_id,
                "command": cmd_str,
                "execution_time": execution_time,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
            self.execution_history.append(error_result)
            logger.error(f"Command execution failed: {e}")
            
            return error_result
    
    def _is_dangerous_command(self, command: str) -> bool:
        """Check if a command is potentially dangerous."""
        dangerous_patterns = [
            "rm -rf /",
            ":(){ :|:& };:",  # Fork bomb
            "dd if=/dev/zero",
            "mkfs.",
            "fdisk",
            "parted",
            "> /dev/sd",
            "chmod 777 /",
            "chown -R root:root /",
            "shutdown",
            "reboot",
            "halt",
            "init 0",
            "kill -9 -1",
            "pkill -f .",
            ":(){ :|: & };:",
            "curl | bash",
            "wget | bash"
        ]
        
        command_lower = command.lower()
        for pattern in dangerous_patterns:
            if pattern.lower() in command_lower:
                return True
        
        return False
    
    def execute_script(
        self, 
        script_content: str, 
        script_type: str = "bash",
        distro: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a script in Linux environment.
        
        Args:
            script_content: Content of the script
            script_type: Type of script (bash, python, etc.)
            distro: WSL distribution to use
            **kwargs: Additional arguments passed to execute_command
            
        Returns:
            Dictionary containing execution results
        """
        try:
            # Create temporary script file
            with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{script_type}', delete=False) as f:
                f.write(script_content)
                temp_script = f.name
            
            # Make script executable if bash
            if script_type == "bash":
                os.chmod(temp_script, 0o755)
                command = f"bash {temp_script}"
            elif script_type == "python":
                command = f"python3 {temp_script}"
            elif script_type == "sh":
                command = f"sh {temp_script}"
            else:
                command = temp_script
            
            # Execute script
            result = self.execute_command(command, distro=distro, **kwargs)
            
            # Clean up temporary file
            try:
                os.unlink(temp_script)
            except Exception as e:
                logger.warning(f"Failed to clean up temp script: {e}")
            
            result["script_type"] = script_type
            result["script_content"] = script_content[:200] + "..." if len(script_content) > 200 else script_content
            
            return result
            
        except Exception as e:
            logger.error(f"Script execution failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "script_type": script_type
            }
    
    def install_package(
        self, 
        package_name: str, 
        package_manager: str = "auto",
        distro: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Install a package in the Linux environment.
        
        Args:
            package_name: Name of the package to install
            package_manager: Package manager to use (auto, apt, yum, pacman, etc.)
            distro: WSL distribution to use
            
        Returns:
            Dictionary containing installation results
        """
        if not self.config.get("auto_install_deps", True):
            return {
                "status": "error",
                "message": "Package installation disabled in config"
            }
        
        # Detect package manager if auto
        if package_manager == "auto":
            package_manager = self._detect_package_manager(distro)
        
        # Build installation command
        install_commands = {
            "apt": f"sudo apt update && sudo apt install -y {package_name}",
            "yum": f"sudo yum install -y {package_name}",
            "dnf": f"sudo dnf install -y {package_name}",
            "pacman": f"sudo pacman -S --noconfirm {package_name}",
            "zypper": f"sudo zypper install -y {package_name}",
            "apk": f"sudo apk add {package_name}"
        }
        
        if package_manager not in install_commands:
            return {
                "status": "error",
                "message": f"Unsupported package manager: {package_manager}"
            }
        
        command = install_commands[package_manager]
        
        logger.info(f"Installing package '{package_name}' using '{package_manager}'")
        
        return self.execute_command(command, distro=distro, timeout=600)  # 10 minutes for package installation
    
    def _detect_package_manager(self, distro: Optional[str] = None) -> str:
        """Detect the package manager available in the Linux environment."""
        # Check for various package managers
        managers_to_check = [
            ("apt", "which apt"),
            ("yum", "which yum"),
            ("dnf", "which dnf"),
            ("pacman", "which pacman"),
            ("zypper", "which zypper"),
            ("apk", "which apk")
        ]
        
        for manager, check_cmd in managers_to_check:
            result = self.execute_command(check_cmd, distro=distro, capture_output=True)
            if result["status"] == "success":
                logger.info(f"Detected package manager: {manager}")
                return manager
        
        # Default to apt for Ubuntu/Debian (common in WSL)
        logger.warning("Could not detect package manager, defaulting to apt")
        return "apt"
    
    def create_virtual_environment(
        self, 
        venv_name: str, 
        python_version: str = "3",
        distro: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a Python virtual environment.
        
        Args:
            venv_name: Name of the virtual environment
            python_version: Python version to use
            distro: WSL distribution to use
            
        Returns:
            Dictionary containing creation results
        """
        working_dir = self.config.get("working_directory", "/tmp/ai_symbiote")
        venv_path = f"{working_dir}/venvs/{venv_name}"
        
        # Create venv directory
        mkdir_result = self.execute_command(f"mkdir -p {working_dir}/venvs", distro=distro)
        if mkdir_result["status"] != "success":
            return mkdir_result
        
        # Install python3-venv if needed
        self.install_package("python3-venv", distro=distro)
        
        # Create virtual environment
        create_cmd = f"python{python_version} -m venv {venv_path}"
        result = self.execute_command(create_cmd, distro=distro)
        
        if result["status"] == "success":
            result["venv_path"] = venv_path
            result["activation_command"] = f"source {venv_path}/bin/activate"
            logger.info(f"Virtual environment '{venv_name}' created at {venv_path}")
        
        return result
    
    def execute_in_venv(
        self, 
        venv_name: str, 
        command: str,
        distro: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a command inside a Python virtual environment.
        
        Args:
            venv_name: Name of the virtual environment
            command: Command to execute
            distro: WSL distribution to use
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing execution results
        """
        working_dir = self.config.get("working_directory", "/tmp/ai_symbiote")
        venv_path = f"{working_dir}/venvs/{venv_name}"
        
        # Build command with venv activation
        venv_command = f"source {venv_path}/bin/activate && {command}"
        
        return self.execute_command(venv_command, distro=distro, **kwargs)
    
    def monitor_system_resources(self, distro: Optional[str] = None) -> Dict[str, Any]:
        """
        Monitor Linux system resources.
        
        Args:
            distro: WSL distribution to use
            
        Returns:
            Dictionary containing system resource information
        """
        commands = {
            "cpu_usage": "top -bn1 | grep 'Cpu(s)' | awk '{print $2}' | cut -d'%' -f1",
            "memory_usage": "free -m | awk 'NR==2{printf \"%.2f%%\", $3*100/$2}'",
            "disk_usage": "df -h / | awk 'NR==2{print $5}'",
            "uptime": "uptime -p",
            "load_average": "uptime | awk -F'load average:' '{print $2}'",
            "process_count": "ps aux | wc -l"
        }
        
        results = {}
        
        for metric, command in commands.items():
            result = self.execute_command(command, distro=distro, capture_output=True)
            if result["status"] == "success":
                results[metric] = result["stdout"].strip()
            else:
                results[metric] = "N/A"
        
        results["timestamp"] = datetime.now().isoformat()
        results["distro"] = distro or self.default_distro
        
        return results
    
    def setup_development_environment(
        self, 
        languages: List[str] = None,
        distro: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Set up a development environment with common tools and languages.
        
        Args:
            languages: List of programming languages to set up
            distro: WSL distribution to use
            
        Returns:
            Dictionary containing setup results
        """
        if languages is None:
            languages = ["python", "nodejs", "git"]
        
        setup_results = {
            "status": "success",
            "installed_packages": [],
            "failed_packages": [],
            "errors": []
        }
        
        # Update package manager
        update_result = self.execute_command("sudo apt update", distro=distro)
        if update_result["status"] != "success":
            setup_results["errors"].append("Failed to update package manager")
        
        # Install common development tools
        common_packages = [
            "curl", "wget", "vim", "nano", "htop", "tree", "unzip", "zip",
            "build-essential", "software-properties-common"
        ]
        
        for package in common_packages:
            result = self.install_package(package, distro=distro)
            if result["status"] == "success":
                setup_results["installed_packages"].append(package)
            else:
                setup_results["failed_packages"].append(package)
                setup_results["errors"].append(f"Failed to install {package}")
        
        # Install language-specific packages
        language_packages = {
            "python": ["python3", "python3-pip", "python3-venv", "python3-dev"],
            "nodejs": ["nodejs", "npm"],
            "git": ["git"],
            "docker": ["docker.io"],
            "java": ["openjdk-11-jdk"],
            "go": ["golang-go"],
            "rust": ["rustc", "cargo"]
        }
        
        for language in languages:
            if language in language_packages:
                for package in language_packages[language]:
                    result = self.install_package(package, distro=distro)
                    if result["status"] == "success":
                        setup_results["installed_packages"].append(f"{language}:{package}")
                    else:
                        setup_results["failed_packages"].append(f"{language}:{package}")
                        setup_results["errors"].append(f"Failed to install {language}:{package}")
        
        # Set up working directory
        working_dir = self.config.get("working_directory", "/tmp/ai_symbiote")
        mkdir_result = self.execute_command(f"mkdir -p {working_dir}", distro=distro)
        if mkdir_result["status"] == "success":
            setup_results["working_directory"] = working_dir
        
        if setup_results["failed_packages"]:
            setup_results["status"] = "partial_success"
        
        logger.info(f"Development environment setup completed: {len(setup_results['installed_packages'])} packages installed")
        
        return setup_results
    
    def get_execution_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get command execution history.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of execution history entries
        """
        if limit is not None:
            return self.execution_history[-limit:]
        return self.execution_history
    
    def get_status(self) -> Dict[str, Any]:
        """Get current Linux execution engine status."""
        # Determine overall status
        if self.is_wsl and len(self.wsl_distros) > 0:
            overall_status = "active"
        elif self.is_wsl and len(self.wsl_distros) == 0:
            overall_status = "wsl_available_no_distros"
        else:
            overall_status = "no_wsl"
        
        return {
            "status": overall_status,  # Add status field that tests expect
            "is_wsl": self.is_wsl,
            "wsl_distros": self.wsl_distros,
            "default_distro": self.default_distro,
            "config": self.config,
            "execution_count": len(self.execution_history),
            "active_sessions": len(self.active_sessions),
            "system_platform": sys.platform
        }


def main():
    """Test the Linux Execution Engine."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Linux Execution Engine")
    parser.add_argument("--command", "-c", help="Command to execute")
    parser.add_argument("--distro", "-d", help="WSL distribution to use")
    parser.add_argument("--setup-dev", action="store_true", help="Set up development environment")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--monitor", action="store_true", help="Monitor system resources")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create execution engine
    engine = LinuxExecutionEngine()
    
    if args.command:
        print(f"Executing command: {args.command}")
        result = engine.execute_command(args.command, distro=args.distro)
        print(json.dumps(result, indent=2))
        
    elif args.setup_dev:
        print("Setting up development environment...")
        result = engine.setup_development_environment(distro=args.distro)
        print(json.dumps(result, indent=2))
        
    elif args.status:
        print("Linux Execution Engine status:")
        status = engine.get_status()
        print(json.dumps(status, indent=2))
        
    elif args.monitor:
        print("System resource monitoring:")
        resources = engine.monitor_system_resources(distro=args.distro)
        print(json.dumps(resources, indent=2))
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()