#!/usr/bin/env python3
"""
Obvivlorum Unified Launcher
Consolidated startup script that replaces all the various batch files and startup scripts
"""

import os
import sys
import json
import argparse
import subprocess
import platform
import signal
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import threading

@dataclass
class LaunchMode:
    """Launch mode configuration"""
    name: str
    description: str
    command: List[str]
    environment: Dict[str, str]
    working_dir: Optional[str] = None
    requires_admin: bool = False
    background: bool = False

class UnifiedLauncher:
    """
    Unified launcher that handles all system startup modes
    Replaces the various batch files and startup scripts
    """
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.config_file = self.base_dir / "launcher_config.json"
        self.modes: Dict[str, LaunchMode] = {}
        self.running_processes: List[subprocess.Popen] = []
        self.shutdown_flag = threading.Event()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self._load_launch_modes()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\nReceived signal {signum}, shutting down...")
        self.shutdown_flag.set()
        self._cleanup_processes()
        sys.exit(0)
    
    def _load_launch_modes(self):
        """Load or create launch mode configurations"""
        default_modes = {
            "core": LaunchMode(
                name="core",
                description="Start core orchestrator (default mode)",
                command=[sys.executable, "core_orchestrator.py"],
                environment={"OBVIVLORUM_MODE": "core"}
            ),
            "legacy": LaunchMode(
                name="legacy",
                description="Start legacy AI symbiote system",
                command=[sys.executable, "ai_symbiote.py"],
                environment={"OBVIVLORUM_MODE": "legacy"}
            ),
            "gui": LaunchMode(
                name="gui",
                description="Start GUI interface",
                command=[sys.executable, "ai_symbiote_gui.py"],
                environment={"OBVIVLORUM_MODE": "gui"}
            ),
            "background": LaunchMode(
                name="background",
                description="Start in background mode",
                command=[sys.executable, "ai_symbiote.py", "--background", "--persistent"],
                environment={"OBVIVLORUM_MODE": "background"},
                background=True
            ),
            "sandbox": LaunchMode(
                name="sandbox",
                description="Start in Docker sandbox",
                command=["docker-compose", "up", "-d"],
                environment={"OBVIVLORUM_MODE": "sandbox"}
            ),
            "test": LaunchMode(
                name="test",
                description="Run system tests",
                command=[sys.executable, "QUICK_SYSTEM_TEST.py"],
                environment={"OBVIVLORUM_MODE": "test"}
            ),
            "web": LaunchMode(
                name="web",
                description="Start web interface",
                command=["python", "-m", "http.server", "8000"],
                environment={"OBVIVLORUM_MODE": "web"},
                working_dir=str(self.base_dir / "web" / "frontend")
            ),
            "parrot": LaunchMode(
                name="parrot",
                description="Start with ParrotOS WSL integration",
                command=[sys.executable, "ai_symbiote.py", "--linux-mode"],
                environment={
                    "OBVIVLORUM_MODE": "parrot",
                    "LINUX_DISTRO": "ParrotOS",
                    "WSL_ENABLED": "true"
                }
            ),
            "security": LaunchMode(
                name="security",
                description="Start security audit mode",
                command=[sys.executable, "security_manager.py", "--audit"],
                environment={"OBVIVLORUM_MODE": "security"},
                requires_admin=True
            ),
            "minimal": LaunchMode(
                name="minimal",
                description="Start minimal core without persistence",
                command=[sys.executable, "core_orchestrator.py", "--minimal"],
                environment={
                    "OBVIVLORUM_MODE": "minimal",
                    "DISABLE_PERSISTENCE": "true"
                }
            )
        }
        
        # Load from config file if exists, otherwise use defaults
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                    
                # Convert config data to LaunchMode objects
                for name, data in config_data.items():
                    self.modes[name] = LaunchMode(
                        name=data["name"],
                        description=data["description"],
                        command=data["command"],
                        environment=data.get("environment", {}),
                        working_dir=data.get("working_dir"),
                        requires_admin=data.get("requires_admin", False),
                        background=data.get("background", False)
                    )
            except Exception as e:
                print(f"Warning: Could not load launcher config: {e}")
                self.modes = default_modes
        else:
            self.modes = default_modes
            self._save_config()
    
    def _save_config(self):
        """Save current launch modes to config file"""
        try:
            config_data = {}
            for name, mode in self.modes.items():
                config_data[name] = {
                    "name": mode.name,
                    "description": mode.description,
                    "command": mode.command,
                    "environment": mode.environment,
                    "working_dir": mode.working_dir,
                    "requires_admin": mode.requires_admin,
                    "background": mode.background
                }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Warning: Could not save launcher config: {e}")
    
    def list_modes(self):
        """List all available launch modes"""
        print("\nAvailable Launch Modes:")
        print("=" * 50)
        
        for name, mode in sorted(self.modes.items()):
            status = ""
            if mode.requires_admin:
                status += " [ADMIN]"
            if mode.background:
                status += " [BACKGROUND]"
                
            print(f"{name:12} - {mode.description}{status}")
        
        print(f"\nTotal: {len(self.modes)} modes available")
    
    def _is_admin(self) -> bool:
        """Check if running with admin privileges"""
        try:
            if platform.system() == "Windows":
                import ctypes
                return ctypes.windll.shell32.IsUserAnAdmin()
            else:
                return os.geteuid() == 0
        except:
            return False
    
    def _check_requirements(self, mode: LaunchMode) -> bool:
        """Check if requirements are met for launch mode"""
        if mode.requires_admin and not self._is_admin():
            print(f"Error: Mode '{mode.name}' requires administrator privileges")
            return False
            
        # Check if command exists
        if mode.command:
            cmd_path = mode.command[0]
            if not self._command_exists(cmd_path):
                print(f"Error: Command '{cmd_path}' not found for mode '{mode.name}'")
                return False
        
        # Check working directory
        if mode.working_dir and not Path(mode.working_dir).exists():
            print(f"Error: Working directory '{mode.working_dir}' does not exist")
            return False
            
        return True
    
    def _command_exists(self, command: str) -> bool:
        """Check if a command exists"""
        try:
            if command == sys.executable or command == "python":
                return True
            
            # Check if it's a Python script in current directory
            if command.endswith('.py') and (self.base_dir / command).exists():
                return True
            
            # Check if command exists in PATH
            result = subprocess.run(
                ["where" if platform.system() == "Windows" else "which", command],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
            
        except:
            return False
    
    def launch_mode(self, mode_name: str, args: List[str] = None) -> bool:
        """Launch a specific mode"""
        if mode_name not in self.modes:
            print(f"Error: Unknown launch mode '{mode_name}'")
            return False
        
        mode = self.modes[mode_name]
        
        # Check requirements
        if not self._check_requirements(mode):
            return False
        
        # Prepare command
        command = mode.command.copy()
        if args:
            command.extend(args)
        
        # Prepare environment
        env = os.environ.copy()
        env.update(mode.environment)
        
        # Set working directory
        cwd = mode.working_dir or str(self.base_dir)
        
        try:
            print(f"Launching mode: {mode.name}")
            print(f"Description: {mode.description}")
            print(f"Command: {' '.join(command)}")
            print(f"Working directory: {cwd}")
            
            if mode.background:
                print("Starting in background mode...")
                
            # Launch process
            if platform.system() == "Windows" and mode.background:
                # Windows background process
                process = subprocess.Popen(
                    command,
                    env=env,
                    cwd=cwd,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            else:
                # Standard process
                process = subprocess.Popen(
                    command,
                    env=env,
                    cwd=cwd,
                    stdout=subprocess.PIPE if mode.background else None,
                    stderr=subprocess.PIPE if mode.background else None
                )
            
            self.running_processes.append(process)
            
            if mode.background:
                print(f"Process started in background (PID: {process.pid})")
                return True
            else:
                # Wait for process completion
                return_code = process.wait()
                print(f"Process completed with return code: {return_code}")
                return return_code == 0
                
        except Exception as e:
            print(f"Error launching mode '{mode_name}': {e}")
            return False
    
    def _cleanup_processes(self):
        """Clean up running processes"""
        for process in self.running_processes:
            try:
                if process.poll() is None:  # Process is still running
                    print(f"Terminating process {process.pid}...")
                    process.terminate()
                    
                    # Wait a bit for graceful termination
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        print(f"Force killing process {process.pid}...")
                        process.kill()
                        
            except Exception as e:
                print(f"Error cleaning up process: {e}")
        
        self.running_processes.clear()
    
    def status(self):
        """Show status of running processes"""
        if not self.running_processes:
            print("No processes currently running.")
            return
        
        print("\nRunning Processes:")
        print("=" * 40)
        
        active_count = 0
        for i, process in enumerate(self.running_processes):
            try:
                status = "RUNNING" if process.poll() is None else "STOPPED"
                print(f"Process {i+1}: PID {process.pid} - {status}")
                if status == "RUNNING":
                    active_count += 1
            except Exception as e:
                print(f"Process {i+1}: Error checking status - {e}")
        
        print(f"\nActive processes: {active_count}/{len(self.running_processes)}")
    
    def stop_all(self):
        """Stop all running processes"""
        if not self.running_processes:
            print("No processes to stop.")
            return
            
        print("Stopping all processes...")
        self._cleanup_processes()
        print("All processes stopped.")
    
    def interactive_menu(self):
        """Show interactive menu for mode selection"""
        while not self.shutdown_flag.is_set():
            print("\n" + "=" * 60)
            print("            OBVIVLORUM UNIFIED LAUNCHER")
            print("=" * 60)
            
            self.list_modes()
            
            print("\nSpecial Commands:")
            print("  status    - Show running processes")
            print("  stop      - Stop all processes")
            print("  quit      - Exit launcher")
            print("  help      - Show this menu")
            
            try:
                choice = input("\nSelect mode or command: ").strip().lower()
                
                if choice in ["quit", "exit", "q"]:
                    break
                elif choice == "status":
                    self.status()
                elif choice == "stop":
                    self.stop_all()
                elif choice in ["help", "h", ""]:
                    continue
                elif choice in self.modes:
                    self.launch_mode(choice)
                else:
                    print(f"Unknown option: {choice}")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("\nShutting down launcher...")
        self._cleanup_processes()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Obvivlorum Unified Launcher",
        epilog="Use --list to see available modes"
    )
    
    parser.add_argument("mode", nargs="?", help="Launch mode to start")
    parser.add_argument("--list", "-l", action="store_true", help="List available modes")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode selection")
    parser.add_argument("--status", "-s", action="store_true", help="Show process status")
    parser.add_argument("--stop", action="store_true", help="Stop all processes")
    parser.add_argument("--args", nargs="*", help="Additional arguments for the selected mode")
    
    args = parser.parse_args()
    
    launcher = UnifiedLauncher()
    
    try:
        if args.list:
            launcher.list_modes()
        elif args.status:
            launcher.status()
        elif args.stop:
            launcher.stop_all()
        elif args.interactive:
            launcher.interactive_menu()
        elif args.mode:
            success = launcher.launch_mode(args.mode, args.args or [])
            sys.exit(0 if success else 1)
        else:
            # Default: start core mode
            print("No mode specified, starting default 'core' mode...")
            success = launcher.launch_mode("core")
            sys.exit(0 if success else 1)
            
    except KeyboardInterrupt:
        print("\nLauncher interrupted by user")
        launcher._cleanup_processes()
        sys.exit(1)
    except Exception as e:
        print(f"Launcher error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()