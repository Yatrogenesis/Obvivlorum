#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI Symbiote Startup Script
==========================

Simple startup script for the AI Symbiote system with interactive configuration.

Author: Francisco Molina
ORCID: https://orcid.org/0009-0008-6093-8267
Email: pako.molina@gmail.com
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path

def print_banner():
    """Print the AI Symbiote banner."""
    banner = """
===============================================================
                       AI SYMBIOTE                           
              Adaptive AI Assistant System                    
                                                             
  [*] AION Protocol Integration                                
  [*] Obvivlorum Bridge                                        
  [*] Cross-Platform Execution                                
  [*] Adaptive Task Facilitation                              
  [*] Security & Safety                                       
                                                             
  Author: Francisco Molina                                    
  Version: 1.0                                                
===============================================================
"""
    print(banner)

def check_requirements():
    """Check system requirements."""
    print("[CHECK] Checking system requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("[ERROR] Python 3.8+ required. Current version:", sys.version)
        return False
    print("[OK] Python version:", sys.version.split()[0])
    
    # Check if we're in the right directory
    if not Path("ai_symbiote.py").exists():
        print("[ERROR] ai_symbiote.py not found. Please run this script from the Obvivlorum directory.")
        return False
    print("[OK] Main script found")
    
    # Check AION directory
    if not Path("AION").exists():
        print("[ERROR] AION directory not found")
        return False
    print("[OK] AION directory found")
    
    # Check required Python packages
    try:
        import numpy
        print("[OK] NumPy available")
    except ImportError:
        print("[ERROR] NumPy not installed. Run: pip install -r AION/requirements.txt")
        return False
    
    return True

def get_user_preferences():
    """Get user preferences for the AI Symbiote."""
    print("\n[CONFIG] Configuration Setup")
    print("=" * 50)
    
    # Get user ID
    default_user = os.environ.get("USER", os.environ.get("USERNAME", "user"))
    user_id = input(f"Enter your user ID [{default_user}]: ").strip()
    if not user_id:
        user_id = default_user
    
    # Get mode preferences
    print("\nSelect operating mode:")
    print("1. Interactive (recommended for first-time users)")
    print("2. Background (runs in background)")
    print("3. Testing (safe mode with limited functionality)")
    
    mode_choice = input("Choose mode [1-3, default=1]: ").strip()
    if mode_choice not in ["1", "2", "3"]:
        mode_choice = "1"
    
    background = mode_choice == "2"
    testing = mode_choice == "3"
    
    # Persistence options (Windows only)
    install_persistence = False
    if sys.platform == "win32" and not testing:
        persist = input("Install Windows persistence? [y/N]: ").strip().lower()
        install_persistence = persist in ["y", "yes"]
    
    # Advanced options
    print("\nAdvanced options:")
    enable_learning = input("Enable adaptive learning? [Y/n]: ").strip().lower()
    enable_learning = enable_learning not in ["n", "no"]
    
    enable_linux = input("Enable Linux execution? [Y/n]: ").strip().lower()
    enable_linux = enable_linux not in ["n", "no"]
    
    enable_suggestions = input("Enable proactive suggestions? [Y/n]: ").strip().lower()
    enable_suggestions = enable_suggestions not in ["n", "no"]
    
    return {
        "user_id": user_id,
        "background": background,
        "testing": testing,
        "install_persistence": install_persistence,
        "enable_learning": enable_learning,
        "enable_linux": enable_linux,
        "enable_suggestions": enable_suggestions
    }

def create_startup_config(preferences):
    """Create a startup configuration file."""
    config = {
        "user_id": preferences["user_id"],
        "components": {
            "aion_protocol": {
                "enabled": True,
                "config_file": "AION/config.json"
            },
            "obvivlorum_bridge": {
                "enabled": True,
                "mock_mode": True
            },
            "windows_persistence": {
                "enabled": sys.platform == "win32" and not preferences["testing"],
                "auto_install": preferences.get("install_persistence", False)
            },
            "linux_executor": {
                "enabled": preferences["enable_linux"],
                "safe_mode": True
            },
            "task_facilitator": {
                "enabled": True,
                "learning_enabled": preferences["enable_learning"],
                "proactive_suggestions": preferences["enable_suggestions"]
            }
        },
        "system": {
            "heartbeat_interval": 60,
            "background_mode": preferences["background"],
            "auto_restart": True
        },
        "security": {
            "safe_mode": preferences["testing"],
            "blocked_commands": ["rm -rf /", "format", "fdisk", "del /f /s /q C:\\*"]
        }
    }
    
    config_file = f"config_{preferences['user_id']}.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"[OK] Configuration saved to {config_file}")
    return config_file

def install_dependencies():
    """Install required dependencies."""
    print("\n[INSTALL] Installing dependencies...")
    
    try:
        # Install AION requirements
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "AION/requirements.txt"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print("[ERROR] Failed to install dependencies:")
            print(result.stderr)
            return False
        
        print("[OK] Dependencies installed successfully")
        return True
        
    except Exception as e:
        print(f"[ERROR] Error installing dependencies: {e}")
        return False

def run_initial_tests(config_file, user_id):
    """Run initial system tests."""
    print("\n[TEST] Running initial tests...")
    
    try:
        # Test system status
        result = subprocess.run([
            sys.executable, "ai_symbiote.py", 
            "--status", 
            "--config", config_file,
            "--user-id", user_id
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            print("[ERROR] System status test failed:")
            print(result.stderr)
            return False
        
        print("[OK] System status test passed")
        
        # Test AION protocol
        result = subprocess.run([
            sys.executable, "ai_symbiote.py",
            "--test-protocol", "ALPHA",
            "--config", config_file,
            "--user-id", user_id
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("[OK] AION protocol test passed")
        else:
            print("[WARN]  AION protocol test had issues (this may be normal)")
        
        return True
        
    except subprocess.TimeoutExpired:
        print("[WARN]  Tests timed out (this may be normal for first run)")
        return True
    except Exception as e:
        print(f"[ERROR] Error running tests: {e}")
        return False

def start_symbiote(config_file, preferences):
    """Start the AI Symbiote system."""
    print("\n[START] Starting AI Symbiote...")
    
    cmd = [
        sys.executable, "ai_symbiote.py",
        "--config", config_file,
        "--user-id", preferences["user_id"]
    ]
    
    if preferences["background"]:
        cmd.append("--background")
    
    if preferences.get("install_persistence", False):
        cmd.append("--persistent")
    
    try:
        if preferences["background"]:
            # Start in background
            process = subprocess.Popen(cmd, 
                                     creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
            print(f"[OK] AI Symbiote started in background (PID: {process.pid})")
            print(f"   Use 'python ai_symbiote.py --status --user-id {preferences['user_id']}' to check status")
        else:
            # Start in interactive mode
            print("[OK] Starting AI Symbiote in interactive mode...")
            print("   Press Ctrl+C to stop")
            print("=" * 50)
            subprocess.run(cmd)
    
    except KeyboardInterrupt:
        print("\n[STOP] AI Symbiote stopped by user")
    except Exception as e:
        print(f"[ERROR] Error starting AI Symbiote: {e}")
        return False
    
    return True

def main():
    """Main startup function."""
    print_banner()
    
    # Check requirements
    if not check_requirements():
        print("\n[ERROR] Requirements check failed. Please fix the issues above and try again.")
        input("Press Enter to exit...")
        return 1
    
    # Get user preferences
    preferences = get_user_preferences()
    
    # Install dependencies
    if not install_dependencies():
        print("\n[ERROR] Dependency installation failed.")
        input("Press Enter to exit...")
        return 1
    
    # Create configuration
    config_file = create_startup_config(preferences)
    
    # Run initial tests
    if not run_initial_tests(config_file, preferences["user_id"]):
        print("\n[WARN]  Initial tests had issues, but continuing...")
    
    # Install persistence if requested
    if preferences.get("install_persistence", False):
        print("\n[SETUP] Installing Windows persistence...")
        try:
            subprocess.run([
                sys.executable, "ai_symbiote.py",
                "--install-persistence",
                "--user-id", preferences["user_id"]
            ], timeout=60)
            print("[OK] Persistence installation completed")
        except Exception as e:
            print(f"[WARN]  Persistence installation had issues: {e}")
    
    # Final confirmation
    print(f"\n[SUCCESS] Setup completed for user: {preferences['user_id']}")
    print("=" * 50)
    
    if preferences["background"]:
        input("Press Enter to start AI Symbiote in background mode...")
    else:
        print("AI Symbiote will start in interactive mode.")
        print("You can stop it anytime with Ctrl+C")
        input("Press Enter to continue...")
    
    # Start the system
    if start_symbiote(config_file, preferences):
        print("\n[OK] AI Symbiote startup completed successfully!")
        return 0
    else:
        print("\n[ERROR] AI Symbiote startup failed.")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n[STOP] Startup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error during startup: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")
        sys.exit(1)