#!/usr/bin/env python3
"""
Execution Mode Manager for Obvivlorum
Handles toggling between different consciousness frameworks and system configurations
"""

import json
import os
import sys
import logging
from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path

class ExecutionMode(Enum):
    """Supported execution modes"""
    STANDARD = "standard"
    TOPOESPECTRO = "topoespectro"
    RESEARCH = "research"
    GUI = "gui"

@dataclass
class ModeConfiguration:
    """Configuration for a specific execution mode"""
    name: str
    description: str
    entry_point: str
    features: List[str]
    dependencies: List[str]
    performance_requirements: Dict[str, Any]
    config_overrides: Dict[str, Any]

class ExecutionModeManager:
    """
    Manages execution modes and configuration toggles for Obvivlorum system
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path(__file__).parent
        self.current_mode = ExecutionMode.STANDARD
        
        # Load configurations
        self.claude_config = self._load_claude_config()
        self.topoespectral_config = self._load_topoespectral_config()
        
        # Initialize logger
        self.logger = self._setup_logger()
        
    def _load_claude_config(self) -> Dict[str, Any]:
        """Load Claude Code configuration"""
        claude_file = self.config_dir.parent / ".claude.json"
        try:
            with open(claude_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.warning(f"Claude config not found at {claude_file}")
            return {}
    
    def _load_topoespectral_config(self) -> Dict[str, Any]:
        """Load Topo-Spectral configuration"""
        topo_file = self.config_dir / "topoespectral_config.json"
        try:
            with open(topo_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.warning(f"Topo-Spectral config not found at {topo_file}")
            return {"topo_spectral_framework": {"enabled": False}}
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for mode manager"""
        logger = logging.getLogger("ExecutionModeManager")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def set_mode(self, mode: ExecutionMode) -> bool:
        """
        Set the execution mode and configure system accordingly
        
        Args:
            mode: Target execution mode
            
        Returns:
            bool: Success status
        """
        try:
            self.logger.info(f"Switching to {mode.value} mode")
            
            # Update configurations based on mode
            if mode == ExecutionMode.STANDARD:
                self._configure_standard_mode()
            elif mode == ExecutionMode.TOPOESPECTRO:
                self._configure_topoespectro_mode()
            elif mode == ExecutionMode.RESEARCH:
                self._configure_research_mode()
            elif mode == ExecutionMode.GUI:
                self._configure_gui_mode()
            else:
                raise ValueError(f"Unsupported mode: {mode}")
            
            self.current_mode = mode
            self.logger.info(f"Successfully switched to {mode.value} mode")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to switch to {mode.value} mode: {e}")
            return False
    
    def _configure_standard_mode(self):
        """Configure system for standard IIT/GWT consciousness metrics"""
        # Disable Topo-Spectral framework
        self.topoespectral_config["topo_spectral_framework"]["enabled"] = False
        
        # Enable fallback metrics
        self.topoespectral_config["fallback_consciousness_metrics"]["enabled"] = True
        
        # Set environment variables
        os.environ["OBVIVLORUM_MODE"] = "standard"
        os.environ["OBVIVLORUM_USE_TOPO_SPECTRAL"] = "false"
        
        # Update performance settings for minimal resource usage
        self._set_performance_mode("minimal")
        
        self.logger.info("Standard mode: Using basic IIT/GWT consciousness metrics")
    
    def _configure_topoespectro_mode(self):
        """Configure system for Topo-Spectral consciousness framework"""
        # Enable Topo-Spectral framework
        self.topoespectral_config["topo_spectral_framework"]["enabled"] = True
        
        # Enable all Topo-Spectral components
        topo_config = self.topoespectral_config["topo_spectral_framework"]
        topo_config["spectral_information_integration"]["enabled"] = True
        topo_config["topological_resilience"]["enabled"] = True
        topo_config["temporal_synchronization"]["enabled"] = True
        
        # Set environment variables
        os.environ["OBVIVLORUM_MODE"] = "topoespectro"
        os.environ["OBVIVLORUM_USE_TOPO_SPECTRAL"] = "true"
        
        # Update performance settings for research workloads
        self._set_performance_mode("research")
        
        self.logger.info("Topo-Spectral mode: Using Francisco Molina's consciousness framework")
    
    def _configure_research_mode(self):
        """Configure system for full research mode with both frameworks"""
        # Enable both frameworks
        self.topoespectral_config["topo_spectral_framework"]["enabled"] = True
        self.topoespectral_config["fallback_consciousness_metrics"]["enabled"] = True
        
        # Enable clinical validation
        topo_config = self.topoespectral_config["topo_spectral_framework"]
        topo_config["clinical_validation"]["enabled"] = True
        
        # Set environment variables
        os.environ["OBVIVLORUM_MODE"] = "research"
        os.environ["OBVIVLORUM_USE_TOPO_SPECTRAL"] = "true"
        os.environ["OBVIVLORUM_RESEARCH_VALIDATION"] = "true"
        
        # Maximum performance settings
        self._set_performance_mode("maximum")
        
        self.logger.info("Research mode: Both IIT/GWT and Topo-Spectral frameworks enabled")
    
    def _configure_gui_mode(self):
        """Configure system for GUI interface with mode selection"""
        # Enable GUI-specific settings
        os.environ["OBVIVLORUM_MODE"] = "gui"
        os.environ["OBVIVLORUM_GUI_MODE_SELECTOR"] = "true"
        
        # Default to standard mode until GUI selection
        self._configure_standard_mode()
        
        self.logger.info("GUI mode: Interface with dynamic mode selection")
    
    def _set_performance_mode(self, performance_level: str):
        """Set performance optimization based on mode"""
        perf_config = self.topoespectral_config["topo_spectral_framework"]["performance_optimization"]
        
        if performance_level == "minimal":
            perf_config["use_numba"] = False
            perf_config["parallel_computation"] = False
            perf_config["max_nodes_standard"] = 500
            
        elif performance_level == "research":
            perf_config["use_numba"] = True
            perf_config["parallel_computation"] = True
            perf_config["max_nodes_standard"] = 1000
            perf_config["max_nodes_research"] = 5000
            
        elif performance_level == "maximum":
            perf_config["use_numba"] = True
            perf_config["parallel_computation"] = True
            perf_config["gpu_acceleration"] = True
            perf_config["max_nodes_research"] = 10000
    
    def get_current_mode(self) -> ExecutionMode:
        """Get current execution mode"""
        return self.current_mode
    
    def is_topo_spectral_enabled(self) -> bool:
        """Check if Topo-Spectral framework is enabled"""
        return self.topoespectral_config.get("topo_spectral_framework", {}).get("enabled", False)
    
    def get_mode_configuration(self, mode: ExecutionMode) -> ModeConfiguration:
        """Get configuration for specified mode"""
        claude_modes = self.claude_config.get("claude_code", {}).get("execution_modes", {})
        mode_data = claude_modes.get(mode.value, {})
        
        return ModeConfiguration(
            name=mode.value,
            description=mode_data.get("description", ""),
            entry_point=mode_data.get("entry_point", ""),
            features=mode_data.get("features", []),
            dependencies=mode_data.get("dependencies", []),
            performance_requirements=self.claude_config.get("claude_code", {}).get(
                "performance_requirements", {}
            ).get(f"{mode.value}_mode", {}),
            config_overrides={}
        )
    
    def validate_mode_requirements(self, mode: ExecutionMode) -> Dict[str, bool]:
        """
        Validate that system meets requirements for specified mode
        
        Returns:
            Dict with validation results for each requirement
        """
        validation = {
            "dependencies": self._check_dependencies(mode),
            "performance": self._check_performance_requirements(mode),
            "configuration": self._check_configuration(mode)
        }
        
        return validation
    
    def _check_dependencies(self, mode: ExecutionMode) -> bool:
        """Check if required dependencies are available"""
        try:
            if mode in [ExecutionMode.TOPOESPECTRO, ExecutionMode.RESEARCH]:
                # Check for Topo-Spectral dependencies
                import ripser
                import persim
                # Numba is optional but recommended
                try:
                    import numba
                    self.logger.info("Numba acceleration available")
                except ImportError:
                    self.logger.info("Numba not available, using standard NumPy")
                
            if mode == ExecutionMode.GUI:
                import tkinter
                
            return True
        except ImportError as e:
            self.logger.warning(f"Missing dependency for {mode.value}: {e}")
            return False
    
    def _check_performance_requirements(self, mode: ExecutionMode) -> bool:
        """Check if system meets performance requirements"""
        # Basic performance check - could be expanded
        try:
            import psutil
            available_ram_gb = psutil.virtual_memory().total / (1024**3)
            cpu_count = psutil.cpu_count()
            
            requirements = self.get_mode_configuration(mode).performance_requirements
            required_ram = requirements.get("ram", "512MB")
            required_cores = requirements.get("cpu_cores", "1")
            
            # Parse RAM requirement (simple parsing)
            if "GB" in required_ram:
                required_ram_gb = float(required_ram.replace("GB", ""))
            else:
                required_ram_gb = float(required_ram.replace("MB", "")) / 1024
            
            return available_ram_gb >= required_ram_gb and cpu_count >= int(required_cores)
            
        except Exception:
            return True  # Assume OK if can't check
    
    def _check_configuration(self, mode: ExecutionMode) -> bool:
        """Check if configuration is valid for mode"""
        if mode == ExecutionMode.TOPOESPECTRO:
            return "topo_spectral_framework" in self.topoespectral_config
        return True
    
    def save_configurations(self):
        """Save current configurations to files"""
        # Save Topo-Spectral config
        topo_file = self.config_dir / "topoespectral_config.json"
        with open(topo_file, 'w') as f:
            json.dump(self.topoespectral_config, f, indent=2)
        
        self.logger.info("Configurations saved")

# Global instance for easy access
mode_manager = ExecutionModeManager()

def get_execution_mode() -> ExecutionMode:
    """Get current execution mode from environment or default"""
    mode_str = os.environ.get("OBVIVLORUM_MODE", "standard")
    try:
        return ExecutionMode(mode_str)
    except ValueError:
        return ExecutionMode.STANDARD

def set_execution_mode(mode: ExecutionMode) -> bool:
    """Set execution mode globally"""
    return mode_manager.set_mode(mode)

def is_topo_spectral_enabled() -> bool:
    """Check if Topo-Spectral framework is enabled"""
    return mode_manager.is_topo_spectral_enabled()

# Example usage and testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Obvivlorum Execution Mode Manager")
    parser.add_argument("--mode", choices=["standard", "topoespectro", "research", "gui"],
                       default="standard", help="Execution mode")
    parser.add_argument("--validate", action="store_true", 
                       help="Validate mode requirements")
    
    args = parser.parse_args()
    
    # Set mode
    target_mode = ExecutionMode(args.mode)
    
    if args.validate:
        validation = mode_manager.validate_mode_requirements(target_mode)
        print(f"Validation results for {target_mode.value} mode:")
        for component, status in validation.items():
            print(f"  {component}: {'✓' if status else '✗'}")
        
        if all(validation.values()):
            print("All requirements met!")
        else:
            print("Some requirements not met. Check dependencies.")
            sys.exit(1)
    
    success = mode_manager.set_mode(target_mode)
    if success:
        print(f"Successfully configured {target_mode.value} mode")
        print(f"Topo-Spectral enabled: {mode_manager.is_topo_spectral_enabled()}")
    else:
        print(f"Failed to configure {target_mode.value} mode")
        sys.exit(1)