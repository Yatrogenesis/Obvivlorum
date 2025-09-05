#!/usr/bin/env python3
"""
Unified Configuration System for Desktop and Web
================================================

Sistema de configuracion unificado que maneja todos los modos,
escalamientos y upgrades tanto para desktop como web.
"""

import json
import os
from typing import Dict, Any, Optional, List
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, asdict

class SystemMode(Enum):
    """Modos de operacion del sistema"""
    STANDARD = "standard"
    TOPOESPECTRO = "topoespectro" 
    RESEARCH = "research"
    GUI_DESKTOP = "gui_desktop"
    WEB_SERVER = "web_server"
    HYBRID = "hybrid"

class ScalingLevel(Enum):
    """Niveles de escalamiento"""
    LOCAL = "local"
    CLOUD_BASIC = "cloud_basic"
    CLOUD_ADVANCED = "cloud_advanced"
    ENTERPRISE = "enterprise"

class AIProvider(Enum):
    """Proveedores de IA disponibles"""
    LOCAL_GGUF = "local_gguf"
    OPENAI_GPT4 = "openai_gpt4" 
    CLAUDE_SONNET = "claude_sonnet"
    HYBRID_MULTI = "hybrid_multi"

@dataclass
class UIConfig:
    """Configuracion de interfaz de usuario"""
    theme: str = "modern_dark"
    animations: bool = True
    transparency: float = 0.95
    window_size: tuple = (1400, 900)
    fonts: Dict[str, str] = None
    
    def __post_init__(self):
        if self.fonts is None:
            self.fonts = {
                "main": "Segoe UI",
                "mono": "Consolas", 
                "title": "Segoe UI Semibold"
            }

@dataclass
class SystemConfig:
    """Configuracion unificada del sistema"""
    mode: SystemMode = SystemMode.STANDARD
    scaling: ScalingLevel = ScalingLevel.LOCAL
    ai_provider: AIProvider = AIProvider.HYBRID_MULTI
    ui_config: UIConfig = None
    aion_protocols: List[str] = None
    performance_settings: Dict[str, Any] = None
    security_level: str = "high"
    
    def __post_init__(self):
        if self.ui_config is None:
            self.ui_config = UIConfig()
        if self.aion_protocols is None:
            self.aion_protocols = ["ALPHA", "BETA", "GAMMA", "DELTA", "OMEGA"]
        if self.performance_settings is None:
            self.performance_settings = {
                "max_memory_gb": 4,
                "cpu_threads": -1,
                "gpu_enabled": True,
                "cache_size_mb": 512
            }

class UnifiedConfigSystem:
    """Sistema de configuracion unificado"""
    
    def __init__(self, config_file: str = "unified_system_config.json"):
        self.config_file = Path(config_file)
        self.config = SystemConfig()
        self.load_config()
    
    def load_config(self) -> None:
        """Cargar configuracion desde archivo"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Convert enums back from strings
                if 'mode' in data:
                    data['mode'] = SystemMode(data['mode'])
                if 'scaling' in data:
                    data['scaling'] = ScalingLevel(data['scaling'])
                if 'ai_provider' in data:
                    data['ai_provider'] = AIProvider(data['ai_provider'])
                
                # Rebuild UI config
                if 'ui_config' in data:
                    data['ui_config'] = UIConfig(**data['ui_config'])
                
                # Update config with loaded data
                for key, value in data.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
                        
            except Exception as e:
                print(f"Error loading config: {e}, using defaults")
    
    def save_config(self) -> None:
        """Guardar configuracion a archivo"""
        try:
            # Convert to dict and handle enums
            config_dict = asdict(self.config)
            config_dict['mode'] = self.config.mode.value
            config_dict['scaling'] = self.config.scaling.value
            config_dict['ai_provider'] = self.config.ai_provider.value
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def get_desktop_config(self) -> Dict[str, Any]:
        """Obtener configuracion especifica para desktop"""
        return {
            "mode": self.config.mode.value,
            "ui": asdict(self.config.ui_config),
            "ai_provider": self.config.ai_provider.value,
            "aion_protocols": self.config.aion_protocols,
            "performance": self.config.performance_settings,
            "scaling": self.config.scaling.value,
            "security": self.config.security_level
        }
    
    def get_web_config(self) -> Dict[str, Any]:
        """Obtener configuracion especifica para web"""
        web_ui = asdict(self.config.ui_config)
        # Adapt for web
        web_ui["responsive"] = True
        web_ui["mobile_friendly"] = True
        
        return {
            "mode": "web_optimized",
            "ui": web_ui,
            "ai_provider": self.config.ai_provider.value,
            "aion_protocols": self.config.aion_protocols,
            "api_endpoints": {
                "chat": "/api/chat",
                "status": "/api/status",
                "config": "/api/config"
            },
            "security": self.config.security_level
        }
    
    def set_brutal_ui_mode(self) -> None:
        """Configurar UI brutalmente impresionante para desktop"""
        self.config.ui_config = UIConfig(
            theme="brutal_modern",
            animations=True,
            transparency=0.92,
            window_size=(1600, 1000),
            fonts={
                "main": "Inter",
                "mono": "JetBrains Mono", 
                "title": "Inter Black"
            }
        )
        
        # Enhanced performance for brutal UI
        self.config.performance_settings.update({
            "ui_framerate": 120,
            "smooth_animations": True,
            "gpu_acceleration": True,
            "anti_aliasing": True,
            "shadow_effects": True
        })
        
        self.save_config()
    
    def get_available_modes(self) -> List[Dict[str, str]]:
        """Obtener lista de modos disponibles"""
        return [
            {"id": "standard", "name": "Standard", "description": "Modo estandar de operacion"},
            {"id": "topoespectro", "name": "Topo-Espectral", "description": "Analisis topo-espectral avanzado"},
            {"id": "research", "name": "Research", "description": "Modo de investigacion cientifica"},
            {"id": "gui_desktop", "name": "GUI Desktop", "description": "Interfaz grafica desktop"},
            {"id": "web_server", "name": "Web Server", "description": "Servidor web completo"},
            {"id": "hybrid", "name": "Hybrid", "description": "Modo hibrido multi-protocolo"}
        ]
    
    def get_scaling_options(self) -> List[Dict[str, str]]:
        """Obtener opciones de escalamiento"""
        return [
            {"id": "local", "name": "Local", "description": "Procesamiento local unicamente"},
            {"id": "cloud_basic", "name": "Cloud Basic", "description": "Cloud basico con fallback local"},
            {"id": "cloud_advanced", "name": "Cloud Advanced", "description": "Cloud avanzado con multiples proveedores"},
            {"id": "enterprise", "name": "Enterprise", "description": "Escalamiento empresarial completo"}
        ]
    
    def upgrade_system(self, target_scaling: str) -> bool:
        """Actualizar nivel de escalamiento del sistema"""
        try:
            new_scaling = ScalingLevel(target_scaling)
            self.config.scaling = new_scaling
            
            # Update performance settings based on scaling
            if new_scaling == ScalingLevel.ENTERPRISE:
                self.config.performance_settings.update({
                    "max_memory_gb": 32,
                    "distributed_processing": True,
                    "redundancy": True,
                    "load_balancing": True
                })
            elif new_scaling == ScalingLevel.CLOUD_ADVANCED:
                self.config.performance_settings.update({
                    "max_memory_gb": 16,
                    "cloud_acceleration": True,
                    "auto_scaling": True
                })
            
            self.save_config()
            return True
        except ValueError:
            return False

# Global instance
config_system = UnifiedConfigSystem()

def get_config() -> UnifiedConfigSystem:
    """Obtener instancia global de configuracion"""
    return config_system