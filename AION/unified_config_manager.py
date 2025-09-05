#!/usr/bin/env python3
"""
UNIFIED CONFIGURATION MANAGER - FASE 6.1
Gestiona todas las opciones del sistema en un solo punto de entrada
"""

import sys
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import platform

logger = logging.getLogger(__name__)

class UnifiedConfigManager:
    """
    GESTOR DE CONFIGURACION UNIFICADO
    
    Caracteristicas:
    1. Un solo punto de entrada para todas las opciones
    2. Deteccion automatica de entorno (local/Colab/Kaggle)
    3. Sistema de advertencias inteligente por volumen de datos
    4. Escalado incremental automatico
    """
    
    def __init__(self):
        self.scaling_level = "local"
        self.holographic_real = False
        self.consciousness_integration = True
        self.matrix_size_max = 200
        self.environment = "local"
        self.config_path = Path("config")
        self.config_path.mkdir(exist_ok=True)
        
        # Detectar entorno automaticamente
        self.environment = self.detect_environment()
        self.scaling_level = self.configure_scaling_level(self.environment)
        
        logger.info(f"UnifiedConfigManager initialized - Environment: {self.environment}, Scaling: {self.scaling_level}")
    
    def detect_environment(self) -> str:
        """Deteccion automatica del entorno de ejecucion"""
        # Google Colab detection
        if 'google.colab' in sys.modules:
            return "colab"
        
        # Kaggle detection  
        if os.path.exists('/kaggle/input') or 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
            return "kaggle"
        
        # Jupyter detection
        if 'ipykernel' in sys.modules:
            return "jupyter"
            
        # Local detection
        return "local"
    
    def configure_scaling_level(self, detected_env: str, user_preference: Optional[str] = None) -> str:
        """Configuracion inteligente del nivel de escalado"""
        if user_preference:
            return self._validate_scaling_level(user_preference, detected_env)
        
        # Auto-configuracion segun entorno
        env_mapping = {
            "local": "level_1_local",
            "colab": "level_2_colab", 
            "kaggle": "level_3_kaggle",
            "jupyter": "level_1_local"
        }
        return env_mapping.get(detected_env, "level_1_local")
    
    def _validate_scaling_level(self, level: str, env: str) -> str:
        """Validacion de nivel de escalado segun entorno"""
        valid_levels = {
            "local": ["level_1_local"],
            "colab": ["level_1_local", "level_2_colab"],
            "kaggle": ["level_1_local", "level_2_colab", "level_3_kaggle"],
            "jupyter": ["level_1_local"]
        }
        
        if level in valid_levels.get(env, []):
            return level
        else:
            logger.warning(f"Invalid scaling level {level} for environment {env}, using default")
            return self.configure_scaling_level(env)
    
    def show_data_volume_warning(self, scaling_level: str) -> bool:
        """Sistema de advertencias por volumen de datos"""
        warnings = {
            "level_2_colab": {
                "message": "  ADVERTENCIA: Escalado a Google Colab (matrices 1024x1024)",
                "details": "Uso de GPU T4/V100, 12GB memoria. Procesamiento intensivo.",
                "matrix_size": 1024,
                "confirmation_required": True
            },
            "level_3_kaggle": {
                "message": " ADVERTENCIA: Escalado a Kaggle (matrices 2048x2048)",
                "details": "Uso masivo: 30GB RAM + GPU. Procesamiento muy intensivo.",
                "matrix_size": 2048,
                "confirmation_required": True
            },
            "level_4_hybrid": {
                "message": " ADVERTENCIA: Escalado Hibrido (matrices 4096x4096+)",
                "details": "Sistema distribuido. Uso de recursos extremo.",
                "matrix_size": 4096,
                "confirmation_required": True
            }
        }
        
        if scaling_level in warnings:
            warning = warnings[scaling_level]
            print(f"\n{warning['message']}")
            print(f"{warning['details']}")
            print(f"Tamano de matriz objetivo: {warning['matrix_size']}x{warning['matrix_size']}")
            
            if warning['confirmation_required']:
                response = input("\nEscriba 'CONFIRMO' para continuar o Enter para cancelar: ")
                return response.upper() == "CONFIRMO"
        
        return True  # No warning needed for local
    
    def get_scaling_config(self, scaling_level: str) -> Dict[str, Any]:
        """Obtener configuracion especifica del nivel de escalado"""
        configs = {
            "level_1_local": {
                "max_matrix_size": 200,
                "memory_limit_gb": 2,
                "use_gpu": False,
                "parallel_threads": min(4, os.cpu_count() or 1),
                "expected_time_ms": 0.01,
                "warning": False
            },
            "level_2_colab": {
                "max_matrix_size": 1024,
                "memory_limit_gb": 12,
                "use_gpu": True,
                "parallel_threads": 8,
                "expected_time_ms": 0.001,
                "warning": True
            },
            "level_3_kaggle": {
                "max_matrix_size": 2048,
                "memory_limit_gb": 30,
                "use_gpu": True,
                "parallel_threads": 16,
                "expected_time_ms": 0.0005,
                "warning": True
            },
            "level_4_hybrid": {
                "max_matrix_size": 4096,
                "memory_limit_gb": 64,
                "use_gpu": True,
                "parallel_threads": 32,
                "expected_time_ms": 0.0001,
                "warning": True,
                "distributed": True
            }
        }
        
        return configs.get(scaling_level, configs["level_1_local"])
    
    def save_config(self) -> str:
        """Guardar configuracion unificada"""
        config = {
            "unified_system": {
                "version": "2.1-UNIFIED",
                "environment": self.environment,
                "scaling_level": self.scaling_level,
                "holographic_real": self.holographic_real,
                "consciousness_integration": self.consciousness_integration,
                "matrix_size_max": self.matrix_size_max,
                "timestamp": str(Path(__file__).stat().st_mtime)
            }
        }
        
        config_file = self.config_path / "unified_system_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        return str(config_file)
    
    def load_config(self, config_file: Optional[str] = None) -> Dict[str, Any]:
        """Cargar configuracion unificada"""
        if config_file is None:
            config_file = self.config_path / "unified_system_config.json"
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.info(f"Config file {config_file} not found, using defaults")
            return {}
    
    def get_system_info(self) -> Dict[str, Any]:
        """Informacion completa del sistema para diagnostico"""
        return {
            "environment": self.environment,
            "scaling_level": self.scaling_level,
            "platform": platform.platform(),
            "python_version": sys.version,
            "cpu_count": os.cpu_count(),
            "scaling_config": self.get_scaling_config(self.scaling_level),
            "holographic_real_enabled": self.holographic_real,
            "consciousness_integration_enabled": self.consciousness_integration
        }

if __name__ == "__main__":
    # Test basico del sistema
    print(" UNIFIED CONFIG MANAGER - TEST")
    
    config_manager = UnifiedConfigManager()
    
    # Mostrar informacion del sistema
    info = config_manager.get_system_info()
    print(f"\n INFORMACION DEL SISTEMA:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test de advertencias
    print(f"\n  TEST DE ADVERTENCIAS:")
    for level in ["level_1_local", "level_2_colab", "level_3_kaggle"]:
        print(f"\nTesting {level}:")
        # Para test automatico, simular confirmacion
        config_manager.scaling_level = level
        # En test no interactivo, skip la confirmacion
        print(f"  Configuracion: {config_manager.get_scaling_config(level)}")
    
    # Guardar configuracion
    config_file = config_manager.save_config()
    print(f"\n Configuracion guardada en: {config_file}")
    
    print("\n UNIFIED CONFIG MANAGER - TEST COMPLETADO")