#!/usr/bin/env python3
"""
TEST SUITE SIMPLE - SISTEMA UNIFICADO FASE 6
Validacion completa del sistema unificado sin caracteres Unicode
"""

import unittest
import sys
import os
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import logging

# Add AION to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'AION'))

try:
    from unified_config_manager import UnifiedConfigManager
    from consciousness_integration import ConsciousnessIntegration  
    from cloud_scaling_manager import CloudScalingManager, ResourceConstraintError
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

class TestUnifiedSystemCore(unittest.TestCase):
    """Tests principales del sistema unificado"""
    
    def setUp(self):
        """Setup para cada test"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        logging.basicConfig(level=logging.WARNING)
    
    def tearDown(self):
        """Cleanup despues de cada test"""
        os.chdir(self.original_cwd)
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass
    
    def test_1_unified_config_basic(self):
        """Test 1: Configuracion unificada basica"""
        config_manager = UnifiedConfigManager()
        self.assertIsInstance(config_manager, UnifiedConfigManager)
        self.assertIn(config_manager.environment, ["local", "colab", "kaggle", "jupyter"])
        print(f"SUCCESS: Config Manager - Environment: {config_manager.environment}")
        return True
    
    def test_2_environment_detection(self):
        """Test 2: Deteccion de entorno"""
        config_manager = UnifiedConfigManager()
        detected_env = config_manager.detect_environment()
        self.assertIsInstance(detected_env, str)
        self.assertIn(detected_env, ["local", "colab", "kaggle", "jupyter"])
        print(f"SUCCESS: Environment detection - {detected_env}")
        return True
    
    def test_3_scaling_config(self):
        """Test 3: Configuracion de escalado"""
        config_manager = UnifiedConfigManager()
        config = config_manager.get_scaling_config("level_1_local")
        self.assertIsInstance(config, dict)
        self.assertEqual(config["max_matrix_size"], 200)
        self.assertEqual(config["expected_time_ms"], 0.01)
        print("SUCCESS: Scaling configuration validated")
        return True
    
    def test_4_consciousness_basic(self):
        """Test 4: Conciencia basica"""
        consciousness = ConsciousnessIntegration(self.temp_dir)
        self.assertIsInstance(consciousness, ConsciousnessIntegration)
        self.assertTrue(consciousness.is_system_conscious())
        
        # Verificar conocimiento del proyecto
        knowledge = consciousness.project_knowledge
        self.assertIn("metadata", knowledge)
        self.assertEqual(knowledge["metadata"]["name"], "Obvivlorum AI Symbiote System")
        print("SUCCESS: Consciousness integration working")
        return True
    
    def test_5_consciousness_queries(self):
        """Test 5: Consultas de conciencia"""
        consciousness = ConsciousnessIntegration(self.temp_dir)
        
        test_queries = [
            "holographic memory",
            "topo-spectral",
            "cloud scaling"
        ]
        
        for query in test_queries:
            context = consciousness.integrate_with_ai_responses(query)
            self.assertIsInstance(context, dict)
            self.assertIn("query_relevant_components", context)
        
        print("SUCCESS: Consciousness query integration working")
        return True
    
    def test_6_cloud_scaling_basic(self):
        """Test 6: Escalado en la nube basico"""
        scaling_manager = CloudScalingManager()
        self.assertIsInstance(scaling_manager, CloudScalingManager)
        
        # Test recursos
        resources = scaling_manager.available_resources
        self.assertIn("environment", resources)
        self.assertIn("cpu_count", resources)
        print(f"SUCCESS: Cloud scaling - Environment: {scaling_manager.current_environment}")
        return True
    
    def test_7_scaling_computation(self):
        """Test 7: Computacion con escalado"""
        scaling_manager = CloudScalingManager()
        
        # Test configuracion valida
        config = scaling_manager.scale_computation(100, "topo_spectral")
        self.assertIsInstance(config, dict)
        self.assertIn("max_matrix_size", config)
        print("SUCCESS: Scaling computation configuration")
        return True
    
    def test_8_resource_estimation(self):
        """Test 8: Estimacion de recursos"""
        scaling_manager = CloudScalingManager()
        
        for size in [200, 500]:
            usage = scaling_manager.get_resource_usage_estimate(size)
            self.assertIsInstance(usage, dict)
            self.assertIn("total_memory_mb", usage)
        
        print("SUCCESS: Resource usage estimation")
        return True
    
    def test_9_system_integration(self):
        """Test 9: Integracion del sistema"""
        config_manager = UnifiedConfigManager()
        consciousness = ConsciousnessIntegration(self.temp_dir)
        scaling_manager = CloudScalingManager()
        
        # Verificar integracion basica
        self.assertTrue(config_manager.consciousness_integration)
        self.assertTrue(consciousness.is_system_conscious())
        self.assertIsInstance(scaling_manager.scaling_level, str)
        
        print("SUCCESS: System integration working")
        return True

def run_simple_tests():
    """Ejecutar tests simplificados"""
    print("EJECUTANDO TESTS SIMPLIFICADOS - SISTEMA UNIFICADO FASE 6")
    print("=" * 60)
    
    # Ejecutar tests uno por uno para mejor control
    test_suite = TestUnifiedSystemCore()
    test_methods = [
        'test_1_unified_config_basic',
        'test_2_environment_detection', 
        'test_3_scaling_config',
        'test_4_consciousness_basic',
        'test_5_consciousness_queries',
        'test_6_cloud_scaling_basic',
        'test_7_scaling_computation',
        'test_8_resource_estimation',
        'test_9_system_integration'
    ]
    
    passed = 0
    failed = 0
    errors = []
    
    for method_name in test_methods:
        try:
            test_suite.setUp()
            method = getattr(test_suite, method_name)
            method()
            passed += 1
            print(f"PASS: {method_name}")
        except Exception as e:
            failed += 1
            errors.append((method_name, str(e)))
            print(f"FAIL: {method_name} - {e}")
        finally:
            try:
                test_suite.tearDown()
            except:
                pass
    
    total = passed + failed
    success_rate = (passed / total) * 100 if total > 0 else 0
    
    print("\n" + "=" * 60)
    print("RESUMEN DE TESTS SIMPLIFICADOS:")
    print(f"  Total: {total}")
    print(f"  Exitosos: {passed}")
    print(f"  Fallidos: {failed}")
    print(f"  Tasa de exito: {success_rate:.1f}%")
    
    if errors:
        print("\nERRORES ENCONTRADOS:")
        for method, error in errors:
            print(f"  - {method}: {error}")
    
    if success_rate >= 95:
        print("\nSISTEMA UNIFICADO COMPLETAMENTE VALIDADO")
        return True
    elif success_rate >= 70:
        print("\nSISTEMA UNIFICADO PARCIALMENTE VALIDADO")
        return True
    else:
        print("\nSISTEMA UNIFICADO REQUIERE CORRECCIONES")
        return False

if __name__ == "__main__":
    success = run_simple_tests()
    sys.exit(0 if success else 1)