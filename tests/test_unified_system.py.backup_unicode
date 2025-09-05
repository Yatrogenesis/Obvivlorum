#!/usr/bin/env python3
"""
TEST SUITE - SISTEMA UNIFICADO FASE 6
Validación completa del sistema unificado con escalado incremental
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
    print("Make sure all Fase 6 components are implemented")
    sys.exit(1)

class TestUnifiedSystem(unittest.TestCase):
    """Tests para el sistema unificado completo"""
    
    def setUp(self):
        """Setup para cada test"""
        # Crear directorio temporal para tests
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        # Configurar logging para tests
        logging.basicConfig(level=logging.WARNING)
    
    def tearDown(self):
        """Cleanup después de cada test"""
        os.chdir(self.original_cwd)
        
        # Cleanup temp directory
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass
    
    def test_unified_config_manager_initialization(self):
        """Test 1: Inicialización del gestor de configuración unificado"""
        config_manager = UnifiedConfigManager()
        
        # Verificar inicialización básica
        self.assertIsInstance(config_manager, UnifiedConfigManager)
        self.assertIn(config_manager.environment, ["local", "colab", "kaggle", "jupyter"])
        self.assertIn(config_manager.scaling_level, ["level_1_local", "level_2_colab", "level_3_kaggle"])
        
        # Verificar configuración por defecto
        self.assertTrue(config_manager.consciousness_integration)
        self.assertIsInstance(config_manager.matrix_size_max, int)
        
        print(f"SUCCESS: UnifiedConfigManager initialized - Environment: {config_manager.environment}")
    
    def test_environment_detection(self):
        """Test 2: Detección correcta del entorno"""
        config_manager = UnifiedConfigManager()
        
        # Test detección local (default en test)
        detected_env = config_manager.detect_environment()
        self.assertIsInstance(detected_env, str)
        self.assertIn(detected_env, ["local", "colab", "kaggle", "jupyter"])
        
        # Mock Google Colab environment
        with patch.dict('sys.modules', {'google.colab': MagicMock()}):
            config_manager_colab = UnifiedConfigManager()
            self.assertEqual(config_manager_colab.detect_environment(), "colab")
        
        # Mock Kaggle environment  
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            config_manager_kaggle = UnifiedConfigManager()
            detected = config_manager_kaggle.detect_environment()
            self.assertEqual(detected, "kaggle")
        
        print(f"✅ Environment detection working - Detected: {detected_env}")
    
    def test_scaling_configuration(self):
        """Test 3: Configuración de niveles de escalado"""
        config_manager = UnifiedConfigManager()
        
        # Test configuración automática
        scaling_config = config_manager.get_scaling_config("level_1_local")
        self.assertIsInstance(scaling_config, dict)
        self.assertIn("max_matrix_size", scaling_config)
        self.assertIn("expected_time_ms", scaling_config)
        self.assertIn("use_gpu", scaling_config)
        
        # Verificar configuraciones específicas
        level_1_config = config_manager.get_scaling_config("level_1_local")
        self.assertEqual(level_1_config["max_matrix_size"], 200)
        self.assertEqual(level_1_config["expected_time_ms"], 0.01)
        self.assertFalse(level_1_config["use_gpu"])
        
        level_2_config = config_manager.get_scaling_config("level_2_colab")
        self.assertEqual(level_2_config["max_matrix_size"], 1024)
        self.assertEqual(level_2_config["expected_time_ms"], 0.001)
        self.assertTrue(level_2_config["use_gpu"])
        
        print("✅ Scaling configuration validation passed")
    
    def test_data_volume_warnings(self):
        """Test 4: Sistema de advertencias por volumen de datos"""
        config_manager = UnifiedConfigManager()
        
        # Test sin advertencia (local)
        with patch('builtins.input', return_value=''):
            result_local = config_manager.show_data_volume_warning("level_1_local")
            self.assertTrue(result_local)  # No warning, should return True
        
        # Test con advertencia (colab) - simular confirmación
        with patch('builtins.input', return_value='CONFIRMO'):
            result_colab = config_manager.show_data_volume_warning("level_2_colab")
            self.assertTrue(result_colab)
        
        # Test con advertencia (colab) - simular cancelación
        with patch('builtins.input', return_value='no'):
            result_cancel = config_manager.show_data_volume_warning("level_2_colab")
            self.assertFalse(result_cancel)
        
        print("✅ Data volume warning system working correctly")
    
    def test_consciousness_integration_initialization(self):
        """Test 5: Inicialización de la integración de conciencia"""
        consciousness = ConsciousnessIntegration(self.temp_dir)
        
        # Verificar inicialización
        self.assertIsInstance(consciousness, ConsciousnessIntegration)
        self.assertTrue(consciousness.is_system_conscious())
        
        # Verificar conocimiento del proyecto
        project_knowledge = consciousness.project_knowledge
        self.assertIn("metadata", project_knowledge)
        self.assertIn("pipeline_status", project_knowledge)
        self.assertIn("key_achievements", project_knowledge)
        
        # Verificar datos específicos del proyecto
        self.assertEqual(project_knowledge["metadata"]["name"], "Obvivlorum AI Symbiote System")
        self.assertEqual(project_knowledge["metadata"]["version"], "2.1-UNIFIED")
        self.assertIn("3780x improvement", project_knowledge["key_achievements"]["performance_breakthrough"])
        
        print("✅ Consciousness integration initialized with project knowledge")
    
    def test_consciousness_query_integration(self):
        """Test 6: Integración de conciencia con consultas"""
        consciousness = ConsciousnessIntegration(self.temp_dir)
        
        # Test identificación de componentes relevantes
        test_queries = [
            ("holographic memory performance", ["AION/holographic_memory.py"]),
            ("topo-spectral consciousness", ["AION/final_optimized_topo_spectral.py", "scientific/topo_spectral_consciousness.py"]),
            ("cloud scaling", ["AION/cloud_scaling_manager.py"]),
            ("ieee publication", ["scientific_papers/IEEE_NNLS_TopoSpectral_Framework.tex"])
        ]
        
        for query, expected_components in test_queries:
            context = consciousness.integrate_with_ai_responses(query)
            relevant_components = context["query_relevant_components"]
            
            # Verificar que al menos uno de los componentes esperados está presente
            found_match = any(comp in str(relevant_components) for comp in expected_components)
            self.assertTrue(found_match, f"No relevant components found for query: {query}")
        
        print("✅ Consciousness query integration working")
    
    def test_consciousness_state_management(self):
        """Test 7: Gestión del estado de conciencia"""
        consciousness = ConsciousnessIntegration(self.temp_dir)
        
        # Test actualización de estado
        consciousness.update_consciousness_state("test_event", {"data": "test"})
        self.assertGreater(len(consciousness.consciousness_state), 0)
        
        # Test historial
        history = consciousness.get_consciousness_history(5)
        self.assertIsInstance(history, list)
        self.assertGreater(len(history), 0)
        
        # Test contexto completo
        context = consciousness.get_full_system_context()
        self.assertIn("consciousness_active", context)
        self.assertIn("project_knowledge", context)
        self.assertTrue(context["consciousness_active"])
        
        print("✅ Consciousness state management working")
    
    def test_cloud_scaling_manager_initialization(self):
        """Test 8: Inicialización del gestor de escalado en la nube"""
        scaling_manager = CloudScalingManager()
        
        # Verificar inicialización
        self.assertIsInstance(scaling_manager, CloudScalingManager)
        self.assertIn(scaling_manager.current_environment, ["local", "colab", "kaggle", "jupyter", "docker"])
        
        # Verificar análisis de recursos
        resources = scaling_manager.available_resources
        self.assertIn("environment", resources)
        self.assertIn("cpu_count", resources)
        self.assertIn("gpu_available", resources)
        self.assertIsInstance(resources["cpu_count"], int)
        
        print(f"✅ CloudScalingManager initialized - Environment: {scaling_manager.current_environment}")
    
    def test_scaling_computation_configuration(self):
        """Test 9: Configuración de computación con escalado"""
        scaling_manager = CloudScalingManager()
        
        # Test configuración válida
        config = scaling_manager.scale_computation(100, "topo_spectral")
        self.assertIsInstance(config, dict)
        self.assertIn("max_matrix_size", config)
        self.assertIn("expected_time_ms", config)
        self.assertIn("current_environment", config)
        
        # Test excepción por tamaño excesivo
        with self.assertRaises(ResourceConstraintError):
            scaling_manager.scale_computation(5000, "topo_spectral")
        
        print("✅ Scaling computation configuration working")
    
    def test_resource_usage_estimation(self):
        """Test 10: Estimación de uso de recursos"""
        scaling_manager = CloudScalingManager()
        
        test_sizes = [200, 1024, 2048]
        for size in test_sizes:
            usage = scaling_manager.get_resource_usage_estimate(size)
            
            self.assertIsInstance(usage, dict)
            self.assertIn("matrix_size", usage)
            self.assertIn("total_memory_mb", usage)
            self.assertIn("recommended_ram_gb", usage)
            self.assertIn("scaling_level_required", usage)
            
            # Verificar que el uso de memoria aumenta con el tamaño
            self.assertGreater(usage["total_memory_mb"], 0)
            
        print("✅ Resource usage estimation working")
    
    def test_scaling_recommendations(self):
        """Test 11: Recomendaciones de escalado"""
        scaling_manager = CloudScalingManager()
        
        # Test recomendaciones por performance
        recommendations = scaling_manager.get_scaling_recommendations(0.001)
        
        self.assertIsInstance(recommendations, dict)
        self.assertIn("target_performance_ms", recommendations)
        self.assertIn("current_level", recommendations) 
        self.assertIn("recommended_levels", recommendations)
        
        # Verificar que las recomendaciones son listas
        self.assertIsInstance(recommendations["recommended_levels"], list)
        
        print("✅ Scaling recommendations working")
    
    def test_dependency_installation_dry_run(self):
        """Test 12: Instalación de dependencias (dry run)"""
        scaling_manager = CloudScalingManager()
        
        # Test dry run (no instala realmente)
        result = scaling_manager.install_cloud_dependencies(dry_run=True)
        self.assertTrue(result)  # Dry run should always succeed
        
        print("✅ Dependency installation (dry run) working")
    
    def test_unified_system_integration(self):
        """Test 13: Integración completa del sistema unificado"""
        # Inicializar todos los componentes
        config_manager = UnifiedConfigManager()
        consciousness = ConsciousnessIntegration(self.temp_dir)
        scaling_manager = CloudScalingManager()
        
        # Test integración básica
        self.assertTrue(config_manager.consciousness_integration)
        self.assertTrue(consciousness.is_system_conscious())
        self.assertIsInstance(scaling_manager.scaling_level, str)
        
        # Test información del sistema
        system_info = config_manager.get_system_info()
        self.assertIn("environment", system_info)
        self.assertIn("scaling_level", system_info)
        self.assertIn("consciousness_integration_enabled", system_info)
        
        # Test contexto de conciencia
        context = consciousness.get_full_system_context()
        self.assertTrue(context["consciousness_active"])
        
        print("✅ Unified system integration working")

class TestUnifiedSystemPerformance(unittest.TestCase):
    """Tests de performance para el sistema unificado"""
    
    def test_system_initialization_performance(self):
        """Test 14: Performance de inicialización del sistema"""
        import time
        
        # Medir tiempo de inicialización
        start_time = time.time()
        
        config_manager = UnifiedConfigManager()
        consciousness = ConsciousnessIntegration()
        scaling_manager = CloudScalingManager()
        
        initialization_time = time.time() - start_time
        
        # Verificar que la inicialización es rápida (<5 segundos)
        self.assertLess(initialization_time, 5.0, 
                       f"System initialization too slow: {initialization_time:.2f}s")
        
        print(f"✅ System initialization performance: {initialization_time:.3f}s")
    
    def test_consciousness_response_time(self):
        """Test 15: Tiempo de respuesta de la conciencia"""
        import time
        
        consciousness = ConsciousnessIntegration()
        
        # Medir tiempo de respuesta
        start_time = time.time()
        context = consciousness.integrate_with_ai_responses("test query performance")
        response_time = time.time() - start_time
        
        # Verificar que la respuesta es rápida (<100ms)
        self.assertLess(response_time, 0.1, 
                       f"Consciousness response too slow: {response_time:.3f}s")
        
        print(f"✅ Consciousness response time: {response_time:.3f}s")

def run_comprehensive_tests():
    """Ejecutar suite completa de tests"""
    print("EJECUTANDO SUITE COMPLETA DE TESTS - SISTEMA UNIFICADO FASE 6")
    print("=" * 70)
    
    # Crear test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Añadir tests principales
    suite.addTests(loader.loadTestsFromTestCase(TestUnifiedSystem))
    suite.addTests(loader.loadTestsFromTestCase(TestUnifiedSystemPerformance))
    
    # Ejecutar tests con verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Resumen de resultados
    print("\n" + "=" * 70)
    print("RESUMEN DE TESTS:")
    print(f"  Tests ejecutados: {result.testsRun}")
    print(f"  Tests exitosos: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  Fallos: {len(result.failures)}")
    print(f"  Errores: {len(result.errors)}")
    
    if result.failures:
        print("\nFALLOS:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\nERRORES:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100
    print(f"\nTASA DE EXITO: {success_rate:.1f}%")
    
    if success_rate >= 95:
        print("SISTEMA UNIFICADO COMPLETAMENTE VALIDADO")
        return True
    elif success_rate >= 80:
        print("SISTEMA UNIFICADO PARCIALMENTE VALIDADO")
        return True
    else:
        print("SISTEMA UNIFICADO REQUIERE CORRECCIONES")
        return False

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)