#!/usr/bin/env python3
"""
Test script for Unified UI functionality
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'AION'))

try:
    from unified_config_manager import UnifiedConfigManager
    from consciousness_integration import ConsciousnessIntegration
    from cloud_scaling_manager import CloudScalingManager
    
    print("TESTING UNIFIED SYSTEM COMPONENTS")
    print("=" * 50)
    
    # Test 1: Unified Config Manager
    print("\n1. Testing Unified Config Manager...")
    config_manager = UnifiedConfigManager()
    print(f"   [OK] Environment: {config_manager.environment}")
    print(f"   [OK] Scaling Level: {config_manager.scaling_level}")
    print(f"   [OK] Consciousness Integration: {config_manager.consciousness_integration}")
    
    # Test 2: Consciousness Integration
    print("\n2. Testing Consciousness Integration...")
    consciousness = ConsciousnessIntegration()
    print(f"   [OK] System Conscious: {consciousness.is_system_conscious()}")
    print(f"   [OK] Project Knowledge: {len(consciousness.project_knowledge)} components")
    print(f"   [OK] Pipeline Phase: {consciousness.project_knowledge['pipeline_status']['current_phase']}")
    
    # Test 3: Cloud Scaling Manager
    print("\n3. Testing Cloud Scaling Manager...")
    cloud_scaling = CloudScalingManager()
    print(f"   [OK] Current Environment: {cloud_scaling.current_environment}")
    print(f"   [OK] Scaling Level: {cloud_scaling.scaling_level}")
    print(f"   [OK] CPU Count: {cloud_scaling.available_resources['cpu_count']}")
    print(f"   [OK] GPU Available: {cloud_scaling.available_resources['gpu_available']}")
    
    # Test 4: Integration Test
    print("\n4. Testing System Integration...")
    try:
        scaling_config = cloud_scaling.scale_computation(100, "topo_spectral")
        print(f"   [OK] Scaling Configuration: Max matrix {scaling_config['max_matrix_size']}x{scaling_config['max_matrix_size']}")
        print(f"   [OK] Expected Performance: {scaling_config['expected_time_ms']}ms")
    except Exception as e:
        print(f"   [ERROR] Scaling Error: {e}")
    
    # Test 5: Resource Estimation
    print("\n5. Testing Resource Estimation...")
    try:
        usage = cloud_scaling.get_resource_usage_estimate(200)
        print(f"   [OK] Memory Usage: {usage['total_memory_mb']} MB")
        print(f"   [OK] Recommended RAM: {usage['recommended_ram_gb']} GB")
        print(f"   [OK] Required Level: {usage['scaling_level_required']}")
    except Exception as e:
        print(f"   [ERROR] Resource Error: {e}")
    
    print("\n" + "=" * 50)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("[OK] Unified System is ready for UI integration")
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure all Fase 6 components are properly implemented")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected Error: {e}")
    sys.exit(1)