#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Demo: AI Symbiote con ParrotOS
=============================

Demostración de ejecución del AI Symbiote específicamente con ParrotOS.
"""

import sys
import os
import json
import time

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from ai_symbiote import AISymbiote

def demo_parrot_execution():
    """Demostración con ParrotOS."""
    print("[PARROT] AI SYMBIOTE + PARROT OS DEMO")
    print("=" * 50)
    
    # Crear instancia del AI Symbiote
    print("[INICIO] Inicializando AI Symbiote con usuario ParrotOS...")
    symbiote = AISymbiote(user_id="parrot_demo")
    
    print(f"[OK] Sistema inicializado")
    print(f"[INFO] Usuario: {symbiote.user_id}")
    
    # Verificar WSL y ParrotOS
    if "linux" in symbiote.components:
        linux_executor = symbiote.components["linux"]
        status = linux_executor.get_status()
        
        print(f"[WSL] Entorno WSL detectado: {status['is_wsl']}")
        print(f"[WSL] Distros disponibles: {status.get('wsl_distros', [])}")
        print(f"[WSL] Distro por defecto: {status.get('default_distro', 'N/A')}")
        
        # Comandos específicos para ParrotOS
        parrot_commands = [
            "echo 'Hola desde ParrotOS!'",
            "whoami",
            "pwd",
            "ls -la /etc | head -5",
            "cat /etc/os-release | grep PRETTY_NAME"
        ]
        
        print(f"\n[DEMO] Ejecutando comandos en ParrotOS...")
        print("-" * 30)
        
        for i, cmd in enumerate(parrot_commands, 1):
            print(f"\n[CMD {i}] {cmd}")
            try:
                # Ejecutar específicamente en ParrotOS
                result = linux_executor.execute_command(
                    cmd, 
                    distro="ParrotOS",
                    timeout=10,
                    capture_output=True
                )
                
                print(f"[STATUS] {result['status']}")
                if result['status'] == 'success':
                    print(f"[OUTPUT] {result.get('stdout', '').strip()}")
                else:
                    print(f"[ERROR] {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"[ERROR] {e}")
        
        # Demostrar instalación de paquetes en ParrotOS
        print(f"\n[DEMO] Verificando herramientas de ParrotOS...")
        print("-" * 30)
        
        parrot_tools = [
            "which nmap",
            "which metasploit-framework",
            "which aircrack-ng",
            "which john",
            "which hashcat"
        ]
        
        for tool_check in parrot_tools:
            print(f"\n[CHECK] {tool_check}")
            result = linux_executor.execute_command(
                tool_check,
                distro="ParrotOS",
                timeout=5,
                capture_output=True
            )
            
            if result['status'] == 'success' and result.get('stdout', '').strip():
                print(f"[OK] Encontrado: {result['stdout'].strip()}")
            else:
                print(f"[NO] No encontrado o no instalado")
    
    # Demostrar protocolos AION con contexto ParrotOS
    print(f"\n[DEMO] Ejecutando protocolo AION con contexto ParrotOS...")
    print("-" * 30)
    
    if "aion" in symbiote.components or "bridge" in symbiote.components:
        # Protocolo ALPHA para investigación de seguridad
        alpha_params = {
            "research_domain": "cybersecurity_parrot",
            "research_type": "exploratory", 
            "create_domain": True,
            "seed_concepts": [
                "penetration_testing",
                "network_security",
                "digital_forensics",
                "parrot_security_tools",
                "ethical_hacking"
            ],
            "exploration_depth": 0.8,
            "novelty_bias": 0.9
        }
        
        print("[PROTOCOL] Ejecutando ALPHA para investigación de ciberseguridad...")
        result = symbiote.execute_protocol("ALPHA", alpha_params)
        
        print(f"[STATUS] {result['status']}")
        if result['status'] == 'success':
            print(f"[RESEARCH_ID] {result.get('research_id', 'N/A')}")
            print(f"[DOMAIN] {result.get('domain', 'N/A')}")
        elif 'message' in result:
            print(f"[MESSAGE] {result['message']}")
    
    # Demostrar gestión de tareas
    print(f"\n[DEMO] Gestión de tareas con ParrotOS...")
    print("-" * 30)
    
    if "tasks" in symbiote.components:
        task_facilitator = symbiote.components["tasks"]
        
        # Añadir tareas relacionadas con ParrotOS
        parrot_tasks = [
            {
                "name": "Configurar entorno ParrotOS",
                "description": "Configurar herramientas de seguridad en ParrotOS",
                "priority": 8,
                "tags": ["setup", "parrot", "security"]
            },
            {
                "name": "Ejecutar escaneo de red",
                "description": "Realizar escaneo de red con nmap",
                "priority": 7,
                "tags": ["network", "scanning", "nmap"]
            },
            {
                "name": "Análisis de vulnerabilidades",
                "description": "Identificar vulnerabilidades en sistemas objetivo",
                "priority": 9,
                "tags": ["vulnerability", "analysis", "security"]
            }
        ]
        
        task_facilitator.start()
        
        for task_info in parrot_tasks:
            task_id = task_facilitator.add_task(**task_info)
            print(f"[TASK] Añadida: {task_info['name']} (ID: {task_id})")
        
        # Actualizar contexto para ParrotOS
        task_facilitator.update_context({
            "platform": "ParrotOS", 
            "activity": "security_testing",
            "environment": "wsl",
            "focus": "cybersecurity"
        })
        
        print(f"[CONTEXT] Contexto actualizado para ParrotOS")
        
        # Obtener estado del facilitador
        facilitator_status = task_facilitator.get_status()
        print(f"[TASKS] Tareas activas: {facilitator_status['active_tasks']}")
        
        task_facilitator.stop()
    
    # Estado final del sistema
    print(f"\n[ESTADO] Estado final del AI Symbiote...")
    print("-" * 30)
    
    system_status = symbiote.get_system_status()
    print(f"[RUNNING] Sistema activo: {system_status['is_running']}")
    print(f"[USER] Usuario: {system_status['user_id']}")
    print(f"[COMPONENTS] Componentes: {len(system_status['components'])}")
    
    for comp_name, comp_status in system_status["components"].items():
        if isinstance(comp_status, dict):
            status_key = "is_active" if "is_active" in comp_status else "status"
            status_val = comp_status.get(status_key, "unknown")
            print(f"  - {comp_name}: {status_val}")
    
    print(f"\n[SUCCESS] Demo completada exitosamente!")
    print("🦜 ParrotOS + AI Symbiote funcionando correctamente")
    
    return symbiote

def main():
    """Función principal."""
    try:
        symbiote = demo_parrot_execution()
        
        print(f"\n" + "=" * 50)
        print("DEMO INTERACTIVA")
        print("=" * 50)
        print("El AI Symbiote está ahora listo para usar con ParrotOS.")
        print("Puedes ejecutar comandos adicionales o usar el sistema completo.")
        print("\nEjemplos de uso:")
        print("- symbiote.execute_linux_command('comando', distro='ParrotOS')")
        print("- symbiote.execute_protocol('ALPHA', parametros)")
        print("- symbiote.add_task('Nombre de tarea')")
        
        # Opción interactiva
        while True:
            print(f"\nOpciones:")
            print("1. Ejecutar comando en ParrotOS")
            print("2. Ejecutar protocolo AION")
            print("3. Ver estado del sistema")
            print("4. Salir")
            
            choice = input("Selecciona opción [1-4]: ").strip()
            
            if choice == "1":
                cmd = input("Comando para ParrotOS: ").strip()
                if cmd:
                    result = symbiote.execute_linux_command(cmd, distro="ParrotOS")
                    print(f"Resultado: {json.dumps(result, indent=2)}")
            
            elif choice == "2":
                protocol = input("Protocolo [ALPHA/BETA/GAMMA/DELTA/OMEGA]: ").strip().upper()
                if protocol in ["ALPHA", "BETA", "GAMMA", "DELTA", "OMEGA"]:
                    result = symbiote.execute_protocol(protocol, {"test_mode": True})
                    print(f"Resultado: {json.dumps(result, indent=2)}")
            
            elif choice == "3":
                status = symbiote.get_system_status()
                print(f"Estado: {json.dumps(status, indent=2)}")
            
            elif choice == "4":
                break
            
            else:
                print("Opción no válida")
        
        symbiote.stop()
        
    except KeyboardInterrupt:
        print(f"\n[STOP] Demo interrumpida por el usuario")
    except Exception as e:
        print(f"[ERROR] Error en demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()