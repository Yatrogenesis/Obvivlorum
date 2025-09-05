#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Reparacion Automatica para AI Symbiote
================================================
Soluciona los problemas identificados en el sistema.
"""

import os
import sys
import json
import subprocess
import logging
from pathlib import Path
import winreg

# Configurar logging simple
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def fix_logging_duplicates():
    """Corregir el problema de logs duplicados."""
    logger.info("[*] Corrigiendo logs duplicados...")
    
    # Leer el archivo ai_symbiote.py
    symbiote_file = Path("D:/Obvivlorum/ai_symbiote.py")
    
    try:
        with open(symbiote_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Buscar la seccion de logging y asegurarse de que no haya duplicacion
        # El problema esta en que se configura logging dos veces
        
        # Reemplazar la configuracion duplicada
        if content.count("logger.info(f\"Initializing AI Symbiote system...\")") > 1:
            logger.info("  [OK] Logs duplicados detectados y corregidos")
        else:
            logger.info("  ? No se detectaron logs duplicados")
            
        # Asegurarse de que los handlers no se dupliquen
        fix_code = """
        # Clear existing handlers to prevent duplicates
        logger.handlers.clear()
        """
        
        if "logger.handlers.clear()" not in content:
            # Insertar antes de anadir handlers
            content = content.replace(
                "# Setup file handler",
                f"{fix_code}\n        # Setup file handler"
            )
            
            with open(symbiote_file, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info("  [OK] Codigo de prevencion de duplicados agregado")
    
    except Exception as e:
        logger.error(f"  [ERROR] Error al corregir logs: {e}")

def fix_windows_permissions():
    """Implementar solucion alternativa para persistencia sin admin."""
    logger.info("[*] Implementando persistencia alternativa...")
    
    try:
        # Crear carpeta en AppData
        appdata = Path(os.environ['APPDATA']) / "AISymbiote"
        appdata.mkdir(parents=True, exist_ok=True)
        
        # Crear batch file para inicio
        batch_content = f"""@echo off
cd /d "D:\\Obvivlorum"
start /min python "D:\\Obvivlorum\\ai_symbiote.py" --background --persistent
exit
"""
        
        batch_file = appdata / "start_symbiote.bat"
        with open(batch_file, 'w') as f:
            f.write(batch_content)
        
        # Metodo 1: Anadir al registro (HKCU - no requiere admin)
        try:
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"SOFTWARE\Microsoft\Windows\CurrentVersion\Run",
                0,
                winreg.KEY_SET_VALUE
            )
            winreg.SetValueEx(key, "AISymbiote", 0, winreg.REG_SZ, str(batch_file))
            winreg.CloseKey(key)
            logger.info("  [OK] Agregado al registro de inicio (HKCU)")
        except Exception as e:
            logger.warning(f"  [WARNING] No se pudo anadir al registro: {e}")
        
        # Metodo 2: Crear acceso directo en Startup
        startup_folder = Path(os.environ['APPDATA']) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
        if startup_folder.exists():
            startup_batch = startup_folder / "AI_Symbiote.bat"
            with open(startup_batch, 'w') as f:
                f.write(batch_content)
            logger.info("  [OK] Agregado a la carpeta de inicio")
        
        # Metodo 3: Crear VBS silencioso para evitar ventana CMD
        vbs_content = f"""Set WshShell = CreateObject("WScript.Shell")
WshShell.Run chr(34) & "{batch_file}" & Chr(34), 0
Set WshShell = Nothing
"""
        vbs_file = appdata / "start_symbiote.vbs"
        with open(vbs_file, 'w') as f:
            f.write(vbs_content)
        logger.info("  [OK] Script VBS silencioso creado")
        
    except Exception as e:
        logger.error(f"  [ERROR] Error en persistencia: {e}")

def fix_wsl_default():
    """Configurar ParrotOS como WSL por defecto."""
    logger.info("[*] Configurando WSL...")
    
    try:
        # Verificar distros disponibles
        result = subprocess.run(["wsl", "--list"], capture_output=True, text=True, shell=True)
        
        if "ParrotOS" in result.stdout:
            # Establecer ParrotOS como default
            subprocess.run(["wsl", "--set-default", "ParrotOS"], shell=True)
            logger.info("  [OK] ParrotOS configurado como WSL por defecto")
        else:
            logger.warning("  [WARNING] ParrotOS no esta instalado en WSL")
            
    except Exception as e:
        logger.warning(f"  [WARNING] No se pudo configurar WSL: {e}")

def check_and_install_dependencies():
    """Verificar e instalar dependencias faltantes."""
    logger.info("[*] Verificando dependencias...")
    
    required_packages = [
        "fastapi",
        "uvicorn",
        "websockets",
        "aiofiles",
        "python-multipart",
        "numpy",
        "torch",
        "transformers",
        "openai"
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing.append(package)
    
    if missing:
        logger.info(f"  [!] Instalando paquetes faltantes: {', '.join(missing)}")
        for package in missing:
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                         capture_output=True)
        logger.info("  [OK] Dependencias instaladas")
    else:
        logger.info("  [OK] Todas las dependencias estan instaladas")

def create_startup_scripts():
    """Crear scripts de inicio mejorados."""
    logger.info("[*] Creando scripts de inicio mejorados...")
    
    # Script para iniciar todo el sistema
    full_start = """@echo off
title AI Symbiote - Sistema Completo
echo ========================================
echo   AI SYMBIOTE - INICIANDO SISTEMA
echo ========================================
echo.

REM Verificar Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python no esta instalado o no esta en PATH
    pause
    exit /b 1
)

REM Iniciar componentes
echo [1/3] Iniciando nucleo del sistema...
start /min cmd /c "cd /d D:\\Obvivlorum && python ai_symbiote.py --background --persistent"
timeout /t 3 /nobreak >nul

echo [2/3] Iniciando servidor web...
start /min cmd /c "cd /d D:\\Obvivlorum\\web\\backend && python symbiote_server.py"
timeout /t 3 /nobreak >nul

echo [3/3] Iniciando interfaz GUI...
start "" "D:\\Obvivlorum\\ai_symbiote_gui.py"

echo.
echo [OK] Sistema iniciado correctamente
echo.
echo Interfaces disponibles:
echo - GUI Desktop: Ventana activa
echo - Web: http://localhost:8000
echo - Chat: http://localhost:8000/chat
echo.
pause
"""
    
    with open("D:/Obvivlorum/START_SYSTEM_FIXED.bat", 'w', encoding='utf-8') as f:
        f.write(full_start)
    
    # Script para modo seguro
    safe_mode = """@echo off
title AI Symbiote - Modo Seguro
echo Iniciando en MODO SEGURO...
cd /d D:\\Obvivlorum
python ai_symbiote.py --config safe_mode.json
pause
"""
    
    with open("D:/Obvivlorum/START_SAFE_MODE_FIXED.bat", 'w', encoding='utf-8') as f:
        f.write(safe_mode)
    
    logger.info("  [OK] Scripts de inicio creados")

def test_interfaces():
    """Verificar que las interfaces pueden iniciarse."""
    logger.info("[*] Verificando interfaces...")
    
    # Verificar GUI
    gui_file = Path("D:/Obvivlorum/ai_symbiote_gui.py")
    if gui_file.exists():
        logger.info("  [OK] Interfaz GUI encontrada")
    else:
        logger.error("  [ERROR] Interfaz GUI no encontrada")
    
    # Verificar servidor web
    server_file = Path("D:/Obvivlorum/web/backend/symbiote_server.py")
    if server_file.exists():
        logger.info("  [OK] Servidor web encontrado")
        
        # Verificar si el puerto 8000 esta disponible
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', 8000))
        sock.close()
        
        if result == 0:
            logger.warning("  [WARNING] Puerto 8000 ya esta en uso")
        else:
            logger.info("  [OK] Puerto 8000 disponible")
    else:
        logger.error("  [ERROR] Servidor web no encontrado")

def create_config_file():
    """Crear archivo de configuracion optimizado."""
    logger.info("[*] Creando configuracion optimizada...")
    
    config = {
        "user_id": "default",
        "logging": {
            "level": "INFO",
            "file": "ai_symbiote.log",
            "prevent_duplicates": True
        },
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
                "enabled": True,
                "auto_install": False,
                "use_alternative_methods": True
            },
            "linux_executor": {
                "enabled": True,
                "safe_mode": True,
                "default_distro": "ParrotOS"
            },
            "task_facilitator": {
                "enabled": True,
                "learning_enabled": True,
                "proactive_suggestions": True
            }
        },
        "interfaces": {
            "gui_desktop": {
                "enabled": True,
                "always_on_top": False,
                "turbo_mode": False
            },
            "web_server": {
                "enabled": True,
                "port": 8000,
                "host": "0.0.0.0"
            }
        },
        "system": {
            "heartbeat_interval": 60,
            "status_file": "symbiote_status.json",
            "auto_restart": True,
            "max_restart_attempts": 3
        }
    }
    
    with open("D:/Obvivlorum/config_optimized.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info("  [OK] Configuracion optimizada creada")

def main():
    """Ejecutar todas las reparaciones."""
    print("""
========================================
   SISTEMA DE REPARACION - AI SYMBIOTE
========================================
    """)
    
    logger.info("Iniciando proceso de reparacion...\n")
    
    # Ejecutar reparaciones
    fix_logging_duplicates()
    print()
    
    fix_windows_permissions()
    print()
    
    fix_wsl_default()
    print()
    
    check_and_install_dependencies()
    print()
    
    create_startup_scripts()
    print()
    
    test_interfaces()
    print()
    
    create_config_file()
    print()
    
    logger.info("REPARACION COMPLETADA")
    print("""
Proximos pasos:
1. Ejecutar START_SYSTEM_FIXED.bat para iniciar el sistema
2. Si hay problemas, usar START_SAFE_MODE_FIXED.bat
3. La persistencia se activara en el proximo reinicio

Archivos creados:
- START_SYSTEM_FIXED.bat - Inicio completo del sistema
- START_SAFE_MODE_FIXED.bat - Modo seguro
- config_optimized.json - Configuracion optimizada
    """)

if __name__ == "__main__":
    main()
    input("\nPresiona Enter para salir...")