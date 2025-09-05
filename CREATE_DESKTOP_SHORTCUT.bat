@echo off
chcp 65001 >nul 2>&1
title AI Symbiote - Crear Acceso Directo de Escritorio
color 0C

echo.
echo  ┌─────────────────────────────────────────────────────────────────┐
echo  │           AI SYMBIOTE - CREAR ACCESO DIRECTO DE ESCRITORIO      │
echo  └─────────────────────────────────────────────────────────────────┘
echo.

echo [INFO] Creando acceso directo en el escritorio...

set "desktop=D:\Users\Propietario\Desktop"
set "shortcut=%desktop%\AI Symbiote - Modo Persistente.lnk"

echo [INFO] Ubicación del escritorio: %desktop%

powershell -Command "$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%shortcut%'); $Shortcut.TargetPath = '%~dp0START_COMPLETE_SYSTEM.bat'; $Shortcut.WorkingDirectory = '%~dp0'; $Shortcut.IconLocation = '%SystemRoot%\System32\shell32.dll,13'; $Shortcut.Description = 'AI Symbiote Sistema Completo Híbrido - Interfaz Persistente'; $Shortcut.Save()"

if exist "%shortcut%" (
    echo.
    echo ┌─────────────────────────────────────────────────────────────────┐
    echo │                    ACCESO DIRECTO CREADO                        │
    echo ├─────────────────────────────────────────────────────────────────┤
    echo │  ✅ Nombre: AI Symbiote - Modo Persistente                     │
    echo │  ✅ Ubicacion: Escritorio de Windows                           │
    echo │  ✅ Funcion: Iniciar sistema completo hibrido                  │
    echo │                                                                 │
    echo │  CARACTERISTICAS DEL ACCESO DIRECTO:                           │
    echo │  • Ejecuta START_COMPLETE_SYSTEM.bat                           │
    echo │  • Activa AI Symbiote Core                                      │
    echo │  • Inicia servidor web hibrido                                  │
    echo │  • Abre interfaz GUI persistente                                │
    echo │  • Abre interfaz web Claude-style                               │
    echo │  • Configura TURBO mode automaticamente                        │
    echo └─────────────────────────────────────────────────────────────────┘
    echo.
    echo [✓] ACCESO DIRECTO CREADO EXITOSAMENTE
    echo.
    echo Puedes hacer doble clic en "AI Symbiote - Modo Persistente" 
    echo desde tu escritorio para iniciar el sistema completo.
    echo.
) else (
    echo [ERROR] No se pudo crear el acceso directo
    echo Verifica que tengas permisos de escritura en el escritorio
)

echo.
pause