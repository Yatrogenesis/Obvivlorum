# 06_lanzar_simbiosis.ps1
# Ejecutar con PowerShell como administrador

# Verificar si se está ejecutando como administrador
if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host "❌ Este script debe ejecutarse como administrador." -ForegroundColor Red
    Write-Host "   Cierra esta ventana y ejecuta PowerShell como administrador." -ForegroundColor Yellow
    pause
    exit 1
}

Write-Host "=== Lanzando entorno simbiótico completo ===" -ForegroundColor Cyan

# Verificar que todos los componentes estén instalados
$componentesOK = wsl -d kali-linux -- bash -c "
    # Verificar TinyLLaMA
    if [ ! -f ~/obvlivorum_simbiosis/llama.cpp/main ] || [ ! -f ~/obvlivorum_simbiosis/llama.cpp/models/tinyllama-1.1b-chat.Q4_K_M.gguf ]; then
        echo 'FALTA_TINYLLAMA'
        exit 1
    fi
    
    # Verificar holomem
    if ! lsmod | grep -q holomem; then
        echo 'FALTA_HOLOMEM'
        exit 1
    fi
    
    # Verificar holomem-util
    if [ ! -f ~/obvlivorum_simbiosis/holomem/holomem-util ]; then
        echo 'FALTA_UTIL'
        exit 1
    fi
    
    # Verificar entorno virtual Python
    if [ ! -d ~/obvlivorum_simbiosis/simbiox ]; then
        echo 'FALTA_VENV'
        exit 1
    fi
    
    # Verificar script de interfaz
    if [ ! -f ~/obvlivorum_simbiosis/simbiosis_cli.py ]; then
        echo 'FALTA_SCRIPT'
        exit 1
    fi
    
    echo 'OK'
"

# Verificar resultado
if ($componentesOK -ne "OK") {
    Write-Host "❌ Faltan componentes necesarios para ejecutar el entorno simbiótico:" -ForegroundColor Red
    
    switch ($componentesOK) {
        "FALTA_TINYLLAMA" { 
            Write-Host "   - TinyLLaMA no está instalado correctamente. Ejecuta el Script 4." -ForegroundColor Yellow 
        }
        "FALTA_HOLOMEM" { 
            Write-Host "   - El módulo holomem no está cargado. Ejecuta los Scripts 2 y 3." -ForegroundColor Yellow 
        }
        "FALTA_UTIL" { 
            Write-Host "   - La utilidad holomem-util no está compilada. Ejecuta el Script 3." -ForegroundColor Yellow 
        }
        "FALTA_VENV" { 
            Write-Host "   - El entorno virtual Python no está creado. Ejecuta el Script 5." -ForegroundColor Yellow 
        }
        "FALTA_SCRIPT" { 
            Write-Host "   - El script de interfaz no existe. Ejecuta el Script 5." -ForegroundColor Yellow 
        }
        default { 
            Write-Host "   - Componentes desconocidos. Verifica todos los scripts anteriores." -ForegroundColor Yellow 
        }
    }
    
    pause
    exit 1
}

# Si el módulo holomem no está cargado, intentar cargarlo
$holomemLoaded = wsl -d kali-linux -- bash -c "lsmod | grep -q holomem && echo 'OK' || echo 'NO'"
if ($holomemLoaded -eq "NO") {
    Write-Host "🔄 El módulo holomem no está cargado. Intentando cargar..." -ForegroundColor Yellow
    $loadResult = wsl -d kali-linux -- bash -c "cd ~/obvlivorum_simbiosis/holomem && sudo insmod holomem.ko"
    
    if (-not $?) {
        Write-Host "❌ Error al cargar el módulo holomem. Verifica el Script 3." -ForegroundColor Red
        pause
        exit 1
    }
}

# Verificar si /dev/holomem existe, de lo contrario crearlo
$deviceExists = wsl -d kali-linux -- bash -c "test -c /dev/holomem && echo 'OK' || echo 'NO'"
if ($deviceExists -eq "NO") {
    Write-Host "🔄 El dispositivo /dev/holomem no existe. Creando..." -ForegroundColor Yellow
    $deviceResult = wsl -d kali-linux -- bash -c "
        MAJOR=\$(grep holomem /proc/devices | cut -d' ' -f1)
        if [ -z \"\$MAJOR\" ]; then
            echo 'ERROR_NO_MAJOR'
            exit 1
        fi
        sudo mknod /dev/holomem c \$MAJOR 0
        sudo chmod 666 /dev/holomem
        test -c /dev/holomem && echo 'OK' || echo 'ERROR_MKNOD'
    "
    
    if ($deviceResult -ne "OK") {
        Write-Host "❌ Error al crear el dispositivo /dev/holomem." -ForegroundColor Red
        Write-Host "   Verifica que el módulo holomem esté cargado correctamente." -ForegroundColor Yellow
        pause
        exit 1
    }
}

# L