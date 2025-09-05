# 01_instalar_WSL2_y_Kali.ps1
# Ejecuta con doble clic o boton derecho > Ejecutar con PowerShell

Write-Host "=== Instalacion Simbiotica: Paso 1 ===" -ForegroundColor Cyan
Write-Host "Verificando WSL2, Virtualizacion y entorno base..."

# 1. Verificar si la virtualizacion esta activada (metodo mejorado)
$virtualizationEnabled = $false

# Metodo 1: CIM Instance
try {
    $cpuInfo = Get-CimInstance Win32_Processor -ErrorAction SilentlyContinue
    if ($cpuInfo.VirtualizationFirmwareEnabled) {
        $virtualizationEnabled = $true
    }
} catch {}

# Metodo 2: Verificar Hyper-V
if (-not $virtualizationEnabled) {
    $hyperv = Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V-All -ErrorAction SilentlyContinue
    if ($hyperv -and $hyperv.State -eq "Enabled") {
        $virtualizationEnabled = $true
    }
}

# Metodo 3: Verificar por servicios
if (-not $virtualizationEnabled) {
    $vmcompute = Get-Service -Name vmcompute -ErrorAction SilentlyContinue
    if ($vmcompute) {
        $virtualizationEnabled = $true
    }
}

if (-not $virtualizationEnabled) {
    Write-Host "Virtualizacion NO detectada." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Opciones disponibles:" -ForegroundColor Cyan
    Write-Host "1. Ejecutar activar_virtualizacion.bat como Administrador"
    Write-Host "2. Activar manualmente en BIOS (requiere reinicio)"
    Write-Host ""
    Write-Host "Intentando activacion automatica..." -ForegroundColor Yellow
    
    # Crear path al BAT
    $batPath = Join-Path $PSScriptRoot "activar_virtualizacion.bat"
    if (Test-Path $batPath) {
        Write-Host "Ejecutando activador de virtualizacion..." -ForegroundColor Green
        Start-Process $batPath -Verb RunAs -Wait
        Write-Host "Vuelve a ejecutar este script despues de reiniciar si es necesario." -ForegroundColor Cyan
    } else {
        Write-Host "No se encontro activar_virtualizacion.bat" -ForegroundColor Red
    }
    exit 1
} else {
    Write-Host "Virtualizacion detectada." -ForegroundColor Green
}

# 2. Verificar si WSL ya esta instalado
$wslFeature = Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux
if ($wslFeature.State -ne "Enabled") {
    Write-Host "Activando WSL..." -ForegroundColor Yellow
    dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
}

# 3. Verificar Plataforma de Maquina Virtual
$vmPlatform = Get-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform
if ($vmPlatform.State -ne "Enabled") {
    Write-Host "Activando plataforma de virtualizacion..." -ForegroundColor Yellow
    dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
}

# 4. Verificar kernel de WSL2
if (-not (Get-Command "wsl.exe" -ErrorAction SilentlyContinue)) {
    Write-Host "WSL no esta disponible en este sistema. Instala manualmente desde:" -ForegroundColor Red
    Write-Host "https://aka.ms/wslstore"
    exit 1
}

# 5. Establecer WSL2 como predeterminado
wsl --set-default-version 2

# 6. Detectar distribuciones Linux instaladas - METODO MEJORADO
Write-Host ""
Write-Host "Detectando distribuciones Linux instaladas..." -ForegroundColor Cyan

$distros = @()

# Ejecutar wsl --list y capturar output con encoding correcto
$wslOutput = & cmd /c "wsl --list 2>nul"

if ($wslOutput) {
    foreach ($line in $wslOutput) {
        if ($line) {
            # Limpiar la linea de caracteres especiales
            $cleanLine = $line -replace '[^\x20-\x7E]', ''
            $cleanLine = $cleanLine.Trim()
            
            # Saltar lineas vacias, headers y Docker
            if ($cleanLine -and 
                $cleanLine -ne "Windows Subsystem for Linux Distributions:" -and
                $cleanLine -notmatch "Windows" -and
                $cleanLine -notmatch "docker" -and
                $cleanLine.Length -gt 0) {
                
                # Quitar asterisco si es la distro por defecto
                $distroName = $cleanLine -replace '^\*\s*', ''
                $distroName = $distroName.Trim()
                
                if ($distroName) {
                    $distros += $distroName
                }
            }
        }
    }
}

# Mostrar lo que encontramos
if ($distros.Count -eq 0) {
    Write-Host "No se encontraron distribuciones Linux instaladas." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Necesitas instalar una distribucion Linux desde Microsoft Store." -ForegroundColor Cyan
    Write-Host "Opciones recomendadas:" -ForegroundColor Yellow
    Write-Host "1. Ubuntu (mas popular)"
    Write-Host "2. Debian (estable)"  
    Write-Host "3. Kali Linux (herramientas de seguridad)"
    Write-Host ""
    
    $choice = Read-Host "Abrir Microsoft Store para instalar Ubuntu? (s/n o 1 para si)"
    if ($choice -match '^[sS1yY]') {
        Start-Process "ms-windows-store://pdp/?productid=9PDXGNCFSCZV"
        Write-Host ""
        Write-Host "Pasos a seguir:" -ForegroundColor Cyan
        Write-Host "1. Instala Ubuntu desde Microsoft Store"
        Write-Host "2. Abrela y configura usuario/password" 
        Write-Host "3. Vuelve a ejecutar este script"
        pause
    }
    exit 0
} else {
    Write-Host ""
    Write-Host "Distribuciones Linux encontradas:" -ForegroundColor Green
    for ($i = 0; $i -lt $distros.Count; $i++) {
        Write-Host "  $($i+1). $($distros[$i])" -ForegroundColor White
    }
    
    if ($distros.Count -eq 1) {
        $selectedDistro = $distros[0]
        Write-Host ""
        Write-Host "Usando: $selectedDistro" -ForegroundColor Green
    } else {
        Write-Host ""
        $selection = Read-Host "Selecciona la distribucion a usar (1-$($distros.Count))"
        $index = [int]$selection - 1
        
        if ($index -ge 0 -and $index -lt $distros.Count) {
            $selectedDistro = $distros[$index]
        } else {
            $selectedDistro = $distros[0]
        }
        
        Write-Host "Seleccionado: $selectedDistro" -ForegroundColor Green
    }
    
    # Guardar distribucion seleccionada para otros scripts
    $configPath = Join-Path $PSScriptRoot "config.txt"
    Set-Content -Path $configPath -Value $selectedDistro -Encoding UTF8
    
    Write-Host ""
    Write-Host "Configuracion guardada. La distribucion '$selectedDistro' sera usada." -ForegroundColor Green
    Write-Host "Continua con el siguiente script." -ForegroundColor Cyan
}