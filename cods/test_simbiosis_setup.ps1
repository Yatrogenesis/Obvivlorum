# test_simbiosis_setup.ps1
# Script de prueba completo para verificar el entorno antes de ejecutar

Write-Host "`n=== VERIFICADOR DE ENTORNO SIMBIOSIS ===" -ForegroundColor Cyan
Write-Host "Este script verificara todos los requisitos antes de ejecutar`n" -ForegroundColor Yellow

$hasErrors = $false

# 1. Verificar PowerShell version
Write-Host "[1/7] Verificando PowerShell..." -ForegroundColor Blue
$psVersion = $PSVersionTable.PSVersion
if ($psVersion.Major -ge 5) {
    Write-Host "  OK: PowerShell $psVersion" -ForegroundColor Green
} else {
    Write-Host "  ERROR: PowerShell version muy antigua ($psVersion)" -ForegroundColor Red
    $hasErrors = $true
}

# 2. Verificar que get_distro.ps1 existe
Write-Host "`n[2/7] Verificando dependencias de scripts..." -ForegroundColor Blue
$getDistroPath = Join-Path $PSScriptRoot "get_distro.ps1"
if (Test-Path $getDistroPath) {
    Write-Host "  OK: get_distro.ps1 encontrado" -ForegroundColor Green
} else {
    Write-Host "  ERROR: get_distro.ps1 no encontrado en $PSScriptRoot" -ForegroundColor Red
    $hasErrors = $true
}

# 3. Verificar WSL instalado
Write-Host "`n[3/7] Verificando instalacion de WSL..." -ForegroundColor Blue
try {
    $wslVersion = wsl --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  OK: WSL instalado" -ForegroundColor Green
        $wslVersion | ForEach-Object { Write-Host "      $_" -ForegroundColor Gray }
    } else {
        throw "WSL no responde"
    }
} catch {
    Write-Host "  ERROR: WSL no esta instalado o no funciona" -ForegroundColor Red
    Write-Host "  Instala WSL con: wsl --install" -ForegroundColor Yellow
    $hasErrors = $true
}

# 4. Verificar distribuciones WSL
Write-Host "`n[4/7] Verificando distribuciones Linux..." -ForegroundColor Blue
try {
    $distros = @()
    $wslList = wsl --list --quiet 2>$null
    
    if ($wslList) {
        foreach ($line in $wslList) {
            $line = $line.Trim()
            $cleanName = $line -replace '[^\x20-\x7E]', ''
            if ($cleanName -and $cleanName -ne "Windows Subsystem for Linux Distributions:" -and 
                $cleanName -ne "docker-desktop" -and $cleanName -ne "docker-desktop-data") {
                $distros += $cleanName
            }
        }
    }
    
    if ($distros.Count -gt 0) {
        Write-Host "  OK: Distribuciones encontradas:" -ForegroundColor Green
        $distros | ForEach-Object { Write-Host "      - $_" -ForegroundColor Gray }
    } else {
        Write-Host "  ERROR: No hay distribuciones Linux instaladas" -ForegroundColor Red
        Write-Host "  Instala una con: wsl --install -d Ubuntu" -ForegroundColor Yellow
        $hasErrors = $true
    }
} catch {
    Write-Host "  ERROR: No se pueden listar las distribuciones" -ForegroundColor Red
    $hasErrors = $true
}

# 5. Obtener y verificar distribucion seleccionada
Write-Host "`n[5/7] Verificando distribucion seleccionada..." -ForegroundColor Blue
try {
    . $getDistroPath
    $selectedDistro = Get-SelectedDistro
    
    if ($selectedDistro) {
        Write-Host "  OK: Distribucion seleccionada: $selectedDistro" -ForegroundColor Green
        
        # Verificar que realmente existe
        $testCommand = wsl -d $selectedDistro -- echo "test" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  OK: La distribucion responde correctamente" -ForegroundColor Green
        } else {
            Write-Host "  ERROR: La distribucion '$selectedDistro' no responde" -ForegroundColor Red
            $hasErrors = $true
        }
    } else {
        Write-Host "  ERROR: No se pudo obtener la distribucion" -ForegroundColor Red
        $hasErrors = $true
    }
} catch {
    Write-Host "  ERROR: Error al obtener la distribucion" -ForegroundColor Red
    $hasErrors = $true
}

# 6. Verificar archivos en WSL
if (-not $hasErrors -and $selectedDistro) {
    Write-Host "`n[6/7] Verificando archivos en WSL..." -ForegroundColor Blue
    
    # Verificar directorio principal
    $checkDir = wsl -d $selectedDistro -- bash -c "if [ -d ~/obvlivorum_simbiosis ]; then echo 'EXISTS'; else echo 'NOT_EXISTS'; fi" 2>$null
    if ($checkDir -eq "EXISTS") {
        Write-Host "  OK: Directorio ~/obvlivorum_simbiosis existe" -ForegroundColor Green
    } else {
        Write-Host "  ERROR: Directorio ~/obvlivorum_simbiosis no existe" -ForegroundColor Red
        Write-Host "  Ejecuta primero el script de configuracion" -ForegroundColor Yellow
        $hasErrors = $true
    }
    
    # Verificar entorno virtual
    $checkVenv = wsl -d $selectedDistro -- bash -c "if [ -d ~/obvlivorum_simbiosis/simbiox ]; then echo 'EXISTS'; else echo 'NOT_EXISTS'; fi" 2>$null
    if ($checkVenv -eq "EXISTS") {
        Write-Host "  OK: Entorno virtual simbiox existe" -ForegroundColor Green
    } else {
        Write-Host "  ERROR: Entorno virtual simbiox no existe" -ForegroundColor Red
        Write-Host "  Necesitas crear el entorno virtual primero" -ForegroundColor Yellow
        $hasErrors = $true
    }
    
    # Verificar script Python
    $checkPy = wsl -d $selectedDistro -- bash -c "if [ -f ~/obvlivorum_simbiosis/simbiosis_cli.py ]; then echo 'EXISTS'; else echo 'NOT_EXISTS'; fi" 2>$null
    if ($checkPy -eq "EXISTS") {
        Write-Host "  OK: Script simbiosis_cli.py existe" -ForegroundColor Green
    } else {
        Write-Host "  ERROR: Script simbiosis_cli.py no existe" -ForegroundColor Red
        $hasErrors = $true
    }
    
    # Verificar Python3
    $pythonVersion = wsl -d $selectedDistro -- python3 --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  OK: Python3 instalado - $pythonVersion" -ForegroundColor Green
    } else {
        Write-Host "  ERROR: Python3 no esta instalado en WSL" -ForegroundColor Red
        Write-Host "  Instala con: sudo apt update && sudo apt install python3 python3-venv" -ForegroundColor Yellow
        $hasErrors = $true
    }
}

# 7. Prueba de ejecucion simulada
Write-Host "`n[7/7] Prueba de comando final..." -ForegroundColor Blue
if (-not $hasErrors) {
    $testCommand = @'
echo "Probando acceso al directorio..."
cd ~/obvlivorum_simbiosis || exit 1
echo "Directorio OK"
if [ -f "simbiox/bin/activate" ]; then
    echo "Archivo de activacion encontrado"
else
    echo "ERROR: Archivo de activacion no encontrado"
    exit 1
fi
'@
    
    $result = wsl -d $selectedDistro -- bash -c $testCommand 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  OK: Prueba de comando exitosa" -ForegroundColor Green
        Write-Host $result -ForegroundColor Gray
    } else {
        Write-Host "  ERROR: La prueba de comando fallo" -ForegroundColor Red
        Write-Host $result -ForegroundColor Red
        $hasErrors = $true
    }
}

# Resumen final
Write-Host "`n=== RESUMEN ===" -ForegroundColor Cyan
if ($hasErrors) {
    Write-Host "RESULTADO: ERRORES DETECTADOS" -ForegroundColor Red
    Write-Host "Por favor, corrige los errores antes de ejecutar 06_lanzar_simbiosis.ps1" -ForegroundColor Yellow
    exit 1
} else {
    Write-Host "RESULTADO: TODO LISTO" -ForegroundColor Green
    Write-Host "Puedes ejecutar con seguridad: .\06_lanzar_simbiosis.ps1" -ForegroundColor Green
    
    Write-Host "`n¿Deseas ejecutar el script ahora? (S/N): " -NoNewline -ForegroundColor Yellow
    $response = Read-Host
    if ($response -eq 'S' -or $response -eq 's') {
        Write-Host "`nEjecutando 06_lanzar_simbiosis.ps1..." -ForegroundColor Cyan
        & "$PSScriptRoot\06_lanzar_simbiosis.ps1"
    }
}