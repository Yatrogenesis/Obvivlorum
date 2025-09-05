# verificar_instalacion.ps1
# Script para verificar que todo este instalado correctamente

Write-Host "=== Verificando instalacion de Simbiosis ===" -ForegroundColor Cyan
Write-Host ""

$errors = 0

# 1. Verificar WSL
Write-Host "Verificando WSL..." -ForegroundColor Yellow
try {
    $wslVersion = wsl --version 2>$null
    if ($wslVersion) {
        Write-Host "[OK] WSL instalado" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] WSL no detectado" -ForegroundColor Red
        $errors++
    }
} catch {
    Write-Host "[ERROR] WSL no disponible" -ForegroundColor Red
    $errors++
}

# 2. Verificar distribucion Linux
Write-Host ""
Write-Host "Verificando distribucion Linux..." -ForegroundColor Yellow
if (Test-Path "$PSScriptRoot\config.txt") {
    $distro = Get-Content "$PSScriptRoot\config.txt" -First 1
    Write-Host "[OK] Distribucion configurada: $distro" -ForegroundColor Green
    
    # Verificar que existe
    $distroExists = wsl -l 2>$null | Select-String $distro
    if ($distroExists) {
        Write-Host "[OK] $distro esta instalado" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] $distro no esta instalado" -ForegroundColor Red
        $errors++
    }
} else {
    Write-Host "[ERROR] No se encontro configuracion de distribucion" -ForegroundColor Red
    $errors++
}

# 3. Verificar archivos del proyecto
Write-Host ""
Write-Host "Verificando archivos del proyecto..." -ForegroundColor Yellow
if ($distro) {
    $files = @(
        "~/obvlivorum_simbiosis/holomem/holomem.c",
        "~/obvlivorum_simbiosis/holomem/holomem-util.c", 
        "~/obvlivorum_simbiosis/holomem/Makefile",
        "~/obvlivorum_simbiosis/simbiosis_cli.py"
    )
    
    foreach ($file in $files) {
        $exists = wsl -d $distro -- bash -c "test -f $file && echo 'exists'"
        if ($exists -eq "exists") {
            Write-Host "[OK] $file" -ForegroundColor Green
        } else {
            Write-Host "[ERROR] Falta: $file" -ForegroundColor Red
            $errors++
        }
    }
}

# 4. Verificar modelos LLM
Write-Host ""
Write-Host "Verificando modelos LLM..." -ForegroundColor Yellow
$llmPath = "D:\Obvivlorum\LLMs"
if (Test-Path $llmPath) {
    $models = Get-ChildItem $llmPath -Filter "*.gguf"
    if ($models.Count -gt 0) {
        Write-Host "[OK] $($models.Count) modelos encontrados:" -ForegroundColor Green
        foreach ($model in $models) {
            Write-Host "     - $($model.Name)" -ForegroundColor White
        }
    } else {
        Write-Host "[ERROR] No se encontraron modelos .gguf" -ForegroundColor Red
        $errors++
    }
} else {
    Write-Host "[ERROR] Carpeta LLMs no encontrada" -ForegroundColor Red
    $errors++
}

# 5. Verificar llama.cpp
Write-Host ""
Write-Host "Verificando llama.cpp..." -ForegroundColor Yellow
if ($distro) {
    $llamaExists = wsl -d $distro -- bash -c "test -d ~/obvlivorum_simbiosis/llama.cpp && echo 'exists'"
    if ($llamaExists -eq "exists") {
        Write-Host "[OK] llama.cpp instalado" -ForegroundColor Green
        
        # Verificar modelo copiado
        $modelExists = wsl -d $distro -- bash -c "test -f ~/obvlivorum_simbiosis/llama.cpp/models/llama-model.gguf && echo 'exists'"
        if ($modelExists -eq "exists") {
            Write-Host "[OK] Modelo local configurado" -ForegroundColor Green
        } else {
            Write-Host "[ADVERTENCIA] Modelo no copiado a llama.cpp" -ForegroundColor Yellow
        }
    } else {
        Write-Host "[ERROR] llama.cpp no instalado" -ForegroundColor Red
        $errors++
    }
}

# Resumen
Write-Host ""
Write-Host "=" * 50 -ForegroundColor Cyan
if ($errors -eq 0) {
    Write-Host "RESULTADO: Todo listo para ejecutar" -ForegroundColor Green
    Write-Host ""
    Write-Host "Para iniciar el sistema, ejecuta:"
    Write-Host "   .\06_lanzar_simbiosis.ps1" -ForegroundColor Cyan
} else {
    Write-Host "RESULTADO: Se encontraron $errors errores" -ForegroundColor Red
    Write-Host ""
    Write-Host "Solucion recomendada:"
    Write-Host "   Ejecuta: .\00_instalar_completo.ps1" -ForegroundColor Yellow
}
Write-Host "=" * 50 -ForegroundColor Cyan