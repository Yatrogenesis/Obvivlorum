# test_instalador.ps1
# Script de prueba para verificar que todo funciona

Write-Host "=== Test del Instalador Simbiosis ===" -ForegroundColor Cyan
Write-Host ""

# Probar deteccion de distribuciones
Write-Host "1. Probando deteccion de distribuciones..." -ForegroundColor Yellow
. "$PSScriptRoot\get_distro.ps1"
$distro = Get-SelectedDistro
Write-Host "Distribucion detectada: $distro" -ForegroundColor Green

# Verificar que WSL funciona con la distro
Write-Host ""
Write-Host "2. Probando conexion WSL..." -ForegroundColor Yellow
try {
    $user = wsl -d $distro whoami 2>$null
    if ($user) {
        Write-Host "Usuario WSL: $($user.Trim())" -ForegroundColor Green
    } else {
        Write-Host "Error conectando a $distro" -ForegroundColor Red
    }
} catch {
    Write-Host "Error en WSL: $($_.Exception.Message)" -ForegroundColor Red
}

# Verificar modelos LLM
Write-Host ""
Write-Host "3. Verificando modelos LLM..." -ForegroundColor Yellow
$llmPath = "D:\Obvivlorum\LLMs"
if (Test-Path $llmPath) {
    $models = Get-ChildItem $llmPath -Filter "*.gguf"
    Write-Host "Modelos encontrados: $($models.Count)" -ForegroundColor Green
    foreach ($model in $models) {
        $sizeMB = [math]::Round($model.Length / 1MB, 0)
        Write-Host "  - $($model.Name) ($sizeMB MB)" -ForegroundColor White
    }
} else {
    Write-Host "Carpeta LLMs no encontrada" -ForegroundColor Red
}

Write-Host ""
Write-Host "=== Test completado ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Si todo aparece en verde, puedes ejecutar:" -ForegroundColor Yellow
Write-Host "   .\00_instalar_completo.ps1" -ForegroundColor Cyan
Write-Host ""
Write-Host "Recuerda responder 's' o '1' para continuar" -ForegroundColor Yellow