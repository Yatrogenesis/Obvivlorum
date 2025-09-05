# 00_instalar_completo.ps1
# Script maestro para instalacion completa del sistema simbiotico
# Ejecutar como administrador

Write-Host "=== SIMBIOSIS OBVIVLORUM - Instalacion Completa ===" -ForegroundColor Cyan
Write-Host "Sistema holografico de IA con memoria persistente"
Write-Host ""

$scripts = @(
    "01_instalar_WSL2_y_Kali.ps1",
    "03_generar_archivos_fuente.ps1", 
    "02_instalar_holomem.ps1",
    "04_instalar_llama_cpp.ps1",
    "05_middleware_e_interfaz.ps1"
)

$currentDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "Orden de instalacion:" -ForegroundColor Yellow
for ($i = 0; $i -lt $scripts.Length; $i++) {
    Write-Host "  $($i+1). $($scripts[$i])" -ForegroundColor White
}
Write-Host ""

$response = Read-Host "Continuar con la instalacion completa? (s/n o 1 para si)"
if ($response -notmatch '^[sS1yY]') {
    Write-Host "Instalacion cancelada" -ForegroundColor Red
    exit 0
}

for ($i = 0; $i -lt $scripts.Length; $i++) {
    $script = $scripts[$i]
    $scriptPath = Join-Path $currentDir $script
    
    Write-Host ""
    Write-Host ("=" * 60) -ForegroundColor Cyan
    Write-Host "EJECUTANDO: $script ($($i+1)/$($scripts.Length))" -ForegroundColor Green
    Write-Host ("=" * 60) -ForegroundColor Cyan
    
    if (Test-Path $scriptPath) {
        try {
            & $scriptPath
            if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne $null) {
                Write-Host "Error en $script (Codigo: $LASTEXITCODE)" -ForegroundColor Red
                $continue = Read-Host "Continuar con el siguiente script? (s/n o 1 para si)"
                if ($continue -notmatch '^[sS1yY]') {
                    Write-Host "Instalacion abortada" -ForegroundColor Red
                    exit $LASTEXITCODE
                }
            } else {
                Write-Host "$script completado exitosamente" -ForegroundColor Green
            }
        }
        catch {
            Write-Host "Error ejecutando $script`: $($_.Exception.Message)" -ForegroundColor Red
            $continue = Read-Host "Continuar con el siguiente script? (s/n o 1 para si)"
            if ($continue -notmatch '^[sS1yY]') {
                Write-Host "Instalacion abortada" -ForegroundColor Red
                exit 1
            }
        }
    } else {
        Write-Host "Script no encontrado: $scriptPath" -ForegroundColor Yellow
    }
    
    if ($i -lt $scripts.Length - 1) {
        Write-Host ""
        Write-Host "Presiona Enter para continuar con el siguiente paso..." -ForegroundColor Yellow
        Read-Host
    }
}

Write-Host ""
Write-Host ("=" * 60) -ForegroundColor Green
Write-Host "INSTALACION COMPLETA FINALIZADA" -ForegroundColor Green
Write-Host ("=" * 60) -ForegroundColor Green
Write-Host ""
Write-Host "Para lanzar el sistema simbiotico, ejecuta:"
Write-Host "   .\06_lanzar_simbiosis.ps1" -ForegroundColor Cyan
Write-Host ""
Write-Host "Comandos disponibles en la interfaz:"
Write-Host "   /memoria - Ver patrones almacenados"
Write-Host "   /salir   - Terminar sesion"
Write-Host ""
Write-Host "El sistema utilizara:" -ForegroundColor Yellow
Write-Host "   - Modelo: Llama-3.2-3B-Instruct (balance velocidad/calidad)"
Write-Host "   - Memoria: Modulo kernel holomem"
Write-Host "   - Interfaz: CLI con sintesis de voz"
