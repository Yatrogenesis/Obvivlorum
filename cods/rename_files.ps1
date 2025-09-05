# Rename script files
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Backup old version
if (Test-Path "$scriptDir\03_generar_archivos_fuente.ps1") {
    Move-Item "$scriptDir\03_generar_archivos_fuente.ps1" "$scriptDir\03_generar_archivos_fuente_old.ps1" -Force
}

# Use new version
if (Test-Path "$scriptDir\03_generar_archivos_fuente_v2.ps1") {
    Move-Item "$scriptDir\03_generar_archivos_fuente_v2.ps1" "$scriptDir\03_generar_archivos_fuente.ps1" -Force
}

Write-Host "Archivos renombrados correctamente" -ForegroundColor Green