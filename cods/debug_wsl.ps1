# debug_wsl.ps1
# Script para diagnosticar el formato de salida de WSL

Write-Host "=== Debug WSL Output ===" -ForegroundColor Cyan
Write-Host ""

Write-Host "1. Probando: wsl --list --quiet" -ForegroundColor Yellow
$output1 = wsl --list --quiet 2>$null
if ($output1) {
    Write-Host "Output:" -ForegroundColor Green
    $i = 0
    foreach ($line in $output1) {
        Write-Host "[$i]: '$line' (Length: $($line.Length))" -ForegroundColor White
        # Mostrar caracteres ASCII
        $bytes = [System.Text.Encoding]::Unicode.GetBytes($line)
        $hex = ($bytes | ForEach-Object { $_.ToString("X2") }) -join " "
        Write-Host "     Hex: $hex" -ForegroundColor Gray
        $i++
    }
} else {
    Write-Host "Sin output" -ForegroundColor Red
}

Write-Host ""
Write-Host "2. Probando: wsl -l -v" -ForegroundColor Yellow
$output2 = wsl -l -v 2>$null
if ($output2) {
    Write-Host "Output:" -ForegroundColor Green
    $i = 0
    foreach ($line in $output2) {
        Write-Host "[$i]: '$line'" -ForegroundColor White
        $i++
    }
} else {
    Write-Host "Sin output" -ForegroundColor Red
}

Write-Host ""
Write-Host "3. Probando: wsl --list" -ForegroundColor Yellow
$output3 = wsl --list 2>$null
if ($output3) {
    Write-Host "Output:" -ForegroundColor Green
    $i = 0
    foreach ($line in $output3) {
        Write-Host "[$i]: '$line'" -ForegroundColor White
        $i++
    }
} else {
    Write-Host "Sin output" -ForegroundColor Red
}

Write-Host ""
Write-Host "4. Intentando parsear distribuciones:" -ForegroundColor Yellow
$distros = @()

# Metodo con encoding
$output = & wsl --list --quiet 2>$null
if ($output) {
    foreach ($item in $output) {
        if ($item) {
            # Convertir de UTF-16 si es necesario
            $text = $item.ToString()
            # Eliminar BOM y caracteres nulos
            $clean = $text -replace '\x00', '' -replace '^\xFEFF', '' -replace '^\xFF\xFE', ''
            $clean = $clean.Trim()
            
            if ($clean -and $clean.Length -gt 0 -and 
                $clean -ne "Windows Subsystem for Linux Distributions:" -and
                $clean -notmatch "docker") {
                Write-Host "Encontrado: '$clean'" -ForegroundColor Green
                $distros += $clean
            }
        }
    }
}

Write-Host ""
Write-Host "Distribuciones detectadas: $($distros.Count)" -ForegroundColor Cyan
foreach ($d in $distros) {
    Write-Host " - $d" -ForegroundColor White
}