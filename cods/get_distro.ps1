# get_distro.ps1
# Funcion helper para obtener la distribucion Linux seleccionada

function Get-SelectedDistro {
    $configPath = Join-Path $PSScriptRoot "config.txt"
    
    if (Test-Path $configPath) {
        $distro = Get-Content $configPath -First 1
        # Limpiar caracteres no visibles
        $distro = $distro -replace '[^\x20-\x7E]', ''
        if ($distro -and $distro -ne "*") {
            return $distro.Trim()
        }
    }
    
    # Si no hay config, buscar cualquier distro instalada
    # Metodo 1: Lista simple
    $wslListSimple = wsl --list --quiet 2>$null
    if ($wslListSimple) {
        foreach ($line in $wslListSimple) {
            $line = $line.Trim()
            if ($line -and $line -ne "Windows Subsystem for Linux Distributions:" -and 
                $line -ne "docker-desktop" -and $line -ne "docker-desktop-data") {
                # Limpiar caracteres no visibles
                $cleanName = $line -replace '[^\x20-\x7E]', ''
                if ($cleanName) {
                    return $cleanName.Trim()
                }
            }
        }
    }
    
    # Metodo 2: Con verbose
    $wslOutput = wsl -l -v 2>$null
    if ($wslOutput) {
        $lines = $wslOutput | Select-Object -Skip 1
        foreach ($line in $lines) {
            if ($line -match '^\s*(\*)?\s*([^\s]+)\s+') {
                $distroName = $matches[2]
                if ($distroName -and $distroName -ne "docker-desktop" -and 
                    $distroName -ne "docker-desktop-data") {
                    return $distroName.Trim()
                }
            }
        }
    }
    
    # Si todo falla, intentar con la primera distro disponible
    Write-Host "ADVERTENCIA: No se pudo detectar distribucion, usando Ubuntu como fallback" -ForegroundColor Yellow
    return "Ubuntu"
}

# Exportar la funcion si se ejecuta como modulo
if ($MyInvocation.InvocationName -ne '.') {
    Get-SelectedDistro
}