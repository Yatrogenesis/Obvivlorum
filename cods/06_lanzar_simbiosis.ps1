# 06_lanzar_simbiosis.ps1
Write-Host "=== Lanzando entorno simbiotico completo ===" -ForegroundColor Cyan

# Obtener distribucion seleccionada
. "$PSScriptRoot\get_distro.ps1"
$distro = Get-SelectedDistro
Write-Host "Usando distribucion: $distro" -ForegroundColor Green

# Ejecutar comandos en WSL con formato correcto
$wslCommand = @'
cd ~/obvlivorum_simbiosis
if [ -d "simbiox" ]; then
    source simbiox/bin/activate
    if [ -f "simbiosis_cli.py" ]; then
        python3 simbiosis_cli.py
    else
        echo "Error: simbiosis_cli.py no encontrado"
        exit 1
    fi
else
    echo "Error: Entorno virtual simbiox no encontrado"
    echo "Ejecuta primero el script de configuracion"
    exit 1
fi
'@

wsl -d $distro -- bash -c $wslCommand
