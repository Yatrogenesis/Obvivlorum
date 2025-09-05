# 02_instalar_holomem.ps1
# Ejecutar con doble clic (PowerShell)
Write-Host "=== Instalacion Simbiotica: Paso 2 ===" -ForegroundColor Cyan

# Obtener distribucion seleccionada
. "$PSScriptRoot\get_distro.ps1"
$distro = Get-SelectedDistro
Write-Host "Usando distribucion: $distro" -ForegroundColor Green

# 1. Crear proyecto dentro de Kali WSL
Write-Host "Preparando entorno dentro de Kali..." -ForegroundColor Yellow
wsl -d $distro -- bash -c "mkdir -p ~/obvlivorum_simbiosis/holomem && cd ~/obvlivorum_simbiosis/holomem"

# 2. Copiar archivos base (esto lo hare en Script 3 que generara estos archivos)
Write-Host "Archivos fuente seran preparados por Script 3." -ForegroundColor Cyan

# 3. Instalar herramientas necesarias
Write-Host "Instalando dependencias..." -ForegroundColor Yellow
wsl -d $distro -- bash -c "sudo apt update && sudo apt install -y build-essential linux-headers-\$(uname -r)"

# 4. Compilar el modulo
Write-Host "Compilando modulo holomem..." -ForegroundColor Yellow
wsl -d $distro -- bash -c "cd ~/obvlivorum_simbiosis/holomem && make"

# 5. Compilar utilidad en espacio de usuario
Write-Host "Compilando utilidad holomem-util..." -ForegroundColor Yellow
wsl -d $distro -- bash -c "cd ~/obvlivorum_simbiosis/holomem && gcc -O2 -o holomem-util holomem-util.c"

# 6. Cargar el modulo
Write-Host "Cargando modulo en kernel..." -ForegroundColor Yellow
wsl -d $distro -- bash -c "sudo insmod ~/obvlivorum_simbiosis/holomem/holomem.ko"

# 7. Verificar
Write-Host "Verificando modulo..." -ForegroundColor Yellow
wsl -d $distro -- bash -c "lsmod | grep holomem && echo 'Modulo cargado correctamente' || echo 'Error al cargar modulo'"

Write-Host ""
Write-Host "Paso 2 completado. Continua con el Script 3 para preparar los archivos fuente." -ForegroundColor Green
