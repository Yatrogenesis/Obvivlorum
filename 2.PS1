# 02_instalar_holomem.ps1
# Ejecutar con PowerShell como administrador

# Verificar si se está ejecutando como administrador
if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host "❌ Este script debe ejecutarse como administrador." -ForegroundColor Red
    Write-Host "   Cierra esta ventana y ejecuta PowerShell como administrador." -ForegroundColor Yellow
    pause
    exit 1
}

Write-Host "=== Instalación Simbiótica: Paso 2 ===" -ForegroundColor Cyan

# 0. Verificar que Kali esté instalado y funcionando
$kaliInstalled = wsl -l -v | Select-String "kali-linux"
if (-not $kaliInstalled) {
    Write-Host "❌ Kali Linux no está instalado o no se encuentra." -ForegroundColor Red
    Write-Host "   Ejecuta primero el Script 1 y asegúrate de completar la instalación de Kali." -ForegroundColor Yellow
    pause
    exit 1
}

# Verificar que Kali sea accesible
try {
    $kaliTest = wsl -d kali-linux -- echo "OK"
    if ($kaliTest -ne "OK") {
        throw "Error de comunicación con Kali Linux"
    }
} catch {
    Write-Host "❌ No se puede acceder a Kali Linux. Verifica la instalación." -ForegroundColor Red
    Write-Host "   Asegúrate de que hayas ejecutado Kali al menos una vez después de instalarlo." -ForegroundColor Yellow
    pause
    exit 1
}

# 1. Crear proyecto dentro de Kali WSL
Write-Host "🔄 Preparando entorno dentro de Kali..." -ForegroundColor Yellow
wsl -d kali-linux -- bash -c "mkdir -p ~/obvlivorum_simbiosis/holomem"

# 2. Instalar herramientas necesarias
Write-Host "🔄 Actualizando repositorios e instalando dependencias..." -ForegroundColor Yellow
$dependenciasExito = wsl -d kali-linux -- bash -c "
    sudo apt update && 
    sudo apt install -y build-essential make gcc linux-headers-\$(uname -r) git wget
"

if (-not $?) {
    Write-Host "❌ Error al escribir los archivos en WSL." -ForegroundColor Red
    pause
    exit 1
}

# Verificar que los archivos se hayan creado correctamente
$filesExist = wsl -d kali-linux -- bash -c "
    test -f $basePath/holomem.c && 
    test -f $basePath/holomem-util.c && 
    test -f $basePath/Makefile && 
    echo 'OK'
"

if ($filesExist -ne "OK") {
    Write-Host "❌ No se pudieron verificar todos los archivos fuente." -ForegroundColor Red
    pause
    exit 1
}

Write-Host "✅ Archivos fuente generados correctamente." -ForegroundColor Green

# Compilar el módulo
Write-Host "🔄 Compilando el módulo holomem..." -ForegroundColor Yellow
$compileResult = wsl -d kali-linux -- bash -c "cd $basePath && make 2>&1"
if (-not $?) {
    Write-Host "❌ Error al compilar el módulo:" -ForegroundColor Red
    Write-Host $compileResult -ForegroundColor Red
    pause
    exit 1
}

# Compilar utilidad de espacio de usuario
Write-Host "🔄 Compilando utilidad holomem-util..." -ForegroundColor Yellow
$compileUtilResult = wsl -d kali-linux -- bash -c "cd $basePath && gcc -Wall -O2 -o holomem-util holomem-util.c 2>&1"
if (-not $?) {
    Write-Host "❌ Error al compilar la utilidad:" -ForegroundColor Red
    Write-Host $compileUtilResult -ForegroundColor Red
    pause
    exit 1
}

# Cargar el módulo
Write-Host "🔄 Cargando módulo en kernel..." -ForegroundColor Yellow
$insertResult = wsl -d kali-linux -- bash -c "cd $basePath && sudo insmod holomem.ko 2>&1"
if (-not $?) {
    Write-Host "❌ Error al cargar el módulo:" -ForegroundColor Red
    Write-Host $insertResult -ForegroundColor Red
    pause
    exit 1
}

# Crear nodo de dispositivo si no existe
$deviceResult = wsl -d kali-linux -- bash -c "
    if [ ! -c /dev/holomem ]; then
        sudo mknod /dev/holomem c \$(grep holomem /proc/devices | cut -d' ' -f1) 0
        sudo chmod 666 /dev/holomem
    fi
"

# Verificar que el módulo esté cargado
$moduleCheck = wsl -d kali-linux -- bash -c "lsmod | grep holomem && echo 'OK'"
if ($moduleCheck -ne "OK") {
    Write-Host "❌ No se pudo verificar que el módulo esté cargado." -ForegroundColor Red
    pause
    exit 1
}

Write-Host "✅ Módulo holomem compilado, cargado y verificado correctamente." -ForegroundColor Green
Write-Host "✅ Paso 3 completado. Continúa con el Script 4 para instalar TinyLLaMA." -ForegroundColor Green
Write-Host "❌ Error instalando dependencias. Verifica la conexión a Internet y los repositorios." -ForegroundColor Red
    pause
    exit 1
}

Write-Host "✅ Dependencias instaladas correctamente." -ForegroundColor Green

# 3. Verificar que exista el directorio para archivos fuente (se crearán en Script 3)
Write-Host "🔄 Verificando estructura de directorios..." -ForegroundColor Yellow
wsl -d kali-linux -- bash -c "
    if [ ! -d ~/obvlivorum_simbiosis/holomem ]; then
        mkdir -p ~/obvlivorum_simbiosis/holomem
    fi
"

Write-Host "✅ Estructura de directorios preparada. Continúa con el Script 3 para generar los archivos fuente." -ForegroundColor Green
Write-Host "⚠️ IMPORTANTE: El módulo se compilará después de generar los archivos fuente en el Script 3." -ForegroundColor Yellow

# 4. Configurar permisos de sudo para evitar contraseñas repetidas
Write-Host "🔄 Configurando permisos para una mejor experiencia de usuario..." -ForegroundColor Yellow
$username = wsl -d kali-linux -- whoami
wsl -d kali-linux -- bash -c "
    if ! sudo grep -q '$username ALL=(ALL) NOPASSWD: ALL' /etc/sudoers; then
        echo '$username ALL=(ALL) NOPASSWD: ALL' | sudo tee -a /etc/sudoers > /dev/null
    fi
"

Write-Host "✅ Paso 2 completado. Ejecuta el Script 3 para generar los archivos fuente del módulo." -ForegroundColor Green