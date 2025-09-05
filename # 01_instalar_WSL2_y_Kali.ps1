# 01_instalar_WSL2_y_Kali.ps1
# Ejecuta con botón derecho > Ejecutar con PowerShell como administrador

# Verificar si se está ejecutando como administrador
if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host "❌ Este script debe ejecutarse como administrador." -ForegroundColor Red
    Write-Host "   Cierra esta ventana y ejecuta PowerShell como administrador." -ForegroundColor Yellow
    pause
    exit 1
}

Write-Host "=== Instalación Simbiótica: Paso 1 ===" -ForegroundColor Cyan
Write-Host "Verificando WSL2, Virtualización y entorno base..."

# Inicializar la variable $needRestart
$needRestart = $false

# 1. Verificar si la virtualización está activada
$cpuInfo = Get-CimInstance Win32_Processor
if (-not $cpuInfo.VirtualizationFirmwareEnabled) {
    Write-Host "❌ Virtualización NO está activada en BIOS. Actívala y vuelve a intentar." -ForegroundColor Red
    Write-Host "   Generalmente se encuentra en la sección 'Advanced Features' o 'CPU Configuration' del BIOS/UEFI." -ForegroundColor Yellow
    pause
    exit 1
} else {
    Write-Host "✅ Virtualización detectada." -ForegroundColor Green
}

# 2. Verificar si WSL ya está instalado
$wslFeature = Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux
if ($wslFeature.State -ne "Enabled") {
    Write-Host "🔄 Activando WSL..." -ForegroundColor Yellow
    dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
    $needRestart = $true
} else {
    Write-Host "✅ WSL ya está activado." -ForegroundColor Green
}

# 3. Verificar Plataforma de Máquina Virtual
$vmPlatform = Get-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform
if ($vmPlatform.State -ne "Enabled") {
    Write-Host "🔄 Activando plataforma de virtualización..." -ForegroundColor Yellow
    dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
    $needRestart = $true
} else {
    Write-Host "✅ Plataforma de virtualización ya está activada." -ForegroundColor Green
}

# 4. Descargar e instalar el paquete de actualización del kernel de Linux para WSL2
$kernelUpdateMsi = "$env:TEMP\wsl_update_x64.msi"
if (-not (Test-Path $kernelUpdateMsi)) {
    Write-Host "🔄 Descargando la actualización del kernel de Linux para WSL2..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri "https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi" -OutFile $kernelUpdateMsi -UseBasicParsing
    Write-Host "🔄 Instalando la actualización del kernel de Linux para WSL2..." -ForegroundColor Yellow
    Start-Process msiexec.exe -Wait -ArgumentList "/I $kernelUpdateMsi /quiet"
    Write-Host "✅ Actualización del kernel de Linux para WSL2 instalada." -ForegroundColor Green
} else {
    Write-Host "✅ La actualización del kernel de Linux para WSL2 ya está disponible localmente." -ForegroundColor Green
}

# 5. Establecer WSL2 como predeterminado
try {
    $wslVersion = (wsl --status | Select-String "Default Version").ToString().Split(":")[1].Trim()
    if ($wslVersion -ne "2") {
        Write-Host "🔄 Estableciendo WSL2 como versión predeterminada..." -ForegroundColor Yellow
        wsl --set-default-version 2
        Write-Host "✅ WSL2 establecido como predeterminado." -ForegroundColor Green
    } else {
        Write-Host "✅ WSL2 ya está establecido como predeterminado." -ForegroundColor Green
    }
} catch {
    Write-Host "🔄 Estableciendo WSL2 como versión predeterminada..." -ForegroundColor Yellow
    wsl --set-default-version 2
    Write-Host "✅ WSL2 establecido como predeterminado." -ForegroundColor Green
}

# 6. Verificar si Kali ya está instalado
$kaliInstalled = wsl -l -v | Select-String "kali-linux"
if (-not $kaliInstalled) {
    Write-Host "🔄 Instalando Kali Linux desde la Microsoft Store..." -ForegroundColor Yellow
    Start-Process "ms-windows-store://pdp/?productid=9PKR34TNCV07"
    Write-Host ""
    Write-Host "🕓 Espera a que la instalación termine en la Microsoft Store." -ForegroundColor Yellow
    Write-Host "   IMPORTANTE: Después de instalar desde la tienda, EJECUTA Kali Linux al menos una vez" -ForegroundColor Yellow
    Write-Host "   para completar la configuración inicial." -ForegroundColor Yellow
    Write-Host "   Luego CIERRA ESA VENTANA y vuelve a ejecutar el Script 2." -ForegroundColor Yellow
    pause
    exit 0
} else {
    Write-Host "✅ Kali ya está instalado. Puedes continuar con el Script 2." -ForegroundColor Green
}

# Verificar si se necesita reiniciar
if ($needRestart) {
    Write-Host "⚠️ Es necesario reiniciar el sistema para completar la instalación." -ForegroundColor Yellow
    $restart = Read-Host "¿Deseas reiniciar ahora? (S/N)"
    if ($restart -eq "S" -or $restart -eq "s") {
        Restart-Computer
    } else {
        Write-Host "❗ Recuerda reiniciar tu sistema antes de continuar con el Script 2." -ForegroundColor Red
    }
}