# Obvivlorum Sistema Unificado de Instalación
# ============================================
# Autor: Sistema Integrado
# Versión: 2.0 Mejorada

# Colores para mejor visualización
$host.UI.RawUI.BackgroundColor = "Black"
$host.UI.RawUI.ForegroundColor = "White"
Clear-Host

# Verificar permisos de administrador
if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host "❌ ERROR: Este script requiere permisos de Administrador" -ForegroundColor Red
    Write-Host ""
    Write-Host "Por favor:" -ForegroundColor Yellow
    Write-Host "1. Cierra esta ventana" -ForegroundColor Yellow
    Write-Host "2. Busca PowerShell en el menú inicio" -ForegroundColor Yellow
    Write-Host "3. Clic derecho -> Ejecutar como administrador" -ForegroundColor Yellow
    Write-Host "4. Navega a D:\Obvivlorum" -ForegroundColor Yellow
    Write-Host "5. Ejecuta: .\Sistema_Unificado.ps1" -ForegroundColor Yellow
    Write-Host ""
    pause
    exit
}

# Funciones auxiliares
function Show-Header {
    Clear-Host
    Write-Host "╔════════════════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║                         OBVIVLORUM SIMBIOSIS v2.0                          ║" -ForegroundColor Cyan
    Write-Host "║                   Sistema Integrado WSL2 + Kali + HoloMem                  ║" -ForegroundColor Cyan
    Write-Host "╚════════════════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
    Write-Host ""
}

function Test-SystemStatus {
    Write-Host "Verificando estado del sistema..." -ForegroundColor Yellow
    Write-Host ""
    
    $status = @{
        Virtualization = $false
        WSL = $false
        WSL2 = $false
        Kali = $false
        Project = $false
        Module = $false
    }
    
    # Verificar virtualización
    $cpu = Get-CimInstance Win32_Processor
    $status.Virtualization = $cpu.VirtualizationFirmwareEnabled
    
    # Verificar WSL
    try {
        $wslStatus = wsl --status 2>$null
        $status.WSL = $?
        
        if ($status.WSL) {
            $status.WSL2 = $wslStatus -match "Default Version.*2"
        }
    } catch {
        $status.WSL = $false
    }
    
    # Verificar Kali
    if ($status.WSL) {
        $kaliCheck = wsl -l -v 2>$null | Select-String "kali-linux"
        $status.Kali = $null -ne $kaliCheck
        
        # Verificar proyecto
        if ($status.Kali) {
            $projectCheck = wsl -d kali-linux -- test -d ~/obvlivorum_simbiosis 2>$null
            $status.Project = $?
            
            # Verificar módulo
            if ($status.Project) {
                $moduleCheck = wsl -d kali-linux -- lsmod 2>$null | Select-String "holomem"
                $status.Module = $null -ne $moduleCheck
            }
        }
    }
    
    # Mostrar estado
    Write-Host "Estado del Sistema:" -ForegroundColor White
    Write-Host "══════════════════" -ForegroundColor Gray
    
    if ($status.Virtualization) {
        Write-Host "[✓] Virtualización habilitada" -ForegroundColor Green
    } else {
        Write-Host "[✗] Virtualización NO habilitada - Requiere activación en BIOS" -ForegroundColor Red
    }
    
    if ($status.WSL) {
        Write-Host "[✓] WSL instalado" -ForegroundColor Green
    } else {
        Write-Host "[✗] WSL no instalado" -ForegroundColor Red
    }
    
    if ($status.WSL2) {
        Write-Host "[✓] WSL2 configurado como predeterminado" -ForegroundColor Green
    } else {
        Write-Host "[✗] WSL2 no configurado" -ForegroundColor Yellow
    }
    
    if ($status.Kali) {
        Write-Host "[✓] Kali Linux instalado" -ForegroundColor Green
    } else {
        Write-Host "[✗] Kali Linux no instalado" -ForegroundColor Red
    }
    
    if ($status.Project) {
        Write-Host "[✓] Proyecto Obvivlorum configurado" -ForegroundColor Green
    } else {
        Write-Host "[✗] Proyecto no configurado" -ForegroundColor Yellow
    }
    
    if ($status.Module) {
        Write-Host "[✓] Módulo HoloMem cargado" -ForegroundColor Green
    } else {
        Write-Host "[✗] Módulo HoloMem no cargado" -ForegroundColor Yellow
    }
    
    Write-Host ""
    return $status
}

function Install-WSL2 {
    Write-Host "Instalando WSL2..." -ForegroundColor Cyan
    
    # Habilitar características
    Write-Host "Habilitando características de Windows..." -ForegroundColor Yellow
    dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
    dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
    
    # Descargar actualizacion del kernel
    $kernelUrl = "https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi"
    $kernelPath = "$env:TEMP\wsl_update_x64.msi"
    
    Write-Host "Descargando actualización del kernel..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri $kernelUrl -OutFile $kernelPath -UseBasicParsing
    
    Write-Host "Instalando actualización del kernel..." -ForegroundColor Yellow
    Start-Process msiexec.exe -Wait -ArgumentList "/I", $kernelPath, "/quiet"
    
    # Establecer WSL2 como predeterminado
    wsl --set-default-version 2
    
    Write-Host "✓ WSL2 instalado correctamente" -ForegroundColor Green
}

function Install-Kali {
    Write-Host "Instalando Kali Linux..." -ForegroundColor Cyan
    
    # Abrir Microsoft Store
    Write-Host "Abriendo Microsoft Store para instalar Kali Linux..." -ForegroundColor Yellow
    Start-Process "ms-windows-store://pdp/?productid=9PKR34TNCV07"
    
    Write-Host ""
    Write-Host "INSTRUCCIONES:" -ForegroundColor Yellow
    Write-Host "1. Instala Kali Linux desde la Microsoft Store" -ForegroundColor White
    Write-Host "2. Una vez instalado, EJECUTA Kali Linux" -ForegroundColor White
    Write-Host "3. Configura usuario y contraseña" -ForegroundColor White
    Write-Host "4. CIERRA la ventana de Kali" -ForegroundColor White
    Write-Host "5. Vuelve aquí y presiona Enter" -ForegroundColor White
    Write-Host ""
    pause
}

function Setup-Project {
    Write-Host "Configurando proyecto Obvivlorum..." -ForegroundColor Cyan
    
    # Crear estructura de directorios
    wsl -d kali-linux -- bash -c "mkdir -p ~/obvlivorum_simbiosis/{holomem,tinyllama,config}"
    
    # Instalar dependencias
    Write-Host "Instalando dependencias..." -ForegroundColor Yellow
    $deps = wsl -d kali-linux -- bash -c "
        sudo apt update && 
        sudo apt install -y build-essential make gcc linux-headers-\`$(uname -r) git wget python3 python3-pip
    "
    
    if ($?) {
        Write-Host "✓ Proyecto configurado correctamente" -ForegroundColor Green
    } else {
        Write-Host "✗ Error configurando proyecto" -ForegroundColor Red
    }
}

function Create-HolomemModule {
    Write-Host "Creando módulo HoloMem..." -ForegroundColor Cyan
    
    $holomemPath = "~/obvlivorum_simbiosis/holomem"
    
    # Crear archivo holomem.c
    $holomemC = @'
#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/fs.h>
#include <linux/uaccess.h>
#include <linux/slab.h>
#include <linux/mutex.h>

#define DEVICE_NAME "holomem"
#define CLASS_NAME "holo"
#define BUFFER_SIZE 1024*1024  // 1MB

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Obvivlorum System");
MODULE_DESCRIPTION("HoloMem - Shared Memory Module");
MODULE_VERSION("2.0");

static int majorNumber;
static struct class* holomemClass = NULL;
static struct device* holomemDevice = NULL;
static char* sharedBuffer = NULL;
static size_t bufferSize = 0;
static DEFINE_MUTEX(holomem_mutex);

static int dev_open(struct inode*, struct file*);
static int dev_release(struct inode*, struct file*);
static ssize_t dev_read(struct file*, char*, size_t, loff_t*);
static ssize_t dev_write(struct file*, const char*, size_t, loff_t*);

static struct file_operations fops = {
    .open = dev_open,
    .read = dev_read,
    .write = dev_write,
    .release = dev_release,
};

static int __init holomem_init(void) {
    printk(KERN_INFO "HoloMem: Initializing module\n");
    
    majorNumber = register_chrdev(0, DEVICE_NAME, &fops);
    if (majorNumber < 0) {
        printk(KERN_ALERT "HoloMem: Failed to register major number\n");
        return majorNumber;
    }
    
    holomemClass = class_create(THIS_MODULE, CLASS_NAME);
    if (IS_ERR(holomemClass)) {
        unregister_chrdev(majorNumber, DEVICE_NAME);
        printk(KERN_ALERT "HoloMem: Failed to create class\n");
        return PTR_ERR(holomemClass);
    }
    
    holomemDevice = device_create(holomemClass, NULL, MKDEV(majorNumber, 0), NULL, DEVICE_NAME);
    if (IS_ERR(holomemDevice)) {
        class_destroy(holomemClass);
        unregister_chrdev(majorNumber, DEVICE_NAME);
        printk(KERN_ALERT "HoloMem: Failed to create device\n");
        return PTR_ERR(holomemDevice);
    }
    
    sharedBuffer = kmalloc(BUFFER_SIZE, GFP_KERNEL);
    if (!sharedBuffer) {
        device_destroy(holomemClass, MKDEV(majorNumber, 0));
        class_unregister(holomemClass);
        class_destroy(holomemClass);
        unregister_chrdev(majorNumber, DEVICE_NAME);
        printk(KERN_ALERT "HoloMem: Failed to allocate memory\n");
        return -ENOMEM;
    }
    
    memset(sharedBuffer, 0, BUFFER_SIZE);
    printk(KERN_INFO "HoloMem: Module loaded successfully\n");
    return 0;
}

static void __exit holomem_exit(void) {
    kfree(sharedBuffer);
    device_destroy(holomemClass, MKDEV(majorNumber, 0));
    class_unregister(holomemClass);
    class_destroy(holomemClass);
    unregister_chrdev(majorNumber, DEVICE_NAME);
    printk(KERN_INFO "HoloMem: Module unloaded\n");
}

static int dev_open(struct inode *inodep, struct file *filep) {
    printk(KERN_INFO "HoloMem: Device opened\n");
    return 0;
}

static ssize_t dev_read(struct file *filep, char *buffer, size_t len, loff_t *offset) {
    int error_count = 0;
    
    mutex_lock(&holomem_mutex);
    
    if (*offset >= bufferSize) {
        mutex_unlock(&holomem_mutex);
        return 0;
    }
    
    if (*offset + len > bufferSize) {
        len = bufferSize - *offset;
    }
    
    error_count = copy_to_user(buffer, sharedBuffer + *offset, len);
    
    if (error_count == 0) {
        *offset += len;
        mutex_unlock(&holomem_mutex);
        return len;
    } else {
        mutex_unlock(&holomem_mutex);
        return -EFAULT;
    }
}

static ssize_t dev_write(struct file *filep, const char *buffer, size_t len, loff_t *offset) {
    mutex_lock(&holomem_mutex);
    
    if (len > BUFFER_SIZE) {
        len = BUFFER_SIZE;
    }
    
    if (copy_from_user(sharedBuffer, buffer, len)) {
        mutex_unlock(&holomem_mutex);
        return -EFAULT;
    }
    
    bufferSize = len;
    mutex_unlock(&holomem_mutex);
    return len;
}

static int dev_release(struct inode *inodep, struct file *filep) {
    printk(KERN_INFO "HoloMem: Device closed\n");
    return 0;
}

module_init(holomem_init);
module_exit(holomem_exit);
'@

    # Crear Makefile
    $makefile = @'
obj-m += holomem.o

all:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
'@

    # Escribir archivos
    Write-Host "Escribiendo archivos del módulo..." -ForegroundColor Yellow
    
    $holomemC | wsl -d kali-linux -- bash -c "cat > $holomemPath/holomem.c"
    $makefile | wsl -d kali-linux -- bash -c "cat > $holomemPath/Makefile"
    
    # Compilar
    Write-Host "Compilando módulo..." -ForegroundColor Yellow
    $compile = wsl -d kali-linux -- bash -c "cd $holomemPath && make"
    
    if ($?) {
        Write-Host "✓ Módulo compilado correctamente" -ForegroundColor Green
        
        # Cargar módulo
        Write-Host "Cargando módulo..." -ForegroundColor Yellow
        wsl -d kali-linux -- bash -c "cd $holomemPath && sudo insmod holomem.ko"
        
        if ($?) {
            Write-Host "✓ Módulo cargado correctamente" -ForegroundColor Green
        }
    } else {
        Write-Host "✗ Error compilando módulo" -ForegroundColor Red
    }
}

# Menú principal
function Show-Menu {
    Write-Host ""
    Write-Host "MENÚ PRINCIPAL" -ForegroundColor Cyan
    Write-Host "═════════════" -ForegroundColor Gray
    Write-Host "1. Instalación completa automática" -ForegroundColor White
    Write-Host "2. Verificar estado del sistema" -ForegroundColor White
    Write-Host "3. Instalar componente específico" -ForegroundColor White
    Write-Host "4. Reparar instalación" -ForegroundColor White
    Write-Host "5. Ejecutar pruebas" -ForegroundColor White
    Write-Host "6. Salir" -ForegroundColor White
    Write-Host ""
}

# Bucle principal
do {
    Show-Header
    $status = Test-SystemStatus
    Show-Menu
    
    $choice = Read-Host "Selecciona una opción (1-6)"
    
    switch ($choice) {
        1 {
            Write-Host ""
            Write-Host "Iniciando instalación completa..." -ForegroundColor Cyan
            
            if (-not $status.Virtualization) {
                Write-Host "✗ La virtualización debe estar habilitada en BIOS primero" -ForegroundColor Red
                pause
                continue
            }
            
            if (-not $status.WSL) {
                Install-WSL2
                Write-Host "⚠ Reinicia el sistema y vuelve a ejecutar este script" -ForegroundColor Yellow
                pause
                exit
            }
            
            if (-not $status.Kali) {
                Install-Kali
                # Recheck status
                $status = Test-SystemStatus
            }
            
            if ($status.Kali -and -not $status.Project) {
                Setup-Project
            }
            
            if ($status.Project -and -not $status.Module) {
                Create-HolomemModule
            }
            
            Write-Host ""
            Write-Host "✓ Instalación completada" -ForegroundColor Green
            pause
        }
        
        2 {
            # Ya se muestra el estado
            pause
        }
        
        3 {
            Write-Host ""
            Write-Host "Componentes disponibles:" -ForegroundColor Cyan
            Write-Host "1. WSL2"
            Write-Host "2. Kali Linux"
            Write-Host "3. Proyecto base"
            Write-Host "4. Módulo HoloMem"
            Write-Host ""
            $comp = Read-Host "Selecciona componente (1-4)"
            
            switch ($comp) {
                1 { Install-WSL2 }
                2 { Install-Kali }
                3 { Setup-Project }
                4 { Create-HolomemModule }
            }
            pause
        }
        
        4 {
            Write-Host ""
            Write-Host "Intentando reparar instalación..." -ForegroundColor Cyan
            
            # Verificar y reparar cada componente
            if ($status.WSL -and $status.Kali) {
                # Reinstalar dependencias
                Setup-Project
                
                # Recargar módulo si es necesario
                if ($status.Project) {
                    wsl -d kali-linux -- bash -c "sudo rmmod holomem 2>/dev/null"
                    Create-HolomemModule
                }
            }
            
            Write-Host "✓ Reparación completada" -ForegroundColor Green
            pause
        }
        
        5 {
            Write-Host ""
            Write-Host "Ejecutando pruebas..." -ForegroundColor Cyan
            
            if ($status.Module) {
                Write-Host "Probando módulo HoloMem..." -ForegroundColor Yellow
                $test = wsl -d kali-linux -- bash -c "echo 'Test' | sudo tee /dev/holomem && sudo cat /dev/holomem"
                if ($test -eq "Test") {
                    Write-Host "✓ Módulo funcionando correctamente" -ForegroundColor Green
                } else {
                    Write-Host "✗ Error en el módulo" -ForegroundColor Red
                }
            }
            
            pause
        }
        
        6 {
            Write-Host ""
            Write-Host "¡Hasta luego!" -ForegroundColor Cyan
            break
        }
        
        default {
            Write-Host "Opción inválida" -ForegroundColor Red
            pause
        }
    }
} while ($choice -ne 6)
