
# --- Start of 1.PS1 ---
# 01_instalar_WSL2_y_Kali.ps1
# Ejecuta con botn derecho > Ejecutar con PowerShell como administrador

# Verificar si se est ejecutando como administrador
if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host " Este script debe ejecutarse como administrador." -ForegroundColor Red
    Write-Host "   Cierra esta ventana y ejecuta PowerShell como administrador." -ForegroundColor Yellow
    pause
    exit 1
}

Write-Host "=== Instalacin Simbitica: Paso 1 ===" -ForegroundColor Cyan
Write-Host "Verificando WSL2, Virtualizacin y entorno base..."

# 1. Verificar si la virtualizacin est activada
$cpuInfo = Get-CimInstance Win32_Processor
if (-not $cpuInfo.VirtualizationFirmwareEnabled) {
    Write-Host " Virtualizacin NO est activada en BIOS. Actvala y vuelve a intentar." -ForegroundColor Red
    Write-Host "   Generalmente se encuentra en la seccin 'Advanced Features' o 'CPU Configuration' del BIOS/UEFI." -ForegroundColor Yellow
    pause
    exit 1
} else {
    Write-Host " Virtualizacin detectada." -ForegroundColor Green
}

# 2. Verificar si WSL ya est instalado
$wslFeature = Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux
if ($wslFeature.State -ne "Enabled") {
    Write-Host " Activando WSL..." -ForegroundColor Yellow
    dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
    $needRestart = $true
} else {
    Write-Host " WSL ya est activado." -ForegroundColor Green
}

# 3. Verificar Plataforma de Mquina Virtual
$vmPlatform = Get-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform
if ($vmPlatform.State -ne "Enabled") {
    Write-Host " Activando plataforma de virtualizacin..." -ForegroundColor Yellow
    dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
    $needRestart = $true
} else {
    Write-Host " Plataforma de virtualizacin ya est activada." -ForegroundColor Green
}

# 4. Descargar e instalar el paquete de actualizacin del kernel de Linux para WSL2
$kernelUpdateMsi = "$env:TEMP\wsl_update_x64.msi"
if (-not (Test-Path $kernelUpdateMsi)) {
    Write-Host " Descargando la actualizacin del kernel de Linux para WSL2..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri "https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi" -OutFile $kernelUpdateMsi -UseBasicParsing
    Write-Host " Instalando la actualizacin del kernel de Linux para WSL2..." -ForegroundColor Yellow
    Start-Process msiexec.exe -Wait -ArgumentList "/I $kernelUpdateMsi /quiet"
    Write-Host " Actualizacin del kernel de Linux para WSL2 instalada." -ForegroundColor Green
} else {
    Write-Host " La actualizacin del kernel de Linux para WSL2 ya est disponible localmente." -ForegroundColor Green
}

# 5. Establecer WSL2 como predeterminado
try {
    $wslVersion = (wsl --status | Select-String "Default Version").ToString().Split(":")[1].Trim()
    if ($wslVersion -ne "2") {
        Write-Host " Estableciendo WSL2 como versin predeterminada..." -ForegroundColor Yellow
        wsl --set-default-version 2
        Write-Host " WSL2 establecido como predeterminado." -ForegroundColor Green
    } else {
        Write-Host " WSL2 ya est establecido como predeterminado." -ForegroundColor Green
    }
} catch {
    Write-Host " Estableciendo WSL2 como versin predeterminada..." -ForegroundColor Yellow
    wsl --set-default-version 2
    Write-Host " WSL2 establecido como predeterminado." -ForegroundColor Green
}

# 6. Verificar si Kali ya est instalado
$kaliInstalled = wsl -l -v | Select-String "kali-linux"
if (-not $kaliInstalled) {
    Write-Host " Instalando Kali Linux desde la Microsoft Store..." -ForegroundColor Yellow
    Start-Process "ms-windows-store://pdp/?productid=9PKR34TNCV07"
    Write-Host ""
    Write-Host " Espera a que la instalacin termine en la Microsoft Store." -ForegroundColor Yellow
    Write-Host "   IMPORTANTE: Despus de instalar desde la tienda, EJECUTA Kali Linux al menos una vez" -ForegroundColor Yellow
    Write-Host "   para completar la configuracin inicial." -ForegroundColor Yellow
    Write-Host "   Luego CIERRA ESA VENTANA y vuelve a ejecutar el Script 2." -ForegroundColor Yellow
    pause
    exit 0
} else {
    Write-Host " Kali ya est instalado. Puedes continuar con el Script 2." -ForegroundColor Green
}

# Verificar si se necesita reiniciar
if ($needRestart) {
    Write-Host " Es necesario reiniciar el sistema para completar la instalacin." -ForegroundColor Yellow
    $restart = Read-Host "Deseas reiniciar ahora? (S/N)"
    if ($restart -eq "S" -or $restart -eq "s") {
        Restart-Computer
    } else {
        Write-Host " Recuerda reiniciar tu sistema antes de continuar con el Script 2." -ForegroundColor Red
    }
}
# --- End of 1.PS1 ---

# --- Start of 2.PS1 ---
# 02_instalar_holomem.ps1
# Ejecutar con PowerShell como administrador

# Verificar si se est ejecutando como administrador
if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host " Este script debe ejecutarse como administrador." -ForegroundColor Red
    Write-Host "   Cierra esta ventana y ejecuta PowerShell como administrador." -ForegroundColor Yellow
    pause
    exit 1
}

Write-Host "=== Instalacin Simbitica: Paso 2 ===" -ForegroundColor Cyan

# 0. Verificar que Kali est instalado y funcionando
$kaliInstalled = wsl -l -v | Select-String "kali-linux"
if (-not $kaliInstalled) {
    Write-Host " Kali Linux no est instalado o no se encuentra." -ForegroundColor Red
    Write-Host "   Ejecuta primero el Script 1 y asegrate de completar la instalacin de Kali." -ForegroundColor Yellow
    pause
    exit 1
}

# Verificar que Kali sea accesible
try {
    $kaliTest = wsl -d kali-linux -- echo "OK"
    if ($kaliTest -ne "OK") {
        throw "Error de comunicacin con Kali Linux"
    }
} catch {
    Write-Host " No se puede acceder a Kali Linux. Verifica la instalacin." -ForegroundColor Red
    Write-Host "   Asegrate de que hayas ejecutado Kali al menos una vez despus de instalarlo." -ForegroundColor Yellow
    pause
    exit 1
}

# 1. Crear proyecto dentro de Kali WSL
Write-Host " Preparando entorno dentro de Kali..." -ForegroundColor Yellow
wsl -d kali-linux -- bash -c "mkdir -p ~/obvlivorum_simbiosis/holomem"

# 2. Instalar herramientas necesarias
Write-Host " Actualizando repositorios e instalando dependencias..." -ForegroundColor Yellow
$dependenciasExito = wsl -d kali-linux -- bash -c "
    sudo apt update && 
    sudo apt install -y build-essential make gcc linux-headers-\$(uname -r) git wget
"

if (-not $?) {
    Write-Host " Error al escribir los archivos en WSL." -ForegroundColor Red
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
    Write-Host " No se pudieron verificar todos los archivos fuente." -ForegroundColor Red
    pause
    exit 1
}

Write-Host " Archivos fuente generados correctamente." -ForegroundColor Green

# Compilar el mdulo
Write-Host " Compilando el mdulo holomem..." -ForegroundColor Yellow
$compileResult = wsl -d kali-linux -- bash -c "cd $basePath && make 2>&1"
if (-not $?) {
    Write-Host " Error al compilar el mdulo:" -ForegroundColor Red
    Write-Host $compileResult -ForegroundColor Red
    pause
    exit 1
}

# Compilar utilidad de espacio de usuario
Write-Host " Compilando utilidad holomem-util..." -ForegroundColor Yellow
$compileUtilResult = wsl -d kali-linux -- bash -c "cd $basePath && gcc -Wall -O2 -o holomem-util holomem-util.c 2>&1"
if (-not $?) {
    Write-Host " Error al compilar la utilidad:" -ForegroundColor Red
    Write-Host $compileUtilResult -ForegroundColor Red
    pause
    exit 1
}

# Cargar el mdulo
Write-Host " Cargando mdulo en kernel..." -ForegroundColor Yellow
$insertResult = wsl -d kali-linux -- bash -c "cd $basePath && sudo insmod holomem.ko 2>&1"
if (-not $?) {
    Write-Host " Error al cargar el mdulo:" -ForegroundColor Red
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

# Verificar que el mdulo est cargado
$moduleCheck = wsl -d kali-linux -- bash -c "lsmod | grep holomem && echo 'OK'"
if ($moduleCheck -ne "OK") {
    Write-Host " No se pudo verificar que el mdulo est cargado." -ForegroundColor Red
    pause
    exit 1
}

Write-Host " Mdulo holomem compilado, cargado y verificado correctamente." -ForegroundColor Green
Write-Host " Paso 3 completado. Contina con el Script 4 para instalar TinyLLaMA." -ForegroundColor Green
Write-Host " Error instalando dependencias. Verifica la conexin a Internet y los repositorios." -ForegroundColor Red
    pause
    exit 1
}

Write-Host " Dependencias instaladas correctamente." -ForegroundColor Green

# 3. Verificar que exista el directorio para archivos fuente (se crearn en Script 3)
Write-Host " Verificando estructura de directorios..." -ForegroundColor Yellow
wsl -d kali-linux -- bash -c "
    if [ ! -d ~/obvlivorum_simbiosis/holomem ]; then
        mkdir -p ~/obvlivorum_simbiosis/holomem
    fi
"

Write-Host " Estructura de directorios preparada. Contina con el Script 3 para generar los archivos fuente." -ForegroundColor Green
Write-Host " IMPORTANTE: El mdulo se compilar despus de generar los archivos fuente en el Script 3." -ForegroundColor Yellow

# 4. Configurar permisos de sudo para evitar contraseas repetidas
Write-Host " Configurando permisos para una mejor experiencia de usuario..." -ForegroundColor Yellow
$username = wsl -d kali-linux -- whoami
wsl -d kali-linux -- bash -c "
    if ! sudo grep -q '$username ALL=(ALL) NOPASSWD: ALL' /etc/sudoers; then
        echo '$username ALL=(ALL) NOPASSWD: ALL' | sudo tee -a /etc/sudoers > /dev/null
    fi
"

Write-Host " Paso 2 completado. Ejecuta el Script 3 para generar los archivos fuente del mdulo." -ForegroundColor Green
# --- End of 2.PS1 ---

# --- Start of 3.PS1 ---
# 03_generar_archivos_fuente.ps1
# Ejecutar con PowerShell como administrador

# Verificar si se est ejecutando como administrador
if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host " Este script debe ejecutarse como administrador." -ForegroundColor Red
    Write-Host "   Cierra esta ventana y ejecuta PowerShell como administrador." -ForegroundColor Yellow
    pause
    exit 1
}

Write-Host "=== Simbiosis: Paso 3 - Generando archivos fuente ===" -ForegroundColor Cyan

# Obtener el nombre de usuario de Kali
$username = wsl -d kali-linux -- whoami
$basePath = "/home/$username/obvlivorum_simbiosis/holomem"

# Crear holomem.c
Write-Host " Generando archivo holomem.c..." -ForegroundColor Yellow
$holomemC = @'
#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/fs.h>
#include <linux/device.h>
#include <linux/uaccess.h>
#include <linux/slab.h>
#include <linux/mutex.h>

#define DEVICE_NAME "holomem"
#define CLASS_NAME "holomem_class"
#define MAX_PATTERNS 256
#define MAX_PATTERN_SIZE 4096

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Obvlivorum Simbiosis");
MODULE_DESCRIPTION("Mdulo para almacenamiento y reconocimiento de patrones en memoria");
MODULE_VERSION("0.1");

static int majorNumber;
static struct class* holomemClass = NULL;
static struct device* holomemDevice = NULL;
static DEFINE_MUTEX(holomem_mutex);

// Estructura para almacenar patrones
typedef struct {
    char* data;
    size_t size;
    char name[64];
    int id;
    bool used;
} pattern_t;

static pattern_t patterns[MAX_PATTERNS];

// Funciones de dispositivo
static int holomem_open(struct inode*, struct file*);
static int holomem_release(struct inode*, struct file*);
static ssize_t holomem_read(struct file*, char*, size_t, loff_t*);
static ssize_t holomem_write(struct file*, const char*, size_t, loff_t*);
static long holomem_ioctl(struct file*, unsigned int, unsigned long);

static struct file_operations fops = {
    .open = holomem_open,
    .read = holomem_read,
    .write = holomem_write,
    .release = holomem_release,
    .unlocked_ioctl = holomem_ioctl,
};

// Inicializacin del mdulo
static int __init holomem_init(void) {
    int i;
    printk(KERN_INFO "HoloMem: Inicializando mdulo\n");
    
    // Inicializar arreglo de patrones
    for (i = 0; i < MAX_PATTERNS; i++) {
        patterns[i].data = NULL;
        patterns[i].size = 0;
        patterns[i].used = false;
        patterns[i].id = i;
        strcpy(patterns[i].name, "empty");
    }
    
    // Registrar nmero de dispositivo
    majorNumber = register_chrdev(0, DEVICE_NAME, &fops);
    if (majorNumber < 0) {
        printk(KERN_ALERT "HoloMem: Error al registrar nmero de dispositivo\n");
        return majorNumber;
    }
    
    // Registrar clase de dispositivo
    holomemClass = class_create(THIS_MODULE, CLASS_NAME);
    if (IS_ERR(holomemClass)) {
        unregister_chrdev(majorNumber, DEVICE_NAME);
        printk(KERN_ALERT "HoloMem: Error al crear clase de dispositivo\n");
        return PTR_ERR(holomemClass);
    }
    
    // Crear dispositivo
    holomemDevice = device_create(holomemClass, NULL, MKDEV(majorNumber, 0), NULL, DEVICE_NAME);
    if (IS_ERR(holomemDevice)) {
        class_destroy(holomemClass);
        unregister_chrdev(majorNumber, DEVICE_NAME);
        printk(KERN_ALERT "HoloMem: Error al crear dispositivo\n");
        return PTR_ERR(holomemDevice);
    }
    
    printk(KERN_INFO "HoloMem: Mdulo cargado correctamente (%d)\n", majorNumber);
    return 0;
}

// Cierre del mdulo
static void __exit holomem_exit(void) {
    int i;
    device_destroy(holomemClass, MKDEV(majorNumber, 0));
    class_unregister(holomemClass);
    class_destroy(holomemClass);
    unregister_chrdev(majorNumber, DEVICE_NAME);
    
    // Liberar memoria de patrones
    for (i = 0; i < MAX_PATTERNS; i++) {
        if (patterns[i].data) {
            kfree(patterns[i].data);
            patterns[i].data = NULL;
        }
    }
    
    printk(KERN_INFO "HoloMem: Mdulo descargado\n");
}

// Abrir dispositivo
static int holomem_open(struct inode* inodep, struct file* filep) {
    if (!mutex_trylock(&holomem_mutex)) {
        printk(KERN_ALERT "HoloMem: Dispositivo ocupado\n");
        return -EBUSY;
    }
    return 0;
}

// Cerrar dispositivo
static int holomem_release(struct inode* inodep, struct file* filep) {
    mutex_unlock(&holomem_mutex);
    return 0;
}

// Leer de dispositivo (devuelve patrones almacenados)
static ssize_t holomem_read(struct file* filep, char* buffer, size_t len, loff_t* offset) {
    int error_count = 0;
    char* message;
    size_t message_size = 0;
    int i;
    
    // Construir mensaje con informacin de patrones
    message = kmalloc(4096, GFP_KERNEL);
    if (!message) {
        return -ENOMEM;
    }
    
    message_size += sprintf(message, "HoloMem: Patrones almacenados (%d slots)\n", MAX_PATTERNS);
    for (i = 0; i < MAX_PATTERNS; i++) {
        if (patterns[i].used) {
            message_size += sprintf(message + message_size, 
                                   "ID: %d, Nombre: %s, Tamao: %zu bytes\n", 
                                   i, patterns[i].name, patterns[i].size);
        }
    }
    
    // Enviar datos al espacio de usuario
    error_count = copy_to_user(buffer, message, message_size);
    if (error_count) {
        kfree(message);
        return -EFAULT;
    }
    
    kfree(message);
    return message_size;
}

// Escribir en dispositivo (almacenar nuevo patrn)
static ssize_t holomem_write(struct file* filep, const char* buffer, size_t len, loff_t* offset) {
    int slot = 0;  // Por defecto, usar el primer slot
    char* kbuffer;
    
    if (len > MAX_PATTERN_SIZE) {
        printk(KERN_ALERT "HoloMem: Patrn demasiado grande (mximo %d bytes)\n", MAX_PATTERN_SIZE);
        return -EINVAL;
    }
    
    // Buscar slot libre
    while (slot < MAX_PATTERNS && patterns[slot].used) {
        slot++;
    }
    
    if (slot >= MAX_PATTERNS) {
        printk(KERN_ALERT "HoloMem: No hay slots libres\n");
        return -ENOSPC;
    }
    
    // Liberar memoria si el slot ya tena datos
    if (patterns[slot].data) {
        kfree(patterns[slot].data);
    }
    
    // Asignar memoria para nuevo patrn
    kbuffer = kmalloc(len, GFP_KERNEL);
    if (!kbuffer) {
        return -ENOMEM;
    }
    
    // Copiar datos desde espacio de usuario
    if (copy_from_user(kbuffer, buffer, len)) {
        kfree(kbuffer);
        return -EFAULT;
    }
    
    // Almacenar patrn
    patterns[slot].data = kbuffer;
    patterns[slot].size = len;
    patterns[slot].used = true;
    sprintf(patterns[slot].name, "pattern_%d", slot);
    
    printk(KERN_INFO "HoloMem: Patrn almacenado en slot %d (%zu bytes)\n", slot, len);
    return len;
}

// IOCTL para operaciones avanzadas
#define HOLOMEM_SET_NAME _IOW('h', 1, char[64])
#define HOLOMEM_DELETE_PATTERN _IOW('h', 2, int)
#define HOLOMEM_COMPARE_PATTERN _IOWR('h', 3, int)

static long holomem_ioctl(struct file* filep, unsigned int cmd, unsigned long arg) {
    int err = 0;
    
    switch (cmd) {
        case HOLOMEM_SET_NAME:
            // Implementar cdigo para asignar nombre a un patrn
            break;
            
        case HOLOMEM_DELETE_PATTERN:
            // Implementar cdigo para eliminar un patrn
            break;
            
        case HOLOMEM_COMPARE_PATTERN:
            // Implementar cdigo para comparar patrones
            break;
            
        default:
            err = -ENOTTY;
    }
    
    return err;
}

module_init(holomem_init);
module_exit(holomem_exit);
'@

# Crear holomem-util.c
Write-Host " Generando archivo holomem-util.c..." -ForegroundColor Yellow
$holomemUtilC = @'
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>

#define DEVICE_FILE "/dev/holomem"
#define MAX_BUFFER_SIZE 4096

#define HOLOMEM_SET_NAME _IOW('h', 1, char[64])
#define HOLOMEM_DELETE_PATTERN _IOW('h', 2, int)
#define HOLOMEM_COMPARE_PATTERN _IOWR('h', 3, int)

void usage(void) {
    printf("Uso: holomem-util [COMANDO] [ARGUMENTOS]\n");
    printf("Comandos disponibles:\n");
    printf("  list                            - Listar patrones almacenados\n");
    printf("  store <archivo> <slot> <nombre> - Almacenar archivo en slot especfico\n");
    printf("  delete <slot>                   - Eliminar patrn de un slot\n");
    printf("  retrieve <slot> <archivo>       - Recuperar patrn a un archivo\n");
    exit(1);
}

// Almacenar contenido de archivo como patrn
int store_pattern(const char* filename, int slot, const char* name) {
    FILE* file;
    char* buffer;
    long file_size;
    int fd, ret;
    
    // Abrir archivo
    file = fopen(filename, "rb");
    if (!file) {
        perror("Error al abrir archivo");
        return -1;
    }
    
    // Obtener tamao del archivo
    fseek(file, 0, SEEK_END);
    file_size = ftell(file);
    rewind(file);
    
    if (file_size > MAX_BUFFER_SIZE) {
        printf("Error: Archivo demasiado grande (mximo %d bytes)\n", MAX_BUFFER_SIZE);
        fclose(file);
        return -1;
    }
    
    // Asignar buffer y leer archivo
    buffer = malloc(file_size);
    if (!buffer) {
        perror("Error al asignar memoria");
        fclose(file);
        return -1;
    }
    
    if (fread(buffer, 1, file_size, file) != file_size) {
        perror("Error al leer archivo");
        free(buffer);
        fclose(file);
        return -1;
    }
    
    fclose(file);
    
    // Abrir dispositivo
    fd = open(DEVICE_FILE, O_WRONLY);
    if (fd < 0) {
        perror("Error al abrir dispositivo");
        free(buffer);
        return -1;
    }
    
    // Escribir datos al dispositivo
    ret = write(fd, buffer, file_size);
    if (ret < 0) {
        perror("Error al escribir al dispositivo");
        free(buffer);
        close(fd);
        return -1;
    }
    
    // Establecer nombre si se proporcion
    if (name) {
        char name_buffer[64];
        strncpy(name_buffer, name, 63);
        name_buffer[63] = '\0';
        
        if (ioctl(fd, HOLOMEM_SET_NAME, name_buffer) < 0) {
            perror("Error al establecer nombre");
        }
    }
    
    printf("Patrn almacenado correctamente (%ld bytes)\n", file_size);
    
    free(buffer);
    close(fd);
    return 0;
}

// Listar patrones almacenados
int list_patterns(void) {
    int fd;
    char buffer[MAX_BUFFER_SIZE];
    int ret;
    
    fd = open(DEVICE_FILE, O_RDONLY);
    if (fd < 0) {
        perror("Error al abrir dispositivo");
        return -1;
    }
    
    ret = read(fd, buffer, MAX_BUFFER_SIZE);
    if (ret < 0) {
        perror("Error al leer del dispositivo");
        close(fd);
        return -1;
    }
    
    buffer[ret] = '\0';
    printf("%s", buffer);
    
    close(fd);
    return 0;
}

// Programa principal
int main(int argc, char** argv) {
    if (argc < 2) {
        usage();
    }
    
    // Procesar comandos
    if (strcmp(argv[1], "list") == 0) {
        return list_patterns();
    } 
    else if (strcmp(argv[1], "store") == 0) {
        if (argc < 4) {
            printf("Error: Argumentos insuficientes para 'store'\n");
            usage();
        }
        
        int slot = atoi(argv[3]);
        const char* name = (argc > 4) ? argv[4] : NULL;
        
        return store_pattern(argv[2], slot, name);
    }
    else if (strcmp(argv[1], "delete") == 0) {
        if (argc < 3) {
            printf("Error: Argumentos insuficientes para 'delete'\n");
            usage();
        }
        
        int slot = atoi(argv[2]);
        int fd = open(DEVICE_FILE, O_WRONLY);
        
        if (fd < 0) {
            perror("Error al abrir dispositivo");
            return -1;
        }
        
        if (ioctl(fd, HOLOMEM_DELETE_PATTERN, &slot) < 0) {
            perror("Error al eliminar patrn");
            close(fd);
            return -1;
        }
        
        printf("Patrn eliminado correctamente\n");
        close(fd);
        return 0;
    }
    else {
        printf("Comando desconocido: %s\n", argv[1]);
        usage();
    }
    
    return 0;
}
'@

# Crear Makefile
Write-Host " Generando archivo Makefile..." -ForegroundColor Yellow
$makefile = @'
obj-m := holomem.o
KDIR := /lib/modules/$(shell uname -r)/build
PWD := $(shell pwd)
ccflags-y := -Wall -Werror -O2

all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean
'@

# Escribir archivos a travs de WSL
Write-Host " Escribiendo archivos en WSL..." -ForegroundColor Yellow
wsl -d kali-linux -- bash -c "
    cat > $basePath/holomem.c << 'EOF'
$holomemC
EOF

    cat > $basePath/holomem-util.c << 'EOF'
$holomemUtilC
EOF

    cat > $basePath/Makefile << 'EOF'
$makefile
EOF
"

if (-not $?) {
# --- End of 3.PS1 ---

# --- Start of 4.PS1 ---
# 04_instalar_tinyllama.ps1
# Ejecutar con PowerShell como administrador

# Verificar si se est ejecutando como administrador
if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host " Este script debe ejecutarse como administrador." -ForegroundColor Red
    Write-Host "   Cierra esta ventana y ejecuta PowerShell como administrador." -ForegroundColor Yellow
    pause
    exit 1
}

Write-Host "=== Simbiosis: Paso 4 - TinyLLaMA ===" -ForegroundColor Cyan

# Verificar espacio disponible
$spaceCheck = wsl -d kali-linux -- bash -c "
    SPACE=\$(df -h /home | awk 'NR==2 {print \$4}' | tr -d 'G')
    if (( \$(echo \"\$SPACE < 2\" | bc -l) )); then
        echo 'ESPACIO_INSUFICIENTE'
    else
        echo 'ESPACIO_OK'
    fi
"

if ($spaceCheck -eq "ESPACIO_INSUFICIENTE") {
    Write-Host " No hay suficiente espacio disponible. Se requieren al menos 2GB libres." -ForegroundColor Red
    pause
    exit 1
}

# Verificar si llama.cpp ya existe
$llamaInstalled = wsl -d kali-linux -- bash -c "
    if [ -d ~/obvlivorum_simbiosis/llama.cpp ]; then
        echo 'EXISTE'
    else
        echo 'NO_EXISTE'
    fi
"

if ($llamaInstalled -eq "EXISTE") {
    Write-Host " llama.cpp ya est instalado. Verificando..." -ForegroundColor Yellow
    
    # Verificar si el ejecutable existe
    $execExists = wsl -d kali-linux -- bash -c "
        if [ -f ~/obvlivorum_simbiosis/llama.cpp/main ]; then
            echo 'EXISTE'
        else
            echo 'NO_EXISTE'
        fi
    "
    
    if ($execExists -eq "NO_EXISTE") {
        Write-Host " El ejecutable no existe. Recompilando..." -ForegroundColor Yellow
        wsl -d kali-linux -- bash -c "
            cd ~/obvlivorum_simbiosis/llama.cpp &&
            make clean &&
            make -j\$(nproc)
        "
        
        if (-not $?) {
            Write-Host " Error al compilar llama.cpp." -ForegroundColor Red
            pause
            exit 1
        }
    } else {
        Write-Host " llama.cpp ya est compilado." -ForegroundColor Green
    }
} else {
    # Instalar dependencias necesarias
    Write-Host " Instalando dependencias para llama.cpp..." -ForegroundColor Yellow
    wsl -d kali-linux -- bash -c "
        sudo apt update &&
        sudo apt install -y build-essential cmake python3-dev libopenblas-dev
    "
    
    if (-not $?) {
        Write-Host " Error al instalar dependencias." -ForegroundColor Red
        pause
        exit 1
    }
    
    # Clonar y compilar llama.cpp
    Write-Host " Clonando repositorio llama.cpp..." -ForegroundColor Yellow
    wsl -d kali-linux -- bash -c "
        cd ~/obvlivorum_simbiosis &&
        git clone https://github.com/ggerganov/llama.cpp &&
        cd llama.cpp &&
        make -j\$(nproc)
    "
    
    if (-not $?) {
        Write-Host " Error al clonar o compilar llama.cpp." -ForegroundColor Red
        pause
        exit 1
    }
    
    Write-Host " llama.cpp clonado y compilado correctamente." -ForegroundColor Green
}

# Verificar si el modelo ya est descargado
$modelExists = wsl -d kali-linux -- bash -c "
    if [ -f ~/obvlivorum_simbiosis/llama.cpp/models/tinyllama-1.1b-chat.Q4_K_M.gguf ]; then
        echo 'EXISTE'
    else
        echo 'NO_EXISTE'
    fi
"

if ($modelExists -eq "NO_EXISTE") {
    # Crear directorio de modelos si no existe
    Write-Host " Descargando modelo TinyLLaMA (esto puede tomar varios minutos)..." -ForegroundColor Yellow
    wsl -d kali-linux -- bash -c "
        mkdir -p ~/obvlivorum_simbiosis/llama.cpp/models &&
        cd ~/obvlivorum_simbiosis/llama.cpp/models &&
        wget --progress=bar:force:noscroll https://huggingface.co/cafune/tinyllama-1.1B-gguf/resolve/main/tinyllama-1.1b-chat.Q4_K_M.gguf
    "
    
    if (-not $?) {
        Write-Host " Error al descargar el modelo. Verifica tu conexin a Internet." -ForegroundColor Red
        pause
        exit 1
    }
    
    # Verificar que el modelo se descarg correctamente
    $modelSizeCheck = wsl -d kali-linux -- bash -c "
        SIZE=\$(du -m ~/obvlivorum_simbiosis/llama.cpp/models/tinyllama-1.1b-chat.Q4_K_M.gguf | cut -f1)
        if (( \$SIZE > 500 )); then
            echo 'TAMAO_OK'
        else
            echo 'TAMAO_INCORRECTO'
        fi
    "
    
    if ($modelSizeCheck -eq "TAMAO_INCORRECTO") {
        Write-Host " El modelo descargado parece incompleto o daado." -ForegroundColor Red
        Write-Host "   Intenta ejecutar nuevamente este script." -ForegroundColor Yellow
        pause
        exit 1
    }
    
    Write-Host " Modelo TinyLLaMA descargado correctamente." -ForegroundColor Green
} else {
    Write-Host " El modelo TinyLLaMA ya est descargado." -ForegroundColor Green
}

# Probar el modelo
Write-Host " Verificando funcionamiento del modelo con una prueba sencilla..." -ForegroundColor Yellow
$testResult = wsl -d kali-linux -- bash -c "
    cd ~/obvlivorum_simbiosis/llama.cpp &&
    echo 'Prueba exitosa' > /tmp/test_result.txt &&
    ./main -m models/tinyllama-1.1b-chat.Q4_K_M.gguf -p 'Hola, responde con 3 palabras:' -n 20 --no-display-prompt -r 'EOM' > /tmp/test_result.txt 2>/dev/null || echo 'ERROR_EJECUCIN'
    cat /tmp/test_result.txt
"

if ($testResult -eq "ERROR_EJECUCIN") {
    Write-Host " No se pudo verificar el funcionamiento del modelo. Esto podra causar problemas ms adelante." -ForegroundColor Yellow
} else {
    Write-Host " Prueba del modelo completada." -ForegroundColor Green
}

Write-Host " TinyLLaMA instalado y listo en ~/obvlivorum_simbiosis/llama.cpp/models" -ForegroundColor Green
Write-Host " Paso 4 completado. Contina con el Script 5 para instalar el middleware e interfaz." -ForegroundColor Green
# --- End of 4.PS1 ---

# --- Start of 5.PS1 ---
# 05_middleware_e_interfaz.ps1
# Ejecutar con PowerShell como administrador

# Verificar si se est ejecutando como administrador
if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host " Este script debe ejecutarse como administrador." -ForegroundColor Red
    Write-Host "   Cierra esta ventana y ejecuta PowerShell como administrador." -ForegroundColor Yellow
    pause
    exit 1
}

Write-Host "=== Simbiosis: Paso 5 - Middleware simbitico + Interfaz GPT ===" -ForegroundColor Cyan

# Verificar instalacin previa de TinyLLaMA
$llamaExists = wsl -d kali-linux -- bash -c "
    if [ -f ~/obvlivorum_simbiosis/llama.cpp/main ] && [ -f ~/obvlivorum_simbiosis/llama.cpp/models/tinyllama-1.1b-chat.Q4_K_M.gguf ]; then
        echo 'OK'
    else
        echo 'FALTA'
    fi
"

if ($llamaExists -ne "OK") {
    Write-Host " No se encontr TinyLLaMA instalado correctamente." -ForegroundColor Red
    Write-Host "   Ejecuta primero el Script 4." -ForegroundColor Yellow
    pause
    exit 1
}

# Verificar mdulo holomem
$holomemExists = wsl -d kali-linux -- bash -c "
    if [ -f ~/obvlivorum_simbiosis/holomem/holomem-util ] && lsmod | grep -q holomem; then
        echo 'OK'
    else
        echo 'FALTA'
    fi
"

if ($holomemExists -ne "OK") {
    Write-Host " El mdulo holomem no est cargado o no se compil correctamente." -ForegroundColor Red
    Write-Host "   Verifica los Scripts 2 y 3." -ForegroundColor Yellow
    pause
    exit 1
}

# Instalar dependencias para el middleware
Write-Host " Instalando dependencias para el middleware..." -ForegroundColor Yellow
$installResult = wsl -d kali-linux -- bash -c "
    sudo apt update &&
    sudo apt install -y python3 python3-pip python3-venv espeak ffmpeg python3-pyaudio portaudio19-dev
"

if (-not $?) {
    Write-Host " Error al instalar dependencias necesarias." -ForegroundColor Red
    Write-Host $installResult -ForegroundColor Red
    pause
    exit 1
}

# Crear entorno virtual Python
Write-Host " Creando entorno virtual Python..." -ForegroundColor Yellow
$venvResult = wsl -d kali-linux -- bash -c "
    cd ~/obvlivorum_simbiosis &&
    python3 -m venv simbiox &&
    source simbiox/bin/activate &&
    pip install -U pip wheel setuptools &&
    pip install pyttsx3 SpeechRecognition sounddevice scipy openai-whisper PyAudio
"

if (-not $?) {
    Write-Host " Error al crear entorno virtual o instalar paquetes Python." -ForegroundColor Red
    Write-Host $venvResult -ForegroundColor Red
    pause
    exit 1
}

# Crear script Python mejorado
Write-Host " Creando script de interfaz simbitica..." -ForegroundColor Yellow

$pythonScript = @'
#!/usr/bin/env python3
import os
import subprocess
import sys
import time
import signal
import threading
import pyttsx3
import traceback

# Rutas absolutas
MODEL_PATH = os.path.expanduser("~/obvlivorum_simbiosis/llama.cpp/models/tinyllama-1.1b-chat.Q4_K_M.gguf")
LLAMA_PATH = os.path.expanduser("~/obvlivorum_simbiosis/llama.cpp/main")
HOLOMEM_UTIL = os.path.expanduser("~/obvlivorum_simbiosis/holomem/holomem-util")

# Configuracin TTS
try:
    engine = pyttsx3.init()
    engine.setProperty("rate", 145)
    engine.setProperty("volume", 1.0)
    TTS_DISPONIBLE = True
except Exception as e:
    print(f"ADVERTENCIA: No se pudo inicializar el motor TTS: {e}")
    TTS_DISPONIBLE = False

# Funcin para verificar componentes
def verificar_componentes():
    componentes_ok = True
    
    # Verificar TinyLLaMA
    if not os.path.isfile(LLAMA_PATH):
        print(f" ERROR: Ejecutable de llama.cpp no encontrado en: {LLAMA_PATH}")
        componentes_ok = False
    
    if not os.path.isfile(MODEL_PATH):
        print(f" ERROR: Modelo TinyLLaMA no encontrado en: {MODEL_PATH}")
        componentes_ok = False
    
    # Verificar holomem
    try:
        result = subprocess.run(["lsmod"], stdout=subprocess.PIPE, text=True)
        if "holomem" not in result.stdout:
            print(" ERROR: Mdulo holomem no cargado en el kernel")
            componentes_ok = False
    except Exception as e:
        print(f" ERROR al verificar mdulo holomem: {e}")
        componentes_ok = False
    
    # Verificar holomem-util
    if not os.path.isfile(HOLOMEM_UTIL):
        print(f" ERROR: Utilidad holomem-util no encontrada en: {HOLOMEM_UTIL}")
        componentes_ok = False
    
    return componentes_ok

# Funcin para manejar interrupcin
def signal_handler(sig, frame):
    print("\nCerrando Simbiosis GPT CLI...")
    sys.exit(0)

# Registrar manejador de seal para Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

# Clase para la interfaz simbitica
class SimbioteInterface:
    def __init__(self):
        self.history = []
        self.context = "Eres un asistente de IA simbitico que responde de manera concisa y til."
    
    def generar_respuesta(self, prompt):
        try:
            # Construir prompt con contexto y historia limitada
            full_prompt = self.context + "\n\n"
            
            # Aadir ltimas 3 interacciones de historia si existen
            for i in range(max(0, len(self.history)-3), len(self.history)):
                full_prompt += f"Usuario: {self.history[i][0]}\nIA: {self.history[i][1]}\n\n"
            
            # Aadir la pregunta actual
            full_prompt += f"Usuario: {prompt}\nIA:"
            
            # Lanzar la interfaz
Write-Host " Lanzando entorno simbitico..." -ForegroundColor Green
Write-Host " Presiona Ctrl+C para salir cuando hayas terminado." -ForegroundColor Green

# Limpiar pantalla para una mejor experiencia
Clear-Host

# Imprimir banner
Write-Host "" -ForegroundColor Cyan
Write-Host "   Simbiosis GPT CLI - Interfaz Neural   " -ForegroundColor Cyan
Write-Host "" -ForegroundColor Cyan
Write-Host ""

# Ejecutar la interfaz
wsl -d kali-linux -- bash -c "
    cd ~/obvlivorum_simbiosis &&
    source simbiox/bin/activate &&
    python3 simbiosis_cli.py
"

# Verificar resultado
if (-not $?) {
    Write-Host " Error al ejecutar la interfaz simbitica." -ForegroundColor Red
    Write-Host "   Revisa los mensajes de error y verifica los componentes." -ForegroundColor Yellow
    pause
    exit 1
}

Write-Host " Sesin simbitica finalizada." -ForegroundColor Greenlamar a TinyLLaMA
            print(" Procesando con TinyLLaMA...")
            cmd = [
                LLAMA_PATH,
                "-m", MODEL_PATH,
                "-p", full_prompt,
                "-n", "256",
                "--temp", "0.7",
                "--no-display-prompt"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f" Error al ejecutar TinyLLaMA: {result.stderr}")
                return "Lo siento, ocurri un error al procesar tu solicitud."
            
            # Limpiar y extraer la respuesta
            response = result.stdout.strip()
            
            # Limitar longitud y eliminar posible texto generado extra
            if len(response) > 500:
                response = response[:500] + "..."
            
            # Almacenar en historial
            self.history.append((prompt, response))
            
            # Guardar en holomem
            self.guardar_en_holomem(prompt, response)
            
            return response
        
        except Exception as e:
            print(f" Error al generar respuesta: {e}")
            traceback.print_exc()
            return "Lo siento, ocurri un error inesperado."
    
    def guardar_en_holomem(self, prompt, response):
        try:
            # Guardar prompt del usuario
            with open("/tmp/user_prompt.txt", "w") as f:
                f.write(prompt)
            
            # Guardar respuesta de la IA
            with open("/tmp/ai_response.txt", "w") as f:
                f.write(response)
            
            # Almacenar en holomem
            subprocess.run([
                HOLOMEM_UTIL, "store", "/tmp/user_prompt.txt", "0", "Usuario-Prompt"
            ], stderr=subprocess.DEVNULL)
            
            subprocess.run([
                HOLOMEM_UTIL, "store", "/tmp/ai_response.txt", "1", "IA-Respuesta"
            ], stderr=subprocess.DEVNULL)
            
        except Exception as e:
            print(f" Error al guardar en holomem: {e}")
    
    def hablar(self, texto):
        if not TTS_DISPONIBLE:
            return
        
        try:
            # Iniciar thread para no bloquear
            def speak_thread():
                engine.say(texto)
                engine.runAndWait()
            
            thread = threading.Thread(target=speak_thread)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            print(f" Error en TTS: {e}")

# Funcin principal
def main():
    # Banner
    print("")
    print("   Simbiosis GPT CLI - Interfaz Neural   ")
    print("")
    
    # Verificar componentes
    if not verificar_componentes():
        print("\n Algunos componentes crticos no estn disponibles.")
        print("   Verifica que hayas ejecutado correctamente los scripts 1-4.")
        return 1
    
    # Inicializar interfaz
    interface = SimbioteInterface()
    
    print("\n Todos los sistemas funcionando. Escribe 'salir' para terminar.")
    print(" Puedes hacer preguntas o pedir tareas al sistema simbitico.\n")
    
    # Bucle principal
    while True:
        try:
            # Obtener entrada de usuario
            prompt = input(" T > ").strip()
            
            # Verificar comando de salida
            if prompt.lower() in ["salir", "exit", "quit", "q"]:
                print(" Cerrando sistema simbitico...")
                break
            
            # Si la entrada est vaca, continuar
            if not prompt:
                continue
            
            # Procesar y generar respuesta
            response = interface.generar_respuesta(prompt)
            
            # Mostrar respuesta
            print("\n IA > " + response + "\n")
            
            # Reproducir respuesta (si TTS est disponible)
            interface.hablar(response)
            
        except KeyboardInterrupt:
            print("\n Cerrando sistema simbitico...")
            break
        except Exception as e:
            print(f"\n Error inesperado: {e}")
            traceback.print_exc()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
'@

# Escribir script Python
wsl -d kali-linux -- bash -c "cat > ~/obvlivorum_simbiosis/simbiosis_cli.py << 'EOF'
$pythonScript
EOF
chmod +x ~/obvlivorum_simbiosis/simbiosis_cli.py
"

if (-not $?) {
    Write-Host " Error al crear el script de interfaz." -ForegroundColor Red
    pause
    exit 1
}

Write-Host " Middleware e interfaz creados correctamente." -ForegroundColor Green
Write-Host " Paso 5 completado. Ejecuta el Script 6 para lanzar el entorno simbitico." -ForegroundColor Green
# --- End of 5.PS1 ---

# --- Start of 6.PS1 ---
# 06_lanzar_simbiosis.ps1
# Ejecutar con PowerShell como administrador

# Verificar si se est ejecutando como administrador
if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host " Este script debe ejecutarse como administrador." -ForegroundColor Red
    Write-Host "   Cierra esta ventana y ejecuta PowerShell como administrador." -ForegroundColor Yellow
    pause
    exit 1
}

Write-Host "=== Lanzando entorno simbitico completo ===" -ForegroundColor Cyan

# Verificar que todos los componentes estn instalados
$componentesOK = wsl -d kali-linux -- bash -c "
    # Verificar TinyLLaMA
    if [ ! -f ~/obvlivorum_simbiosis/llama.cpp/main ] || [ ! -f ~/obvlivorum_simbiosis/llama.cpp/models/tinyllama-1.1b-chat.Q4_K_M.gguf ]; then
        echo 'FALTA_TINYLLAMA'
        exit 1
    fi
    
    # Verificar holomem
    if ! lsmod | grep -q holomem; then
        echo 'FALTA_HOLOMEM'
        exit 1
    fi
    
    # Verificar holomem-util
    if [ ! -f ~/obvlivorum_simbiosis/holomem/holomem-util ]; then
        echo 'FALTA_UTIL'
        exit 1
    fi
    
    # Verificar entorno virtual Python
    if [ ! -d ~/obvlivorum_simbiosis/simbiox ]; then
        echo 'FALTA_VENV'
        exit 1
    fi
    
    # Verificar script de interfaz
    if [ ! -f ~/obvlivorum_simbiosis/simbiosis_cli.py ]; then
        echo 'FALTA_SCRIPT'
        exit 1
    fi
    
    echo 'OK'
"

# Verificar resultado
if ($componentesOK -ne "OK") {
    Write-Host " Faltan componentes necesarios para ejecutar el entorno simbitico:" -ForegroundColor Red
    
    switch ($componentesOK) {
        "FALTA_TINYLLAMA" { 
            Write-Host "   - TinyLLaMA no est instalado correctamente. Ejecuta el Script 4." -ForegroundColor Yellow 
        }
        "FALTA_HOLOMEM" { 
            Write-Host "   - El mdulo holomem no est cargado. Ejecuta los Scripts 2 y 3." -ForegroundColor Yellow 
        }
        "FALTA_UTIL" { 
            Write-Host "   - La utilidad holomem-util no est compilada. Ejecuta el Script 3." -ForegroundColor Yellow 
        }
        "FALTA_VENV" { 
            Write-Host "   - El entorno virtual Python no est creado. Ejecuta el Script 5." -ForegroundColor Yellow 
        }
        "FALTA_SCRIPT" { 
            Write-Host "   - El script de interfaz no existe. Ejecuta el Script 5." -ForegroundColor Yellow 
        }
        default { 
            Write-Host "   - Componentes desconocidos. Verifica todos los scripts anteriores." -ForegroundColor Yellow 
        }
    }
    
    pause
    exit 1
}

# Si el mdulo holomem no est cargado, intentar cargarlo
$holomemLoaded = wsl -d kali-linux -- bash -c "lsmod | grep -q holomem && echo 'OK' || echo 'NO'"
if ($holomemLoaded -eq "NO") {
    Write-Host " El mdulo holomem no est cargado. Intentando cargar..." -ForegroundColor Yellow
    $loadResult = wsl -d kali-linux -- bash -c "cd ~/obvlivorum_simbiosis/holomem && sudo insmod holomem.ko"
    
    if (-not $?) {
        Write-Host " Error al cargar el mdulo holomem. Verifica el Script 3." -ForegroundColor Red
        pause
        exit 1
    }
}

# Verificar si /dev/holomem existe, de lo contrario crearlo
$deviceExists = wsl -d kali-linux -- bash -c "test -c /dev/holomem && echo 'OK' || echo 'NO'"
if ($deviceExists -eq "NO") {
    Write-Host " El dispositivo /dev/holomem no existe. Creando..." -ForegroundColor Yellow
    $deviceResult = wsl -d kali-linux -- bash -c "
        MAJOR=\$(grep holomem /proc/devices | cut -d' ' -f1)
        if [ -z \"\$MAJOR\" ]; then
            echo 'ERROR_NO_MAJOR'
            exit 1
        fi
        sudo mknod /dev/holomem c \$MAJOR 0
        sudo chmod 666 /dev/holomem
        test -c /dev/holomem && echo 'OK' || echo 'ERROR_MKNOD'
    "
    
    if ($deviceResult -ne "OK") {
        Write-Host " Error al crear el dispositivo /dev/holomem." -ForegroundColor Red
        Write-Host "   Verifica que el mdulo holomem est cargado correctamente." -ForegroundColor Yellow
        pause
        exit 1
    }
}

# L
# --- End of 6.PS1 ---
