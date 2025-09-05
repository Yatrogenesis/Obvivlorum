# 03_generar_archivos_fuente.ps1
# Ejecutar con PowerShell como administrador

# Verificar si se está ejecutando como administrador
if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host "❌ Este script debe ejecutarse como administrador." -ForegroundColor Red
    Write-Host "   Cierra esta ventana y ejecuta PowerShell como administrador." -ForegroundColor Yellow
    pause
    exit 1
}

Write-Host "=== Simbiosis: Paso 3 - Generando archivos fuente ===" -ForegroundColor Cyan

# Obtener el nombre de usuario de Kali
$username = wsl -d kali-linux -- whoami
$basePath = "/home/$username/obvlivorum_simbiosis/holomem"

# Crear holomem.c
Write-Host "🔄 Generando archivo holomem.c..." -ForegroundColor Yellow
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
MODULE_DESCRIPTION("Módulo para almacenamiento y reconocimiento de patrones en memoria");
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

// Inicialización del módulo
static int __init holomem_init(void) {
    int i;
    printk(KERN_INFO "HoloMem: Inicializando módulo\n");
    
    // Inicializar arreglo de patrones
    for (i = 0; i < MAX_PATTERNS; i++) {
        patterns[i].data = NULL;
        patterns[i].size = 0;
        patterns[i].used = false;
        patterns[i].id = i;
        strcpy(patterns[i].name, "empty");
    }
    
    // Registrar número de dispositivo
    majorNumber = register_chrdev(0, DEVICE_NAME, &fops);
    if (majorNumber < 0) {
        printk(KERN_ALERT "HoloMem: Error al registrar número de dispositivo\n");
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
    
    printk(KERN_INFO "HoloMem: Módulo cargado correctamente (%d)\n", majorNumber);
    return 0;
}

// Cierre del módulo
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
    
    printk(KERN_INFO "HoloMem: Módulo descargado\n");
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
    
    // Construir mensaje con información de patrones
    message = kmalloc(4096, GFP_KERNEL);
    if (!message) {
        return -ENOMEM;
    }
    
    message_size += sprintf(message, "HoloMem: Patrones almacenados (%d slots)\n", MAX_PATTERNS);
    for (i = 0; i < MAX_PATTERNS; i++) {
        if (patterns[i].used) {
            message_size += sprintf(message + message_size, 
                                   "ID: %d, Nombre: %s, Tamaño: %zu bytes\n", 
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

// Escribir en dispositivo (almacenar nuevo patrón)
static ssize_t holomem_write(struct file* filep, const char* buffer, size_t len, loff_t* offset) {
    int slot = 0;  // Por defecto, usar el primer slot
    char* kbuffer;
    
    if (len > MAX_PATTERN_SIZE) {
        printk(KERN_ALERT "HoloMem: Patrón demasiado grande (máximo %d bytes)\n", MAX_PATTERN_SIZE);
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
    
    // Liberar memoria si el slot ya tenía datos
    if (patterns[slot].data) {
        kfree(patterns[slot].data);
    }
    
    // Asignar memoria para nuevo patrón
    kbuffer = kmalloc(len, GFP_KERNEL);
    if (!kbuffer) {
        return -ENOMEM;
    }
    
    // Copiar datos desde espacio de usuario
    if (copy_from_user(kbuffer, buffer, len)) {
        kfree(kbuffer);
        return -EFAULT;
    }
    
    // Almacenar patrón
    patterns[slot].data = kbuffer;
    patterns[slot].size = len;
    patterns[slot].used = true;
    sprintf(patterns[slot].name, "pattern_%d", slot);
    
    printk(KERN_INFO "HoloMem: Patrón almacenado en slot %d (%zu bytes)\n", slot, len);
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
            // Implementar código para asignar nombre a un patrón
            break;
            
        case HOLOMEM_DELETE_PATTERN:
            // Implementar código para eliminar un patrón
            break;
            
        case HOLOMEM_COMPARE_PATTERN:
            // Implementar código para comparar patrones
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
Write-Host "🔄 Generando archivo holomem-util.c..." -ForegroundColor Yellow
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
    printf("  store <archivo> <slot> <nombre> - Almacenar archivo en slot específico\n");
    printf("  delete <slot>                   - Eliminar patrón de un slot\n");
    printf("  retrieve <slot> <archivo>       - Recuperar patrón a un archivo\n");
    exit(1);
}

// Almacenar contenido de archivo como patrón
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
    
    // Obtener tamaño del archivo
    fseek(file, 0, SEEK_END);
    file_size = ftell(file);
    rewind(file);
    
    if (file_size > MAX_BUFFER_SIZE) {
        printf("Error: Archivo demasiado grande (máximo %d bytes)\n", MAX_BUFFER_SIZE);
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
    
    // Establecer nombre si se proporcionó
    if (name) {
        char name_buffer[64];
        strncpy(name_buffer, name, 63);
        name_buffer[63] = '\0';
        
        if (ioctl(fd, HOLOMEM_SET_NAME, name_buffer) < 0) {
            perror("Error al establecer nombre");
        }
    }
    
    printf("Patrón almacenado correctamente (%ld bytes)\n", file_size);
    
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
            perror("Error al eliminar patrón");
            close(fd);
            return -1;
        }
        
        printf("Patrón eliminado correctamente\n");
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
Write-Host "🔄 Generando archivo Makefile..." -ForegroundColor Yellow
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

# Escribir archivos a través de WSL
Write-Host "🔄 Escribiendo archivos en WSL..." -ForegroundColor Yellow
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