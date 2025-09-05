# 03_generar_archivos_fuente_v2.ps1
# Version mejorada con manejo robusto de archivos
# Ejecutar con PowerShell (doble clic)

Write-Host "=== Simbiosis: Paso 3 - Generando archivos fuente ===" -ForegroundColor Cyan

# Obtener distribucion seleccionada
. "$PSScriptRoot\get_distro.ps1"
$distro = Get-SelectedDistro
Write-Host "Usando distribucion: $distro" -ForegroundColor Green

# Obtener usuario de Linux y limpiar output
$linuxUser = (wsl -d $distro whoami 2>$null).Trim()
if (-not $linuxUser) {
    Write-Host "Error obteniendo usuario de $distro" -ForegroundColor Red
    exit 1
}
Write-Host "Usuario Linux: $linuxUser" -ForegroundColor Cyan

# Crear directorio via WSL
Write-Host "Creando estructura de directorios..." -ForegroundColor Yellow
$result = wsl -d $distro -- bash -c "mkdir -p ~/obvlivorum_simbiosis/holomem && echo 'OK'"
if ($result -ne "OK") {
    Write-Host "Error creando directorios en $distro" -ForegroundColor Red
    exit 1
}

Write-Host "Generando archivos fuente..." -ForegroundColor Yellow

# Metodo alternativo: crear archivos directamente via echo en WSL
Write-Host "Creando holomem.c..." -ForegroundColor Cyan

# Crear holomem.c linea por linea (mas seguro)
wsl -d $distro -- bash -c @'
cat > ~/obvlivorum_simbiosis/holomem/holomem.c << 'EOF'
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/proc_fs.h>
#include <linux/uaccess.h>
#include <linux/slab.h>
#include <linux/vmalloc.h>

#define MODULE_NAME "holomem"
#define PROC_ENTRY "holomem"
#define MAX_PATTERNS 1024
#define MAX_PATTERN_SIZE 4096

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Obvivlorum Symbiosis");
MODULE_DESCRIPTION("Holographic Memory Pattern Storage");

struct pattern_entry {
    char *data;
    size_t size;
    int priority;
    char label[64];
    struct pattern_entry *next;
};

static struct pattern_entry *pattern_head = NULL;
static struct proc_dir_entry *proc_entry;
static int pattern_count = 0;

static int store_pattern(const char *data, size_t size, int priority, const char *label) {
    struct pattern_entry *entry;
    
    if (pattern_count >= MAX_PATTERNS || size > MAX_PATTERN_SIZE)
        return -ENOMEM;
    
    entry = kmalloc(sizeof(struct pattern_entry), GFP_KERNEL);
    if (!entry)
        return -ENOMEM;
    
    entry->data = kmalloc(size + 1, GFP_KERNEL);
    if (!entry->data) {
        kfree(entry);
        return -ENOMEM;
    }
    
    memcpy(entry->data, data, size);
    entry->data[size] = '\0';
    entry->size = size;
    entry->priority = priority;
    strncpy(entry->label, label, 63);
    entry->label[63] = '\0';
    
    entry->next = pattern_head;
    pattern_head = entry;
    pattern_count++;
    
    return 0;
}

static ssize_t holomem_write(struct file *file, const char __user *buffer, size_t count, loff_t *pos) {
    char *kernel_buf;
    char *data_start;
    int priority = 0;
    char label[64] = "default";
    
    if (count > MAX_PATTERN_SIZE)
        return -EINVAL;
    
    kernel_buf = kmalloc(count + 1, GFP_KERNEL);
    if (!kernel_buf)
        return -ENOMEM;
    
    if (copy_from_user(kernel_buf, buffer, count)) {
        kfree(kernel_buf);
        return -EFAULT;
    }
    kernel_buf[count] = '\0';
    
    // Parse: "PRIORITY:LABEL:DATA"
    data_start = strchr(kernel_buf, ':');
    if (data_start) {
        *data_start = '\0';
        priority = simple_strtol(kernel_buf, NULL, 10);
        data_start++;
        
        char *label_end = strchr(data_start, ':');
        if (label_end) {
            *label_end = '\0';
            strncpy(label, data_start, 63);
            data_start = label_end + 1;
        }
    } else {
        data_start = kernel_buf;
    }
    
    if (store_pattern(data_start, strlen(data_start), priority, label) == 0) {
        printk(KERN_INFO "holomem: Pattern stored - %s (priority %d)\n", label, priority);
        kfree(kernel_buf);
        return count;
    }
    
    kfree(kernel_buf);
    return -ENOMEM;
}

static ssize_t holomem_read(struct file *file, char __user *buffer, size_t count, loff_t *pos) {
    struct pattern_entry *entry;
    char *output;
    size_t total_len = 0;
    ssize_t ret;
    
    if (*pos > 0)
        return 0;
    
    output = vmalloc(MAX_PATTERNS * 128);
    if (!output)
        return -ENOMEM;
    
    total_len += sprintf(output + total_len, "Holomem Patterns (%d total):\n", pattern_count);
    
    for (entry = pattern_head; entry != NULL; entry = entry->next) {
        total_len += sprintf(output + total_len, "[%s] P:%d Size:%zu - %.80s\n",
                           entry->label, entry->priority, entry->size, entry->data);
        if (total_len >= count - 100) break;
    }
    
    ret = simple_read_from_buffer(buffer, count, pos, output, total_len);
    vfree(output);
    return ret;
}

static const struct proc_ops holomem_proc_ops = {
    .proc_read = holomem_read,
    .proc_write = holomem_write,
};

static int __init holomem_init(void) {
    proc_entry = proc_create(PROC_ENTRY, 0666, NULL, &holomem_proc_ops);
    if (!proc_entry) {
        printk(KERN_ERR "holomem: Failed to create proc entry\n");
        return -ENOMEM;
    }
    
    printk(KERN_INFO "holomem: Holographic memory module loaded\n");
    return 0;
}

static void __exit holomem_exit(void) {
    struct pattern_entry *entry, *next;
    
    if (proc_entry)
        proc_remove(proc_entry);
    
    for (entry = pattern_head; entry != NULL; entry = next) {
        next = entry->next;
        kfree(entry->data);
        kfree(entry);
    }
    
    printk(KERN_INFO "holomem: Module unloaded, %d patterns released\n", pattern_count);
}

module_init(holomem_init);
module_exit(holomem_exit);
EOF
'@

Write-Host "Creando holomem-util.c..." -ForegroundColor Cyan

wsl -d $distro -- bash -c @'
cat > ~/obvlivorum_simbiosis/holomem/holomem-util.c << 'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>

#define PROC_PATH "/proc/holomem"
#define MAX_BUFFER 8192

void usage(char *progname) {
    printf("Holomem Utility - Symbiotic Pattern Storage\n");
    printf("Usage:\n");
    printf("  %s store <file> <priority> <label>  - Store pattern from file\n", progname);
    printf("  %s list                             - List all patterns\n", progname);
    printf("  %s query <text>                     - Query patterns\n", progname);
    printf("\nExamples:\n");
    printf("  %s store pattern.txt 5 GPT-Response\n", progname);
    printf("  %s list\n", progname);
}

int store_pattern(const char *filename, int priority, const char *label) {
    FILE *file;
    FILE *proc_file;
    char buffer[MAX_BUFFER];
    char write_buffer[MAX_BUFFER + 128];
    size_t bytes_read;
    
    file = fopen(filename, "r");
    if (!file) {
        perror("Error opening input file");
        return -1;
    }
    
    bytes_read = fread(buffer, 1, sizeof(buffer) - 1, file);
    buffer[bytes_read] = '\0';
    fclose(file);
    
    // Remove newlines for cleaner storage
    for (int i = 0; buffer[i]; i++) {
        if (buffer[i] == '\n') buffer[i] = ' ';
    }
    
    proc_file = fopen(PROC_PATH, "w");
    if (!proc_file) {
        perror("Error opening /proc/holomem");
        return -1;
    }
    
    sprintf(write_buffer, "%d:%s:%s", priority, label, buffer);
    fputs(write_buffer, proc_file);
    fclose(proc_file);
    
    printf("Pattern stored: [%s] Priority %d\n", label, priority);
    return 0;
}

int list_patterns() {
    FILE *proc_file;
    char buffer[1024];
    
    proc_file = fopen(PROC_PATH, "r");
    if (!proc_file) {
        perror("Error reading /proc/holomem");
        return -1;
    }
    
    while (fgets(buffer, sizeof(buffer), proc_file)) {
        printf("%s", buffer);
    }
    
    fclose(proc_file);
    return 0;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }
    
    if (strcmp(argv[1], "store") == 0) {
        if (argc != 5) {
            usage(argv[0]);
            return 1;
        }
        return store_pattern(argv[2], atoi(argv[3]), argv[4]);
    }
    else if (strcmp(argv[1], "list") == 0) {
        return list_patterns();
    }
    else {
        usage(argv[0]);
        return 1;
    }
}
EOF
'@

Write-Host "Creando Makefile..." -ForegroundColor Cyan

wsl -d $distro -- bash -c @'
cat > ~/obvlivorum_simbiosis/holomem/Makefile << 'EOF'
obj-m := holomem.o
KDIR := /lib/modules/$(shell uname -r)/build
PWD := $(shell pwd)
ccflags-y := -Wall -Werror -O2
all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules
clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean
EOF
'@

# Verificar que los archivos se crearon
Write-Host ""
Write-Host "Verificando archivos creados..." -ForegroundColor Yellow
$files = wsl -d $distro -- bash -c "ls -la ~/obvlivorum_simbiosis/holomem/"
Write-Host $files

Write-Host ""
Write-Host "Archivos generados correctamente" -ForegroundColor Green