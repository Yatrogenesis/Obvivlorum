# Módulo de Memoria Holográfica para Linux

## Descripción

Este módulo de kernel implementa un sistema de almacenamiento holográfico para Linux, donde la información se distribuye en un espacio de memoria compartido utilizando principios similares a la holografía óptica. Cada patrón se almacena con redundancia y se puede recuperar incluso cuando partes del espacio de memoria están dañadas o corruptas.

El almacenamiento holográfico permite no solo recuperación robusta ante fallos, sino también recuperación asociativa basada en similitud, donde se pueden encontrar patrones que coinciden parcialmente con una consulta dada.

## Características

- **Almacenamiento Distribuido**: La información se distribuye en un espacio holográfico con redundancia incorporada (factor 7).
- **Recuperación Asociativa**: Búsqueda de patrones similares basada en coherencia.
- **Codificación Óptima**: Implementación FFT para convolución eficiente.
- **Múltiples Tipos de Codificación**: Soporta codificación simbólica, emocional y contextual.
- **Optimización para Hardware**: Detección de capacidades AVX y optimizaciones correspondientes.
- **Sistema de Caché**: Caché LRU para patrones frecuentemente accedidos.
- **Monitoreo Integrado**: Métricas completas disponibles vía sysfs y proc.
- **Resistencia a Ataques**: Validación exhaustiva de entradas y auditoría de operaciones sospechosas.

## Requisitos de Hardware

### Requisitos Mínimos
- **Procesador**: Cualquier CPU x86-64 o ARM compatible con Linux (kernel 5.0+)
- **Memoria**: 256MB RAM disponible (128MB para espacio holográfico + sobrecarga)
- **Almacenamiento**: 5MB para el módulo y utilidades
- **Kernel Linux**: 5.0+ (recomendado 5.10+)

### Requisitos Recomendados
- **Procesador**: CPU x86-64 con soporte AVX/AVX2 para optimizaciones
- **Memoria**: 1GB+ RAM disponible
- **Kernel Linux**: 5.15+

### Comprobación de Compatibilidad
El módulo detectará automáticamente las capacidades del procesador e informará durante la carga:
```
HoloMem: CPU features - AVX: Yes, AVX2: Yes, AVX512: No
```

## Instalación

### Compilación
```bash
# Compilación estándar
make

# Compilación con símbolos de depuración
make DEBUG=1

# Compilación con optimizaciones AVX
make OPTIMIZE_AVX=1
```

### Instalación
```bash
sudo make install
```

### Carga del Módulo
```bash
sudo modprobe holomem

# Alternativamente
sudo insmod holomem.ko
```

## Uso

### API del Dispositivo
El módulo crea el archivo de dispositivo `/dev/holomem` que puede utilizarse a través de comandos ioctl:

- `HOLOMEM_IOCSTOREPAT`: Almacena un patrón
- `HOLOMEM_IOCRETPAT`: Recupera un patrón almacenado
- `HOLOMEM_IOCSEARCHPAT`: Busca patrones similares a una consulta
- `HOLOMEM_IOCGETSTAT`: Obtiene estadísticas del sistema
- `HOLOMEM_IOCFLUSH`: Limpia todos los patrones almacenados

### Utilidad de Línea de Comandos
Se proporciona una utilidad `holomem-util` para interactuar con el módulo:

```bash
# Almacenar un archivo como patrón (tipo 0=simbólico, 1=emocional, 2=contextual)
./holomem-util store archivo.dat 0 "Nombre del patrón"

# Recuperar un patrón por ID
./holomem-util retrieve 1 archivo_recuperado.dat

# Buscar patrones similares (umbral 0-10000)
./holomem-util search consulta.dat 0 8000

# Ver estadísticas
./holomem-util stats

# Limpiar la memoria
./holomem-util flush
```

### Monitoreo del Sistema

#### A través de procfs
```bash
cat /proc/holomem
```

#### A través de sysfs
```bash
# Ver número de patrones almacenados
cat /sys/kernel/holomem/pattern_count

# Ver operaciones de almacenamiento
cat /sys/kernel/holomem/store_ops

# Ver tasa de aciertos de caché
cat /sys/kernel/holomem/cache_hit_rate
```

## Arquitectura del Módulo

```
+-------------------+     +-------------------+     +-------------------+
| Interfaz Usuario  |     | Sistema de Caché  |     | Métricas y        |
| - IOCTL           |<--->| - LRU Cache       |<--->| Monitoreo         |
| - procfs          |     | - Gestión Eficiente|     | - sysfs           |
| - sysfs           |     +-------------------+     | - Estadísticas     |
+-------------------+             ^                  +-------------------+
        ^                         |                          ^
        |                         v                          |
        v                 +-------------------+              v
+-------------------+     | Núcleo Holográfico|     +-------------------+
| Gestión Patrones  |<--->| - Codificación FFT|<--->| Seguridad y       |
| - Hash Table      |     | - Convolución     |     | Auditoría         |
| - Lista Enlazada  |     | - Espacio Común   |     | - Validación      |
+-------------------+     +-------------------+     | - Detección        |
                                   ^                +-------------------+
                                   |
                                   v
                          +-------------------+
                          | Optimizaciones HW |
                          | - Detección CPU   |
                          | - Optimización AVX|
                          +-------------------+
```

## Limitaciones Actuales

- Tamaño de entrada máximo: 4MB por patrón
- Memoria holográfica fija: 128MB (configurable en compilación)
- Máximo 2048 patrones simultáneos
- La codificación/decodificación no es perfecta (aproximación holográfica)
- No persiste después de descargar el módulo (solo almacenamiento en memoria)

## Solución de Problemas

### El módulo no carga
Verifique los mensajes del kernel:
```bash
dmesg | grep HoloMem
```

### Rendimiento lento en operaciones FFT
Verifique si las capacidades AVX están siendo detectadas:
```bash
cat /proc/holomem | grep "CPU features"
```
Si no se detectan correctamente, recompile con `OPTIMIZE_AVX=1`.

### Error "No space left on device"
El límite de 2048 patrones se ha alcanzado. Ejecute:
```bash
echo "flush" > /dev/holomem
```

## Licencia

Este software se distribuye bajo la licencia GNU General Public License v2.
