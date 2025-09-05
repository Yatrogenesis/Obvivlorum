#!/bin/bash
# Script de prueba para validación del módulo de memoria holográfica

echo "===== Prueba del Módulo de Memoria Holográfica ====="

# Comprobar permisos de superusuario
if [ $(id -u) -ne 0 ]; then
    echo "Este script requiere permisos de superusuario"
    exit 1
fi

# Verificar compilación del módulo
if [ ! -f "holomem.ko" ]; then
    echo "Error: El módulo no está compilado. Ejecute 'make' primero."
    exit 1
fi

# Comprobar si ya está cargado
if lsmod | grep -q "holomem"; then
    echo "Descargando módulo existente..."
    rmmod holomem
    # Esperar a que se descargue completamente
    sleep 1
fi

# Cargar el módulo
echo "Cargando módulo..."
insmod holomem.ko

# Verificar que se ha cargado correctamente
if ! lsmod | grep -q "holomem"; then
    echo "Error: El módulo no se cargó correctamente"
    exit 1
fi

# Esperar a que se inicialice completamente
sleep 1

# Verificar que se crearon los archivos
if [ ! -c /dev/holomem ]; then
    echo "Error: Archivo de dispositivo no creado"
    rmmod holomem
    exit 1
fi

if [ ! -f /proc/holomem ]; then
    echo "Error: Archivo proc no creado"
    rmmod holomem
    exit 1
fi

if [ ! -d /sys/kernel/holomem ]; then
    echo "Error: Directorio sysfs no creado"
    rmmod holomem
    exit 1
fi

echo "Archivos del módulo creados correctamente."

# Verificar permisos
if [ ! -r /dev/holomem ] || [ ! -w /dev/holomem ]; then
    echo "Error: Permisos incorrectos en /dev/holomem"
    rmmod holomem
    exit 1
fi

echo "Permisos correctos."

# Comprobar stats iniciales
echo "Comprobando estadísticas iniciales..."
cat /proc/holomem | grep "Stored patterns"
cat /sys/kernel/holomem/pattern_count

# Verificar funcionamiento con la utilidad
if [ -f "./holomem-util" ]; then
    echo "Ejecutando pruebas con la utilidad..."
    
    # Crear archivo de prueba
    echo "Creando archivo de prueba..."
    dd if=/dev/urandom of=test_data.bin bs=1K count=64
    
    # Almacenar patrón
    echo "Almacenando patrón..."
    ./holomem-util store test_data.bin 0 "Patrón de prueba"
    
    # Verificar que se almacenó
    echo "Verificando almacenamiento..."
    cat /proc/holomem | grep "Patrón de prueba"
    
    # Recuperar patrón
    echo "Recuperando patrón..."
    ./holomem-util retrieve 1 test_recovered.bin
    
    # Buscar patrón similar
    echo "Buscando patrón similar..."
    ./holomem-util search test_data.bin 0 8000
    
    # Verificar estadísticas
    echo "Verificando estadísticas después de operaciones..."
    ./holomem-util stats
    
    # Limpiar
    echo "Limpiando datos de prueba..."
    rm -f test_data.bin test_recovered.bin
else
    echo "No se encontró la utilidad holomem-util. Omitiendo pruebas de funcionalidad."
fi

# Verificar que sysfs muestra métricas
echo "Verificando métricas en sysfs..."
cat /sys/kernel/holomem/store_ops
cat /sys/kernel/holomem/retrieve_ops
cat /sys/kernel/holomem/search_ops
cat /sys/kernel/holomem/cache_hit_rate

# Probar flush a través del archivo
echo "Probando comando flush..."
echo "flush" > /dev/holomem

# Verificar stats después de flush
echo "Verificando estadísticas después de flush..."
cat /proc/holomem | grep "Stored patterns"

# Comprobar uso de memoria
echo "Comprobando uso de memoria del módulo..."
grep holomem /proc/modules
grep holomem /proc/slabinfo 2>/dev/null || echo "No hay información de slab"

# Descargar módulo
echo "Descargando módulo..."
rmmod holomem

# Verificar descarga
if lsmod | grep -q "holomem"; then
    echo "Error: El módulo no se descargó correctamente"
    exit 1
fi

echo "Prueba completada con éxito"
exit 0
