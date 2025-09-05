# 🧠 SIMBIOSIS OBVIVLORUM

Sistema holográfico de IA con memoria persistente que integra un LLM local con un módulo de kernel para almacenamiento de patrones.

## 🎯 Características

- **Memoria Holográfica**: Módulo kernel `holomem` para almacenamiento persistente de patrones
- **IA Local**: Utiliza modelos Llama locales (sin conexión a internet)
- **Interfaz Intuitiva**: CLI con síntesis de voz y comandos especiales
- **Multiplataforma**: Funciona en Windows con WSL2 + Kali Linux

## 📦 Modelos Disponibles

El sistema puede usar cualquiera de estos modelos (ya incluidos):

| Modelo | Tamaño | Velocidad | Calidad | Recomendado |
|--------|--------|-----------|---------|-------------|
| Llama-3.2-1B | 773MB | ⚡⚡⚡ | ⭐⭐ | Para sistemas lentos |
| **Llama-3.2-3B** | 1.9GB | ⚡⚡ | ⭐⭐⭐ | **✅ Por defecto** |
| Meta-Llama-3-8B | 4.6GB | ⚡ | ⭐⭐⭐⭐ | Para mejor calidad |
| Phi-3-mini | 2.4GB | ⚡⚡ | ⭐⭐⭐ | Alternativo |

## 🚀 Instalación Rápida

### Opción 1: Instalación Automática
```powershell
# Ejecutar como administrador
.\cods\00_instalar_completo.ps1
```

### Opción 2: Instalación Manual
```powershell
# Paso a paso
.\cods\01_instalar_WSL2_y_Kali.ps1
.\cods\03_generar_archivos_fuente.ps1  
.\cods\02_instalar_holomem.ps1
.\cods\04_instalar_llama_cpp.ps1
.\cods\05_middleware_e_interfaz.ps1
```

## 🎮 Uso

### Lanzar el Sistema
```powershell
.\cods\06_lanzar_simbiosis.ps1
```

### Comandos Disponibles
- `¿Qué es la IA?` - Pregunta normal
- `/memoria` - Ver patrones almacenados
- `/salir` - Terminar sesión

### Ejemplo de Sesión
```
🧠 Tú: ¿Cuál es el sentido de la vida?
🤖 Procesando con Llama-3.2-3B...
🤖 IA: El sentido de la vida es una pregunta fundamental...
💾 Patrón almacenado: Conversacion

🧠 Tú: /memoria
📚 Memoria holográfica:
[Conversacion] P:5 Size:234 - Q: ¿Cuál es el sentido de la vida? A: El sentido...
```

## 🔧 Arquitectura Técnica

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Windows CLI   │───▶│  WSL2 + Kali     │───▶│  Kernel Module  │
│   (PowerShell)  │    │  (Python + C++)  │    │   (holomem)     │ 
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         └───── llama.cpp ───────┴────── /proc/holomem ──┘
```

### Componentes

1. **holomem.ko**: Módulo kernel para almacenamiento de patrones
2. **holomem-util**: Utilidad CLI para interactuar con el módulo
3. **llama.cpp**: Motor de inferencia para modelos LLM
4. **simbiosis_cli.py**: Interfaz principal con síntesis de voz

## 🛠️ Desarrollo

### Estructura de Archivos
```
cods/
├── 00_instalar_completo.ps1     # 🎯 Script maestro
├── 01_instalar_WSL2_y_Kali.ps1  # Configuración base
├── 02_instalar_holomem.ps1      # Compilación módulo
├── 03_generar_archivos_fuente.ps1 # Código fuente
├── 04_instalar_llama_cpp.ps1    # Motor LLM
├── 05_middleware_e_interfaz.ps1 # Interfaz Python
└── 06_lanzar_simbiosis.ps1      # Lanzador
```

### Personalización de Modelo

Para cambiar el modelo LLM, edita el script 4:
```powershell
# En 04_instalar_llama_cpp.ps1
$sourcePath = "D:\Obvivlorum\LLMs\Meta-Llama-3-8B-Instruct.Q4_0.gguf"  # Cambiar aquí
```

## 🔍 Solución de Problemas

### Error: "Virtualización NO está activada"
- Activar VT-x/AMD-V en BIOS
- Reiniciar y volver a ejecutar

### Error: "No se encontró Kali Linux"
- Instalar desde Microsoft Store
- Configurar usuario inicial

### Error: "Módulo holomem no carga"
```bash
# Verificar en WSL
sudo dmesg | grep holomem
lsmod | grep holomem
```

### Error: "Modelo no responde"
- Verificar que el archivo .gguf existe
- Comprobar permisos de lectura
- Probar con modelo más pequeño

## 🎨 Personalización

### Cambiar Velocidad de Voz
```python
# En simbiosis_cli.py
engine.setProperty("rate", 120)  # Más lento
engine.setProperty("rate", 180)  # Más rápido
```

### Agregar Comandos Personalizados
```python
# En simbiosis_cli.py
if prompt.lower() == "/ayuda":
    print("🆘 Comandos disponibles: /memoria, /salir")
    continue
```

## 📊 Rendimiento

| Componente | RAM | CPU | Tiempo Respuesta |
|------------|-----|-----|------------------|
| Llama-3.2-1B | 1GB | Bajo | 2-5s |
| Llama-3.2-3B | 3GB | Medio | 5-10s |
| Meta-Llama-8B | 6GB | Alto | 10-20s |

## 🤝 Contribución

1. Fork del repositorio
2. Crear rama feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit cambios (`git commit -am 'Agregar funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crear Pull Request

## 📜 Licencia

GPL v3 - Software libre para uso, modificación y distribución.

## 🏆 Créditos

- **LLaMA**: Meta AI
- **llama.cpp**: Georgi Gerganov
- **WSL2**: Microsoft
- **Kali Linux**: Offensive Security

---
*🧠 "La simbiosis entre humano y máquina comienza con la memoria compartida"*