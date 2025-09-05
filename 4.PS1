# 04_instalar_tinyllama.ps1
# Ejecutar con PowerShell como administrador

# Verificar si se está ejecutando como administrador
if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host "❌ Este script debe ejecutarse como administrador." -ForegroundColor Red
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
    Write-Host "❌ No hay suficiente espacio disponible. Se requieren al menos 2GB libres." -ForegroundColor Red
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
    Write-Host "🔄 llama.cpp ya está instalado. Verificando..." -ForegroundColor Yellow
    
    # Verificar si el ejecutable existe
    $execExists = wsl -d kali-linux -- bash -c "
        if [ -f ~/obvlivorum_simbiosis/llama.cpp/main ]; then
            echo 'EXISTE'
        else
            echo 'NO_EXISTE'
        fi
    "
    
    if ($execExists -eq "NO_EXISTE") {
        Write-Host "🔄 El ejecutable no existe. Recompilando..." -ForegroundColor Yellow
        wsl -d kali-linux -- bash -c "
            cd ~/obvlivorum_simbiosis/llama.cpp &&
            make clean &&
            make -j\$(nproc)
        "
        
        if (-not $?) {
            Write-Host "❌ Error al compilar llama.cpp." -ForegroundColor Red
            pause
            exit 1
        }
    } else {
        Write-Host "✅ llama.cpp ya está compilado." -ForegroundColor Green
    }
} else {
    # Instalar dependencias necesarias
    Write-Host "🔄 Instalando dependencias para llama.cpp..." -ForegroundColor Yellow
    wsl -d kali-linux -- bash -c "
        sudo apt update &&
        sudo apt install -y build-essential cmake python3-dev libopenblas-dev
    "
    
    if (-not $?) {
        Write-Host "❌ Error al instalar dependencias." -ForegroundColor Red
        pause
        exit 1
    }
    
    # Clonar y compilar llama.cpp
    Write-Host "🔄 Clonando repositorio llama.cpp..." -ForegroundColor Yellow
    wsl -d kali-linux -- bash -c "
        cd ~/obvlivorum_simbiosis &&
        git clone https://github.com/ggerganov/llama.cpp &&
        cd llama.cpp &&
        make -j\$(nproc)
    "
    
    if (-not $?) {
        Write-Host "❌ Error al clonar o compilar llama.cpp." -ForegroundColor Red
        pause
        exit 1
    }
    
    Write-Host "✅ llama.cpp clonado y compilado correctamente." -ForegroundColor Green
}

# Verificar si el modelo ya está descargado
$modelExists = wsl -d kali-linux -- bash -c "
    if [ -f ~/obvlivorum_simbiosis/llama.cpp/models/tinyllama-1.1b-chat.Q4_K_M.gguf ]; then
        echo 'EXISTE'
    else
        echo 'NO_EXISTE'
    fi
"

if ($modelExists -eq "NO_EXISTE") {
    # Crear directorio de modelos si no existe
    Write-Host "🔄 Descargando modelo TinyLLaMA (esto puede tomar varios minutos)..." -ForegroundColor Yellow
    wsl -d kali-linux -- bash -c "
        mkdir -p ~/obvlivorum_simbiosis/llama.cpp/models &&
        cd ~/obvlivorum_simbiosis/llama.cpp/models &&
        wget --progress=bar:force:noscroll https://huggingface.co/cafune/tinyllama-1.1B-gguf/resolve/main/tinyllama-1.1b-chat.Q4_K_M.gguf
    "
    
    if (-not $?) {
        Write-Host "❌ Error al descargar el modelo. Verifica tu conexión a Internet." -ForegroundColor Red
        pause
        exit 1
    }
    
    # Verificar que el modelo se descargó correctamente
    $modelSizeCheck = wsl -d kali-linux -- bash -c "
        SIZE=\$(du -m ~/obvlivorum_simbiosis/llama.cpp/models/tinyllama-1.1b-chat.Q4_K_M.gguf | cut -f1)
        if (( \$SIZE > 500 )); then
            echo 'TAMAÑO_OK'
        else
            echo 'TAMAÑO_INCORRECTO'
        fi
    "
    
    if ($modelSizeCheck -eq "TAMAÑO_INCORRECTO") {
        Write-Host "❌ El modelo descargado parece incompleto o dañado." -ForegroundColor Red
        Write-Host "   Intenta ejecutar nuevamente este script." -ForegroundColor Yellow
        pause
        exit 1
    }
    
    Write-Host "✅ Modelo TinyLLaMA descargado correctamente." -ForegroundColor Green
} else {
    Write-Host "✅ El modelo TinyLLaMA ya está descargado." -ForegroundColor Green
}

# Probar el modelo
Write-Host "🔄 Verificando funcionamiento del modelo con una prueba sencilla..." -ForegroundColor Yellow
$testResult = wsl -d kali-linux -- bash -c "
    cd ~/obvlivorum_simbiosis/llama.cpp &&
    echo 'Prueba exitosa' > /tmp/test_result.txt &&
    ./main -m models/tinyllama-1.1b-chat.Q4_K_M.gguf -p 'Hola, responde con 3 palabras:' -n 20 --no-display-prompt -r 'EOM' > /tmp/test_result.txt 2>/dev/null || echo 'ERROR_EJECUCIÓN'
    cat /tmp/test_result.txt
"

if ($testResult -eq "ERROR_EJECUCIÓN") {
    Write-Host "⚠️ No se pudo verificar el funcionamiento del modelo. Esto podría causar problemas más adelante." -ForegroundColor Yellow
} else {
    Write-Host "✅ Prueba del modelo completada." -ForegroundColor Green
}

Write-Host "✅ TinyLLaMA instalado y listo en ~/obvlivorum_simbiosis/llama.cpp/models" -ForegroundColor Green
Write-Host "✅ Paso 4 completado. Continúa con el Script 5 para instalar el middleware e interfaz." -ForegroundColor Green