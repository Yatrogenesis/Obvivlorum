# 05_middleware_e_interfaz.ps1
Write-Host "=== Simbiosis: Paso 5 - Middleware simbiotico + Interfaz GPT ===" -ForegroundColor Cyan

# Obtener distribucion seleccionada
. "$PSScriptRoot\get_distro.ps1"
$distro = Get-SelectedDistro
Write-Host "Usando distribucion: $distro" -ForegroundColor Green

# Instalar dependencias con formato correcto
$installCommand = @'
echo "Instalando dependencias del sistema..."
sudo apt update
sudo apt install -y python3 python3-pip python3-venv espeak ffmpeg
if [ $? -eq 0 ]; then
    echo "Dependencias instaladas correctamente"
else
    echo "Error instalando dependencias"
    exit 1
fi

echo "Configurando entorno virtual Python..."
cd ~/obvlivorum_simbiosis
if [ ! -d "simbiox" ]; then
    python3 -m venv simbiox
fi

source simbiox/bin/activate
if [ $? -ne 0 ]; then
    echo "Error activando entorno virtual"
    exit 1
fi

echo "Instalando paquetes Python..."
pip install -U pip
pip install pyttsx3 SpeechRecognition sounddevice scipy openai-whisper
if [ $? -eq 0 ]; then
    echo "Paquetes Python instalados correctamente"
else
    echo "Error instalando paquetes Python"
    exit 1
fi
'@

Write-Host "Instalando dependencias (requiere contraseña de sudo)..." -ForegroundColor Yellow
wsl -d $distro -- bash -c $installCommand

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error durante la instalacion de dependencias" -ForegroundColor Red
    exit 1
}

# Crear script Python simbiotico
# Obtener usuario de Linux y limpiar output
$linuxUser = (wsl -d $distro whoami 2>$null).Trim()
if (-not $linuxUser) {
    Write-Host "Error obteniendo usuario de $distro" -ForegroundColor Red
    exit 1
}
Write-Host "Usuario Linux: $linuxUser" -ForegroundColor Cyan

# Crear archivo Python via WSL
Write-Host "Creando interfaz Python..." -ForegroundColor Yellow

$pythonScript = @'
import os
import subprocess
import pyttsx3
import time
import re

model_path = "~/obvlivorum_simbiosis/llama.cpp/models/llama-model.gguf"
llama_path = "~/obvlivorum_simbiosis/llama.cpp/main"

# Configurar sintesis de voz
try:
    engine = pyttsx3.init()
    engine.setProperty("rate", 145)
    voice_enabled = True
except:
    print("Sintesis de voz no disponible")
    voice_enabled = False

def clean_llm_output(text):
    """Limpia la salida del LLM removiendo metadatos"""
    # Remover informacion de sistema y timing
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        if not any(x in line.lower() for x in ['llama_print_timings', 'load time', 'sample time', 'prompt eval', 'eval time']):
            cleaned_lines.append(line)
    
    result = '\n'.join(cleaned_lines).strip()
    return result if result else "Error procesando respuesta"

def store_in_holomem(text, label="IA-Response"):
    """Almacena texto en memoria holografica"""
    try:
        with open('/tmp/pattern.txt', 'w') as f:
            f.write(text)
        
        cmd = f'~/obvlivorum_simbiosis/holomem/holomem-util store /tmp/pattern.txt 5 {label}'
        subprocess.run(["bash", "-c", cmd], check=True)
        print(f"Patron almacenado: {label}")
    except Exception as e:
        print(f"Error almacenando en holomem: {e}")

print("=== SIMBIOSIS IA - Sistema Holografico ===")
print("Comandos especiales:")
print("  /memoria - Ver patrones almacenados")
print("  /limpiar - Limpiar memoria")
print("  /salir - Terminar sesion")
print("=" * 50)

while True:
    try:
        prompt = input("\nTu: ").strip()
        
        if prompt.lower() in ["/salir", "exit", "quit"]:
            print("Terminando simbiosis...")
            break
        
        if prompt.lower() == "/memoria":
            print("\nMemoria holografica:")
            try:
                result = subprocess.run(["bash", "-c", "~/obvlivorum_simbiosis/holomem/holomem-util list"], 
                                      capture_output=True, text=True)
                print(result.stdout if result.stdout else "Memoria vacia")
            except:
                print("Error accediendo a memoria")
            continue
        
        if not prompt:
            continue
        
        print("Procesando con Llama-3.2-3B...")
        
        # Construir comando mejorado
        escaped_prompt = prompt.replace('"', '\\"')
        cmd = f'{llama_path} -m {model_path} -p "Usuario: {escaped_prompt}\nAsistente:" -n 300 -t 4 --temp 0.7'
        
        result = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            output = "Error procesando solicitud"
        else:
            output = clean_llm_output(result.stdout)
        
        print(f"\nIA: {output}")
        
        # Sintesis de voz si esta disponible
        if voice_enabled and len(output) < 500:
            try:
                engine.say(output)
                engine.runAndWait()
            except:
                pass
        
        # Almacenar en memoria holografica
        store_in_holomem(f"Q: {prompt}\nA: {output}", "Conversacion")
        
    except KeyboardInterrupt:
        print("\nSesion interrumpida")
        break
    except subprocess.TimeoutExpired:
        print("Timeout - El modelo tardo demasiado")
    except Exception as e:
        print(f"Error inesperado: {e}")

print("Simbiosis terminada")
'@

# Crear comando para guardar el script
$createScriptCommand = @"
cd ~/obvlivorum_simbiosis
cat > simbiosis_cli.py << 'EOFPYTHON'
$pythonScript
EOFPYTHON

chmod +x simbiosis_cli.py
if [ -f simbiosis_cli.py ]; then
    echo "Script Python creado correctamente"
    ls -la simbiosis_cli.py
else
    echo "Error creando script Python"
    exit 1
fi
"@

wsl -d $distro -- bash -c $createScriptCommand

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nMiddleware e interfaz creados exitosamente!" -ForegroundColor Green
    Write-Host "Ejecuta el Script 6 (.\06_lanzar_simbiosis.ps1) para lanzarlo." -ForegroundColor Yellow
} else {
    Write-Host "Error creando la interfaz Python" -ForegroundColor Red
    exit 1
}