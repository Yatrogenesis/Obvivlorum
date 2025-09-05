# 05_middleware_e_interfaz.ps1
# Ejecutar con PowerShell como administrador

# Verificar si se está ejecutando como administrador
if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host "❌ Este script debe ejecutarse como administrador." -ForegroundColor Red
    Write-Host "   Cierra esta ventana y ejecuta PowerShell como administrador." -ForegroundColor Yellow
    pause
    exit 1
}

Write-Host "=== Simbiosis: Paso 5 - Middleware simbiótico + Interfaz GPT ===" -ForegroundColor Cyan

# Verificar instalación previa de TinyLLaMA
$llamaExists = wsl -d kali-linux -- bash -c "
    if [ -f ~/obvlivorum_simbiosis/llama.cpp/main ] && [ -f ~/obvlivorum_simbiosis/llama.cpp/models/tinyllama-1.1b-chat.Q4_K_M.gguf ]; then
        echo 'OK'
    else
        echo 'FALTA'
    fi
"

if ($llamaExists -ne "OK") {
    Write-Host "❌ No se encontró TinyLLaMA instalado correctamente." -ForegroundColor Red
    Write-Host "   Ejecuta primero el Script 4." -ForegroundColor Yellow
    pause
    exit 1
}

# Verificar módulo holomem
$holomemExists = wsl -d kali-linux -- bash -c "
    if [ -f ~/obvlivorum_simbiosis/holomem/holomem-util ] && lsmod | grep -q holomem; then
        echo 'OK'
    else
        echo 'FALTA'
    fi
"

if ($holomemExists -ne "OK") {
    Write-Host "❌ El módulo holomem no está cargado o no se compiló correctamente." -ForegroundColor Red
    Write-Host "   Verifica los Scripts 2 y 3." -ForegroundColor Yellow
    pause
    exit 1
}

# Instalar dependencias para el middleware
Write-Host "🔄 Instalando dependencias para el middleware..." -ForegroundColor Yellow
$installResult = wsl -d kali-linux -- bash -c "
    sudo apt update &&
    sudo apt install -y python3 python3-pip python3-venv espeak ffmpeg python3-pyaudio portaudio19-dev
"

if (-not $?) {
    Write-Host "❌ Error al instalar dependencias necesarias." -ForegroundColor Red
    Write-Host $installResult -ForegroundColor Red
    pause
    exit 1
}

# Crear entorno virtual Python
Write-Host "🔄 Creando entorno virtual Python..." -ForegroundColor Yellow
$venvResult = wsl -d kali-linux -- bash -c "
    cd ~/obvlivorum_simbiosis &&
    python3 -m venv simbiox &&
    source simbiox/bin/activate &&
    pip install -U pip wheel setuptools &&
    pip install pyttsx3 SpeechRecognition sounddevice scipy openai-whisper PyAudio
"

if (-not $?) {
    Write-Host "❌ Error al crear entorno virtual o instalar paquetes Python." -ForegroundColor Red
    Write-Host $venvResult -ForegroundColor Red
    pause
    exit 1
}

# Crear script Python mejorado
Write-Host "🔄 Creando script de interfaz simbiótica..." -ForegroundColor Yellow

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

# Configuración TTS
try:
    engine = pyttsx3.init()
    engine.setProperty("rate", 145)
    engine.setProperty("volume", 1.0)
    TTS_DISPONIBLE = True
except Exception as e:
    print(f"ADVERTENCIA: No se pudo inicializar el motor TTS: {e}")
    TTS_DISPONIBLE = False

# Función para verificar componentes
def verificar_componentes():
    componentes_ok = True
    
    # Verificar TinyLLaMA
    if not os.path.isfile(LLAMA_PATH):
        print(f"❌ ERROR: Ejecutable de llama.cpp no encontrado en: {LLAMA_PATH}")
        componentes_ok = False
    
    if not os.path.isfile(MODEL_PATH):
        print(f"❌ ERROR: Modelo TinyLLaMA no encontrado en: {MODEL_PATH}")
        componentes_ok = False
    
    # Verificar holomem
    try:
        result = subprocess.run(["lsmod"], stdout=subprocess.PIPE, text=True)
        if "holomem" not in result.stdout:
            print("❌ ERROR: Módulo holomem no cargado en el kernel")
            componentes_ok = False
    except Exception as e:
        print(f"❌ ERROR al verificar módulo holomem: {e}")
        componentes_ok = False
    
    # Verificar holomem-util
    if not os.path.isfile(HOLOMEM_UTIL):
        print(f"❌ ERROR: Utilidad holomem-util no encontrada en: {HOLOMEM_UTIL}")
        componentes_ok = False
    
    return componentes_ok

# Función para manejar interrupción
def signal_handler(sig, frame):
    print("\nCerrando Simbiosis GPT CLI...")
    sys.exit(0)

# Registrar manejador de señal para Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

# Clase para la interfaz simbiótica
class SimbioteInterface:
    def __init__(self):
        self.history = []
        self.context = "Eres un asistente de IA simbiótico que responde de manera concisa y útil."
    
    def generar_respuesta(self, prompt):
        try:
            # Construir prompt con contexto y historia limitada
            full_prompt = self.context + "\n\n"
            
            # Añadir últimas 3 interacciones de historia si existen
            for i in range(max(0, len(self.history)-3), len(self.history)):
                full_prompt += f"Usuario: {self.history[i][0]}\nIA: {self.history[i][1]}\n\n"
            
            # Añadir la pregunta actual
            full_prompt += f"Usuario: {prompt}\nIA:"
            
            # Lanzar la interfaz
Write-Host "🚀 Lanzando entorno simbiótico..." -ForegroundColor Green
Write-Host "✅ Presiona Ctrl+C para salir cuando hayas terminado." -ForegroundColor Green

# Limpiar pantalla para una mejor experiencia
Clear-Host

# Imprimir banner
Write-Host "╔═══════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  🧠 Simbiosis GPT CLI - Interfaz Neural   ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Ejecutar la interfaz
wsl -d kali-linux -- bash -c "
    cd ~/obvlivorum_simbiosis &&
    source simbiox/bin/activate &&
    python3 simbiosis_cli.py
"

# Verificar resultado
if (-not $?) {
    Write-Host "❌ Error al ejecutar la interfaz simbiótica." -ForegroundColor Red
    Write-Host "   Revisa los mensajes de error y verifica los componentes." -ForegroundColor Yellow
    pause
    exit 1
}

Write-Host "✅ Sesión simbiótica finalizada." -ForegroundColor Greenlamar a TinyLLaMA
            print("⏳ Procesando con TinyLLaMA...")
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
                print(f"⚠️ Error al ejecutar TinyLLaMA: {result.stderr}")
                return "Lo siento, ocurrió un error al procesar tu solicitud."
            
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
            print(f"❌ Error al generar respuesta: {e}")
            traceback.print_exc()
            return "Lo siento, ocurrió un error inesperado."
    
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
            print(f"⚠️ Error al guardar en holomem: {e}")
    
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
            print(f"⚠️ Error en TTS: {e}")

# Función principal
def main():
    # Banner
    print("╔═══════════════════════════════════════════╗")
    print("║  🧠 Simbiosis GPT CLI - Interfaz Neural   ║")
    print("╚═══════════════════════════════════════════╝")
    
    # Verificar componentes
    if not verificar_componentes():
        print("\n❌ Algunos componentes críticos no están disponibles.")
        print("   Verifica que hayas ejecutado correctamente los scripts 1-4.")
        return 1
    
    # Inicializar interfaz
    interface = SimbioteInterface()
    
    print("\n✅ Todos los sistemas funcionando. Escribe 'salir' para terminar.")
    print("🔄 Puedes hacer preguntas o pedir tareas al sistema simbiótico.\n")
    
    # Bucle principal
    while True:
        try:
            # Obtener entrada de usuario
            prompt = input("👤 Tú > ").strip()
            
            # Verificar comando de salida
            if prompt.lower() in ["salir", "exit", "quit", "q"]:
                print("✅ Cerrando sistema simbiótico...")
                break
            
            # Si la entrada está vacía, continuar
            if not prompt:
                continue
            
            # Procesar y generar respuesta
            response = interface.generar_respuesta(prompt)
            
            # Mostrar respuesta
            print("\n🤖 IA > " + response + "\n")
            
            # Reproducir respuesta (si TTS está disponible)
            interface.hablar(response)
            
        except KeyboardInterrupt:
            print("\n✅ Cerrando sistema simbiótico...")
            break
        except Exception as e:
            print(f"\n❌ Error inesperado: {e}")
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
    Write-Host "❌ Error al crear el script de interfaz." -ForegroundColor Red
    pause
    exit 1
}

Write-Host "✅ Middleware e interfaz creados correctamente." -ForegroundColor Green
Write-Host "✅ Paso 5 completado. Ejecuta el Script 6 para lanzar el entorno simbiótico." -ForegroundColor Green