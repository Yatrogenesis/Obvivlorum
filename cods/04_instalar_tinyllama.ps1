# 04_instalar_llama_cpp.ps1
Write-Host "=== Simbiosis: Paso 4 - Configurando LLM Local ===" -ForegroundColor Cyan

# Obtener distribucion seleccionada
. "$PSScriptRoot\get_distro.ps1"
$distro = Get-SelectedDistro
Write-Host "Usando distribucion: $distro" -ForegroundColor Green

# Instalar llama.cpp
Write-Host "Instalando llama.cpp..." -ForegroundColor Yellow
wsl -d $distro -- bash -c "
cd ~/obvlivorum_simbiosis &&
git clone https://github.com/ggerganov/llama.cpp &&
cd llama.cpp &&
make -j$(nproc)
"

# Crear directorio de modelos y copiar modelo local
Write-Host "Configurando modelos locales..." -ForegroundColor Yellow
wsl -d $distro -- bash -c "
cd ~/obvlivorum_simbiosis/llama.cpp &&
mkdir -p models
"

# Copiar el modelo mas eficiente desde Windows
$sourcePath = "D:\Obvivlorum\LLMs\Llama-3.2-3B-Instruct-Q4_0.gguf"
$wslUserPath = (wsl -d $distro whoami 2>$null).Trim()
$targetPath = "\\wsl$\$distro\home\$wslUserPath\obvlivorum_simbiosis\llama.cpp\models\llama-model.gguf"

Write-Host "Copiando modelo Llama-3.2-3B (balance entre velocidad y calidad)..." -ForegroundColor Cyan
Copy-Item -Path $sourcePath -Destination $targetPath -Force

Write-Host "Llama.cpp y modelo configurados correctamente" -ForegroundColor Green
Write-Host "Modelo disponible en: ~/obvlivorum_simbiosis/llama.cpp/models/llama-model.gguf" -ForegroundColor Cyan
