#!/usr/bin/env python3
"""
Model Downloader - Download GGUF models
======================================
"""

import os
import requests
import sys
from tqdm import tqdm

def download_file(url, filename):
    """Download file with progress bar."""
    print(f"Downloading {filename}...")
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                progress_bar.update(len(chunk))

def main():
    """Download recommended GGUF model."""
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    
    # Recommended lightweight but capable models
    models = {
        "1": {
            "name": "Llama-2-7B-Chat-GGUF (Q4_K_M) - 4.1GB",
            "url": "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf",
            "filename": "llama-2-7b-chat.Q4_K_M.gguf",
            "size": "4.1GB"
        },
        "2": {
            "name": "TinyLlama-1.1B-Chat-GGUF (Q4_K_M) - 669MB",
            "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            "size": "669MB"
        },
        "3": {
            "name": "Phi-2-GGUF (Q4_K_M) - 1.6GB",
            "url": "https://huggingface.co/microsoft/phi-2-gguf/resolve/main/phi-2.Q4_K_M.gguf",
            "filename": "phi-2.Q4_K_M.gguf",
            "size": "1.6GB"
        }
    }
    
    print("=== AI Symbiote Model Downloader ===")
    print("\nModelos disponibles:")
    for key, model in models.items():
        print(f"{key}. {model['name']}")
    
    choice = input("\nSelecciona un modelo (1-3): ").strip()
    
    if choice not in models:
        print("Opción inválida")
        return
    
    model = models[choice]
    model_path = os.path.join(models_dir, model['filename'])
    
    if os.path.exists(model_path):
        print(f"El modelo {model['filename']} ya existe.")
        return
    
    print(f"\nDescargando: {model['name']}")
    print(f"Tamaño: {model['size']}")
    print(f"Destino: {model_path}")
    
    confirm = input("\n¿Continuar con la descarga? (y/n): ").lower()
    if confirm != 'y':
        print("Descarga cancelada")
        return
    
    try:
        download_file(model['url'], model_path)
        print(f"\n✓ Modelo descargado exitosamente: {model['filename']}")
        print("El modelo está listo para usar con AI Symbiote")
    
    except Exception as e:
        print(f"\n[ERROR] Error descargando modelo: {e}")
        if os.path.exists(model_path):
            os.remove(model_path)

if __name__ == "__main__":
    main()