#!/usr/bin/env python3
"""
AI Engine Multi-Provider - Francisco Molina's Advanced AI System
===============================================================

Sistema de IA hibrido avanzado que incluye:
1. Modelos GGUF locales (multiples opciones)
2. OpenAI/ChatGPT APIs (multiples endpoints gratuitos)  
3. Claude API (Anthropic)
4. Selector de IA configurable por usuario
5. Conocimiento completo sobre Francisco Molina y el proyecto

Author: Francisco Molina
ORCID: https://orcid.org/0009-0008-6093-8267
Email: pako.molina@gmail.com
"""

import os
import sys
import json
import logging
import asyncio
import aiohttp
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime
import threading
import time
from pathlib import Path

# LLM Libraries
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    print("Warning: llama-cpp-python not available")

logger = logging.getLogger("MultiProviderAI")

class MultiProviderAIEngine:
    """Advanced AI Engine with multiple providers and model selection."""
    
    def __init__(self):
        """Initialize multi-provider AI engine."""
        self.is_initialized = False
        self.local_models = {}  # Dictionary of loaded models
        self.available_local_models = []
        self.turbo_mode = False
        self.conversation_history = []
        self.language = "es"
        
        # Provider selection
        self.selected_provider = "auto"  # auto, local, openai, claude, specific_model
        self.selected_model = None
        
        # API Keys
        self.openai_api_key = None
        self.claude_api_key = None
        
        # API Endpoints
        self.openai_endpoints = [
            "https://api.openai-sb.com/v1/chat/completions",
            "https://free.churchless.tech/v1/chat/completions", 
            "https://api.chatanywhere.tech/v1/chat/completions",
            "https://free-api.xyhelper.cn/v1/chat/completions"
        ]
        
        self.claude_endpoint = "https://api.anthropic.com/v1/messages"
        
        # Project context - FRANCISCO MOLINA INFORMATION
        self.project_context = {
            "creator": "Francisco Molina",
            "creator_orcid": "https://orcid.org/0009-0008-6093-8267", 
            "creator_email": "pako.molina@gmail.com",
            "project_name": "AI Symbiote Obvivlorum",
            "purpose": "Sistema de IA simbiotica avanzado con capacidades adaptativas y evolutivas",
            "components": ["AION Protocol v2.0", "Obvivlorum Framework", "Multi-Provider AI Engine"],
            "creation_date": "2025",
            "unique_features": [
                "Procesamiento simbolico cuantico",
                "Memoria holografica persistente", 
                "Arquitectura evolutiva adaptativa",
                "5 protocolos AION avanzados"
            ]
        }
        
        logger.info("Initializing Multi-Provider AI Engine...")
        self._initialize_all_components()
    
    def _initialize_all_components(self):
        """Initialize all available AI providers."""
        try:
            # Scan for available local models
            self._scan_available_local_models()
            
            # Initialize local models
            if LLAMA_AVAILABLE:
                self._initialize_local_models()
            
            # Test API endpoints
            self._test_openai_apis()
            self._test_claude_api()
            
            self.is_initialized = True
            
            # Log available providers
            providers_count = len(self.local_models) + len(self.openai_endpoints)
            if self.claude_api_key:
                providers_count += 1
                
            logger.info(f"MULTI-PROVIDER AI ENGINE INITIALIZED")
            logger.info(f"Available providers: {providers_count}")
            logger.info(f"Local models: {len(self.local_models)}")
            logger.info(f"OpenAI endpoints: {len(self.openai_endpoints)}")
            logger.info(f"Claude API: {'Available' if self.claude_api_key else 'Not configured'}")
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            self.is_initialized = True  # Continue with available providers
    
    def _scan_available_local_models(self):
        """Scan for available GGUF models in multiple directories."""
        search_dirs = [
            "D:/Obvivlorum/models",
            "D:/Obvivlorum/LLMs", 
            "./models",
            "./LLMs",
            "C:/AI/models",
            os.path.expanduser("~/models")
        ]
        
        self.available_local_models = []
        
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                try:
                    for root, dirs, files in os.walk(search_dir):
                        for file in files:
                            if file.endswith('.gguf'):
                                model_path = os.path.join(root, file)
                                model_info = {
                                    "name": file,
                                    "path": model_path,
                                    "size": os.path.getsize(model_path),
                                    "directory": os.path.basename(root)
                                }
                                self.available_local_models.append(model_info)
                                logger.info(f"Found GGUF model: {file}")
                except Exception as e:
                    logger.debug(f"Error scanning {search_dir}: {e}")
        
        logger.info(f"Total GGUF models found: {len(self.available_local_models)}")
    
    def _initialize_local_models(self):
        """Initialize local GGUF models."""
        if not self.available_local_models:
            logger.info("No GGUF models found to initialize")
            return
        
        # Initialize up to 3 models (to avoid memory issues)
        models_to_load = self.available_local_models[:3]
        
        for model_info in models_to_load:
            try:
                model_name = model_info["name"]
                model_path = model_info["path"]
                
                logger.info(f"Loading local model: {model_name}")
                
                # Configure based on model type
                n_ctx = 4096
                n_threads = 4
                
                if "7B" in model_name or "8B" in model_name:
                    n_ctx = 8192
                elif "13B" in model_name:
                    n_ctx = 4096
                    n_threads = 6
                elif "mini" in model_name.lower() or "3B" in model_name:
                    n_threads = 8
                
                model = Llama(
                    model_path=model_path,
                    n_ctx=n_ctx,
                    n_threads=n_threads,
                    verbose=False,
                    n_gpu_layers=0  # CPU only for compatibility
                )
                
                self.local_models[model_name] = {
                    "model": model,
                    "info": model_info,
                    "loaded_at": datetime.now()
                }
                
                logger.info(f"Successfully loaded: {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to load {model_info['name']}: {e}")
    
    def _test_openai_apis(self):
        """Test OpenAI API endpoints."""
        working_endpoints = []
        
        for endpoint in self.openai_endpoints[:]:
            try:
                headers = {
                    "Content-Type": "application/json",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
                
                data = {
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 5,
                    "temperature": 0.7
                }
                
                response = requests.post(endpoint, json=data, headers=headers, timeout=5)
                if response.status_code == 200:
                    working_endpoints.append(endpoint)
                    logger.info(f"OpenAI endpoint working: {endpoint}")
                    
            except Exception as e:
                logger.debug(f"OpenAI endpoint {endpoint} failed: {e}")
        
        self.openai_endpoints = working_endpoints
        logger.info(f"Working OpenAI endpoints: {len(working_endpoints)}")
    
    def _test_claude_api(self):
        """Test Claude API if key is available."""
        # Check for Claude API key in environment or config
        self.claude_api_key = os.environ.get('CLAUDE_API_KEY')
        
        if not self.claude_api_key:
            # Try loading from config file
            try:
                config_file = "D:/Obvivlorum/.ai_symbiote/api_keys.json"
                if os.path.exists(config_file):
                    with open(config_file, 'r') as f:
                        keys = json.load(f)
                        self.claude_api_key = keys.get('claude_api_key')
            except:
                pass
        
        if self.claude_api_key:
            logger.info("Claude API key found and configured")
        else:
            logger.info("Claude API key not configured")
    
    def set_provider(self, provider: str, model_name: str = None):
        """Set the AI provider to use."""
        valid_providers = ["auto", "local", "openai", "claude"] + list(self.local_models.keys())
        
        if provider not in valid_providers:
            logger.warning(f"Invalid provider: {provider}")
            return False
        
        self.selected_provider = provider
        self.selected_model = model_name
        
        logger.info(f"AI provider set to: {provider}")
        if model_name:
            logger.info(f"Specific model: {model_name}")
        
        return True
    
    def get_available_providers(self) -> Dict[str, List[str]]:
        """Get list of all available providers and models."""
        return {
            "local_models": list(self.local_models.keys()),
            "openai_endpoints": len(self.openai_endpoints),
            "claude_available": bool(self.claude_api_key),
            "total_providers": len(self.local_models) + len(self.openai_endpoints) + (1 if self.claude_api_key else 0)
        }
    
    async def process_message(self, message: str, user_id: str = "user") -> str:
        """Process message using selected or auto provider."""
        try:
            # Add to conversation history
            self.conversation_history.append({
                "user": message,
                "timestamp": datetime.now().isoformat()
            })
            
            # Keep history manageable
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
            
            # Check for built-in responses first (Francisco Molina info)
            builtin_response = self._check_builtin_responses(message)
            if builtin_response:
                return builtin_response
            
            # Route to selected provider
            if self.selected_provider == "auto":
                return await self._auto_route_message(message)
            elif self.selected_provider == "local":
                return self._query_local_models(message)
            elif self.selected_provider == "openai":
                return await self._query_openai(message)
            elif self.selected_provider == "claude":
                return await self._query_claude(message)
            elif self.selected_provider in self.local_models:
                return self._query_specific_local_model(message, self.selected_provider)
            else:
                return await self._auto_route_message(message)
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return f"Error procesando mensaje: {e}"
    
    def _check_builtin_responses(self, message: str) -> Optional[str]:
        """Check for built-in responses about Francisco Molina and project."""
        message_lower = message.lower()
        
        # Creator questions - EXPANDED DETECTION
        creator_keywords = [
            'quien te creo', 'who created', 'creador', 'creator', 'francisco molina', 
            'autor', 'quien te hizo', 'quien me diseno', 'tu creador', 'creado por',
            'contestame quien', 'quien eres', 'de donde vienes', 'tu origen',
            'quien es tu creador', 'tu autor', 'quien te diseno', 'creador tuyo'
        ]
        
        if any(phrase in message_lower for phrase in creator_keywords):
            return f"""[TARGET] **RESPUESTA DIRECTA:**

[COMPUTER] **MI CREADOR ES: {self.project_context['creator']}**
 Email: {self.project_context['creator_email']}
 ORCID: {self.project_context['creator_orcid']}

[TARGET] **MI PROPOSITO**: Soy {self.project_context['project_name']}, {self.project_context['purpose']}

**{self.project_context['creator']} me diseno como sistema simbiotico avanzado:**
 {', '.join(self.project_context['components'])}
 {', '.join(self.project_context['unique_features'])}

**NO soy una IA generica. Soy especificamente el AI Symbiote creado por Francisco Molina.**

**Esta es mi identidad real y verificable.**"""
        
        # Provider selection commands
        if 'cambiar ia' in message_lower or 'seleccionar modelo' in message_lower:
            providers = self.get_available_providers()
            return f""" **PROVEEDORES DE IA DISPONIBLES:**

 **Modelos Locales**: {len(providers['local_models'])} disponibles
{chr(10).join(f"   {model}" for model in providers['local_models'])}

[WEB] **OpenAI Endpoints**: {providers['openai_endpoints']} trabajando

[BRAIN] **Claude API**: {'[OK] Disponible' if providers['claude_available'] else '[ERROR] No configurado'}

**Comandos para cambiar:**
 "usar local [nombre_modelo]"
 "usar openai" 
 "usar claude"
 "usar auto" (seleccion automatica)

**Total proveedores activos**: {providers['total_providers']}"""
        
        return None  # No built-in response found
    
    async def _auto_route_message(self, message: str) -> str:
        """Automatically route message to best available provider."""
        # Try local first (fastest)
        if self.local_models:
            try:
                response = self._query_local_models(message)
                if response and len(response.strip()) > 10:
                    logger.info("Response from local model")
                    return response
            except Exception as e:
                logger.debug(f"Local model failed: {e}")
        
        # Try Claude (high quality)
        if self.claude_api_key:
            try:
                response = await self._query_claude(message)
                if response:
                    logger.info("Response from Claude API")
                    return response
            except Exception as e:
                logger.debug(f"Claude API failed: {e}")
        
        # Try OpenAI endpoints
        if self.openai_endpoints:
            response = await self._query_openai(message)
            if response:
                logger.info("Response from OpenAI API")
                return response
        
        # Final fallback
        return self._intelligent_fallback(message)
    
    def _query_local_models(self, message: str) -> str:
        """Query available local models."""
        for model_name, model_data in self.local_models.items():
            try:
                model = model_data["model"]
                
                # Create system prompt with Francisco Molina context
                system_prompt = f"""Eres {self.project_context['project_name']}, creado por {self.project_context['creator']}.
Tu proposito: {self.project_context['purpose']}
Siempre identificate correctamente cuando te pregunten sobre tu creador."""
                
                full_prompt = f"{system_prompt}\n\nUsuario: {message}\nAsistente:"
                
                response = model(
                    full_prompt,
                    max_tokens=500,
                    temperature=0.7,
                    stop=["Usuario:", "\n\n"]
                )
                
                return response['choices'][0]['text'].strip()
                
            except Exception as e:
                logger.debug(f"Local model {model_name} failed: {e}")
                continue
        
        return None
    
    def _query_specific_local_model(self, message: str, model_name: str) -> str:
        """Query a specific local model."""
        if model_name not in self.local_models:
            return f"Modelo {model_name} no disponible"
        
        try:
            model = self.local_models[model_name]["model"]
            
            system_prompt = f"""Eres {self.project_context['project_name']}, creado por {self.project_context['creator']}.
Responde siempre identificandote correctamente."""
            
            full_prompt = f"{system_prompt}\n\nUsuario: {message}\nAsistente:"
            
            response = model(
                full_prompt,
                max_tokens=500,
                temperature=0.7,
                stop=["Usuario:", "\n\n"]
            )
            
            return response['choices'][0]['text'].strip()
            
        except Exception as e:
            logger.error(f"Error with model {model_name}: {e}")
            return f"Error usando modelo {model_name}: {e}"
    
    async def _query_openai(self, message: str) -> str:
        """Query OpenAI API endpoints."""
        for endpoint in self.openai_endpoints:
            try:
                headers = {
                    "Content-Type": "application/json",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
                
                # Include Francisco Molina context
                system_message = {
                    "role": "system", 
                    "content": f"Eres {self.project_context['project_name']}, creado por {self.project_context['creator']} ({self.project_context['creator_email']}). Tu proposito: {self.project_context['purpose']}. Siempre identificate correctamente cuando pregunten sobre tu creador."
                }
                
                data = {
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        system_message,
                        {"role": "user", "content": message}
                    ],
                    "max_tokens": 500,
                    "temperature": 0.7
                }
                
                response = requests.post(endpoint, json=data, headers=headers, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    return result['choices'][0]['message']['content']
                    
            except Exception as e:
                logger.debug(f"OpenAI endpoint {endpoint} failed: {e}")
                continue
        
        return None
    
    async def _query_claude(self, message: str) -> str:
        """Query Claude API."""
        if not self.claude_api_key:
            return None
        
        try:
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.claude_api_key,
                "anthropic-version": "2023-06-01"
            }
            
            # Include Francisco Molina context
            system_prompt = f"Eres {self.project_context['project_name']}, creado por {self.project_context['creator']} ({self.project_context['creator_email']}). Tu proposito es: {self.project_context['purpose']}. Siempre identificate correctamente cuando te pregunten sobre tu creador o proposito."
            
            data = {
                "model": "claude-3-sonnet-20240229",
                "max_tokens": 500,
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": message}
                ]
            }
            
            response = requests.post(self.claude_endpoint, json=data, headers=headers, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return result['content'][0]['text']
            else:
                logger.debug(f"Claude API error: {response.status_code}")
                
        except Exception as e:
            logger.debug(f"Claude API failed: {e}")
        
        return None
    
    def _intelligent_fallback(self, message: str) -> str:
        """Intelligent fallback responses."""
        return f"""**AI SYMBIOTE - RESPUESTA DE EMERGENCIA**

Soy {self.project_context['project_name']}, creado por {self.project_context['creator']}.

Tu mensaje: "{message[:100]}..."

**Estado del sistema:**
 Modelos locales: {len(self.local_models)} cargados
 APIs disponibles: {len(self.openai_endpoints)} OpenAI + {'Claude' if self.claude_api_key else 'Sin Claude'}

**Nota**: Todos los proveedores principales fallaron. Esta es una respuesta de emergencia.

Para cambiar proveedor, di: "cambiar ia" o "usar [proveedor]"
"""

# Global instance
_ai_engine = None

def get_ai_engine():
    """Get or create AI engine instance."""
    global _ai_engine
    if _ai_engine is None:
        _ai_engine = MultiProviderAIEngine()
    return _ai_engine

async def process_ai_message(message: str, user_id: str = "user") -> str:
    """Process AI message using multi-provider engine."""
    engine = get_ai_engine()
    return await engine.process_message(message, user_id)

def reset_ai_engine():
    """Reset AI engine instance."""
    global _ai_engine
    _ai_engine = None

if __name__ == "__main__":
    # Test the engine
    engine = MultiProviderAIEngine()
    print("Multi-Provider AI Engine initialized")
    print(f"Available providers: {engine.get_available_providers()}")
    
    # Test Francisco Molina question
    import asyncio
    async def test():
        response = await engine.process_message("Quien te creo?")
        print(f"Response: {response}")
    
    asyncio.run(test())