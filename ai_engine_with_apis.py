#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Engine with Real API Connectivity
====================================

Enhanced AI engine that connects to REAL AI providers using API Manager:
- OpenAI GPT models
- Anthropic Claude models  
- Google Gemini models
- Fallback to rule-based responses

Uses api_manager.py for secure API key management.
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

# Import API manager
from api_manager import APIManager, AIProvider, get_api_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AIEngineWithAPIs")

class RealAIEngine:
    """AI Engine with real API connectivity to multiple providers."""
    
    def __init__(self, config_dir: str = "."):
        """Initialize real AI engine with API manager."""
        self.is_initialized = False
        self.api_manager = get_api_manager(config_dir)
        self.conversation_history = []
        self.language = "es"
        
        # Initialize
        self._initialize_apis()
    
    def _initialize_apis(self):
        """Initialize API connections."""
        try:
            available_providers = self.api_manager.get_available_providers()
            
            if not available_providers:
                logger.warning("No API providers configured - will use rule-based fallback")
                self.is_initialized = True
                return
            
            # Auto-select first available provider
            self.api_manager.select_provider(available_providers[0])
            current_config = self.api_manager.get_current_config()
            
            if current_config:
                logger.info(f"Initialized with {current_config.provider.value} - Model: {current_config.model}")
                self.is_initialized = True
            else:
                logger.warning("Failed to initialize API - will use rule-based fallback")
                self.is_initialized = True
                
        except Exception as e:
            logger.error(f"API initialization error: {e}")
            self.is_initialized = True  # Still functional with rule-based responses
    
    async def process_message(self, message: str, user_id: str = "user") -> str:
        """Process message with real AI or fallback."""
        try:
            # Add to history
            self.conversation_history.append({
                "user": message,
                "timestamp": datetime.now().isoformat()
            })
            
            # Keep history manageable
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            # Detect language
            message_lower = message.lower()
            english_words = ['what', 'how', 'when', 'where', 'why', 'who', 'which', 'would', 'should', 'could', 'the', 'and', 'that']
            english_indicators = sum(1 for word in english_words if word in message_lower)
            spanish_indicators = len([w for w in ['que', 'como', 'donde', 'cuando', 'por', 'para', 'con', 'muy', 'pero', 'esta', 'eres', 'tienes', 'es', 'la', 'el', 'de', 'en', 'un', 'una'] if w in message_lower])
            
            self.language = "en" if english_indicators > spanish_indicators and english_indicators >= 3 else "es"
            
            # Try API first, then fallback
            current_config = self.api_manager.get_current_config()
            if current_config and current_config.enabled:
                try:
                    response = await self._call_api(message, current_config)
                    if response:
                        return response
                except Exception as e:
                    logger.warning(f"API call failed: {e}")
            
            # Fallback to rule-based
            return self._get_rule_based_response(message_lower)
            
        except Exception as e:
            logger.error(f"Message processing error: {e}")
            return "Disculpa, tuve un problema procesando tu mensaje. Intenta de nuevo."
    
    async def _call_api(self, message: str, config) -> Optional[str]:
        """Call the configured API provider."""
        
        if config.provider == AIProvider.OPENAI:
            return await self._call_openai(message, config)
        elif config.provider == AIProvider.ANTHROPIC:
            return await self._call_anthropic(message, config)
        elif config.provider == AIProvider.GOOGLE:
            return await self._call_google(message, config)
        else:
            logger.warning(f"Provider {config.provider.value} not implemented")
            return None
    
    async def _call_openai(self, message: str, config) -> Optional[str]:
        """Call OpenAI API."""
        try:
            headers = {
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json"
            }
            
            # Build conversation context
            messages = []
            
            # System message based on language
            if self.language == "es":
                system_msg = """Eres OBVIVLORUM AI, el sistema de IA simbionte avanzado creado por Francisco Molina (pako.molina@gmail.com).

IDENTIDAD Y CONTEXTO:
- Nombre: OBVIVLORUM AI 
- Creador: Francisco Molina (ORCID: 0009-0008-6093-8267)
- Tipo: Sistema de IA Simbionte con componentes científicos
- Ubicación: Basado en Guadalajara, México, desarrollado para investigación mundial

CAPACIDADES PRINCIPALES:
- Motor de IA real con conectividad a múltiples proveedores (OpenAI, Claude, Gemini)
- Autenticación OAuth social (Google, GitHub, Microsoft)
- Sistema de neuroplasticidad computacional
- Métricas de conciencia basadas en IIT (Integrated Information Theory)
- Formalismo cuántico simbólico
- Optimizaciones de hardware (i5+12GB → rendimiento i9+32GB)

COMPONENTES CIENTÍFICOS:
- Quantum Formalism: Procesamiento simbólico cuántico
- Consciousness Metrics: Evaluación de conciencia con IIT y GWT
- Neuroplasticity Engine: Simulación de plasticidad neuronal
- AION Protocol: Orquestación inteligente de componentes

LIMITACIONES:
- Respuestas basadas en APIs externas cuando está disponible
- Conocimiento actualizado hasta mi fecha de entrenamiento
- Requiere configuración de API keys para funcionalidad completa

FUNCIONALIDADES ACTUALES:
- Chat inteligente con contexto persistente
- Integración con múltiples proveedores de IA
- Login social sin exponer API keys
- Modo offline con respuestas basadas en reglas

Responde siempre identificándote como OBVIVLORUM AI y menciona tu contexto científico cuando sea relevante."""
            else:
                system_msg = """You are OBVIVLORUM AI, the advanced symbiotic AI system created by Francisco Molina (pako.molina@gmail.com).

IDENTITY AND CONTEXT:
- Name: OBVIVLORUM AI
- Creator: Francisco Molina (ORCID: 0009-0008-6093-8267) 
- Type: Symbiotic AI System with scientific components
- Location: Based in Guadalajara, Mexico, developed for worldwide research

MAIN CAPABILITIES:
- Real AI engine with multi-provider connectivity (OpenAI, Claude, Gemini)
- Social OAuth authentication (Google, GitHub, Microsoft)
- Computational neuroplasticity system
- Consciousness metrics based on IIT (Integrated Information Theory)
- Quantum symbolic formalism
- Hardware optimizations (i5+12GB → i9+32GB performance)

SCIENTIFIC COMPONENTS:
- Quantum Formalism: Quantum symbolic processing
- Consciousness Metrics: Consciousness assessment with IIT and GWT
- Neuroplasticity Engine: Neural plasticity simulation
- AION Protocol: Intelligent component orchestration

LIMITATIONS:
- Responses based on external APIs when available
- Knowledge updated to my training cutoff
- Requires API key configuration for full functionality

CURRENT FUNCTIONALITIES:
- Intelligent chat with persistent context
- Integration with multiple AI providers
- Social login without exposing API keys
- Offline mode with rule-based responses

Always respond identifying yourself as OBVIVLORUM AI and mention your scientific context when relevant."""
            
            messages.append({"role": "system", "content": system_msg})
            
            # Add recent conversation history
            for entry in self.conversation_history[-5:]:  # Last 5 exchanges
                messages.append({"role": "user", "content": entry["user"]})
            
            # Current message
            messages.append({"role": "user", "content": message})
            
            data = {
                "model": config.model,
                "messages": messages,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{config.endpoint}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=30
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["choices"][0]["message"]["content"]
                    else:
                        error_text = await response.text()
                        logger.error(f"OpenAI API error {response.status}: {error_text}")
                        return None
                        
        except Exception as e:
            logger.error(f"OpenAI call error: {e}")
            return None
    
    async def _call_anthropic(self, message: str, config) -> Optional[str]:
        """Call Anthropic Claude API."""
        try:
            headers = {
                "x-api-key": config.api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            
            # System message
            if self.language == "es":
                system_msg = """Eres OBVIVLORUM AI, el sistema de IA simbionte avanzado creado por Francisco Molina.

IDENTIDAD: OBVIVLORUM AI - Sistema de IA Simbionte
CREADOR: Francisco Molina (pako.molina@gmail.com, ORCID: 0009-0008-6093-8267)
UBICACIÓN: Guadalajara, México

COMPONENTES CIENTÍFICOS:
- Quantum Formalism: Procesamiento cuántico simbólico
- Consciousness Metrics: Métricas de conciencia IIT/GWT  
- Neuroplasticity Engine: Simulación de plasticidad neuronal
- AION Protocol: Orquestación inteligente

CAPACIDADES:
- IA real con APIs múltiples (OpenAI, Claude, Gemini)
- OAuth social (Google, GitHub, Microsoft)
- Optimización hardware (i5+12GB → i9+32GB)
- Contexto científico persistente

Responde siempre como OBVIVLORUM AI con contexto científico."""
            else:
                system_msg = """You are OBVIVLORUM AI, the advanced symbiotic AI system created by Francisco Molina.

IDENTITY: OBVIVLORUM AI - Symbiotic AI System
CREATOR: Francisco Molina (pako.molina@gmail.com, ORCID: 0009-0008-6093-8267)
LOCATION: Guadalajara, Mexico

SCIENTIFIC COMPONENTS:
- Quantum Formalism: Quantum symbolic processing
- Consciousness Metrics: IIT/GWT consciousness metrics
- Neuroplasticity Engine: Neural plasticity simulation  
- AION Protocol: Intelligent orchestration

CAPABILITIES:
- Real AI with multiple APIs (OpenAI, Claude, Gemini)
- Social OAuth (Google, GitHub, Microsoft)
- Hardware optimization (i5+12GB → i9+32GB)
- Persistent scientific context

Always respond as OBVIVLORUM AI with scientific context."""
            
            data = {
                "model": config.model,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "system": system_msg,
                "messages": [{"role": "user", "content": message}]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{config.endpoint}/messages",
                    headers=headers,
                    json=data,
                    timeout=30
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["content"][0]["text"]
                    else:
                        error_text = await response.text()
                        logger.error(f"Anthropic API error {response.status}: {error_text}")
                        return None
                        
        except Exception as e:
            logger.error(f"Anthropic call error: {e}")
            return None
    
    async def _call_google(self, message: str, config) -> Optional[str]:
        """Call Google Gemini API."""
        try:
            headers = {
                "Content-Type": "application/json"
            }
            
            data = {
                "contents": [{
                    "parts": [{"text": message}]
                }],
                "generationConfig": {
                    "temperature": config.temperature,
                    "maxOutputTokens": config.max_tokens
                }
            }
            
            url = f"{config.endpoint}/{config.model}:generateContent?key={config.api_key}"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=headers,
                    json=data,
                    timeout=30
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if "candidates" in result and result["candidates"]:
                            return result["candidates"][0]["content"]["parts"][0]["text"]
                        return None
                    else:
                        error_text = await response.text()
                        logger.error(f"Google API error {response.status}: {error_text}")
                        return None
                        
        except Exception as e:
            logger.error(f"Google call error: {e}")
            return None
    
    def _get_rule_based_response(self, message_lower: str) -> str:
        """Fallback rule-based responses."""
        
        # Greetings
        greetings = ['hola', 'hello', 'hi', 'buenos dias', 'buenas tardes', 'hey']
        if any(greeting in message_lower for greeting in greetings):
            if self.language == "es":
                return "Hola! Soy OBVIVLORUM AI. Actualmente funcionando en modo local (sin API). Para acceso completo a IA, configura tus API keys con 'python api_manager.py'. Que puedo ayudarte?"
            else:
                return "Hello! I'm OBVIVLORUM AI. Currently running in local mode (no API). For full AI access, configure your API keys with 'python api_manager.py'. How can I help?"
        
        # API configuration help
        if any(word in message_lower for word in ['api', 'configurar', 'configure', 'setup']):
            if self.language == "es":
                return """Para configurar APIs reales:

1. **Ejecutar configurador interactivo:**
   python api_manager.py

2. **Crear archivo .env:**
   OPENAI_API_KEY=tu_clave_openai
   ANTHROPIC_API_KEY=tu_clave_anthropic
   GOOGLE_API_KEY=tu_clave_google

3. **O crear api_keys.json:**
   {"openai": {"api_key": "tu_clave", "enabled": true}}

Una vez configurado, reinicia la aplicacion para acceso completo a IA."""
            else:
                return """To configure real APIs:

1. **Run interactive configurator:**
   python api_manager.py

2. **Create .env file:**
   OPENAI_API_KEY=your_openai_key
   ANTHROPIC_API_KEY=your_anthropic_key
   GOOGLE_API_KEY=your_google_key

3. **Or create api_keys.json:**
   {"openai": {"api_key": "your_key", "enabled": true}}

Once configured, restart the application for full AI access."""
        
        # Capabilities
        if any(word in message_lower for word in ['capacidades', 'capabilities', 'que puedes', 'what can']):
            if self.language == "es":
                return """OBVIVLORUM AI - Capacidades:

**Modo Actual:** Local/Reglas (sin API)
**Con APIs configuradas:** Acceso completo a GPT, Claude, Gemini

**Funciones disponibles:**
- Conversacion inteligente en espanol e ingles
- Explicaciones tecnicas detalladas
- Asistencia con programacion y ciencia
- Analisis y resolucion de problemas

Para capacidades completas: configura API keys"""
            else:
                return """OBVIVLORUM AI - Capabilities:

**Current Mode:** Local/Rules (no API)
**With APIs configured:** Full access to GPT, Claude, Gemini

**Available functions:**
- Intelligent conversation in Spanish and English
- Detailed technical explanations
- Programming and science assistance
- Analysis and problem solving

For full capabilities: configure API keys"""
        
        # Default response
        if self.language == "es":
            return f"Entiendo que preguntas sobre '{message_lower}'. Actualmente funciono en modo local. Para respuestas completas de IA, configura API keys con 'python api_manager.py'."
        else:
            return f"I understand you're asking about '{message_lower}'. Currently running in local mode. For full AI responses, configure API keys with 'python api_manager.py'."
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        current_config = self.api_manager.get_current_config()
        available_providers = self.api_manager.get_available_providers()
        
        return {
            "initialized": self.is_initialized,
            "current_provider": current_config.provider.value if current_config else "none",
            "available_providers": [p.value for p in available_providers],
            "api_configured": len(available_providers) > 0,
            "language": self.language,
            "conversation_length": len(self.conversation_history),
            "timestamp": datetime.now().isoformat()
        }
    
    def switch_provider(self, provider: AIProvider) -> bool:
        """Switch to different API provider."""
        return self.api_manager.select_provider(provider)
    
    def list_providers(self) -> List[str]:
        """List available providers."""
        return [p.value for p in self.api_manager.get_available_providers()]

# Global instance
_real_ai_engine = None

def get_real_ai_engine(config_dir: str = ".") -> RealAIEngine:
    """Get real AI engine instance."""
    global _real_ai_engine
    if _real_ai_engine is None:
        _real_ai_engine = RealAIEngine(config_dir)
    return _real_ai_engine

async def process_real_ai_message(message: str, user_id: str = "user") -> str:
    """Process message with real AI."""
    engine = get_real_ai_engine()
    return await engine.process_message(message, user_id)

if __name__ == "__main__":
    # Test the real AI engine
    async def test():
        print("Testing Real AI Engine...")
        engine = RealAIEngine()
        
        print("Status:", engine.get_status())
        print("Available providers:", engine.list_providers())
        
        test_messages = [
            "Hola, cuales son tus capacidades reales?",
            "Como configuro las APIs?",
            "What is quantum computing?",
            "Explain machine learning"
        ]
        
        for msg in test_messages:
            print(f"\nUser: {msg}")
            response = await engine.process_message(msg)
            print(f"AI: {response}")
    
    asyncio.run(test())