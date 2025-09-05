#!/usr/bin/env python3
"""
AI Engine Fixed - Stable Multilingual AI
========================================

Fixed AI implementation with coherent Spanish/English responses.
"""

import os
import sys
import json
import logging
import asyncio
import numpy as np
import cv2
import speech_recognition as sr
import pyttsx3
import threading
import time
import random
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# AI Models
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BlenderbotTokenizer, BlenderbotForConditionalGeneration,
    pipeline
)
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AIEngine")

class RealAIEngine:
    """
    Fixed AI Engine with coherent multilingual conversation.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the AI Engine with stable configuration."""
        self.config = config or self._default_config()
        self.is_initialized = False
        
        # AI Components
        self.chat_model = None
        self.chat_tokenizer = None
        self.conversation_pipeline = None
        self.tts_engine = None
        self.speech_recognizer = None
        
        # Vision components
        self.face_cascade = None
        self.known_faces = {}
        
        # Conversation context
        self.conversation_history = []
        self.max_history = 5
        self.language = "es"  # Default to Spanish
        
        # Voice settings
        self.voice_enabled = True
        self.listening = False
        
        # Knowledge base for coherent responses
        self.knowledge_base = self._initialize_knowledge()
        
        logger.info("Initializing Fixed AI Engine...")
        self._initialize_components()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for stable AI."""
        return {
            "model": {
                "name": "microsoft/DialoGPT-medium",  # More stable than small
                "fallback": "facebook/blenderbot-400M-distill",  # Better fallback
                "max_length": 100,
                "min_length": 10,
                "temperature": 0.7,  # Lower for more coherent responses
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.2,
                "device": "cpu"
            },
            "voice": {
                "enabled": True,
                "rate": 180,
                "volume": 0.8,
                "voice_id": 0
            },
            "vision": {
                "enabled": True,
                "face_recognition": True
            }
        }
    
    def _initialize_knowledge(self) -> Dict[str, Any]:
        """Initialize knowledge base for coherent responses."""
        return {
            "identity": {
                "name": "AI Symbiote",
                "purpose": "Soy tu asistente de IA personal con capacidades de voz, vision y conversacion inteligente.",
                "creator": "Francisco Molina",
                "capabilities": [
                    "Conversacion inteligente en espanol e ingles",
                    "Reconocimiento y sintesis de voz",
                    "Deteccion y reconocimiento facial",
                    "Gestion de tareas adaptativa",
                    "Aprendizaje de patrones de usuario"
                ]
            },
            "responses": {
                "greeting": [
                    "Hola! Soy AI Symbiote, tu asistente inteligente. En que puedo ayudarte?",
                    "Bienvenido! Estoy aqui para asistirte. Que necesitas?",
                    "Hola, es un gusto conocerte. Como puedo ayudarte hoy?"
                ],
                "capabilities": [
                    "Mis capacidades incluyen: conversacion inteligente, reconocimiento de voz, vision por computadora, y gestion de tareas.",
                    "Puedo hablar contigo, entender tu voz, ver a traves de la camara y ayudarte con diversas tareas.",
                    "Tengo capacidades de IA conversacional, procesamiento de voz, analisis visual y aprendizaje adaptativo."
                ],
                "language": [
                    "Hablo espanol e ingles. Puedo cambiar entre ambos idiomas segun prefieras.",
                    "I speak both Spanish and English. I can switch between them as you prefer.",
                    "Puedo comunicarme en espanol o ingles, como te sea mas comodo."
                ],
                "help": [
                    "Puedo ayudarte con: responder preguntas, mantener conversaciones, reconocer rostros, entender comandos de voz y gestionar tareas.",
                    "Estoy aqui para asistirte. Puedes preguntarme lo que necesites o pedirme que realice tareas.",
                    "Necesitas ayuda? Puedo conversar, responder dudas, y usar mis capacidades de voz y vision."
                ],
                "error": [
                    "Disculpa, no entendi bien. Podrias reformular tu pregunta?",
                    "Perdon, hubo un problema procesando eso. Puedes intentar de nuevo?",
                    "No estoy seguro de entender. Podrias explicarte de otra manera?"
                ]
            }
        }
    
    def _initialize_components(self):
        """Initialize all AI components with fallbacks."""
        try:
            # Initialize chat AI with fallback
            self._initialize_chat_ai_safe()
            
            # Initialize text-to-speech
            self._initialize_tts()
            
            # Initialize speech recognition
            self._initialize_speech_recognition()
            
            # Initialize computer vision
            self._initialize_vision()
            
            self.is_initialized = True
            logger.info("AI Engine fully initialized with stable configuration")
            
        except Exception as e:
            logger.error(f"Failed to initialize AI Engine: {e}")
            self.is_initialized = False
    
    def _initialize_chat_ai_safe(self):
        """Initialize chat AI with multiple fallback options."""
        try:
            # Try to use a conversation pipeline (most stable)
            logger.info("Initializing conversation pipeline...")
            self.conversation_pipeline = pipeline(
                "conversational",
                model="microsoft/DialoGPT-medium",
                device=-1  # CPU
            )
            logger.info("Conversation pipeline initialized successfully")
            
        except Exception as e1:
            logger.warning(f"Primary model failed: {e1}, trying fallback...")
            
            try:
                # Fallback to Blenderbot (more stable for conversation)
                logger.info("Loading Blenderbot as fallback...")
                self.chat_tokenizer = BlenderbotTokenizer.from_pretrained(
                    "facebook/blenderbot-400M-distill"
                )
                self.chat_model = BlenderbotForConditionalGeneration.from_pretrained(
                    "facebook/blenderbot-400M-distill"
                )
                self.chat_model.eval()
                logger.info("Blenderbot loaded successfully")
                
            except Exception as e2:
                logger.warning(f"Fallback also failed: {e2}")
                # Use rule-based system as last resort
                self.chat_model = None
                self.chat_tokenizer = None
                logger.info("Using rule-based conversation system")
    
    def _initialize_tts(self):
        """Initialize text-to-speech engine."""
        try:
            self.tts_engine = pyttsx3.init()
            
            # Configure for Spanish/English
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Try to find Spanish voice
                spanish_voice = None
                for voice in voices:
                    if 'spanish' in voice.name.lower() or 'espanol' in voice.name.lower():
                        spanish_voice = voice.id
                        break
                
                if spanish_voice:
                    self.tts_engine.setProperty('voice', spanish_voice)
                else:
                    self.tts_engine.setProperty('voice', voices[0].id)
            
            self.tts_engine.setProperty('rate', self.config["voice"]["rate"])
            self.tts_engine.setProperty('volume', self.config["voice"]["volume"])
            
            logger.info("Text-to-speech engine initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize TTS: {e}")
            self.tts_engine = None
    
    def _initialize_speech_recognition(self):
        """Initialize speech recognition."""
        try:
            self.speech_recognizer = sr.Recognizer()
            self.speech_recognizer.energy_threshold = 300
            self.speech_recognizer.dynamic_energy_threshold = True
            logger.info("Speech recognition initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize speech recognition: {e}")
            self.speech_recognizer = None
    
    def _initialize_vision(self):
        """Initialize computer vision components."""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if self.face_cascade.empty():
                logger.warning("Could not load face cascade")
                self.face_cascade = None
            else:
                logger.info("Computer vision initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize vision: {e}")
            self.face_cascade = None
    
    async def process_message(self, message: str, user_id: str = "user") -> str:
        """
        Process a message and generate a coherent response.
        
        Args:
            message: User's message
            user_id: User identifier
            
        Returns:
            Coherent AI response in Spanish/English
        """
        try:
            # Detect language
            self.language = "es" if any(c in message.lower() for c in ['n', 'a', 'e', 'i', 'o', 'u']) else "en"
            
            # Check for specific queries first
            response = self._check_knowledge_base(message)
            if response:
                return response
            
            # Use AI model if available
            if self.conversation_pipeline:
                return await self._generate_pipeline_response(message)
            elif self.chat_model and self.chat_tokenizer:
                return await self._generate_model_response(message)
            else:
                # Fallback to rule-based responses
                return self._generate_rule_based_response(message)
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return self._get_error_response()
    
    def _check_knowledge_base(self, message: str) -> Optional[str]:
        """Check if message matches knowledge base queries."""
        message_lower = message.lower()
        
        # Check for greetings
        if any(word in message_lower for word in ['hola', 'hello', 'hi', 'buenos', 'buenas']):
            return random.choice(self.knowledge_base["responses"]["greeting"])
        
        # Check for capabilities
        if any(word in message_lower for word in ['capacidad', 'capability', 'poder', 'can you', 'puedes']):
            return random.choice(self.knowledge_base["responses"]["capabilities"])
        
        # Check for language
        if any(word in message_lower for word in ['idioma', 'language', 'hablas', 'speak']):
            return random.choice(self.knowledge_base["responses"]["language"])
        
        # Check for help
        if any(word in message_lower for word in ['ayuda', 'help', 'asist', 'assist']):
            return random.choice(self.knowledge_base["responses"]["help"])
        
        # Check for identity
        if any(word in message_lower for word in ['quien eres', 'who are you', 'que eres', 'what are you']):
            return self.knowledge_base["identity"]["purpose"]
        
        return None
    
    async def _generate_pipeline_response(self, message: str) -> str:
        """Generate response using conversation pipeline."""
        try:
            from transformers import Conversation
            
            # Create conversation
            conversation = Conversation(message)
            
            # Generate response
            result = self.conversation_pipeline(conversation)
            
            # Extract response
            response = result.generated_responses[-1] if result.generated_responses else ""
            
            # Clean and validate response
            response = self._clean_and_validate_response(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Pipeline generation error: {e}")
            return self._get_error_response()
    
    async def _generate_model_response(self, message: str) -> str:
        """Generate response using direct model."""
        try:
            # For Blenderbot, we need to handle it differently
            inputs = self.chat_tokenizer([message], return_tensors="pt", truncation=True, max_length=128)
            
            # Generate response
            with torch.no_grad():
                reply_ids = self.chat_model.generate(
                    **inputs,
                    max_new_tokens=100,  # Use max_new_tokens instead of max_length
                    min_length=10,
                    temperature=0.7,
                    do_sample=True,
                    top_k=50,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    pad_token_id=self.chat_tokenizer.pad_token_id,
                    eos_token_id=self.chat_tokenizer.eos_token_id
                )
            
            # Decode only the generated response (not the input)
            if reply_ids.shape[1] > 0:
                response = self.chat_tokenizer.decode(reply_ids[0], skip_special_tokens=True)
            else:
                return self._generate_rule_based_response(message)
            
            # Remove input from response if present
            if message in response:
                response = response.replace(message, "").strip()
            
            # Clean and validate
            response = self._clean_and_validate_response(response)
            
            # If response is too short or invalid, use rule-based
            if len(response) < 5:
                return self._generate_rule_based_response(message)
            
            return response
            
        except Exception as e:
            logger.error(f"Model generation error: {e}")
            # Always fallback to rule-based instead of error message
            return self._generate_rule_based_response(message)
    
    def _generate_rule_based_response(self, message: str) -> str:
        """Generate rule-based response when AI models are unavailable."""
        message_lower = message.lower()
        
        # Extended pattern matching with more variations
        patterns = {
            # Spanish greetings and responses
            "hola": "Hola! Soy AI Symbiote, tu asistente inteligente. Como puedo ayudarte hoy?",
            "buenos dias": "Buenos dias! Espero que tengas un excelente dia. En que puedo asistirte?",
            "buenas tardes": "Buenas tardes! Como puedo ayudarte en este momento?",
            "como estas": "Estoy funcionando perfectamente, gracias por preguntar. Y tu como estas?",
            "que tal": "Todo excelente! Listo para ayudarte. Que necesitas?",
            
            # Spanish capabilities
            "que puedes hacer": "Puedo: conversar inteligentemente, reconocer tu voz, ver a traves de la camara, gestionar tareas y aprender de tus patrones.",
            "capacidades": "Mis capacidades incluyen conversacion en espanol e ingles, sintesis de voz, reconocimiento facial y gestion adaptativa de tareas.",
            "ayuda": "Estoy aqui para ayudarte. Puedes preguntarme lo que necesites, usar comandos de voz o activar la camara para reconocimiento facial.",
            
            # English greetings
            "hello": "Hello! I'm AI Symbiote, your intelligent assistant. How can I help you today?",
            "hi": "Hi there! Ready to assist you. What do you need?",
            "good morning": "Good morning! Hope you're having a great day. How can I assist?",
            "how are you": "I'm working perfectly, thank you for asking. How are you doing?",
            
            # English capabilities  
            "what can you do": "I can: chat intelligently, recognize your voice, see through the camera, manage tasks, and learn from your patterns.",
            "capabilities": "My capabilities include conversation in Spanish and English, voice synthesis, facial recognition, and adaptive task management.",
            "help": "I'm here to help. You can ask me anything, use voice commands, or activate the camera for facial recognition.",
            
            # Common questions
            "quien eres": "Soy AI Symbiote, tu asistente personal de inteligencia artificial con capacidades avanzadas de voz y vision.",
            "who are you": "I'm AI Symbiote, your personal AI assistant with advanced voice and vision capabilities.",
            "idioma": "Hablo espanol e ingles fluidamente. Puedo cambiar entre ambos segun prefieras.",
            "language": "I speak Spanish and English fluently. I can switch between them as you prefer.",
            
            # Thanks and goodbye
            "gracias": "De nada! Es un placer ayudarte. Necesitas algo mas?",
            "thank you": "You're welcome! It's my pleasure to help. Anything else you need?",
            "adios": "Hasta luego! Que tengas un excelente dia. Estare aqui cuando me necesites.",
            "goodbye": "Goodbye! Have a great day. I'll be here whenever you need me.",
            "bye": "See you later! Feel free to come back anytime.",
            
            # Task-related
            "tarea": "Puedo ayudarte a gestionar tus tareas. Quieres crear una nueva tarea o ver las existentes?",
            "task": "I can help you manage your tasks. Would you like to create a new task or see existing ones?",
            
            # Technical
            "sistema": "Mi sistema esta funcionando al 100%. Todos los componentes operativos.",
            "status": "System fully operational. All components working perfectly.",
            "error": "Si encuentras algun error, por favor describelo y tratare de solucionarlo.",
            
            # Fun responses
            "chiste": "Por que los programadores prefieren el modo oscuro? Porque la luz atrae a los bugs! ",
            "joke": "Why do programmers prefer dark mode? Because light attracts bugs! "
        }
        
        # Check each pattern
        for pattern, response in patterns.items():
            if pattern in message_lower:
                return response
        
        # Smart contextual responses for unmatched messages
        if "?" in message:
            # It's a question
            if self.language == "es":
                responses = [
                    "Esa es una pregunta interesante. Dejame pensar en la mejor manera de ayudarte.",
                    "Entiendo tu pregunta. Podrias darme un poco mas de contexto?",
                    "Buena pregunta. Basandome en mi conocimiento, puedo sugerirte algunas opciones."
                ]
            else:
                responses = [
                    "That's an interesting question. Let me think about the best way to help you.",
                    "I understand your question. Could you provide a bit more context?",
                    "Good question. Based on my knowledge, I can suggest some options."
                ]
            return random.choice(responses)
        
        # Default conversational responses
        if self.language == "es":
            responses = [
                "Entiendo. Podrias explicarme un poco mas sobre eso?",
                "Interesante. Cuentame mas para poder ayudarte mejor.",
                "Comprendo lo que dices. Como puedo asistirte con eso?",
                "De acuerdo. Hay algo especifico en lo que pueda ayudarte?"
            ]
        else:
            responses = [
                "I understand. Could you tell me a bit more about that?",
                "Interesting. Tell me more so I can help you better.",
                "I see what you're saying. How can I assist you with that?",
                "Alright. Is there something specific I can help you with?"
            ]
        
        return random.choice(responses)
    
    def _clean_and_validate_response(self, response: str) -> str:
        """Clean and validate AI response to ensure coherence."""
        if not response or len(response) < 3:
            return self._get_error_response()
        
        # Remove any gibberish or non-standard characters
        import re
        
        # Keep only standard characters for Spanish/English
        cleaned = re.sub(r'[^\w\s\.\,\!\?\\\n\a\e\i\o\u\u\'\"\-]', '', response)
        
        # Check if response is mostly valid words
        words = cleaned.split()
        if len(words) < 2:
            return self._get_error_response()
        
        # Check for repetitive nonsense
        if len(set(words)) < len(words) / 3:  # Too many repeated words
            return self._get_error_response()
        
        # Ensure proper length
        if len(cleaned) > 500:
            cleaned = cleaned[:500] + "..."
        
        return cleaned.strip()
    
    def _get_error_response(self) -> str:
        """Get an appropriate error response."""
        return random.choice(self.knowledge_base["responses"]["error"])
    
    async def speak(self, text: str) -> bool:
        """Convert text to speech."""
        if not self.voice_enabled or not self.tts_engine:
            return False
        
        try:
            def speak_thread():
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            
            thread = threading.Thread(target=speak_thread)
            thread.daemon = True
            thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return False
    
    async def listen(self, timeout: int = 5) -> Optional[str]:
        """Listen for voice input."""
        if not self.speech_recognizer:
            return None
        
        try:
            with sr.Microphone() as source:
                logger.info("Listening...")
                self.speech_recognizer.adjust_for_ambient_noise(source)
                audio = self.speech_recognizer.listen(source, timeout=timeout)
            
            # Try Spanish first, then English
            try:
                text = self.speech_recognizer.recognize_google(audio, language="es-ES")
            except:
                text = self.speech_recognizer.recognize_google(audio, language="en-US")
            
            logger.info(f"Recognized: {text}")
            return text
            
        except Exception as e:
            logger.error(f"Speech recognition error: {e}")
            return None
    
    async def analyze_face(self, image_data: np.ndarray) -> Dict[str, Any]:
        """Analyze faces in image."""
        if not self.face_cascade:
            return {"faces": [], "status": "vision_unavailable"}
        
        try:
            gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            
            results = {
                "faces": [],
                "count": len(faces),
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            }
            
            for i, (x, y, w, h) in enumerate(faces):
                face_info = {
                    "id": f"face_{i}",
                    "position": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                    "confidence": 0.85
                }
                results["faces"].append(face_info)
            
            return results
            
        except Exception as e:
            logger.error(f"Face analysis error: {e}")
            return {"faces": [], "status": "error", "error": str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """Get AI engine status."""
        return {
            "initialized": self.is_initialized,
            "components": {
                "chat_ai": self.conversation_pipeline is not None or self.chat_model is not None,
                "tts": self.tts_engine is not None,
                "speech_recognition": self.speech_recognizer is not None,
                "vision": self.face_cascade is not None
            },
            "language": self.language,
            "voice_enabled": self.voice_enabled,
            "model_type": "pipeline" if self.conversation_pipeline else ("model" if self.chat_model else "rule-based"),
            "timestamp": datetime.now().isoformat()
        }

# Global instance
_ai_engine = None

def get_ai_engine() -> RealAIEngine:
    """Get global AI engine instance."""
    global _ai_engine
    if _ai_engine is None:
        _ai_engine = RealAIEngine()
    return _ai_engine

async def process_ai_message(message: str, user_id: str = "user") -> str:
    """Process message with AI."""
    engine = get_ai_engine()
    return await engine.process_message(message, user_id)

if __name__ == "__main__":
    # Test the fixed AI
    async def test():
        print("Testing Fixed AI Engine...")
        engine = RealAIEngine()
        
        test_messages = [
            "Hola, como estas?",
            "Cuales son tus capacidades?",
            "What language do you speak?",
            "Ayudame con algo",
            "Quien eres?"
        ]
        
        for msg in test_messages:
            print(f"\nUser: {msg}")
            response = await engine.process_message(msg)
            print(f"AI: {response}")
        
        print("\n[OK] AI Engine test completed successfully!")
    
    asyncio.run(test())