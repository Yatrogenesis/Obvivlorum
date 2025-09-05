#!/usr/bin/env python3
"""
AI Engine - Real AI Implementation
==================================

Real AI implementation using local models (DialoGPT/GPT2) with full voice 
and vision capabilities.
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
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# AI Models
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    pipeline, GPT2LMHeadModel, GPT2Tokenizer
)
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AIEngine")

class RealAIEngine:
    """
    Real AI Engine with conversational AI, voice synthesis, and computer vision.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the AI Engine with real capabilities."""
        self.config = config or self._default_config()
        self.is_initialized = False
        
        # AI Components
        self.chat_model = None
        self.chat_tokenizer = None
        self.tts_engine = None
        self.speech_recognizer = None
        
        # Vision components
        self.face_cascade = None
        self.known_faces = {}
        
        # Conversation context
        self.conversation_history = []
        self.max_history = 10
        
        # Voice settings
        self.voice_enabled = True
        self.listening = False
        
        logger.info("Initializing Real AI Engine...")
        self._initialize_components()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for AI Engine."""
        return {
            "model": {
                "name": "microsoft/DialoGPT-small",  # Lightweight conversational AI
                "max_length": 200,
                "temperature": 0.8,
                "top_p": 0.9,
                "device": "cpu"  # Use CPU for compatibility
            },
            "voice": {
                "enabled": True,
                "rate": 180,
                "volume": 0.8,
                "voice_id": 0
            },
            "vision": {
                "enabled": True,
                "face_recognition": True,
                "emotion_detection": False  # Can be added later
            },
            "memory": {
                "remember_users": True,
                "context_window": 5,
                "save_conversations": True
            }
        }
    
    def _initialize_components(self):
        """Initialize all AI components."""
        try:
            # Initialize chat AI
            self._initialize_chat_ai()
            
            # Initialize text-to-speech
            self._initialize_tts()
            
            # Initialize speech recognition
            self._initialize_speech_recognition()
            
            # Initialize computer vision
            self._initialize_vision()
            
            self.is_initialized = True
            logger.info("AI Engine fully initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize AI Engine: {e}")
            self.is_initialized = False
    
    def _initialize_chat_ai(self):
        """Initialize the conversational AI model."""
        try:
            model_name = self.config["model"]["name"]
            logger.info(f"Loading AI model: {model_name}")
            
            # Load tokenizer and model
            self.chat_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.chat_model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Add pad token if it doesn't exist
            if self.chat_tokenizer.pad_token is None:
                self.chat_tokenizer.pad_token = self.chat_tokenizer.eos_token
            
            # Set to evaluation mode
            self.chat_model.eval()
            
            logger.info("Chat AI model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize chat AI: {e}")
            # Fallback to a simpler model
            try:
                logger.info("Falling back to GPT-2 model...")
                self.chat_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
                self.chat_model = GPT2LMHeadModel.from_pretrained('gpt2')
                self.chat_tokenizer.pad_token = self.chat_tokenizer.eos_token
                logger.info("GPT-2 model loaded as fallback")
            except Exception as e2:
                logger.error(f"Fallback model also failed: {e2}")
    
    def _initialize_tts(self):
        """Initialize text-to-speech engine."""
        try:
            self.tts_engine = pyttsx3.init()
            
            # Configure voice settings
            voices = self.tts_engine.getProperty('voices')
            if voices:
                voice_id = self.config["voice"]["voice_id"]
                if voice_id < len(voices):
                    self.tts_engine.setProperty('voice', voices[voice_id].id)
            
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
            # Load face detection cascade
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
        Process a text message and generate an AI response.
        
        Args:
            message: User's message
            user_id: User identifier
            
        Returns:
            AI-generated response
        """
        if not self.is_initialized or not self.chat_model:
            return "I apologize, but my AI systems are currently offline. Please try again later."
        
        try:
            # Add to conversation history
            self.conversation_history.append(f"User: {message}")
            
            # Keep history manageable
            if len(self.conversation_history) > self.max_history:
                self.conversation_history = self.conversation_history[-self.max_history:]
            
            # Build conversation context
            context = "\n".join(self.conversation_history)
            context += "\nAI:"
            
            # Tokenize input
            inputs = self.chat_tokenizer.encode(context, return_tensors="pt")
            
            # Generate response
            with torch.no_grad():
                outputs = self.chat_model.generate(
                    inputs,
                    max_length=inputs.shape[1] + self.config["model"]["max_length"],
                    temperature=self.config["model"]["temperature"],
                    top_p=self.config["model"]["top_p"],
                    do_sample=True,
                    pad_token_id=self.chat_tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            # Decode response
            response = self.chat_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the AI's response
            if "AI:" in response:
                ai_response = response.split("AI:")[-1].strip()
            else:
                ai_response = response.strip()
            
            # Clean up the response
            ai_response = self._clean_response(ai_response)
            
            # Add to conversation history
            self.conversation_history.append(f"AI: {ai_response}")
            
            logger.info(f"Generated response for user {user_id}")
            return ai_response
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return "I encountered an error processing your message. Let me try to help you differently."
    
    def _clean_response(self, response: str) -> str:
        """Clean and improve the AI response."""
        # Remove repetitive text
        lines = response.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and line not in cleaned_lines:
                cleaned_lines.append(line)
        
        cleaned_response = ' '.join(cleaned_lines)
        
        # Remove user prompts that might have leaked
        if "User:" in cleaned_response:
            cleaned_response = cleaned_response.split("User:")[0].strip()
        
        # Ensure response isn't empty
        if not cleaned_response or len(cleaned_response) < 5:
            cleaned_response = "I understand. How can I help you further?"
        
        return cleaned_response[:500]  # Limit length
    
    async def speak(self, text: str) -> bool:
        """
        Convert text to speech.
        
        Args:
            text: Text to speak
            
        Returns:
            Success status
        """
        if not self.voice_enabled or not self.tts_engine:
            return False
        
        try:
            # Run TTS in a separate thread to avoid blocking
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
        """
        Listen for voice input and convert to text.
        
        Args:
            timeout: Listening timeout in seconds
            
        Returns:
            Transcribed text or None
        """
        if not self.speech_recognizer:
            return None
        
        try:
            with sr.Microphone() as source:
                logger.info("Listening for voice input...")
                self.speech_recognizer.adjust_for_ambient_noise(source)
                audio = self.speech_recognizer.listen(source, timeout=timeout)
            
            # Use Google Speech Recognition
            text = self.speech_recognizer.recognize_google(audio)
            logger.info(f"Recognized speech: {text}")
            return text
            
        except sr.UnknownValueError:
            logger.warning("Could not understand audio")
            return None
        except sr.RequestError as e:
            logger.error(f"Speech recognition error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in speech recognition: {e}")
            return None
    
    async def analyze_face(self, image_data: np.ndarray) -> Dict[str, Any]:
        """
        Analyze faces in an image.
        
        Args:
            image_data: OpenCV image array
            
        Returns:
            Analysis results
        """
        if not self.face_cascade:
            return {"faces": [], "status": "vision_unavailable"}
        
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
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
                    "confidence": 0.8  # Placeholder confidence
                }
                results["faces"].append(face_info)
            
            logger.info(f"Detected {len(faces)} face(s)")
            return results
            
        except Exception as e:
            logger.error(f"Face analysis error: {e}")
            return {"faces": [], "status": "error", "error": str(e)}
    
    def set_voice_enabled(self, enabled: bool):
        """Enable or disable voice output."""
        self.voice_enabled = enabled
        logger.info(f"Voice output {'enabled' if enabled else 'disabled'}")
    
    def clear_conversation(self):
        """Clear conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current AI engine status."""
        return {
            "initialized": self.is_initialized,
            "components": {
                "chat_ai": self.chat_model is not None,
                "tts": self.tts_engine is not None,
                "speech_recognition": self.speech_recognizer is not None,
                "vision": self.face_cascade is not None
            },
            "voice_enabled": self.voice_enabled,
            "conversation_length": len(self.conversation_history),
            "timestamp": datetime.now().isoformat()
        }

# Global AI Engine instance
_ai_engine = None

def get_ai_engine() -> RealAIEngine:
    """Get the global AI engine instance."""
    global _ai_engine
    if _ai_engine is None:
        _ai_engine = RealAIEngine()
    return _ai_engine

async def process_ai_message(message: str, user_id: str = "user") -> str:
    """Convenience function to process a message with the AI."""
    engine = get_ai_engine()
    return await engine.process_message(message, user_id)

def initialize_ai():
    """Initialize the global AI engine."""
    global _ai_engine
    _ai_engine = RealAIEngine()
    return _ai_engine.is_initialized

if __name__ == "__main__":
    # Test the AI engine
    async def test_ai():
        engine = RealAIEngine()
        
        if engine.is_initialized:
            print("AI Engine initialized successfully!")
            
            # Test conversation
            response = await engine.process_message("Hello, how are you?")
            print(f"AI: {response}")
            
            # Test TTS if available
            if engine.tts_engine:
                await engine.speak("Hello! I can speak now.")
            
            print("AI Engine test completed.")
        else:
            print("Failed to initialize AI Engine")
    
    asyncio.run(test_ai())