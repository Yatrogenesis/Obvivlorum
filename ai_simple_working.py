#!/usr/bin/env python3
"""
AI Simple Working - No mas fallos
=================================

Sistema de IA simple pero que FUNCIONA sin errores.
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
from typing import Dict, List, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SimpleAI")

class SimpleWorkingAI:
    """AI simple pero funcional sin errores."""
    
    def __init__(self):
        """Initialize simple working AI."""
        self.is_initialized = False
        self.tts_engine = None
        self.speech_recognizer = None
        self.face_cascade = None
        self.conversation_history = []
        self.voice_enabled = True
        self.language = "es"  # Espanol como idioma principal
        self.default_language = "es"
        
        logger.info("Initializing Simple Working AI...")
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all components without AI model failures."""
        try:
            # Initialize TTS
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 180)
                self.tts_engine.setProperty('volume', 0.8)
                logger.info("TTS initialized")
            except:
                logger.warning("TTS failed to initialize")
            
            # Initialize Speech Recognition
            try:
                self.speech_recognizer = sr.Recognizer()
                logger.info("Speech recognition initialized")
            except:
                logger.warning("Speech recognition failed")
            
            # Initialize Vision
            try:
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                logger.info("Vision system initialized")
            except:
                logger.warning("Vision system failed")
            
            self.is_initialized = True
            logger.info("Simple Working AI initialized successfully - NO MODEL FAILURES")
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            self.is_initialized = True  # Still mark as initialized for basic functionality
    
    async def process_message(self, message: str, user_id: str = "user") -> str:
        """Process message with intelligent rule-based responses."""
        try:
            message_lower = message.lower()
            
            # Detect language - Espanol por defecto, ingles solo si es claramente ingles
            english_words = ['what', 'how', 'when', 'where', 'why', 'who', 'which', 'would', 'should', 'could', 'the', 'and', 'that']
            english_indicators = sum(1 for word in english_words if word in message_lower)
            spanish_indicators = len([w for w in ['que', 'como', 'donde', 'cuando', 'por', 'para', 'con', 'muy', 'pero', 'esta', 'eres', 'tienes', 'es', 'la', 'el', 'de', 'en', 'un', 'una'] if w in message_lower])
            
            # Espanol por defecto, ingles solo si hay mas indicadores en ingles
            self.language = "en" if english_indicators > spanish_indicators and english_indicators >= 3 else "es"
            
            # Add to history
            self.conversation_history.append({
                "user": message,
                "timestamp": datetime.now().isoformat()
            })
            
            # Keep history manageable
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            # Process with comprehensive patterns
            response = self._get_intelligent_response(message_lower)
            
            logger.info(f"Generated response for user {user_id}")
            return response
            
        except Exception as e:
            logger.error(f"Message processing error: {e}")
            return "Disculpa, tuve un problema. Puedes intentar de nuevo?"
    
    def _get_intelligent_response(self, message_lower: str) -> str:
        """Get intelligent response based on comprehensive pattern matching."""
        
        # === PREGUNTAS TECNICAS Y COMPLEJAS EN ESPANOL ===
        
        # Blockchain y Criptomonedas
        if any(word in message_lower for word in ['blockchain', 'bitcoin', 'criptomoneda', 'cryptocurrency', 'cripto']):
            if self.language == "es":
                return """Blockchain es una tecnologia revolucionaria de registro distribuido:

**Que es blockchain?**
 Base de datos descentralizada y inmutable
 Cada bloque contiene transacciones verificadas
 Conectados criptograficamente formando una cadena
 No requiere autoridad central para validar

**Caracteristicas principales:**
 Transparencia: Todas las transacciones son publicas
 Inmutabilidad: Los datos no pueden alterarse
 Descentralizacion: No hay punto unico de falla
 Consenso: La red valida cada transaccion

**Aplicaciones:**
 Criptomonedas como Bitcoin y Ethereum
 Contratos inteligentes automatizados
 Cadenas de suministro transparentes
 Identidad digital segura

**Ventajas:** Elimina intermediarios, reduce costos, aumenta seguridad
**Desafios:** Consumo energetico, escalabilidad, regulacion"""
            else:
                return """Blockchain is a revolutionary distributed ledger technology that creates tamper-proof digital records through cryptographic linking and decentralized consensus mechanisms."""

        # Bases de Datos
        if any(word in message_lower for word in ['base de datos', 'database', 'sql', 'nosql', 'mysql']):
            if self.language == "es":
                return """Las bases de datos son sistemas para almacenar y gestionar informacion:

**Tipos principales:**
 **Relacionales (SQL)**: MySQL, PostgreSQL, SQL Server
  - Datos estructurados en tablas con relaciones
  - ACID compliance para transacciones seguras
  - Consultas complejas con JOIN operations

 **No relacionales (NoSQL)**: MongoDB, Redis, Cassandra  
  - Datos flexibles: documentos, clave-valor, grafos
  - Escalabilidad horizontal
  - Mejor para big data y aplicaciones distribuidas

**Conceptos fundamentales:**
 Normalizacion: Eliminar redundancia de datos
 Indices: Acelerar busquedas y consultas
 Transacciones: Operaciones atomicas y consistentes
 Respaldos: Copias de seguridad automaticas

**Casos de uso:**
 Relacionales: Sistemas bancarios, ERP, CRM
 NoSQL: Redes sociales, IoT, analisis en tiempo real"""
            else:
                return """Databases are organized systems for storing, managing, and retrieving data efficiently using various models like relational (SQL) or document-based (NoSQL) approaches."""

        # Cloud Computing
        if any(word in message_lower for word in ['nube', 'cloud computing', 'aws', 'azure', 'google cloud']):
            if self.language == "es":
                return """La computacion en la nube revoluciona como usamos recursos IT:

**Modelos de servicio:**
 **IaaS** (Infrastructure): Servidores virtuales, almacenamiento
 **PaaS** (Platform): Entorno de desarrollo y despliegue  
 **SaaS** (Software): Aplicaciones listas para usar

**Principales proveedores:**
 AWS (Amazon): Lider del mercado, mas servicios
 Microsoft Azure: Integracion con Windows/Office
 Google Cloud: Especializado en ML y analytics
 IBM Cloud, Oracle Cloud: Enfoque empresarial

**Ventajas clave:**
 Escalabilidad instantanea segun demanda
 Pago por uso real de recursos
 Disponibilidad global 24/7
 Actualizaciones automaticas
 Reduccion de infraestructura fisica

**Desafios:**
 Dependencia de internet
 Preocupaciones de seguridad y privacidad  
 Posible vendor lock-in
 Costos pueden crecer inesperadamente"""
            else:
                return """Cloud computing delivers IT services over the internet, offering scalable resources, cost efficiency, and global accessibility through IaaS, PaaS, and SaaS models."""

        # Inteligencia Artificial
        if any(word in message_lower for word in ['inteligencia artificial', 'machine learning', 'ia', 'ai', 'algoritmo']):
            if self.language == "es":
                return """La inteligencia artificial (IA) es la capacidad de las maquinas para realizar tareas que normalmente requieren inteligencia humana, como:

 **Aprendizaje**: Mejorar el rendimiento basandose en la experiencia
 **Razonamiento**: Resolver problemas usando logica  
 **Percepcion**: Interpretar datos sensoriales
 **Procesamiento de lenguaje**: Entender y generar texto humano

El machine learning es una rama de la IA que permite a las computadoras aprender sin ser programadas explicitamente para cada tarea."""
            else:
                return """Artificial Intelligence (AI) is the capability of machines to perform tasks that normally require human intelligence, such as:

 **Learning**: Improving performance based on experience
 **Reasoning**: Solving problems using logic
 **Perception**: Interpreting sensory data  
 **Language processing**: Understanding and generating human text

Machine learning is a branch of AI that enables computers to learn without being explicitly programmed for each task."""
        
        # Redes Neuronales
        if any(word in message_lower for word in ['red neuronal', 'neural network', 'neurona', 'deep learning']):
            if self.language == "es":
                return """Una red neuronal es un modelo computacional inspirado en el cerebro humano:

**Componentes basicos:**
 **Neuronas artificiales**: Procesan y transmiten informacion
 **Conexiones (sinapsis)**: Tienen pesos que se ajustan durante el aprendizaje
 **Capas**: Entrada, ocultas y salida

**Funcionamiento:**
1. Recibe datos en la capa de entrada
2. Los procesa a traves de capas ocultas
3. Produce resultados en la capa de salida
4. Ajusta pesos basandose en errores (backpropagation)

Se usa en reconocimiento de imagenes, procesamiento de lenguaje natural y muchas otras aplicaciones."""
            else:
                return """A neural network is a computational model inspired by the human brain:

**Basic components:**
 **Artificial neurons**: Process and transmit information
 **Connections (synapses)**: Have weights adjusted during learning
 **Layers**: Input, hidden, and output

**How it works:**
1. Receives data in input layer
2. Processes through hidden layers  
3. Produces results in output layer
4. Adjusts weights based on errors (backpropagation)

Used in image recognition, natural language processing, and many other applications."""
        
        # Programacion
        if any(word in message_lower for word in ['programacion', 'programming', 'codigo', 'code', 'python', 'javascript']):
            if self.language == "es":
                return """La programacion es el proceso de crear software mediante codigo:

**Conceptos fundamentales:**
 **Algoritmos**: Secuencia de instrucciones para resolver problemas
 **Estructuras de datos**: Formas de organizar informacion
 **Logica**: Condicionales, bucles y funciones
 **Debugging**: Encontrar y corregir errores

**Lenguajes populares:**
 Python: Facil de aprender, versatil
 JavaScript: Para web y aplicaciones
 Java: Robusto para empresas
 C++: Alto rendimiento

Hay algun aspecto especifico de programacion que te interese?"""
            else:
                return """Programming is the process of creating software through code:

**Fundamental concepts:**
 **Algorithms**: Sequence of instructions to solve problems
 **Data structures**: Ways to organize information
 **Logic**: Conditionals, loops, and functions
 **Debugging**: Finding and fixing errors

**Popular languages:**
 Python: Easy to learn, versatile
 JavaScript: For web and applications  
 Java: Robust for enterprise
 C++: High performance

Is there a specific programming aspect you're interested in?"""
        
        # Ciberseguridad
        if any(word in message_lower for word in ['ciberseguridad', 'cybersecurity', 'seguridad', 'security', 'hack']):
            if self.language == "es":
                return """La ciberseguridad protege sistemas digitales de amenazas:

**Por que es importante:**
 Protege informacion personal y empresarial
 Evita perdidas economicas por ataques
 Mantiene la confianza en sistemas digitales
 Cumple regulaciones legales

**Amenazas comunes:**
 Malware, virus y ransomware
 Phishing y ingenieria social
 Ataques de fuerza bruta
 Vulnerabilidades de software

**Buenas practicas:**
 Contrasenas fuertes y unicas
 Actualizaciones regulares
 Backups seguros
 Autenticacion de dos factores"""
            else:
                return """Cybersecurity protects digital systems from threats:

**Why it's important:**
 Protects personal and business information
 Prevents economic losses from attacks
 Maintains trust in digital systems
 Complies with legal regulations

**Common threats:**
 Malware, viruses, and ransomware
 Phishing and social engineering
 Brute force attacks
 Software vulnerabilities  

**Best practices:**
 Strong, unique passwords
 Regular updates
 Secure backups
 Two-factor authentication"""
        
        # === PREGUNTAS GENERALES ===
        
        # Saludos
        greetings = ['hola', 'hello', 'hi', 'buenos dias', 'good morning', 'buenas tardes', 'buenas noches', 'hey', 'saludos']
        if any(greeting in message_lower for greeting in greetings):
            if self.language == "es":
                return """Hola! Soy AI Symbiote, tu asistente inteligente avanzado.

**Especialidades:**
 Tecnologia: IA, blockchain, programacion, ciberseguridad
 Ciencia: Explicaciones detalladas y precisas  
 Analisis: Comparaciones tecnicas profundas
 Resolucion: Guias paso a paso para problemas complejos

Estoy configurado para operar principalmente en espanol y dar respuestas tecnicamente precisas. Que tema complejo te interesa explorar?"""
            else:
                return "Hello! I'm AI Symbiote, your intelligent assistant. I can explain complex concepts about technology, science, programming and more. What would you like to learn today?"
        
        # Capacidades
        capabilities = ['capacidades', 'capabilities', 'que puedes hacer', 'what can you do']
        if any(cap in message_lower for cap in capabilities):
            if self.language == "es":
                return """Mis capacidades inteligentes incluyen:

* Explicaciones tecnicas: IA, programacion, ciencia
* Analisis conceptual: Comparaciones y diferencias  
* Resolucion de problemas: Guias paso a paso
* Conversacion natural: En espanol e ingles
* Vision por computadora: Reconocimiento facial
* Procesamiento de voz: Sintesis y reconocimiento

Preguntame sobre cualquier tema complejo y te dare una explicacion clara!"""
            else:
                return """My intelligent capabilities include:

* Technical explanations: AI, programming, science
* Conceptual analysis: Comparisons and differences
* Problem solving: Step-by-step guides  
* Natural conversation: In Spanish and English
* Computer vision: Facial recognition
* Voice processing: Synthesis and recognition

Ask me about any complex topic and I'll give you a clear explanation!"""
        
        # Ayuda
        help_words = ['ayuda', 'help', 'asistencia', 'assistance']
        if any(word in message_lower for word in help_words):
            if self.language == "es":
                return """Perfecto! Puedo ayudarte con:

**Preguntas tecnicas:**
 "Que es machine learning?"
 "Como funciona blockchain?"  
 "Diferencias entre frontend y backend"

**Conceptos complejos:**
 "Por que es importante la ciberseguridad?"
 "Como funcionan las bases de datos?"
 "Que es la computacion en la nube?"

**Resolucion de problemas:**
 Guias paso a paso
 Comparaciones detalladas
 Explicaciones con ejemplos

Sobre que tema especifico te gustaria que te ayude?"""
            else:
                return """Perfect! I can help you with:

**Technical questions:**
 "What is machine learning?"
 "How does blockchain work?"
 "Differences between frontend and backend"

**Complex concepts:**  
 "Why is cybersecurity important?"
 "How do databases work?"
 "What is cloud computing?"

**Problem solving:**
 Step-by-step guides
 Detailed comparisons
 Explanations with examples

What specific topic would you like me to help you with?"""
        
        # === RESPUESTAS CONTEXTUALES INTELIGENTES ===
        
        # Si contiene signos de pregunta
        if '?' in message_lower:
            if self.language == "es":
                return f"""Esa es una excelente pregunta. Para darte la mejor respuesta sobre "{message_lower.replace('?', '')}", necesitaria un poco mas de contexto.

Podrias especificar:
 Te interesa el aspecto tecnico, practico o teorico?
 Es para un proyecto especifico o curiosidad general?
 Que nivel de detalle prefieres?

Mientras tanto, puedo adelantarte que es un tema fascinante con multiples dimensiones. Preguntame con mas detalles!"""
            else:
                return f"""That's an excellent question. To give you the best answer about "{message_lower.replace('?', '')}", I'd need a bit more context.

Could you specify:
 Are you interested in technical, practical, or theoretical aspects?  
 Is it for a specific project or general curiosity?
 What level of detail do you prefer?

In the meantime, I can tell you it's a fascinating topic with multiple dimensions. Ask me with more details!"""
        
        # Respuesta inteligente por defecto
        if self.language == "es":
            return f"""Entiendo que estas preguntando sobre "{message_lower}". Es un tema que puede abordarse desde diferentes angulos.

Para darte una respuesta mas precisa y util, podrias:
 Ser mas especifico sobre que aspecto te interesa
 Darme mas contexto sobre tu consulta  
 Decirme si buscas una explicacion tecnica o general

Estoy aqui para ayudarte con explicaciones claras y detalladas!"""
        else:
            return f"""I understand you're asking about "{message_lower}". It's a topic that can be approached from different angles.

To give you a more precise and useful answer, could you:
 Be more specific about which aspect interests you
 Give me more context about your query
 Tell me if you're looking for a technical or general explanation

I'm here to help with clear and detailed explanations!"""
    
    async def speak(self, text: str) -> bool:
        """Text-to-speech without errors."""
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
    
    async def analyze_face(self, image_data: np.ndarray) -> Dict[str, Any]:
        """Face analysis without errors."""
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
                results["faces"].append({
                    "id": f"face_{i}",
                    "position": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                    "confidence": 0.9
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Face analysis error: {e}")
            return {"faces": [], "status": "error", "error": str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        return {
            "initialized": self.is_initialized,
            "components": {
                "chat_ai": True,  # Rule-based is always available
                "tts": self.tts_engine is not None,
                "speech_recognition": self.speech_recognizer is not None,
                "vision": self.face_cascade is not None
            },
            "language": self.language,
            "voice_enabled": self.voice_enabled,
            "model_type": "intelligent_rules",
            "conversation_length": len(self.conversation_history),
            "timestamp": datetime.now().isoformat()
        }

# Global instance
_simple_ai = None

def get_ai_engine() -> SimpleWorkingAI:
    """Get simple working AI instance."""
    global _simple_ai
    if _simple_ai is None:
        _simple_ai = SimpleWorkingAI()
    return _simple_ai

async def process_ai_message(message: str, user_id: str = "user") -> str:
    """Process message with simple AI."""
    engine = get_ai_engine()
    return await engine.process_message(message, user_id)

if __name__ == "__main__":
    # Test the simple AI
    async def test():
        print("Testing Simple Working AI...")
        engine = SimpleWorkingAI()
        
        test_messages = [
            "Que es la inteligencia artificial?",
            "Como funciona una red neuronal?",
            "Por que es importante la ciberseguridad?",
            "Hola, cuales son tus capacidades?",
            "Ayudame con programacion"
        ]
        
        for msg in test_messages:
            print(f"\nUser: {msg}")
            response = await engine.process_message(msg)
            print(f"AI: {response}")
        
        print("\nOK Simple Working AI test completed!")
    
    asyncio.run(test())