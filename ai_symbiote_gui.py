#!/usr/bin/env python3
"""
AI Symbiote GUI - Persistent Desktop Interface
=============================================

Interfaz de escritorio persistente estilo Visual Basic para AI Symbiote.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import asyncio
import sys
import os
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

try:
    from ai_engine_multi_provider import MultiProviderAIEngine
    AI_AVAILABLE = True
    print("[OK] AI Engine Multi-Provider cargado - Modelos locales + OpenAI + Claude API")
except ImportError:
    AI_AVAILABLE = False
    print("Warning: AI engine not available, using demo mode")

class AISymbioteGUI:
    """AI Symbiote Desktop Interface."""
    
    def __init__(self):
        """Initialize the GUI."""
        self.root = tk.Tk()
        self.setup_window()
        self.setup_widgets()
        self.setup_styles()
        
        self.ai_engine = None
        self.conversation_history = []
        self.turbo_mode = False
        
        if AI_AVAILABLE:
            self.ai_engine = MultiProviderAIEngine()
        
        # Start the interface and show capabilities
        self.add_system_message("🤖 AI Symbiote v3.0 iniciado - Interfaz persistente activa")
        if AI_AVAILABLE:
            self.add_system_message("[OK] Motor Multi-Provider: Modelos locales GGUF + OpenAI + Claude API")
            self.add_system_message("[TARGET] Capacidades: Reconocimiento facial, voz, modo TURBO, selección de IA")
            self.add_system_message("[FAST] Comandos: 'TURBO ON/OFF', 'usar local/openai/claude', 'activa voz'")
            self.add_system_message("[BRAIN] Francisco Molina es mi creador - Sistema AI Symbiote Obvivlorum")
        else:
            self.add_system_message("[WARNING] Modo demo - AI engine no disponible")
    
    def setup_window(self):
        """Setup main window properties."""
        self.root.title("AI Symbiote v3.0 - Interfaz Inteligente")
        self.root.geometry("1000x700")
        self.root.minsize(800, 600)
        
        # Set window icon (if available)
        try:
            self.root.iconbitmap("icon.ico")
        except:
            pass
        
        # Keep window always on top option
        self.always_on_top = tk.BooleanVar()
        
        # Configure window closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_widgets(self):
        """Setup all GUI widgets."""
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # === TOP PANEL ===
        top_frame = ttk.Frame(main_frame)
        top_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        top_frame.columnconfigure(1, weight=1)
        
        # Status label
        self.status_label = ttk.Label(top_frame, text="Estado: Conectado", foreground="green")
        self.status_label.grid(row=0, column=0, sticky=tk.W)
        
        # TURBO button
        self.turbo_button = ttk.Button(top_frame, text="[ROCKET] TURBO OFF", command=self.toggle_turbo)
        self.turbo_button.grid(row=0, column=2, padx=(10, 0))
        
        # Always on top checkbox
        ttk.Checkbutton(top_frame, text="Siempre visible", variable=self.always_on_top, 
                       command=self.toggle_always_on_top).grid(row=0, column=3, padx=(10, 0))
        
        # === CHAT AREA ===
        chat_frame = ttk.LabelFrame(main_frame, text="Conversación", padding="5")
        chat_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        chat_frame.columnconfigure(0, weight=1)
        chat_frame.rowconfigure(0, weight=1)
        
        # Chat display
        self.chat_display = scrolledtext.ScrolledText(chat_frame, 
                                                     wrap=tk.WORD, 
                                                     width=80, 
                                                     height=25,
                                                     font=('Consolas', 10))
        self.chat_display.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # === INPUT AREA ===
        input_frame = ttk.LabelFrame(main_frame, text="Mensaje", padding="5")
        input_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        input_frame.columnconfigure(0, weight=1)
        
        # Input field
        self.input_field = tk.Text(input_frame, height=3, font=('Consolas', 10))
        self.input_field.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        
        # Send button
        self.send_button = ttk.Button(input_frame, text="Enviar", command=self.send_message)
        self.send_button.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Bind Enter key
        self.input_field.bind('<Control-Return>', lambda e: self.send_message())
        
        # === INFO PANEL ===
        info_frame = ttk.LabelFrame(main_frame, text="Estado del Sistema", padding="5")
        info_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E))
        info_frame.columnconfigure(1, weight=1)
        
        # Model info
        ttk.Label(info_frame, text="Modelo:").grid(row=0, column=0, sticky=tk.W)
        self.model_label = ttk.Label(info_frame, text="Híbrido (Local + API)")
        self.model_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        # Performance info
        ttk.Label(info_frame, text="Rendimiento:").grid(row=1, column=0, sticky=tk.W)
        self.perf_label = ttk.Label(info_frame, text="Normal")
        self.perf_label.grid(row=1, column=1, sticky=tk.W, padx=(10, 0))
    
    def setup_styles(self):
        """Setup custom styles."""
        style = ttk.Style()
        
        # Configure button style for TURBO
        style.configure('Turbo.TButton', foreground='white', background='red')
    
    def add_message(self, sender: str, message: str, color: str = "black"):
        """Add message to chat display."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {sender}: {message}\n\n"
        
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, formatted_message)
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
    
    def add_system_message(self, message: str):
        """Add system message."""
        self.add_message("SISTEMA", message, "blue")
    
    def send_message(self):
        """Send user message and get AI response."""
        user_input = self.input_field.get(1.0, tk.END).strip()
        
        if not user_input:
            return
        
        # Clear input
        self.input_field.delete(1.0, tk.END)
        
        # Add user message
        self.add_message("TÚ", user_input)
        
        # Update status
        self.status_label.config(text="Estado: Procesando...", foreground="orange")
        self.send_button.config(state=tk.DISABLED)
        
        # Process AI response in background
        threading.Thread(target=self.process_ai_response, args=(user_input,), daemon=True).start()
    
    def process_ai_response(self, message: str):
        """Process AI response in background thread."""
        try:
            if AI_AVAILABLE:
                # Check for special GUI commands first
                message_lower = message.lower()
                if any(cmd in message_lower for cmd in ['capacidades', 'funciones', 'qué puedes hacer']):
                    response = self.get_symbiote_capabilities()
                elif any(cmd in message_lower for cmd in ['modelo', 'qué modelo', 'que modelo']):
                    response = self.get_model_info()
                else:
                    # Use new multi-provider AI engine
                    response = self.ai_engine.process_message(message)
            else:
                # Demo response
                response = f"[DEMO] Esta es una respuesta de demostración a tu mensaje: '{message}'. En el modo completo, aquí aparecería la respuesta real de AI Symbiote."
            
            # Update GUI in main thread
            self.root.after(0, self.update_with_response, response)
            
        except Exception as e:
            error_msg = f"Error procesando mensaje: {str(e)}"
            self.root.after(0, self.update_with_response, error_msg)
    
    def update_with_response(self, response: str):
        """Update GUI with AI response."""
        # Add AI response
        self.add_message("AI SYMBIOTE", response)
        
        # Update status
        self.status_label.config(text="Estado: Listo", foreground="green")
        self.send_button.config(state=tk.NORMAL)
        
        # Focus back to input
        self.input_field.focus()
    
    def get_symbiote_capabilities(self):
        """Return detailed symbiote capabilities."""
        return """🤖 AI SYMBIOTE v3.0 - CAPACIDADES COMPLETAS:

[BRAIN] INTELIGENCIA HÍBRIDA:
• Modelo Phi-3-mini local (2.4GB) - Respuestas rápidas e inteligentes
• ChatGPT API como fallback automático
• Reglas expertas especializadas

[TARGET] FUNCIONES COMO SYMBIOTE:
• Reconocimiento facial en tiempo real con OpenCV
• Reconocimiento de voz y síntesis de audio
• Procesamiento de lenguaje natural avanzado
• Integración completa con sistema operativo Windows
• Modo TURBO para optimización de rendimiento

[FAST] COMANDOS ESPECIALES:
• "TURBO ON/OFF" - Activar/desactivar modo alta velocidad
• "activa reconocimiento facial" - Iniciar cámara y detección
• "activa reconocimiento de voz" - Configurar micrófono y STT
• "¿cuáles son tus capacidades?" - Esta información

[WEB] INTERFACES DISPONIBLES:
• GUI Persistente (esta ventana) - Interfaz nativa Windows
• Web Interface en http://localhost:8000 - Estilo Claude avanzado
• API RESTful completa para desarrolladores

💾 CARACTERÍSTICAS TÉCNICAS:
• Funciona completamente offline (modelo local)
• Sin límites de uso ni restricciones
• Privacidad total - nada sale de tu PC
• Persistencia automática al arrancar Windows"""
    
    def get_model_info(self):
        """Return model information."""
        if AI_AVAILABLE and self.ai_engine:
            current_provider = self.ai_engine.current_provider
            available_models = len(self.ai_engine.local_models)
            return f"""[TARGET] INFORMACIÓN DEL SISTEMA AI:

[CHART] PROVEEDOR ACTUAL: {current_provider.upper()}
• Francisco Molina es mi creador
• Sistema AI Symbiote Obvivlorum
• Proyecto AION con protocolo multi-provider

🔄 PROVEEDORES DISPONIBLES:
1. Modelos Locales ({available_models} modelos GGUF)
2. OpenAI API (GPT-3.5/4)
3. Claude API (Sonnet/Haiku)

[FAST] COMANDOS DE CAMBIO:
• 'usar local' - Modelos locales GGUF
• 'usar openai' - API de OpenAI 
• 'usar claude' - API de Claude

[BRAIN] CONTEXTO DEL PROYECTO:
Creado por Francisco Molina como sistema simbiótico adaptativo."""
        else:
            return "Sistema AI no disponible - Modo demo activo"
    
    def toggle_turbo(self):
        """Toggle TURBO mode."""
        if not AI_AVAILABLE:
            messagebox.showwarning("Advertencia", "Modo TURBO requiere AI engine activo")
            return
        
        try:
            if self.turbo_mode:
                # Disable TURBO
                response = self.ai_engine.disable_turbo_mode()
                self.turbo_button.config(text="[ROCKET] TURBO OFF")
                self.perf_label.config(text="Normal")
                self.turbo_mode = False
            else:
                # Enable TURBO
                response = self.ai_engine.enable_turbo_mode()
                self.turbo_button.config(text="[HOT] TURBO ON", style='Turbo.TButton')
                self.perf_label.config(text="Alto rendimiento")
                self.turbo_mode = True
            
            self.add_system_message(response)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en modo TURBO: {e}")
    
    def toggle_always_on_top(self):
        """Toggle always on top."""
        self.root.attributes('-topmost', self.always_on_top.get())
    
    def on_closing(self):
        """Handle window closing."""
        if messagebox.askokcancel("Salir", "¿Cerrar AI Symbiote?"):
            # Disable TURBO mode if active
            if self.turbo_mode and AI_AVAILABLE and self.ai_engine:
                self.ai_engine.disable_turbo_mode()
            
            self.root.destroy()
    
    def run(self):
        """Run the GUI."""
        self.add_system_message("Interfaz lista - Escribe tu mensaje y presiona Enter o clic en Enviar")
        
        # Focus on input field
        self.input_field.focus()
        
        # Start main loop
        self.root.mainloop()

def main():
    """Main function."""
    print("Iniciando AI Symbiote GUI...")
    
    try:
        app = AISymbioteGUI()
        app.run()
    except Exception as e:
        print(f"Error iniciando interfaz: {e}")
        input("Presiona Enter para salir...")

if __name__ == "__main__":
    main()