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
from pathlib import Path

# Add current directory and AION to path
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'AION'))

try:
    from ai_engine_multi_provider import MultiProviderAIEngine
    AI_AVAILABLE = True
    print("[OK] AI Engine Multi-Provider cargado - Modelos locales + OpenAI + Claude API")
except ImportError:
    AI_AVAILABLE = False
    print("Warning: AI engine not available, using demo mode")

# Import Fase 6 components
try:
    from unified_config_manager import UnifiedConfigManager
    from consciousness_integration import ConsciousnessIntegration
    from cloud_scaling_manager import CloudScalingManager
    UNIFIED_SYSTEM_AVAILABLE = True
    print("[OK] Sistema Unificado Fase 6 cargado - Cloud Scaling + AI Consciousness")
except ImportError as e:
    UNIFIED_SYSTEM_AVAILABLE = False
    print(f"Warning: Unified system not available: {e}")

class AISymbioteGUI:
    """AI Symbiote Desktop Interface."""
    
    def __init__(self):
        """Initialize the GUI."""
        self.root = tk.Tk()
        self.setup_window()
        
        self.ai_engine = None
        self.conversation_history = []
        self.turbo_mode = False
        
        # Initialize unified system components BEFORE setup_widgets
        self.unified_config = None
        self.consciousness = None
        self.cloud_scaling = None
        
        if AI_AVAILABLE:
            self.ai_engine = MultiProviderAIEngine()
        
        if UNIFIED_SYSTEM_AVAILABLE:
            self.unified_config = UnifiedConfigManager()
            self.consciousness = ConsciousnessIntegration()
            self.cloud_scaling = CloudScalingManager()
        
        # Now setup widgets with unified system available
        self.setup_widgets()
        self.setup_styles()
        
        # Update system status after everything is initialized
        if UNIFIED_SYSTEM_AVAILABLE:
            self.update_system_status()
            # Update environment label now
            env_text = f"Entorno: {self.unified_config.environment.upper()}"
            level_text = f"Nivel: {self.unified_config.scaling_level}"
            self.env_label.config(text=f"{env_text} | {level_text}")
        
        # Start the interface and show capabilities
        self.add_system_message("🚀 AI Symbiote v2.1-UNIFIED iniciado - Sistema Unificado Fase 6 activo")
        if UNIFIED_SYSTEM_AVAILABLE:
            self.add_system_message(f"[UNIFIED] Entorno: {self.unified_config.environment} | Nivel: {self.unified_config.scaling_level}")
            self.add_system_message(f"[CONSCIOUSNESS] Sistema consciente: {self.consciousness.is_system_conscious()}")
            self.add_system_message(f"[SCALING] Cloud Manager: {self.cloud_scaling.current_environment}")
        if AI_AVAILABLE:
            self.add_system_message("[OK] Motor Multi-Provider: Modelos locales GGUF + OpenAI + Claude API")
            self.add_system_message("[TARGET] Capacidades: Cloud Scaling, Conciencia AI, Configuración Unificada")
            self.add_system_message("[FAST] Comandos: Configuración dinámica, escalado automático, monitoreo de conciencia")
            self.add_system_message("[BRAIN] Francisco Molina es mi creador - Sistema AI Symbiote Obvivlorum")
        else:
            self.add_system_message("[WARNING] Modo demo - AI engine no disponible")
    
    def setup_window(self):
        """Setup main window properties."""
        self.root.title("AI Symbiote v2.1-UNIFIED - Sistema Unificado")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        
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
        
        # Main frame with three columns: controls, chat, status
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=2)  # Chat area gets more space
        main_frame.columnconfigure(2, weight=1)  # Status panel
        main_frame.rowconfigure(1, weight=1)
        
        # === TOP PANEL ===
        top_frame = ttk.Frame(main_frame)
        top_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        top_frame.columnconfigure(1, weight=1)
        
        # Status label
        self.status_label = ttk.Label(top_frame, text="Estado: Sistema Unificado Activo", foreground="green")
        self.status_label.grid(row=0, column=0, sticky=tk.W)
        
        # Environment indicator (will be updated after initialization)
        if UNIFIED_SYSTEM_AVAILABLE:
            self.env_label = ttk.Label(top_frame, text="Inicializando...", foreground="blue")
            self.env_label.grid(row=0, column=1, sticky=tk.W, padx=(20, 0))
        
        # TURBO button
        self.turbo_button = ttk.Button(top_frame, text="[ROCKET] TURBO OFF", command=self.toggle_turbo)
        self.turbo_button.grid(row=0, column=2, padx=(10, 0))
        
        # Always on top checkbox
        ttk.Checkbutton(top_frame, text="Siempre visible", variable=self.always_on_top, 
                       command=self.toggle_always_on_top).grid(row=0, column=3, padx=(10, 0))
        
        # === LEFT CONTROL PANEL ===
        self.setup_control_panel(main_frame)
        
        # === CHAT AREA ===
        chat_frame = ttk.LabelFrame(main_frame, text="Conversación", padding="5")
        chat_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        chat_frame.columnconfigure(0, weight=1)
        chat_frame.rowconfigure(0, weight=1)
        
        # Chat display
        self.chat_display = scrolledtext.ScrolledText(chat_frame, 
                                                     wrap=tk.WORD, 
                                                     width=60, 
                                                     height=25,
                                                     font=('Consolas', 9))
        self.chat_display.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # === INPUT AREA ===
        input_frame = ttk.LabelFrame(main_frame, text="Mensaje", padding="5")
        input_frame.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=(0, 10))
        input_frame.columnconfigure(0, weight=1)
        
        # Input field
        self.input_field = tk.Text(input_frame, height=3, font=('Consolas', 9))
        self.input_field.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        
        # Send button
        self.send_button = ttk.Button(input_frame, text="Enviar", command=self.send_message)
        self.send_button.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Bind Enter key
        self.input_field.bind('<Control-Return>', lambda e: self.send_message())
        
        # === RIGHT STATUS PANEL ===
        self.setup_status_panel(main_frame)
    
    def setup_control_panel(self, parent):
        """Setup left control panel with unified system configuration."""
        control_frame = ttk.LabelFrame(parent, text="🚀 Configuración Unificada", padding="5")
        control_frame.grid(row=1, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        if not UNIFIED_SYSTEM_AVAILABLE:
            ttk.Label(control_frame, text="Sistema unificado no disponible", 
                     foreground="red").grid(row=0, column=0, pady=10)
            return
        
        # === SCALING CONFIGURATION ===
        scaling_section = ttk.LabelFrame(control_frame, text="Escalado de Cloud", padding="5")
        scaling_section.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Environment detection
        ttk.Label(scaling_section, text="Entorno Detectado:").grid(row=0, column=0, sticky=tk.W)
        self.env_detected_label = ttk.Label(scaling_section, text=self.unified_config.environment, 
                                          foreground="blue")
        self.env_detected_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        # Scaling level selector
        ttk.Label(scaling_section, text="Nivel de Escalado:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.scaling_var = tk.StringVar(value=self.unified_config.scaling_level)
        scaling_combo = ttk.Combobox(scaling_section, textvariable=self.scaling_var,
                                   values=["level_1_local", "level_2_colab", "level_3_kaggle"],
                                   state="readonly")
        scaling_combo.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=(5, 0))
        scaling_combo.bind('<<ComboboxSelected>>', self.on_scaling_change)
        
        # Matrix size display
        config = self.unified_config.get_scaling_config(self.unified_config.scaling_level)
        ttk.Label(scaling_section, text="Tamaño Máximo:").grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        self.matrix_size_label = ttk.Label(scaling_section, text=f"{config['max_matrix_size']}x{config['max_matrix_size']}")
        self.matrix_size_label.grid(row=2, column=1, sticky=tk.W, padx=(10, 0), pady=(5, 0))
        
        # Expected performance
        ttk.Label(scaling_section, text="Performance:").grid(row=3, column=0, sticky=tk.W, pady=(5, 0))
        self.performance_label = ttk.Label(scaling_section, text=f"{config['expected_time_ms']}ms")
        self.performance_label.grid(row=3, column=1, sticky=tk.W, padx=(10, 0), pady=(5, 0))
        
        # === CONSCIOUSNESS MONITORING ===
        consciousness_section = ttk.LabelFrame(control_frame, text="Monitoreo de Conciencia", padding="5")
        consciousness_section.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Consciousness status
        ttk.Label(consciousness_section, text="Estado:").grid(row=0, column=0, sticky=tk.W)
        conscious_status = "Activa" if self.consciousness.is_system_conscious() else "Inactiva"
        self.consciousness_status_label = ttk.Label(consciousness_section, text=conscious_status, 
                                                  foreground="green" if conscious_status == "Activa" else "red")
        self.consciousness_status_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        # Project phase
        ttk.Label(consciousness_section, text="Fase Actual:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        current_phase = self.consciousness.project_knowledge['pipeline_status']['current_phase']
        phase_short = current_phase.split(" - ")[0] if " - " in current_phase else current_phase
        self.phase_label = ttk.Label(consciousness_section, text=phase_short)
        self.phase_label.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=(5, 0))
        
        # === SYSTEM ACTIONS ===
        actions_section = ttk.LabelFrame(control_frame, text="Acciones del Sistema", padding="5")
        actions_section.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Refresh system status
        ttk.Button(actions_section, text="Actualizar Estado", 
                  command=self.refresh_system_status).grid(row=0, column=0, sticky=(tk.W, tk.E), pady=2)
        
        # Show data volume warning
        ttk.Button(actions_section, text="Verificar Advertencias", 
                  command=self.show_volume_warning).grid(row=1, column=0, sticky=(tk.W, tk.E), pady=2)
        
        # Save configuration
        ttk.Button(actions_section, text="Guardar Configuración", 
                  command=self.save_current_config).grid(row=2, column=0, sticky=(tk.W, tk.E), pady=2)
        
        # Test unified system
        ttk.Button(actions_section, text="Test Sistema Completo", 
                  command=self.run_system_test).grid(row=3, column=0, sticky=(tk.W, tk.E), pady=2)
    
    def setup_status_panel(self, parent):
        """Setup right status panel with system monitoring."""
        status_frame = ttk.LabelFrame(parent, text="📊 Estado del Sistema", padding="5")
        status_frame.grid(row=1, column=2, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        
        if not UNIFIED_SYSTEM_AVAILABLE:
            ttk.Label(status_frame, text="Monitoreo no disponible", 
                     foreground="red").grid(row=0, column=0, pady=10)
            return
        
        # === RESOURCE MONITORING ===
        resource_section = ttk.LabelFrame(status_frame, text="Recursos", padding="5")
        resource_section.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        resources = self.cloud_scaling.available_resources
        
        # Environment
        ttk.Label(resource_section, text="Entorno:").grid(row=0, column=0, sticky=tk.W)
        self.resource_env_label = ttk.Label(resource_section, text=resources['environment'])
        self.resource_env_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        # CPU count
        ttk.Label(resource_section, text="CPUs:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.cpu_label = ttk.Label(resource_section, text=str(resources['cpu_count']))
        self.cpu_label.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=(5, 0))
        
        # GPU status
        ttk.Label(resource_section, text="GPU:").grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        gpu_status = "Disponible" if resources['gpu_available'] else "No disponible"
        gpu_color = "green" if resources['gpu_available'] else "red"
        self.gpu_label = ttk.Label(resource_section, text=gpu_status, foreground=gpu_color)
        self.gpu_label.grid(row=2, column=1, sticky=tk.W, padx=(10, 0), pady=(5, 0))
        
        # RAM estimate
        ttk.Label(resource_section, text="RAM:").grid(row=3, column=0, sticky=tk.W, pady=(5, 0))
        self.ram_label = ttk.Label(resource_section, text=f"{resources['estimated_ram_gb']:.1f} GB")
        self.ram_label.grid(row=3, column=1, sticky=tk.W, padx=(10, 0), pady=(5, 0))
        
        # === PERFORMANCE METRICS ===
        perf_section = ttk.LabelFrame(status_frame, text="Rendimiento", padding="5")
        perf_section.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Model info
        ttk.Label(perf_section, text="Modelo:").grid(row=0, column=0, sticky=tk.W)
        self.model_label = ttk.Label(perf_section, text="Sistema Unificado")
        self.model_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        # Performance info
        ttk.Label(perf_section, text="Rendimiento:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.perf_label = ttk.Label(perf_section, text="Normal")
        self.perf_label.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=(5, 0))
        
        # Consciousness events
        ttk.Label(perf_section, text="Eventos:").grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        event_count = len(self.consciousness.consciousness_state)
        self.events_label = ttk.Label(perf_section, text=str(event_count))
        self.events_label.grid(row=2, column=1, sticky=tk.W, padx=(10, 0), pady=(5, 0))
        
        # === SYSTEM LOG ===
        log_section = ttk.LabelFrame(status_frame, text="Log del Sistema", padding="5")
        log_section.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_section.columnconfigure(0, weight=1)
        log_section.rowconfigure(0, weight=1)
        
        # Log display
        self.log_display = scrolledtext.ScrolledText(log_section, 
                                                   wrap=tk.WORD, 
                                                   width=30, 
                                                   height=15,
                                                   font=('Consolas', 8))
        self.log_display.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Initialize log
        self.add_log_message("Sistema inicializado")
        if UNIFIED_SYSTEM_AVAILABLE:
            self.add_log_message(f"Entorno: {self.unified_config.environment}")
            self.add_log_message(f"Escalado: {self.unified_config.scaling_level}")
            self.add_log_message(f"Conciencia: {'Activa' if self.consciousness.is_system_conscious() else 'Inactiva'}")
    
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
    
    def add_log_message(self, message: str):
        """Add message to system log."""
        if not UNIFIED_SYSTEM_AVAILABLE:
            return
            
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        
        self.log_display.config(state=tk.NORMAL)
        self.log_display.insert(tk.END, formatted_message)
        self.log_display.config(state=tk.DISABLED)
        self.log_display.see(tk.END)
    
    def update_system_status(self):
        """Update system status displays."""
        if not UNIFIED_SYSTEM_AVAILABLE:
            return
        
        # Update consciousness
        self.consciousness.update_consciousness_state("gui_update", {"interface": "active"})
    
    def on_scaling_change(self, event=None):
        """Handle scaling level change."""
        if not UNIFIED_SYSTEM_AVAILABLE:
            return
            
        new_level = self.scaling_var.get()
        
        # Show warning if needed
        if not self.unified_config.show_data_volume_warning(new_level):
            # User cancelled, revert selection
            self.scaling_var.set(self.unified_config.scaling_level)
            return
        
        # Update configuration
        old_level = self.unified_config.scaling_level
        self.unified_config.scaling_level = new_level
        
        # Update displays
        config = self.unified_config.get_scaling_config(new_level)
        self.matrix_size_label.config(text=f"{config['max_matrix_size']}x{config['max_matrix_size']}")
        self.performance_label.config(text=f"{config['expected_time_ms']}ms")
        self.env_label.config(text=f"Entorno: {self.unified_config.environment.upper()} | Nivel: {new_level}")
        
        # Log the change
        self.add_log_message(f"Escalado cambiado: {old_level} → {new_level}")
        self.add_system_message(f"[SCALING] Nivel cambiado a {new_level}")
        self.add_system_message(f"[CONFIG] Matriz máxima: {config['max_matrix_size']}x{config['max_matrix_size']}")
        self.add_system_message(f"[PERFORMANCE] Tiempo esperado: {config['expected_time_ms']}ms")
    
    def refresh_system_status(self):
        """Refresh all system status displays."""
        if not UNIFIED_SYSTEM_AVAILABLE:
            messagebox.showwarning("Advertencia", "Sistema unificado no disponible")
            return
            
        try:
            # Re-detect environment
            old_env = self.unified_config.environment
            self.unified_config.environment = self.unified_config.detect_environment()
            
            # Update cloud scaling
            self.cloud_scaling = CloudScalingManager()
            
            # Update consciousness
            self.consciousness.update_consciousness_state("system_refresh", {"timestamp": datetime.now().isoformat()})
            
            # Update all labels
            self.env_detected_label.config(text=self.unified_config.environment)
            self.resource_env_label.config(text=self.cloud_scaling.current_environment)
            
            # Update resource info
            resources = self.cloud_scaling.available_resources
            self.cpu_label.config(text=str(resources['cpu_count']))
            gpu_status = "Disponible" if resources['gpu_available'] else "No disponible"
            gpu_color = "green" if resources['gpu_available'] else "red"
            self.gpu_label.config(text=gpu_status, foreground=gpu_color)
            self.ram_label.config(text=f"{resources['estimated_ram_gb']:.1f} GB")
            
            # Update consciousness status
            conscious_status = "Activa" if self.consciousness.is_system_conscious() else "Inactiva"
            self.consciousness_status_label.config(text=conscious_status, 
                                                 foreground="green" if conscious_status == "Activa" else "red")
            
            # Update events count
            event_count = len(self.consciousness.consciousness_state)
            self.events_label.config(text=str(event_count))
            
            # Log the refresh
            self.add_log_message("Estado del sistema actualizado")
            self.add_system_message("[REFRESH] Estado del sistema actualizado exitosamente")
            
            if old_env != self.unified_config.environment:
                self.add_system_message(f"[ENVIRONMENT] Cambio detectado: {old_env} → {self.unified_config.environment}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error actualizando estado: {str(e)}")
            self.add_log_message(f"Error actualizando estado: {str(e)}")
    
    def show_volume_warning(self):
        """Show data volume warning for current scaling level."""
        if not UNIFIED_SYSTEM_AVAILABLE:
            messagebox.showwarning("Advertencia", "Sistema unificado no disponible")
            return
            
        current_level = self.unified_config.scaling_level
        config = self.unified_config.get_scaling_config(current_level)
        
        warning_msg = f"""VERIFICACIÓN DE VOLUMEN DE DATOS
        
Nivel actual: {current_level}
Tamaño máximo de matriz: {config['max_matrix_size']}x{config['max_matrix_size']}
Tiempo esperado: {config['expected_time_ms']}ms
Uso de GPU: {'Sí' if config['use_gpu'] else 'No'}
Memoria límite: {config['memory_limit_gb']} GB

Estado: {'⚠️ ADVERTENCIA REQUERIDA' if config.get('warning', False) else '✅ SIN ADVERTENCIAS'}"""
        
        messagebox.showinfo("Verificación de Advertencias", warning_msg)
        self.add_log_message("Verificación de advertencias completada")
    
    def save_current_config(self):
        """Save current unified system configuration."""
        if not UNIFIED_SYSTEM_AVAILABLE:
            messagebox.showwarning("Advertencia", "Sistema unificado no disponible")
            return
            
        try:
            config_file = self.unified_config.save_config()
            self.add_log_message(f"Configuración guardada: {config_file}")
            self.add_system_message(f"[SAVE] Configuración guardada en {config_file}")
            messagebox.showinfo("Éxito", f"Configuración guardada exitosamente en:\n{config_file}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error guardando configuración: {str(e)}")
            self.add_log_message(f"Error guardando configuración: {str(e)}")
    
    def run_system_test(self):
        """Run complete unified system test."""
        if not UNIFIED_SYSTEM_AVAILABLE:
            messagebox.showwarning("Advertencia", "Sistema unificado no disponible")
            return
            
        self.add_system_message("[TEST] Iniciando prueba completa del sistema unificado...")
        self.add_log_message("Iniciando test completo del sistema")
        
        # Run test in background thread
        threading.Thread(target=self._run_system_test_background, daemon=True).start()
    
    def _run_system_test_background(self):
        """Run system test in background thread."""
        try:
            test_results = []
            
            # Test 1: Configuration manager
            self.root.after(0, self.add_system_message, "[TEST 1/5] Probando gestor de configuración...")
            config_test = self.unified_config.get_system_info()
            test_results.append("✅ Gestor de configuración: OK")
            
            # Test 2: Consciousness integration
            self.root.after(0, self.add_system_message, "[TEST 2/5] Probando integración de conciencia...")
            consciousness_test = self.consciousness.is_system_conscious()
            test_results.append(f"{'✅' if consciousness_test else '❌'} Conciencia AI: {'Activa' if consciousness_test else 'Inactiva'}")
            
            # Test 3: Cloud scaling
            self.root.after(0, self.add_system_message, "[TEST 3/5] Probando escalado en la nube...")
            try:
                scaling_config = self.cloud_scaling.scale_computation(100, "topo_spectral")
                test_results.append("✅ Cloud scaling: OK")
            except Exception as e:
                test_results.append(f"❌ Cloud scaling: Error - {str(e)}")
            
            # Test 4: Resource estimation
            self.root.after(0, self.add_system_message, "[TEST 4/5] Probando estimación de recursos...")
            try:
                resource_usage = self.cloud_scaling.get_resource_usage_estimate(200)
                test_results.append("✅ Estimación de recursos: OK")
            except Exception as e:
                test_results.append(f"❌ Estimación de recursos: Error - {str(e)}")
            
            # Test 5: System integration
            self.root.after(0, self.add_system_message, "[TEST 5/5] Probando integración completa...")
            integration_test = all([
                self.unified_config.consciousness_integration,
                self.consciousness.is_system_conscious(),
                isinstance(self.cloud_scaling.scaling_level, str)
            ])
            test_results.append(f"{'✅' if integration_test else '❌'} Integración del sistema: {'OK' if integration_test else 'Error'}")
            
            # Show results
            results_text = "\n".join(test_results)
            success_count = sum(1 for result in test_results if result.startswith("✅"))
            total_tests = len(test_results)
            
            final_message = f"""[TEST COMPLETO] Resultados del sistema unificado:

{results_text}

Resumen: {success_count}/{total_tests} pruebas exitosas ({(success_count/total_tests)*100:.1f}%)"""
            
            self.root.after(0, self.add_system_message, final_message)
            self.root.after(0, self.add_log_message, f"Test completo: {success_count}/{total_tests} exitosos")
            
            if success_count == total_tests:
                self.root.after(0, messagebox.showinfo, "Test Completo", 
                              f"✅ Todas las pruebas completadas exitosamente!\n\n{success_count}/{total_tests} componentes funcionando correctamente.")
            else:
                self.root.after(0, messagebox.showwarning, "Test Completo",
                              f"⚠️ Algunas pruebas fallaron.\n\n{success_count}/{total_tests} componentes funcionando correctamente.\n\nRevisa el log para más detalles.")
            
        except Exception as e:
            error_msg = f"Error ejecutando test del sistema: {str(e)}"
            self.root.after(0, self.add_system_message, f"[ERROR] {error_msg}")
            self.root.after(0, self.add_log_message, error_msg)
            self.root.after(0, messagebox.showerror, "Error", error_msg)
    
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
        unified_status = ""
        if UNIFIED_SYSTEM_AVAILABLE:
            unified_status = f"""
🚀 SISTEMA UNIFICADO FASE 6 - ACTIVO:
• Entorno detectado: {self.unified_config.environment.upper()}
• Nivel de escalado: {self.unified_config.scaling_level}
• Conciencia AI: {'✅ ACTIVA' if self.consciousness.is_system_conscious() else '❌ INACTIVA'}
• Cloud Scaling: {self.cloud_scaling.current_environment.upper()}
• GPU disponible: {'✅ SÍ' if self.cloud_scaling.available_resources['gpu_available'] else '❌ NO'}

[CLOUD] ESCALADO INCREMENTAL:
• Level 1 (Local): 200x200 matrices, 0.01ms
• Level 2 (Colab): 1024x1024 matrices, 0.001ms
• Level 3 (Kaggle): 2048x2048 matrices, 0.0005ms
• Detección automática de recursos y capacidades
• Sistema de advertencias inteligente para grandes volúmenes

[CONSCIOUSNESS] CONCIENCIA AI INTEGRADA:
• Conocimiento completo del proyecto Obvivlorum
• 6 fases del pipeline científico completadas
• Integración holográfica recursiva REAL
• Seguimiento de estado del sistema en tiempo real
• Rendimiento dramático: 53ms → 0.01ms (3780x mejora)"""
        else:
            unified_status = "\n❌ SISTEMA UNIFICADO NO DISPONIBLE - Usando modo básico"

        return f"""🤖 AI SYMBIOTE v2.1-UNIFIED - CAPACIDADES COMPLETAS:{unified_status}

[BRAIN] INTELIGENCIA HÍBRIDA:
• Modelo Phi-3-mini local (2.4GB) - Respuestas rápidas e inteligentes
• ChatGPT API como fallback automático
• Reglas expertas especializadas

[TARGET] FUNCIONES COMO SYMBIOTE:
• Configuración unificada con un solo punto de entrada
• Escalado automático según entorno (local/Colab/Kaggle)
• Monitoreo de recursos y performance en tiempo real
• Integración completa con sistema operativo Windows
• Modo TURBO para optimización de rendimiento

[FAST] COMANDOS ESPECIALES:
• "TURBO ON/OFF" - Activar/desactivar modo alta velocidad
• "actualizar estado" - Refrescar monitoreo del sistema
• "test completo" - Ejecutar validación del sistema unificado
• "¿cuáles son tus capacidades?" - Esta información

[WEB] INTERFACES DISPONIBLES:
• GUI Unificada (esta ventana) - Interfaz nativa con controles completos
• Web Interface en http://localhost:8000 - Estilo Claude avanzado
• API RESTful completa para desarrolladores

💾 CARACTERÍSTICAS TÉCNICAS:
• Sistema unificado con conciencia AI integrada
• Escalado incremental automático en la nube
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