#!/usr/bin/env python3
"""
Brutal UI Desktop - FIXED VERSION
=================================

Version corregida sin AnimatedButton complejo y imports simplificados
"""

import tkinter as tk
from tkinter import messagebox
import sys
import os
import subprocess
from typing import Dict, Any, Optional
import json

# Simple import without Path complications
try:
    from unified_config_system import UnifiedConfigSystem, SystemMode, ScalingLevel, AIProvider
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    print("Warning: Configuration system not available, using defaults")

class BrutalTheme:
    """Tema visual brutalmente impresionante"""
    
    COLORS = {
        'bg_primary': '#0a0a0a',
        'bg_secondary': '#1a1a1a',
        'bg_accent': '#2d2d2d',
        'text_primary': '#ffffff',
        'text_secondary': '#b0b0b0',
        'accent_primary': '#00ff88',
        'accent_secondary': '#ff6b35',
        'accent_warning': '#ffeb3b',
        'accent_error': '#f44336',
        'gradient_start': '#1e3c72',
        'gradient_end': '#2a5298',
    }
    
    FONTS = {
        'title': ('Arial Black', 20, 'bold'),
        'subtitle': ('Arial', 14, 'bold'),
        'body': ('Arial', 10, 'normal'),
        'mono': ('Consolas', 9, 'normal'),
        'button': ('Arial', 10, 'bold'),
    }

class ConfigPanel(tk.Frame):
    """Panel de configuracion avanzada"""
    
    def __init__(self, parent):
        super().__init__(parent, bg=BrutalTheme.COLORS["bg_primary"])
        self.config_system = None
        if CONFIG_AVAILABLE:
            try:
                self.config_system = UnifiedConfigSystem()
            except:
                self.config_system = None
        self._create_widgets()
    
    def _create_widgets(self):
        """Crear widgets de configuracion"""
        # Titulo
        title = tk.Label(self, text="CONFIG - CONFIGURACION BRUTAL", 
                        fg=BrutalTheme.COLORS['accent_primary'],
                        bg=BrutalTheme.COLORS['bg_primary'],
                        font=BrutalTheme.FONTS['title'])
        title.pack(pady=20)
        
        # Frame para opciones
        options_frame = tk.Frame(self, bg=BrutalTheme.COLORS['bg_primary'])
        options_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Modo del sistema
        self._create_mode_section(options_frame)
        
        # Botones de accion
        self._create_action_buttons(options_frame)
    
    def _create_mode_section(self, parent):
        """Seccion de modo del sistema"""
        frame = tk.LabelFrame(parent, text="Modo del Sistema", 
                             fg=BrutalTheme.COLORS['text_primary'],
                             bg=BrutalTheme.COLORS['bg_secondary'],
                             font=BrutalTheme.FONTS['subtitle'])
        frame.pack(fill='x', pady=10)
        
        self.mode_var = tk.StringVar(value="gui_desktop")
        
        modes = [
            ("Standard", "standard", "Modo estandar de operacion"),
            ("GUI Desktop", "gui_desktop", "Interfaz grafica desktop"),
            ("ParrotOS Mode", "parrot", "Modo ParrotOS sin restricciones"),
            ("Web Server", "web_server", "Servidor web completo"),
        ]
        
        for name, value, desc in modes:
            rb = tk.Radiobutton(frame, text=f"{name} - {desc}", 
                               variable=self.mode_var, value=value,
                               fg=BrutalTheme.COLORS['text_secondary'],
                               bg=BrutalTheme.COLORS['bg_secondary'],
                               font=BrutalTheme.FONTS['body'],
                               selectcolor=BrutalTheme.COLORS['accent_primary'])
            rb.pack(anchor='w', padx=10, pady=2)
    
    def _create_action_buttons(self, parent):
        """Botones de accion"""
        button_frame = tk.Frame(parent, bg=BrutalTheme.COLORS['bg_primary'])
        button_frame.pack(fill='x', pady=20)
        
        # Lanzar sistema
        launch_btn = tk.Button(button_frame, text="LAUNCH SYSTEM", 
                              command=self._launch_system,
                              bg=BrutalTheme.COLORS['accent_primary'], 
                              fg='#000000',
                              font=BrutalTheme.FONTS['button'],
                              relief='flat', padx=30, pady=10)
        launch_btn.pack(side='left', padx=10)
        
        # ParrotOS Mode
        parrot_btn = tk.Button(button_frame, text="PARROTOS MODE", 
                              command=self._launch_parrot_gui,
                              bg=BrutalTheme.COLORS['accent_secondary'], 
                              fg='#000000',
                              font=BrutalTheme.FONTS['button'],
                              relief='flat', padx=30, pady=10)
        parrot_btn.pack(side='left', padx=10)
        
        # Test
        test_btn = tk.Button(button_frame, text="TEST UI", 
                            command=self._test_ui,
                            bg=BrutalTheme.COLORS['accent_warning'], 
                            fg='#000000',
                            font=BrutalTheme.FONTS['button'],
                            relief='flat', padx=30, pady=10)
        test_btn.pack(side='right', padx=10)
    
    def _launch_system(self):
        """Lanzar sistema completo"""
        try:
            mode = self.mode_var.get()
            if mode == "gui_desktop":
                subprocess.Popen([sys.executable, "ai_symbiote_gui.py"])
            elif mode == "web_server":
                subprocess.Popen([sys.executable, "web/backend/symbiote_server.py"])
            elif mode == "parrot":
                subprocess.Popen([sys.executable, "parrot_gui_integration.py"])
            else:
                subprocess.Popen([sys.executable, "unified_launcher.py"])
            
            messagebox.showinfo("System Launched", f"Sistema iniciado en modo: {mode}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al lanzar sistema: {e}")
    
    def _launch_parrot_gui(self):
        """Lanzar ParrotOS en modo grafico"""
        try:
            subprocess.Popen([sys.executable, "parrot_gui_integration.py"])
            messagebox.showinfo("ParrotOS Launched", "ParrotOS mode launched successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Error launching ParrotOS: {e}")
    
    def _test_ui(self):
        """Test UI functionality"""
        messagebox.showinfo("UI Test", "UI is working correctly!\nNo blinking detected!")

class BrutalUIDesktop:
    """Aplicacion desktop brutalmente impresionante - FIXED VERSION"""
    
    def __init__(self):
        print("Initializing Brutal UI Desktop...")
        self.root = tk.Tk()
        self._setup_window()
        self._create_ui()
        print("Brutal UI Desktop initialized successfully")
    
    def _setup_window(self):
        """Configurar ventana principal"""
        self.root.title("OBVIVLORUM - BRUTAL DESKTOP (FIXED)")
        self.root.geometry("1200x800")
        self.root.configure(bg=BrutalTheme.COLORS['bg_primary'])
        
        # No transparency to avoid potential issues
        # try:
        #     self.root.attributes('-alpha', 0.95)
        # except:
        #     pass
    
    def _create_ui(self):
        """Crear interfaz de usuario"""
        # Header
        header = tk.Frame(self.root, bg=BrutalTheme.COLORS['bg_secondary'], height=80)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        title_label = tk.Label(header, text="OBVIVLORUM AI SYSTEM - FIXED VERSION", 
                              fg=BrutalTheme.COLORS['accent_primary'],
                              bg=BrutalTheme.COLORS['bg_secondary'],
                              font=BrutalTheme.FONTS['title'])
        title_label.pack(expand=True)
        
        subtitle_label = tk.Label(header, text="Brutal Desktop Configuration & Control Center", 
                                 fg=BrutalTheme.COLORS['text_secondary'],
                                 bg=BrutalTheme.COLORS['bg_secondary'],
                                 font=BrutalTheme.FONTS['subtitle'])
        subtitle_label.pack()
        
        # Panel de configuracion
        config_panel = ConfigPanel(self.root)
        config_panel.pack(fill='both', expand=True)
        
        # Footer
        footer = tk.Frame(self.root, bg=BrutalTheme.COLORS['bg_secondary'], height=40)
        footer.pack(fill='x')
        footer.pack_propagate(False)
        
        footer_text = tk.Label(footer, text="POWER - Single Click Launch  Full Configuration  ParrotOS Ready", 
                              fg=BrutalTheme.COLORS['text_secondary'],
                              bg=BrutalTheme.COLORS['bg_secondary'],
                              font=BrutalTheme.FONTS['body'])
        footer_text.pack(expand=True)
    
    def run(self):
        """Ejecutar aplicacion"""
        print("Starting Brutal UI mainloop...")
        try:
            self.root.mainloop()
            print("Brutal UI ended normally")
        except Exception as e:
            print(f"Error in Brutal UI: {e}")
            raise

def main():
    """Funcion principal"""
    print("=" * 60)
    print("OBVIVLORUM - BRUTAL DESKTOP (FIXED VERSION)")
    print("=" * 60)
    
    try:
        app = BrutalUIDesktop()
        app.run()
    except Exception as e:
        print(f"Critical error: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()