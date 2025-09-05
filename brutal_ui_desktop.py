#!/usr/bin/env python3
"""
Brutal UI Desktop - Brutalmente Impresionante
==============================================

Interfaz desktop con diseño UI/UX brutalmente impresionante
Configuración completa, launcher unificado y soporte ParrotOS
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import tkinter.font as tkFont
import threading
import asyncio
import sys
import os
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
import json

# Import our unified systems
sys.path.insert(0, str(Path(__file__).parent))
from unified_config_system import UnifiedConfigSystem, SystemMode, ScalingLevel, AIProvider

class BrutalTheme:
    """Tema visual brutalmente impresionante"""
    
    COLORS = {
        'bg_primary': '#0a0a0a',        # Negro profundo
        'bg_secondary': '#1a1a1a',      # Negro carbón
        'bg_accent': '#2d2d2d',         # Gris oscuro
        'text_primary': '#ffffff',       # Blanco puro
        'text_secondary': '#b0b0b0',     # Gris claro
        'accent_primary': '#00ff88',     # Verde neón
        'accent_secondary': '#ff6b35',   # Naranja vibrante
        'accent_warning': '#ffeb3b',     # Amarillo brillante
        'accent_error': '#f44336',       # Rojo intenso
        'gradient_start': '#1e3c72',     # Azul profundo
        'gradient_end': '#2a5298',       # Azul medio
    }
    
    FONTS = {
        'title': ('Inter Black', 24, 'bold'),
        'subtitle': ('Inter SemiBold', 16, 'bold'),
        'body': ('Inter', 12, 'normal'),
        'mono': ('JetBrains Mono', 11, 'normal'),
        'button': ('Inter Medium', 11, 'bold'),
    }

class AnimatedButton(tk.Canvas):
    """Botón con animaciones brutales"""
    
    def __init__(self, parent, text, command=None, **kwargs):
        super().__init__(parent, highlightthickness=0, **kwargs)
        self.text = text
        self.command = command
        self.hover_scale = 1.0
        
        self.config(bg=BrutalTheme.COLORS['bg_secondary'])
        self.bind('<Button-1>', self._on_click)
        self.bind('<Enter>', self._on_enter)
        self.bind('<Leave>', self._on_leave)
        
        self._draw_button()
    
    def _draw_button(self):
        """Dibujar botón con efectos"""
        self.delete('all')
        w, h = self.winfo_reqwidth() or 200, self.winfo_reqheight() or 40
        
        # Gradiente de fondo
        for i in range(h):
            color_ratio = i / h
            r1, g1, b1 = self._hex_to_rgb(BrutalTheme.COLORS['gradient_start'])
            r2, g2, b2 = self._hex_to_rgb(BrutalTheme.COLORS['gradient_end'])
            
            r = int(r1 + (r2 - r1) * color_ratio)
            g = int(g1 + (g2 - g1) * color_ratio)
            b = int(b1 + (b2 - b1) * color_ratio)
            
            color = f'#{r:02x}{g:02x}{b:02x}'
            self.create_line(0, i, w, i, fill=color, width=1)
        
        # Borde brillante
        self.create_rectangle(2, 2, w-2, h-2, outline=BrutalTheme.COLORS['accent_primary'], width=2)
        
        # Texto con sombra
        shadow_x, shadow_y = 2, 2
        self.create_text(w//2 + shadow_x, h//2 + shadow_y, text=self.text, 
                        fill='#000000', font=BrutalTheme.FONTS['button'])
        self.create_text(w//2, h//2, text=self.text, 
                        fill=BrutalTheme.COLORS['text_primary'], font=BrutalTheme.FONTS['button'])
    
    def _hex_to_rgb(self, hex_color):
        """Convertir hex a RGB"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def _on_click(self, event):
        """Manejar click"""
        if self.command:
            self.command()
    
    def _on_enter(self, event):
        """Animación hover"""
        self.config(cursor='hand2')
        # Aquí podrías agregar más efectos de hover
    
    def _on_leave(self, event):
        """Salir de hover"""
        self.config(cursor='')

class ConfigPanel(ttk.Frame):
    """Panel de configuración avanzada"""
    
    def __init__(self, parent, config_system: UnifiedConfigSystem):
        super().__init__(parent)
        self.config_system = config_system
        self._create_widgets()
    
    def _create_widgets(self):
        """Crear widgets de configuración"""
        # Título
        title = tk.Label(self, text="⚙️ CONFIGURACIÓN BRUTAL", 
                        fg=BrutalTheme.COLORS['accent_primary'],
                        bg=BrutalTheme.COLORS['bg_primary'],
                        font=BrutalTheme.FONTS['title'])
        title.pack(pady=20)
        
        # Frame para opciones
        options_frame = tk.Frame(self, bg=BrutalTheme.COLORS['bg_primary'])
        options_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Modo del sistema
        self._create_mode_section(options_frame)
        
        # Escalamiento
        self._create_scaling_section(options_frame)
        
        # Proveedor IA
        self._create_ai_section(options_frame)
        
        # ParrotOS Integration
        self._create_parrot_section(options_frame)
        
        # Botones de acción
        self._create_action_buttons(options_frame)
    
    def _create_mode_section(self, parent):
        """Sección de modo del sistema"""
        frame = tk.LabelFrame(parent, text="Modo del Sistema", 
                             fg=BrutalTheme.COLORS['text_primary'],
                             bg=BrutalTheme.COLORS['bg_secondary'],
                             font=BrutalTheme.FONTS['subtitle'])
        frame.pack(fill='x', pady=10)
        
        self.mode_var = tk.StringVar(value=self.config_system.config.mode.value)
        
        for mode_info in self.config_system.get_available_modes():
            rb = tk.Radiobutton(frame, text=f"{mode_info['name']} - {mode_info['description']}", 
                               variable=self.mode_var, value=mode_info['id'],
                               fg=BrutalTheme.COLORS['text_secondary'],
                               bg=BrutalTheme.COLORS['bg_secondary'],
                               font=BrutalTheme.FONTS['body'],
                               selectcolor=BrutalTheme.COLORS['accent_primary'])
            rb.pack(anchor='w', padx=10, pady=2)
    
    def _create_scaling_section(self, parent):
        """Sección de escalamiento"""
        frame = tk.LabelFrame(parent, text="Escalamiento", 
                             fg=BrutalTheme.COLORS['text_primary'],
                             bg=BrutalTheme.COLORS['bg_secondary'],
                             font=BrutalTheme.FONTS['subtitle'])
        frame.pack(fill='x', pady=10)
        
        self.scaling_var = tk.StringVar(value=self.config_system.config.scaling.value)
        
        for scaling_info in self.config_system.get_scaling_options():
            rb = tk.Radiobutton(frame, text=f"{scaling_info['name']} - {scaling_info['description']}", 
                               variable=self.scaling_var, value=scaling_info['id'],
                               fg=BrutalTheme.COLORS['text_secondary'],
                               bg=BrutalTheme.COLORS['bg_secondary'],
                               font=BrutalTheme.FONTS['body'],
                               selectcolor=BrutalTheme.COLORS['accent_secondary'])
            rb.pack(anchor='w', padx=10, pady=2)
    
    def _create_ai_section(self, parent):
        """Sección de IA"""
        frame = tk.LabelFrame(parent, text="Proveedor de IA", 
                             fg=BrutalTheme.COLORS['text_primary'],
                             bg=BrutalTheme.COLORS['bg_secondary'],
                             font=BrutalTheme.FONTS['subtitle'])
        frame.pack(fill='x', pady=10)
        
        self.ai_var = tk.StringVar(value=self.config_system.config.ai_provider.value)
        
        ai_options = [
            ("Local GGUF", "local_gguf", "Modelos locales optimizados"),
            ("OpenAI GPT-4", "openai_gpt4", "API de OpenAI"),
            ("Claude Sonnet", "claude_sonnet", "API de Anthropic"),
            ("Multi-Provider", "hybrid_multi", "Múltiples proveedores inteligentes")
        ]
        
        for name, value, desc in ai_options:
            rb = tk.Radiobutton(frame, text=f"{name} - {desc}", 
                               variable=self.ai_var, value=value,
                               fg=BrutalTheme.COLORS['text_secondary'],
                               bg=BrutalTheme.COLORS['bg_secondary'],
                               font=BrutalTheme.FONTS['body'],
                               selectcolor=BrutalTheme.COLORS['accent_warning'])
            rb.pack(anchor='w', padx=10, pady=2)
    
    def _create_parrot_section(self, parent):
        """Sección ParrotOS"""
        frame = tk.LabelFrame(parent, text="🦜 ParrotOS Integration", 
                             fg=BrutalTheme.COLORS['accent_primary'],
                             bg=BrutalTheme.COLORS['bg_secondary'],
                             font=BrutalTheme.FONTS['subtitle'])
        frame.pack(fill='x', pady=10)
        
        self.parrot_enabled = tk.BooleanVar(value=True)
        parrot_cb = tk.Checkbutton(frame, text="Habilitar modo ParrotOS sin restricciones", 
                                  variable=self.parrot_enabled,
                                  fg=BrutalTheme.COLORS['text_primary'],
                                  bg=BrutalTheme.COLORS['bg_secondary'],
                                  font=BrutalTheme.FONTS['body'])
        parrot_cb.pack(anchor='w', padx=10, pady=5)
        
        # Botón para lanzar ParrotOS
        parrot_btn = AnimatedButton(frame, "🚀 LANZAR PARROTOS GRÁFICO", 
                                   command=self._launch_parrot_gui,
                                   width=300, height=40)
        parrot_btn.pack(pady=10)
    
    def _create_action_buttons(self, parent):
        """Botones de acción"""
        button_frame = tk.Frame(parent, bg=BrutalTheme.COLORS['bg_primary'])
        button_frame.pack(fill='x', pady=20)
        
        # Aplicar configuración
        apply_btn = AnimatedButton(button_frame, "💾 APLICAR CONFIGURACIÓN", 
                                  command=self._apply_config,
                                  width=250, height=50)
        apply_btn.pack(side='left', padx=10)
        
        # Modo brutal UI
        brutal_btn = AnimatedButton(button_frame, "🔥 MODO BRUTAL UI", 
                                   command=self._enable_brutal_mode,
                                   width=250, height=50)
        brutal_btn.pack(side='left', padx=10)
        
        # Lanzar sistema
        launch_btn = AnimatedButton(button_frame, "🚀 LANZAR SISTEMA", 
                                   command=self._launch_system,
                                   width=250, height=50)
        launch_btn.pack(side='right', padx=10)
    
    def _apply_config(self):
        """Aplicar configuración"""
        try:
            # Update config based on selections
            self.config_system.config.mode = SystemMode(self.mode_var.get())
            self.config_system.config.scaling = ScalingLevel(self.scaling_var.get())
            self.config_system.config.ai_provider = AIProvider(self.ai_var.get())
            
            self.config_system.save_config()
            messagebox.showinfo("✅ Éxito", "Configuración aplicada correctamente")
            
        except Exception as e:
            messagebox.showerror("❌ Error", f"Error al aplicar configuración: {e}")
    
    def _enable_brutal_mode(self):
        """Habilitar modo brutal UI"""
        self.config_system.set_brutal_ui_mode()
        messagebox.showinfo("🔥 Brutal Mode", "Modo Brutal UI activado. Reinicia la aplicación para ver los cambios.")
    
    def _launch_system(self):
        """Lanzar sistema completo"""
        try:
            mode = self.mode_var.get()
            if mode == "gui_desktop":
                subprocess.Popen([sys.executable, "ai_symbiote_gui.py"])
            elif mode == "web_server":
                subprocess.Popen([sys.executable, "web/backend/symbiote_server.py"])
            else:
                subprocess.Popen([sys.executable, "unified_launcher.py", f"--mode={mode}"])
            
            messagebox.showinfo("🚀 Sistema Lanzado", f"Sistema iniciado en modo: {mode}")
            
        except Exception as e:
            messagebox.showerror("❌ Error", f"Error al lanzar sistema: {e}")
    
    def _launch_parrot_gui(self):
        """Lanzar ParrotOS en modo gráfico"""
        try:
            # Check if running on Linux
            if os.name != 'posix':
                messagebox.showinfo("ℹ️ Info", "ParrotOS mode is designed for Linux systems")
                return
            
            # Launch ParrotOS tools with GUI
            parrot_commands = [
                "sudo python3 demo_parrot.py --gui",
                "python3 linux_executor.py --parrot-mode --gui"
            ]
            
            for cmd in parrot_commands:
                try:
                    subprocess.Popen(cmd.split())
                    break
                except FileNotFoundError:
                    continue
            else:
                messagebox.showwarning("⚠️ Warning", "ParrotOS components not found")
            
        except Exception as e:
            messagebox.showerror("❌ Error", f"Error launching ParrotOS: {e}")

class BrutalUIDesktop:
    """Aplicación desktop brutalmente impresionante"""
    
    def __init__(self):
        self.config_system = UnifiedConfigSystem()
        self.root = tk.Tk()
        self._setup_window()
        self._create_ui()
    
    def _setup_window(self):
        """Configurar ventana principal"""
        self.root.title("🔥 OBVIVLORUM - BRUTAL DESKTOP")
        self.root.geometry("1600x1000")
        self.root.configure(bg=BrutalTheme.COLORS['bg_primary'])
        
        # Configurar transparencia si está disponible
        try:
            self.root.attributes('-alpha', 0.95)
        except:
            pass
        
        # Icono de la ventana (si existe)
        try:
            self.root.iconbitmap('icon.ico')
        except:
            pass
    
    def _create_ui(self):
        """Crear interfaz de usuario"""
        # Header con gradiente
        header = tk.Frame(self.root, bg=BrutalTheme.COLORS['bg_secondary'], height=80)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        title_label = tk.Label(header, text="🧠 OBVIVLORUM AI SYSTEM", 
                              fg=BrutalTheme.COLORS['accent_primary'],
                              bg=BrutalTheme.COLORS['bg_secondary'],
                              font=BrutalTheme.FONTS['title'])
        title_label.pack(expand=True)
        
        subtitle_label = tk.Label(header, text="Brutal Desktop Configuration & Control Center", 
                                 fg=BrutalTheme.COLORS['text_secondary'],
                                 bg=BrutalTheme.COLORS['bg_secondary'],
                                 font=BrutalTheme.FONTS['subtitle'])
        subtitle_label.pack()
        
        # Panel de configuración
        config_panel = ConfigPanel(self.root, self.config_system)
        config_panel.pack(fill='both', expand=True)
        config_panel.configure(bg=BrutalTheme.COLORS['bg_primary'])
        
        # Footer
        footer = tk.Frame(self.root, bg=BrutalTheme.COLORS['bg_secondary'], height=40)
        footer.pack(fill='x')
        footer.pack_propagate(False)
        
        footer_text = tk.Label(footer, text="⚡ Single Click Launch • Full Configuration • ParrotOS Ready ⚡", 
                              fg=BrutalTheme.COLORS['text_secondary'],
                              bg=BrutalTheme.COLORS['bg_secondary'],
                              font=BrutalTheme.FONTS['body'])
        footer_text.pack(expand=True)
    
    def run(self):
        """Ejecutar aplicación"""
        self.root.mainloop()

def main():
    """Función principal"""
    app = BrutalUIDesktop()
    app.run()

if __name__ == "__main__":
    main()