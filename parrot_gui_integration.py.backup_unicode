#!/usr/bin/env python3
"""
ParrotOS GUI Integration - Sin Restricciones
============================================

Integración gráfica completa con ParrotOS para testing
sin límites ni restricciones según instrucciones del usuario
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import subprocess
import threading
import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

class ParrotToolsGUI:
    """GUI para herramientas de ParrotOS sin restricciones"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.setup_window()
        self.process_list = []
        self.output_widgets = {}
    
    def setup_window(self):
        """Configurar ventana principal"""
        self.root.title("🦜 ParrotOS Integration - Unrestricted Mode")
        self.root.geometry("1400x800")
        self.root.configure(bg='#0f1419')
        
        # Tema oscuro ParrotOS
        self.colors = {
            'bg': '#0f1419',
            'panel': '#1a1f2e', 
            'accent': '#00d9ff',
            'success': '#00ff5f',
            'warning': '#ff8700',
            'error': '#ff5f5f',
            'text': '#ffffff'
        }
        
        self.create_ui()
    
    def create_ui(self):
        """Crear interfaz de usuario"""
        # Header
        header = tk.Frame(self.root, bg=self.colors['panel'], height=60)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        title = tk.Label(header, text="🦜 PARROTOS UNRESTRICTED MODE", 
                        fg=self.colors['accent'], bg=self.colors['panel'],
                        font=('JetBrains Mono', 16, 'bold'))
        title.pack(expand=True)
        
        subtitle = tk.Label(header, text="Testing Mode - No Limits • No Restrictions • Full Access", 
                           fg=self.colors['success'], bg=self.colors['panel'],
                           font=('JetBrains Mono', 10))
        subtitle.pack()
        
        # Main content
        main_frame = tk.Frame(self.root, bg=self.colors['bg'])
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left panel - Tools
        self.create_tools_panel(main_frame)
        
        # Right panel - Output
        self.create_output_panel(main_frame)
        
        # Bottom panel - Command line
        self.create_command_panel(main_frame)
    
    def create_tools_panel(self, parent):
        """Panel de herramientas"""
        tools_frame = tk.LabelFrame(parent, text="🔧 ParrotOS Tools", 
                                   fg=self.colors['text'], bg=self.colors['panel'],
                                   font=('JetBrains Mono', 12, 'bold'))
        tools_frame.pack(side='left', fill='y', padx=(0, 5))
        
        # Categorías de herramientas
        categories = {
            "Network Analysis": [
                ("Nmap Port Scan", "nmap -sS -O"),
                ("Netstat Connections", "netstat -tulpn"),
                ("Network Discovery", "arp-scan -l"),
                ("WiFi Networks", "iwlist scan")
            ],
            "System Information": [
                ("System Info", "uname -a && lscpu"),
                ("Memory Usage", "free -h && df -h"),
                ("Running Processes", "ps aux"),
                ("Network Interfaces", "ip addr show")
            ],
            "Security Testing": [
                ("Security Headers", "curl -I"),
                ("DNS Lookup", "dig"),
                ("Trace Route", "traceroute"),
                ("Port Connectivity", "nc -zv")
            ],
            "Custom Commands": [
                ("Custom Shell", "bash"),
                ("Python REPL", "python3"),
                ("File Explorer", "ls -la"),
                ("Text Editor", "nano")
            ]
        }
        
        # Crear botones por categoría
        for category, tools in categories.items():
            cat_frame = tk.LabelFrame(tools_frame, text=category,
                                     fg=self.colors['accent'], bg=self.colors['panel'],
                                     font=('JetBrains Mono', 10, 'bold'))
            cat_frame.pack(fill='x', padx=5, pady=5)
            
            for tool_name, base_command in tools:
                btn = tk.Button(cat_frame, text=tool_name,
                               command=lambda cmd=base_command, name=tool_name: self.run_tool(cmd, name),
                               bg=self.colors['bg'], fg=self.colors['text'],
                               font=('JetBrains Mono', 9),
                               relief='flat', padx=5, pady=2)
                btn.pack(fill='x', padx=2, pady=1)
    
    def create_output_panel(self, parent):
        """Panel de salida"""
        output_frame = tk.LabelFrame(parent, text="📟 Output Terminal", 
                                    fg=self.colors['text'], bg=self.colors['panel'],
                                    font=('JetBrains Mono', 12, 'bold'))
        output_frame.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        # Notebook para múltiples terminales
        self.notebook = ttk.Notebook(output_frame)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Terminal principal
        self.create_terminal_tab("Main Terminal")
    
    def create_terminal_tab(self, tab_name):
        """Crear pestaña de terminal"""
        frame = tk.Frame(self.notebook, bg=self.colors['bg'])
        self.notebook.add(frame, text=tab_name)
        
        # Terminal output
        terminal = scrolledtext.ScrolledText(frame, 
                                           bg='#000000', fg='#00ff00',
                                           font=('JetBrains Mono', 10),
                                           insertbackground='#00ff00')
        terminal.pack(fill='both', expand=True)
        
        # Info inicial
        terminal.insert(tk.END, f"🦜 ParrotOS Terminal - {tab_name}\n")
        terminal.insert(tk.END, "=" * 50 + "\n")
        terminal.insert(tk.END, "Ready for unrestricted testing!\n\n")
        
        self.output_widgets[tab_name] = terminal
        return terminal
    
    def create_command_panel(self, parent):
        """Panel de línea de comandos"""
        cmd_frame = tk.Frame(parent, bg=self.colors['bg'])
        cmd_frame.pack(fill='x', pady=(10, 0))
        
        # Input personalizado
        input_frame = tk.Frame(cmd_frame, bg=self.colors['panel'])
        input_frame.pack(fill='x')
        
        tk.Label(input_frame, text="🦜 $ ", 
                fg=self.colors['accent'], bg=self.colors['panel'],
                font=('JetBrains Mono', 12, 'bold')).pack(side='left')
        
        self.command_entry = tk.Entry(input_frame, 
                                    bg=self.colors['bg'], fg=self.colors['text'],
                                    font=('JetBrains Mono', 11),
                                    insertbackground=self.colors['accent'])
        self.command_entry.pack(side='left', fill='x', expand=True)
        self.command_entry.bind('<Return>', self.execute_custom_command)
        
        # Botones de acción
        btn_frame = tk.Frame(input_frame, bg=self.colors['panel'])
        btn_frame.pack(side='right')
        
        execute_btn = tk.Button(btn_frame, text="Execute", 
                               command=self.execute_custom_command,
                               bg=self.colors['success'], fg='#000000',
                               font=('JetBrains Mono', 9, 'bold'))
        execute_btn.pack(side='left', padx=2)
        
        clear_btn = tk.Button(btn_frame, text="Clear", 
                             command=self.clear_terminal,
                             bg=self.colors['warning'], fg='#000000',
                             font=('JetBrains Mono', 9, 'bold'))
        clear_btn.pack(side='left', padx=2)
        
        new_tab_btn = tk.Button(btn_frame, text="New Tab", 
                               command=self.create_new_tab,
                               bg=self.colors['accent'], fg='#000000',
                               font=('JetBrains Mono', 9, 'bold'))
        new_tab_btn.pack(side='left', padx=2)
    
    def run_tool(self, command: str, tool_name: str):
        """Ejecutar herramienta sin restricciones"""
        def execute():
            try:
                # Crear nueva pestaña para esta herramienta
                tab_name = f"{tool_name}"
                if tab_name not in self.output_widgets:
                    terminal = self.create_terminal_tab(tab_name)
                else:
                    terminal = self.output_widgets[tab_name]
                
                # Cambiar a la pestaña
                for i, tab_id in enumerate(self.notebook.tabs()):
                    if self.notebook.tab(tab_id, 'text') == tab_name:
                        self.notebook.select(i)
                        break
                
                terminal.insert(tk.END, f"🚀 Executing: {command}\n")
                terminal.insert(tk.END, "-" * 50 + "\n")
                terminal.see(tk.END)
                
                # Mostrar diálogo de parámetros si es necesario
                if command.endswith(("curl -I", "dig", "traceroute", "nc -zv")):
                    target = tk.simpledialog.askstring("Target", f"Enter target for {tool_name}:")
                    if target:
                        command = f"{command} {target}"
                    else:
                        terminal.insert(tk.END, "❌ Operation cancelled\n\n")
                        return
                
                # Ejecutar comando
                if os.name == 'posix':  # Linux/Unix
                    process = subprocess.Popen(command, shell=True, 
                                             stdout=subprocess.PIPE, 
                                             stderr=subprocess.STDOUT, 
                                             universal_newlines=True, 
                                             bufsize=1)
                    
                    # Leer salida en tiempo real
                    while True:
                        line = process.stdout.readline()
                        if not line and process.poll() is not None:
                            break
                        if line:
                            terminal.insert(tk.END, line)
                            terminal.see(tk.END)
                            self.root.update()
                    
                    terminal.insert(tk.END, f"\n✅ Command completed (exit code: {process.returncode})\n\n")
                    
                else:
                    # Windows - mostrar mensaje informativo
                    terminal.insert(tk.END, "ℹ️ ParrotOS tools are designed for Linux systems\n")
                    terminal.insert(tk.END, f"Command that would run: {command}\n")
                    terminal.insert(tk.END, "💡 Use WSL or Linux VM for full functionality\n\n")
                
            except Exception as e:
                terminal.insert(tk.END, f"❌ Error: {e}\n\n")
            
            terminal.see(tk.END)
        
        # Ejecutar en hilo separado para no bloquear UI
        thread = threading.Thread(target=execute)
        thread.daemon = True
        thread.start()
    
    def execute_custom_command(self, event=None):
        """Ejecutar comando personalizado"""
        command = self.command_entry.get().strip()
        if not command:
            return
        
        # Obtener terminal actual
        current_tab = self.notebook.select()
        tab_text = self.notebook.tab(current_tab, 'text')
        terminal = self.output_widgets[tab_text]
        
        terminal.insert(tk.END, f"🦜 $ {command}\n")
        
        # Limpiar entrada
        self.command_entry.delete(0, tk.END)
        
        # Ejecutar comando
        self.run_tool(command, "Custom Command")
    
    def clear_terminal(self):
        """Limpiar terminal actual"""
        current_tab = self.notebook.select()
        tab_text = self.notebook.tab(current_tab, 'text')
        terminal = self.output_widgets[tab_text]
        terminal.delete(1.0, tk.END)
        terminal.insert(tk.END, f"🦜 ParrotOS Terminal - {tab_text}\n")
        terminal.insert(tk.END, "=" * 50 + "\n")
        terminal.insert(tk.END, "Terminal cleared. Ready for new commands!\n\n")
    
    def create_new_tab(self):
        """Crear nueva pestaña"""
        tab_count = len(self.output_widgets) + 1
        tab_name = f"Terminal {tab_count}"
        self.create_terminal_tab(tab_name)
        
        # Cambiar a la nueva pestaña
        self.notebook.select(len(self.notebook.tabs()) - 1)
    
    def run(self):
        """Ejecutar aplicación"""
        # Mostrar advertencia/info inicial
        messagebox.showinfo("🦜 ParrotOS Mode", 
                           "ParrotOS Integration Mode Activated\n\n"
                           "⚠️ This mode provides unrestricted access for testing\n"
                           "🔧 All tools available without limitations\n"
                           "🚀 Use responsibly for legitimate testing only")
        
        self.root.mainloop()

def main():
    """Función principal"""
    # Verificar si estamos en Linux para funcionalidad completa
    if os.name != 'posix':
        print("⚠️ Warning: ParrotOS tools work best on Linux systems")
        print("💡 Consider using WSL or a Linux VM for full functionality")
    
    app = ParrotToolsGUI()
    app.run()

if __name__ == "__main__":
    import tkinter.simpledialog
    main()