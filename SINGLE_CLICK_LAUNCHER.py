#!/usr/bin/env python3
"""
🚀 OBVIVLORUM - SINGLE CLICK LAUNCHER
====================================

Lanzador unificado de un solo clic con selector de modo
Prioriza Desktop, Web solo bajo demanda
"""

import tkinter as tk
from tkinter import ttk, messagebox
import sys
import os
import subprocess
from pathlib import Path
from typing import Dict, Any

class ModeSelectorDialog:
    """Dialog para seleccionar modo de arranque"""
    
    def __init__(self):
        self.selected_mode = None
        self.root = tk.Tk()
        self._setup_dialog()
    
    def _setup_dialog(self):
        """Configurar dialog"""
        self.root.title("🚀 OBVIVLORUM LAUNCHER")
        self.root.geometry("600x400")
        self.root.configure(bg='#0a0a0a')
        self.root.resizable(False, False)
        
        # Centrar ventana
        self.root.transient()
        self.root.grab_set()
        
        self._create_ui()
    
    def _create_ui(self):
        """Crear interfaz"""
        # Header
        header = tk.Frame(self.root, bg='#1a1a1a', height=80)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        title = tk.Label(header, text="🧠 OBVIVLORUM", 
                        fg='#00ff88', bg='#1a1a1a',
                        font=('Inter Black', 20, 'bold'))
        title.pack(expand=True)
        
        subtitle = tk.Label(header, text="Selecciona modo de arranque", 
                           fg='#b0b0b0', bg='#1a1a1a',
                           font=('Inter', 12))
        subtitle.pack()
        
        # Opciones de modo
        modes_frame = tk.Frame(self.root, bg='#0a0a0a')
        modes_frame.pack(expand=True, fill='both', padx=40, pady=20)
        
        # Modo Desktop (Predeterminado/Recomendado)
        self._create_mode_option(modes_frame, 
            title="🖥️ DESKTOP MODE", 
            subtitle="Interfaz Gráfica Brutal • Configuración Completa • ParrotOS Ready",
            advantages="✅ UI/UX Impresionante\n✅ Control Total\n✅ Configuración Avanzada\n✅ ParrotOS Integration\n✅ Rendimiento Óptimo",
            disadvantages="❌ Solo local (no web access)\n❌ Requiere GUI disponible",
            command="desktop",
            recommended=True
        )
        
        # Modo Web
        self._create_mode_option(modes_frame,
            title="🌐 WEB MODE", 
            subtitle="Servidor Web • API REST • Acceso Remoto",
            advantages="✅ Acceso desde cualquier dispositivo\n✅ API REST completa\n✅ Multi-usuario\n✅ Responsive design",
            disadvantages="❌ UI menos avanzada\n❌ Requiere navegador\n❌ Mayor uso de recursos",
            command="web",
            recommended=False
        )
        
        # Botón de lanzamiento
        launch_frame = tk.Frame(self.root, bg='#0a0a0a')
        launch_frame.pack(fill='x', padx=40, pady=20)
        
        self.launch_button = tk.Button(launch_frame, text="🚀 LANZAR DESKTOP (RECOMENDADO)", 
                                      command=lambda: self._launch_mode("desktop"),
                                      bg='#00ff88', fg='#000000',
                                      font=('Inter Medium', 12, 'bold'),
                                      relief='flat', padx=20, pady=10)
        self.launch_button.pack()
    
    def _create_mode_option(self, parent, title, subtitle, advantages, disadvantages, command, recommended):
        """Crear opción de modo"""
        frame = tk.Frame(parent, bg='#1a1a1a', relief='ridge', bd=2)
        frame.pack(fill='x', pady=10)
        
        # Header del modo
        header_frame = tk.Frame(frame, bg='#1a1a1a')
        header_frame.pack(fill='x', padx=15, pady=10)
        
        title_label = tk.Label(header_frame, text=title, 
                              fg='#00ff88' if recommended else '#ff6b35',
                              bg='#1a1a1a',
                              font=('Inter SemiBold', 14, 'bold'))
        title_label.pack(anchor='w')
        
        if recommended:
            rec_label = tk.Label(header_frame, text="⭐ RECOMENDADO", 
                                fg='#ffeb3b', bg='#1a1a1a',
                                font=('Inter', 10, 'bold'))
            rec_label.pack(anchor='w')
        
        subtitle_label = tk.Label(header_frame, text=subtitle, 
                                 fg='#b0b0b0', bg='#1a1a1a',
                                 font=('Inter', 10))
        subtitle_label.pack(anchor='w')
        
        # Ventajas y desventajas
        details_frame = tk.Frame(frame, bg='#1a1a1a')
        details_frame.pack(fill='x', padx=15, pady=(0, 10))
        
        # Ventajas
        adv_frame = tk.Frame(details_frame, bg='#1a1a1a')
        adv_frame.pack(side='left', fill='both', expand=True)
        
        adv_label = tk.Label(adv_frame, text=advantages, 
                            fg='#ffffff', bg='#1a1a1a',
                            font=('JetBrains Mono', 9),
                            justify='left')
        adv_label.pack(anchor='w')
        
        # Desventajas
        disadv_frame = tk.Frame(details_frame, bg='#1a1a1a')
        disadv_frame.pack(side='right', fill='both', expand=True)
        
        disadv_label = tk.Label(disadv_frame, text=disadvantages, 
                               fg='#ffffff', bg='#1a1a1a',
                               font=('JetBrains Mono', 9),
                               justify='left')
        disadv_label.pack(anchor='w')
        
        # Botón de selección
        select_btn = tk.Button(frame, text=f"Seleccionar {title.split()[1]}", 
                              command=lambda: self._select_mode(command),
                              bg='#2d2d2d', fg='#ffffff',
                              font=('Inter', 10),
                              relief='flat', padx=15, pady=5)
        select_btn.pack(pady=(0, 10))
    
    def _select_mode(self, mode):
        """Seleccionar modo"""
        if mode == "desktop":
            self.launch_button.config(text="🚀 LANZAR DESKTOP MODE", 
                                    command=lambda: self._launch_mode("desktop"),
                                    bg='#00ff88')
        else:
            self.launch_button.config(text="🚀 LANZAR WEB MODE", 
                                    command=lambda: self._launch_mode("web"),
                                    bg='#ff6b35')
    
    def _launch_mode(self, mode):
        """Lanzar modo seleccionado"""
        self.selected_mode = mode
        self.root.destroy()
    
    def show(self):
        """Mostrar dialog y retornar modo seleccionado"""
        self.root.mainloop()
        return self.selected_mode or "desktop"  # Default a desktop

class SingleClickLauncher:
    """Lanzador principal de un solo clic"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        os.chdir(self.project_root)  # Cambiar al directorio del proyecto
    
    def check_dependencies(self) -> Dict[str, bool]:
        """Verificar dependencias del sistema"""
        deps = {}
        
        # Python components
        deps['ai_engine'] = (self.project_root / "ai_engine_multi_provider.py").exists()
        deps['gui_desktop'] = (self.project_root / "ai_symbiote_gui.py").exists()
        deps['brutal_ui'] = (self.project_root / "brutal_ui_desktop.py").exists()
        deps['web_backend'] = (self.project_root / "web" / "backend" / "symbiote_server.py").exists()
        deps['core_orchestrator'] = (self.project_root / "core_orchestrator.py").exists()
        deps['config_system'] = (self.project_root / "unified_config_system.py").exists()
        
        return deps
    
    def launch_desktop_mode(self):
        """Lanzar modo desktop"""
        print("🚀 Launching DESKTOP MODE...")
        
        try:
            # Verificar dependencias
            deps = self.check_dependencies()
            
            if not deps['brutal_ui']:
                print("⚠️ Brutal UI not found, falling back to standard GUI...")
                if deps['gui_desktop']:
                    subprocess.Popen([sys.executable, "ai_symbiote_gui.py"])
                else:
                    print("❌ No GUI components found!")
                    return False
            else:
                # Lanzar Brutal UI Desktop
                subprocess.Popen([sys.executable, "brutal_ui_desktop.py"])
            
            print("✅ Desktop mode launched successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error launching desktop mode: {e}")
            return False
    
    def launch_web_mode(self):
        """Lanzar modo web"""
        print("🌐 Launching WEB MODE...")
        
        try:
            deps = self.check_dependencies()
            
            if not deps['web_backend']:
                print("❌ Web backend not found!")
                return False
            
            # Lanzar servidor web
            web_server_path = self.project_root / "web" / "backend" / "symbiote_server.py"
            subprocess.Popen([sys.executable, str(web_server_path)])
            
            print("✅ Web server launched!")
            print("🌐 Access at: http://localhost:8000")
            return True
            
        except Exception as e:
            print(f"❌ Error launching web mode: {e}")
            return False
    
    def run(self):
        """Ejecutar launcher principal"""
        print("=" * 60)
        print("🧠 OBVIVLORUM - SINGLE CLICK LAUNCHER")
        print("=" * 60)
        
        # Mostrar selector de modo
        selector = ModeSelectorDialog()
        selected_mode = selector.show()
        
        print(f"Selected mode: {selected_mode}")
        
        # Lanzar modo seleccionado
        if selected_mode == "web":
            success = self.launch_web_mode()
        else:  # Default to desktop
            success = self.launch_desktop_mode()
        
        if not success:
            print("❌ Launch failed!")
            input("Press Enter to exit...")
        else:
            print("🎉 System launched successfully!")
            print("Check your desktop/browser for the application.")

def main():
    """Función principal"""
    launcher = SingleClickLauncher()
    launcher.run()

if __name__ == "__main__":
    main()