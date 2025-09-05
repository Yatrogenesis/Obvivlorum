#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Simple GUI - 100% Working Version
====================================
GUI version of ai_simple_working.py - guaranteed to work without Unicode issues
"""

import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk
import threading
import asyncio
import sys
import os

# Import the working AI engine
try:
    from ai_simple_working import SimpleWorkingAI, process_ai_message
except ImportError:
    print("ERROR: Could not import ai_simple_working.py")
    sys.exit(1)

class SimpleGUIWorking:
    """Simple GUI for the working AI engine."""
    
    def __init__(self):
        """Initialize GUI."""
        self.root = tk.Tk()
        self.ai_engine = None
        self.setup_window()
        self.create_ui()
        
        # Initialize AI engine
        threading.Thread(target=self.init_ai_engine, daemon=True).start()
        
    def setup_window(self):
        """Setup main window."""
        self.root.title("OBVIVLORUM AI - Simple Working Version")
        self.root.geometry("800x600")
        self.root.configure(bg='#1a1a1a')
        
    def create_ui(self):
        """Create user interface."""
        # Title
        title_frame = tk.Frame(self.root, bg='#1a1a1a')
        title_frame.pack(fill='x', padx=10, pady=(10, 5))
        
        title = tk.Label(title_frame, 
                        text="OBVIVLORUM AI - Simple Working Version", 
                        fg='#00ff88', bg='#1a1a1a',
                        font=('Arial', 14, 'bold'))
        title.pack()
        
        # Status
        self.status_label = tk.Label(title_frame, 
                                   text="Status: Initializing AI Engine...", 
                                   fg='#ffaa00', bg='#1a1a1a',
                                   font=('Arial', 10))
        self.status_label.pack()
        
        # Chat area
        chat_frame = tk.Frame(self.root, bg='#1a1a1a')
        chat_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame, wrap=tk.WORD,
            bg='#000000', fg='#ffffff',
            font=('Consolas', 10),
            insertbackground='#ffffff'
        )
        self.chat_display.pack(fill='both', expand=True)
        
        # Input area
        input_frame = tk.Frame(self.root, bg='#1a1a1a')
        input_frame.pack(fill='x', padx=10, pady=5)
        
        self.input_field = tk.Entry(input_frame, 
                                   bg='#333333', fg='#ffffff',
                                   font=('Arial', 11),
                                   insertbackground='#ffffff')
        self.input_field.pack(side='left', fill='x', expand=True, padx=(0, 10))
        self.input_field.bind('<Return>', self.send_message)
        
        self.send_button = tk.Button(input_frame, text="Send", 
                                   command=self.send_message,
                                   bg='#00ff88', fg='#000000',
                                   font=('Arial', 10, 'bold'),
                                   padx=20)
        self.send_button.pack(side='right')
        
        # Control buttons
        control_frame = tk.Frame(self.root, bg='#1a1a1a')
        control_frame.pack(fill='x', padx=10, pady=5)
        
        self.clear_button = tk.Button(control_frame, text="Clear Chat",
                                    command=self.clear_chat,
                                    bg='#ff6666', fg='#ffffff',
                                    font=('Arial', 9))
        self.clear_button.pack(side='left', padx=(0, 10))
        
        self.status_button = tk.Button(control_frame, text="AI Status",
                                     command=self.show_ai_status,
                                     bg='#6666ff', fg='#ffffff',
                                     font=('Arial', 9))
        self.status_button.pack(side='left')
        
        self.config_button = tk.Button(control_frame, text="Config",
                                     command=self.show_config,
                                     bg='#ff8800', fg='#ffffff',
                                     font=('Arial', 9))
        self.config_button.pack(side='left', padx=(10, 0))
        
        # Initial messages
        self.add_system_message("OBVIVLORUM AI Simple Working Version")
        self.add_system_message("No Unicode issues - Stable version for i5+12GB systems")
        
    def init_ai_engine(self):
        """Initialize AI engine in background."""
        try:
            self.ai_engine = SimpleWorkingAI()
            self.root.after(0, lambda: self.status_label.config(
                text="Status: AI Engine Ready!", fg='#00ff88'))
            self.root.after(0, lambda: self.add_system_message(
                "AI Engine initialized successfully - Ready for chat!"))
        except Exception as e:
            self.root.after(0, lambda: self.status_label.config(
                text=f"Status: AI Engine Error - {e}", fg='#ff6666'))
            self.root.after(0, lambda: self.add_system_message(
                f"ERROR: Failed to initialize AI engine: {e}"))
    
    def add_message(self, sender, message, color='#ffffff'):
        """Add message to chat display."""
        self.chat_display.config(state='normal')
        self.chat_display.insert(tk.END, f"{sender}: ", 'sender')
        self.chat_display.insert(tk.END, f"{message}\n\n", 'message')
        
        # Configure tags for colors
        self.chat_display.tag_configure('sender', foreground=color, font=('Arial', 10, 'bold'))
        self.chat_display.tag_configure('message', foreground='#ffffff')
        
        self.chat_display.see(tk.END)
        self.chat_display.config(state='disabled')
    
    def add_system_message(self, message):
        """Add system message."""
        self.add_message("SYSTEM", message, '#00ff88')
    
    def send_message(self, event=None):
        """Send message to AI."""
        user_input = self.input_field.get().strip()
        if not user_input:
            return
        
        if not self.ai_engine:
            self.add_message("ERROR", "AI Engine not initialized yet", '#ff6666')
            return
        
        # Add user message
        self.add_message("YOU", user_input, '#ffaa00')
        self.input_field.delete(0, tk.END)
        
        # Show thinking
        self.add_message("AI", "Processing your message...", '#888888')
        
        # Process in background
        threading.Thread(target=self.process_message_background, args=(user_input,), daemon=True).start()
    
    def process_message_background(self, user_input):
        """Process message in background thread."""
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Get AI response
            response = loop.run_until_complete(
                self.ai_engine.process_message(user_input)
            )
            
            # Update UI on main thread
            self.root.after(0, lambda: self.replace_last_message("AI", response, '#00dd88'))
            
        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            self.root.after(0, lambda: self.replace_last_message("ERROR", error_msg, '#ff6666'))
    
    def replace_last_message(self, sender, message, color):
        """Replace the last message in chat."""
        self.chat_display.config(state='normal')
        
        # Remove last message
        lines = self.chat_display.get(1.0, tk.END).strip().split('\n')
        if lines and "Processing your message..." in lines[-1]:
            # Find and remove the "Processing..." line
            self.chat_display.delete('end-3l', 'end-1l')
        
        # Add new message
        self.chat_display.insert(tk.END, f"{sender}: ", 'sender')
        self.chat_display.insert(tk.END, f"{message}\n\n", 'message')
        
        # Configure colors
        self.chat_display.tag_configure('sender', foreground=color, font=('Arial', 10, 'bold'))
        self.chat_display.tag_configure('message', foreground='#ffffff')
        
        self.chat_display.see(tk.END)
        self.chat_display.config(state='disabled')
    
    def clear_chat(self):
        """Clear chat display."""
        self.chat_display.config(state='normal')
        self.chat_display.delete(1.0, tk.END)
        self.chat_display.config(state='disabled')
        self.add_system_message("Chat cleared")
    
    def show_ai_status(self):
        """Show AI engine status."""
        if not self.ai_engine:
            self.add_message("STATUS", "AI Engine not initialized", '#ff6666')
            return
        
        try:
            status = self.ai_engine.get_status()
            status_text = f"""AI Engine Status:
- Initialized: {status['initialized']}
- Language: {status['language']}
- Voice Enabled: {status['voice_enabled']}
- Model Type: {status['model_type']}
- Components:
  * Chat AI: {status['components']['chat_ai']}
  * TTS: {status['components']['tts']}
  * Speech Recognition: {status['components']['speech_recognition']}
  * Vision: {status['components']['vision']}
- Conversation Length: {status['conversation_length']}"""
            
            self.add_message("STATUS", status_text, '#66ddff')
            
        except Exception as e:
            self.add_message("ERROR", f"Failed to get AI status: {e}", '#ff6666')
    
    def show_config(self):
        """Show configuration dialog."""
        config_dialog = tk.Toplevel(self.root)
        config_dialog.title("OBVIVLORUM Configuration")
        config_dialog.geometry("500x400")
        config_dialog.configure(bg='#1a1a1a')
        config_dialog.resizable(False, False)
        
        # Center dialog
        config_dialog.update_idletasks()
        x = (config_dialog.winfo_screenwidth() // 2) - (250)
        y = (config_dialog.winfo_screenheight() // 2) - (200)
        config_dialog.geometry(f"500x400+{x}+{y}")
        
        # Title
        title = tk.Label(config_dialog,
                        text="OBVIVLORUM Configuration",
                        bg='#1a1a1a',
                        fg='#00ff88',
                        font=('Arial', 16, 'bold'))
        title.pack(pady=20)
        
        # Configuration options
        config_text = """Configuration Options:

CURRENT MODE: Simple Working Version
• Rule-based responses only
• No API keys required
• Ultra-lightweight and stable
• Optimized for i5+12GB systems

UPGRADE OPTIONS:

1. Real AI with APIs:
   • Run: python api_manager.py
   • Configure OpenAI, Claude, or Gemini keys
   • Switch to ai_gui_with_oauth.py

2. OAuth Social Login:
   • Run: python oauth_manager.py  
   • Setup Google/GitHub/Microsoft login
   • Automatic API access via your accounts

3. TinyLlama Local AI:
   • Run: python ai_engine_tinyllama.py
   • 1.1B parameter model
   • Completely local and private

4. Commercial Launcher:
   • Run: python OBVIVLORUM_LAUNCHER.py
   • Professional mode selection interface
   • Hardware optimization included

SYSTEM OPTIMIZATIONS:
• Unicode cleanup completed
• Memory management optimized
• Process priority enhanced
• Startup time minimized"""
        
        info_label = tk.Label(config_dialog,
                            text=config_text,
                            bg='#1a1a1a',
                            fg='#ffffff',
                            font=('Consolas', 9),
                            justify='left')
        info_label.pack(padx=20, pady=20)
        
        # Close button
        close_button = tk.Button(config_dialog,
                               text="Close",
                               command=config_dialog.destroy,
                               bg='#555555',
                               fg='#ffffff',
                               font=('Arial', 10),
                               padx=20,
                               pady=8)
        close_button.pack(pady=20)
    
    def run(self):
        """Start the GUI."""
        try:
            print("Starting Simple GUI Working...")
            self.root.mainloop()
        except KeyboardInterrupt:
            print("GUI interrupted by user")
        except Exception as e:
            print(f"GUI error: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main function."""
    print("=" * 60)
    print("OBVIVLORUM AI - SIMPLE GUI WORKING")
    print("100% Stable - No Unicode Issues")
    print("Optimized for i5+12GB systems")
    print("=" * 60)
    
    try:
        app = SimpleGUIWorking()
        app.run()
    except Exception as e:
        print(f"Critical error: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()