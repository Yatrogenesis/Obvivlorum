#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OBVIVLORUM AI GUI with Claude OAuth Authentication
=================================================

GUI version of OBVIVLORUM AI with direct Claude/Anthropic authentication.
No localhost redirect - connects directly to Claude.ai
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import json
import logging
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from oauth_manager import OAuthManager, OAuthProvider
    from ai_simple_working import SimpleWorkingAI
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ClaudeGUI")

class ClaudeAuthDialog:
    """Dialog for Claude/Anthropic authentication."""
    
    def __init__(self, parent):
        """Initialize authentication dialog."""
        self.parent = parent
        self.result = False
        self.user_info = None
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Claude AI Authentication")
        self.dialog.geometry("500x400")
        self.dialog.resizable(False, False)
        self.dialog.configure(bg='#1a1a1a')
        
        # Center dialog
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - 250
        y = (self.dialog.winfo_screenheight() // 2) - 200
        self.dialog.geometry(f"500x400+{x}+{y}")
        
        self.create_ui()
        
        # Make modal
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Start OAuth process
        self.start_authentication()
    
    def create_ui(self):
        """Create authentication UI."""
        # Header
        header_frame = tk.Frame(self.dialog, bg='#1a1a1a')
        header_frame.pack(fill='x', padx=20, pady=20)
        
        title_label = tk.Label(header_frame,
                              text="Claude AI Authentication",
                              bg='#1a1a1a',
                              fg='#D4512A',
                              font=('Segoe UI', 18, 'bold'))
        title_label.pack()
        
        subtitle_label = tk.Label(header_frame,
                                 text="Connect with your Anthropic account",
                                 bg='#1a1a1a',
                                 fg='#888888',
                                 font=('Segoe UI', 10))
        subtitle_label.pack(pady=(5, 0))
        
        # Status area
        self.status_frame = tk.Frame(self.dialog, bg='#1a1a1a')
        self.status_frame.pack(fill='both', expand=True, padx=20)
        
        self.status_label = tk.Label(self.status_frame,
                                   text="Initializing Claude authentication...",
                                   bg='#1a1a1a',
                                   fg='#ffffff',
                                   font=('Segoe UI', 11),
                                   wraplength=460,
                                   justify='left')
        self.status_label.pack(pady=20)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.status_frame, mode='indeterminate')
        self.progress.pack(fill='x', pady=10)
        self.progress.start()
        
        # Buttons
        button_frame = tk.Frame(self.dialog, bg='#1a1a1a')
        button_frame.pack(fill='x', padx=20, pady=20)
        
        self.cancel_button = tk.Button(button_frame,
                                     text="Cancel",
                                     bg='#444444',
                                     fg='#ffffff',
                                     font=('Segoe UI', 10),
                                     padx=20,
                                     pady=5,
                                     border=0,
                                     command=self.cancel)
        self.cancel_button.pack(side='right')
        
        self.retry_button = tk.Button(button_frame,
                                    text="Retry",
                                    bg='#D4512A',
                                    fg='#ffffff',
                                    font=('Segoe UI', 10),
                                    padx=20,
                                    pady=5,
                                    border=0,
                                    command=self.retry,
                                    state='disabled')
        self.retry_button.pack(side='right', padx=(0, 10))
    
    def start_authentication(self):
        """Start Claude authentication in background thread."""
        def auth_thread():
            try:
                self.update_status("Connecting to Claude AI...")
                
                oauth = OAuthManager()
                
                self.update_status("Starting Claude authentication flow...\nBrowser will open for login.")
                
                # Start OAuth flow
                result = oauth.start_oauth_flow(OAuthProvider.CLAUDE)
                
                if result:
                    # Get user info
                    token = oauth.get_token(OAuthProvider.CLAUDE)
                    if token and token.user_info:
                        self.user_info = token.user_info
                        user_name = self.user_info.get('name', self.user_info.get('email', 'Claude User'))
                        self.update_status(f"Authentication successful!\nWelcome, {user_name}!")
                        self.result = True
                        
                        # Auto-close after success
                        self.dialog.after(2000, self.close)
                    else:
                        self.update_status("Authentication completed but no user information available.")
                        self.result = True
                        self.dialog.after(2000, self.close)
                else:
                    self.update_status("Authentication failed or was cancelled.\nPlease try again.")
                    self.enable_retry()
                    
            except Exception as e:
                logger.error(f"Authentication error: {e}")
                self.update_status(f"Authentication error: {str(e)}\nPlease try again.")
                self.enable_retry()
        
        # Start in background thread
        threading.Thread(target=auth_thread, daemon=True).start()
    
    def update_status(self, message):
        """Update status message."""
        def update():
            self.status_label.config(text=message)
            if "successful" in message.lower():
                self.progress.stop()
                self.status_label.config(fg='#00ff88')
            elif "failed" in message.lower() or "error" in message.lower():
                self.progress.stop()
                self.status_label.config(fg='#ff6b6b')
        
        self.dialog.after(0, update)
    
    def enable_retry(self):
        """Enable retry button."""
        def enable():
            self.retry_button.config(state='normal')
            self.progress.stop()
        
        self.dialog.after(0, enable)
    
    def retry(self):
        """Retry authentication."""
        self.retry_button.config(state='disabled')
        self.progress.start()
        self.start_authentication()
    
    def cancel(self):
        """Cancel authentication."""
        self.result = False
        self.close()
    
    def close(self):
        """Close dialog."""
        self.dialog.destroy()

class ClaudeGUI:
    """Main GUI for Claude AI integration."""
    
    def __init__(self):
        """Initialize the GUI."""
        self.root = tk.Tk()
        self.root.title("OBVIVLORUM AI - Claude Mode")
        self.root.geometry("800x600")
        self.root.configure(bg='#1a1a1a')
        
        # State
        self.authenticated = False
        self.user_info = None
        self.ai_engine = None
        
        self.create_ui()
        self.start_authentication()
    
    def create_ui(self):
        """Create the main UI."""
        # Header
        header_frame = tk.Frame(self.root, bg='#1a1a1a', height=80)
        header_frame.pack(fill='x', padx=20, pady=(20, 0))
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(header_frame,
                              text="OBVIVLORUM AI",
                              bg='#1a1a1a',
                              fg='#ffffff',
                              font=('Segoe UI', 20, 'bold'))
        title_label.pack(side='left', pady=20)
        
        subtitle_label = tk.Label(header_frame,
                                 text="Claude Mode",
                                 bg='#1a1a1a',
                                 fg='#D4512A',
                                 font=('Segoe UI', 14))
        subtitle_label.pack(side='left', padx=(20, 0), pady=20)
        
        # Status
        self.status_label = tk.Label(header_frame,
                                   text="Not authenticated",
                                   bg='#1a1a1a',
                                   fg='#888888',
                                   font=('Segoe UI', 10))
        self.status_label.pack(side='right', pady=20)
        
        # Chat area
        chat_frame = tk.Frame(self.root, bg='#1a1a1a')
        chat_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Chat display
        self.chat_display = scrolledtext.ScrolledText(chat_frame,
                                                     bg='#2d2d2d',
                                                     fg='#ffffff',
                                                     font=('Consolas', 10),
                                                     wrap=tk.WORD,
                                                     state='disabled')
        self.chat_display.pack(fill='both', expand=True, pady=(0, 10))
        
        # Input area
        input_frame = tk.Frame(chat_frame, bg='#1a1a1a')
        input_frame.pack(fill='x')
        
        self.input_entry = tk.Entry(input_frame,
                                   bg='#2d2d2d',
                                   fg='#ffffff',
                                   font=('Segoe UI', 11),
                                   insertbackground='#ffffff')
        self.input_entry.pack(side='left', fill='x', expand=True, padx=(0, 10))
        self.input_entry.bind('<Return>', self.send_message)
        
        self.send_button = tk.Button(input_frame,
                                   text="Send",
                                   bg='#D4512A',
                                   fg='#ffffff',
                                   font=('Segoe UI', 10, 'bold'),
                                   padx=20,
                                   pady=5,
                                   border=0,
                                   command=self.send_message)
        self.send_button.pack(side='right')
        
        # Initially disable input
        self.input_entry.config(state='disabled')
        self.send_button.config(state='disabled')
        
        # Add initial message
        self.add_chat_message("System", "Initializing Claude AI integration...", "#888888")
    
    def start_authentication(self):
        """Start authentication process."""
        # Show authentication dialog
        auth_dialog = ClaudeAuthDialog(self.root)
        self.root.wait_window(auth_dialog.dialog)
        
        if auth_dialog.result:
            self.authenticated = True
            self.user_info = auth_dialog.user_info
            
            # Update status
            if self.user_info:
                user_name = self.user_info.get('name', self.user_info.get('email', 'Claude User'))
                self.status_label.config(text=f"Authenticated as {user_name}", fg='#00ff88')
                self.add_chat_message("System", f"Welcome {user_name}! Claude AI is ready.", "#00ff88")
            else:
                self.status_label.config(text="Authenticated", fg='#00ff88')
                self.add_chat_message("System", "Claude AI authentication successful!", "#00ff88")
            
            # Enable input
            self.input_entry.config(state='normal')
            self.send_button.config(state='normal')
            self.input_entry.focus()
            
            # Initialize AI engine
            self.initialize_ai_engine()
        else:
            # Authentication failed
            self.add_chat_message("System", "Authentication failed. Please restart to try again.", "#ff6b6b")
            self.status_label.config(text="Authentication failed", fg='#ff6b6b')
    
    def initialize_ai_engine(self):
        """Initialize the AI engine."""
        try:
            self.ai_engine = SimpleWorkingAI()
            self.add_chat_message("System", "AI engine initialized and ready for conversation.", "#888888")
        except Exception as e:
            logger.error(f"Failed to initialize AI engine: {e}")
            self.add_chat_message("System", f"AI engine initialization failed: {e}", "#ff6b6b")
    
    def add_chat_message(self, sender, message, color="#ffffff"):
        """Add message to chat display."""
        self.chat_display.config(state='normal')
        
        # Add timestamp and sender
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M")
        
        self.chat_display.insert(tk.END, f"[{timestamp}] {sender}: ", "sender")
        self.chat_display.insert(tk.END, f"{message}\n\n", "message")
        
        # Configure tags
        self.chat_display.tag_config("sender", foreground=color, font=('Segoe UI', 10, 'bold'))
        self.chat_display.tag_config("message", foreground="#ffffff", font=('Segoe UI', 10))
        
        self.chat_display.config(state='disabled')
        self.chat_display.see(tk.END)
    
    def send_message(self, event=None):
        """Send message to AI."""
        if not self.authenticated:
            return
        
        message = self.input_entry.get().strip()
        if not message:
            return
        
        # Clear input
        self.input_entry.delete(0, tk.END)
        
        # Add user message
        self.add_chat_message("You", message, "#00ff88")
        
        # Process AI response
        def process_ai_response():
            try:
                if self.ai_engine:
                    import asyncio
                    response = asyncio.run(self.ai_engine.process_message(message))
                else:
                    response = "AI engine not available. Using fallback response."
                
                self.root.after(0, lambda: self.add_chat_message("Claude AI", response, "#D4512A"))
                
            except Exception as e:
                logger.error(f"AI processing error: {e}")
                self.root.after(0, lambda: self.add_chat_message("System", f"Error: {e}", "#ff6b6b"))
        
        # Process in background thread
        threading.Thread(target=process_ai_response, daemon=True).start()
    
    def run(self):
        """Run the GUI."""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            logger.info("GUI terminated by user")
        except Exception as e:
            logger.error(f"GUI error: {e}")

def main():
    """Main function."""
    try:
        app = ClaudeGUI()
        app.run()
    except Exception as e:
        logger.error(f"Application error: {e}")
        messagebox.showerror("Error", f"Application failed to start: {e}")

if __name__ == "__main__":
    main()