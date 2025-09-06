#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OBVIVLORUM AI - Real Claude Integration
======================================

100% functional OBVIVLORUM AI with real Claude.ai integration using browser automation.
No fake responses, no API key requirements - just pure OAuth + browser automation.

This is the version that actually works like Claude Code.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import asyncio
import threading
import json
import logging
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from real_claude_client import RealClaudeClient, PLAYWRIGHT_AVAILABLE
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install playwright: pip install playwright")
    print("Then run: playwright install chromium")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OBVIVLORUM")

class OBVIVLORUM_AI:
    """Main OBVIVLORUM AI application with real Claude integration."""
    
    def __init__(self):
        """Initialize the application."""
        if not PLAYWRIGHT_AVAILABLE:
            messagebox.showerror(
                "Dependencies Missing",
                "Playwright is required for real Claude integration.\n\n"
                "Please install it with:\n"
                "pip install playwright\n"
                "playwright install chromium"
            )
            sys.exit(1)
            
        self.root = tk.Tk()
        self.root.title("OBVIVLORUM AI - Real Claude Integration")
        self.root.geometry("1000x700")
        self.root.configure(bg='#0d1117')
        
        # State
        self.claude_client: Optional[RealClaudeClient] = None
        self.authenticated = False
        self.message_count = 0
        
        # Style configuration
        self.colors = {
            'bg': '#0d1117',
            'surface': '#161b22',
            'primary': '#f78166',
            'success': '#3fb950',
            'warning': '#d29922',
            'error': '#f85149',
            'text': '#f0f6fc',
            'text_muted': '#8b949e',
            'border': '#30363d'
        }
        
        self.setup_ui()
        self.setup_async_loop()
        
        # Auto-start authentication
        self.root.after(1000, self.start_authentication)
        
    def setup_ui(self):
        """Set up the user interface."""
        # Header
        header_frame = tk.Frame(self.root, bg=self.colors['bg'], height=100)
        header_frame.pack(fill='x', padx=20, pady=20)
        header_frame.pack_propagate(False)
        
        # Title and logo
        title_frame = tk.Frame(header_frame, bg=self.colors['bg'])
        title_frame.pack(side='left', fill='y')
        
        title_label = tk.Label(
            title_frame,
            text="OBVIVLORUM AI",
            bg=self.colors['bg'],
            fg=self.colors['text'],
            font=('Segoe UI', 24, 'bold')
        )
        title_label.pack(anchor='w')
        
        subtitle_label = tk.Label(
            title_frame,
            text="Real Claude Integration • Browser Automation • 100% Functional",
            bg=self.colors['bg'],
            fg=self.colors['text_muted'],
            font=('Segoe UI', 10)
        )
        subtitle_label.pack(anchor='w', pady=(5, 0))
        
        # Status panel
        status_frame = tk.Frame(header_frame, bg=self.colors['surface'], relief='solid', bd=1)
        status_frame.pack(side='right', fill='y', padx=(20, 0))
        
        self.status_label = tk.Label(
            status_frame,
            text="● Initializing...",
            bg=self.colors['surface'],
            fg=self.colors['warning'],
            font=('Segoe UI', 11, 'bold'),
            padx=15,
            pady=10
        )
        self.status_label.pack()
        
        self.stats_label = tk.Label(
            status_frame,
            text="Messages: 0 | Session: Starting",
            bg=self.colors['surface'],
            fg=self.colors['text_muted'],
            font=('Segoe UI', 9),
            padx=15,
            pady=(0, 10)
        )
        self.stats_label.pack()
        
        # Main content area
        main_frame = tk.Frame(self.root, bg=self.colors['bg'])
        main_frame.pack(fill='both', expand=True, padx=20, pady=(0, 20))
        
        # Chat area
        chat_frame = tk.Frame(main_frame, bg=self.colors['surface'], relief='solid', bd=1)
        chat_frame.pack(fill='both', expand=True, pady=(0, 15))
        
        # Chat header
        chat_header = tk.Frame(chat_frame, bg=self.colors['surface'], height=40)
        chat_header.pack(fill='x', padx=1, pady=1)
        chat_header.pack_propagate(False)
        
        chat_title = tk.Label(
            chat_header,
            text="💬 Real Claude AI Chat",
            bg=self.colors['surface'],
            fg=self.colors['text'],
            font=('Segoe UI', 12, 'bold'),
            padx=15
        )
        chat_title.pack(side='left', fill='y')
        
        # Buttons
        button_frame = tk.Frame(chat_header, bg=self.colors['surface'])
        button_frame.pack(side='right', padx=15, pady=8)
        
        self.clear_button = tk.Button(
            button_frame,
            text="Clear Chat",
            bg=self.colors['border'],
            fg=self.colors['text'],
            font=('Segoe UI', 9),
            padx=10,
            pady=2,
            relief='flat',
            command=self.clear_chat,
            cursor='hand2'
        )
        self.clear_button.pack(side='right', padx=(10, 0))
        
        self.new_conversation_button = tk.Button(
            button_frame,
            text="New Chat",
            bg=self.colors['primary'],
            fg='white',
            font=('Segoe UI', 9, 'bold'),
            padx=10,
            pady=2,
            relief='flat',
            command=self.new_conversation,
            cursor='hand2'
        )
        self.new_conversation_button.pack(side='right')
        
        # Chat display
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            bg=self.colors['bg'],
            fg=self.colors['text'],
            font=('Consolas', 11),
            wrap=tk.WORD,
            state='disabled',
            padx=15,
            pady=10,
            relief='flat',
            insertbackground=self.colors['text']
        )
        self.chat_display.pack(fill='both', expand=True, padx=1, pady=(0, 1))
        
        # Input area
        input_frame = tk.Frame(main_frame, bg=self.colors['surface'], relief='solid', bd=1)
        input_frame.pack(fill='x')
        
        # Input field
        input_container = tk.Frame(input_frame, bg=self.colors['surface'])
        input_container.pack(fill='x', padx=15, pady=15)
        
        self.input_entry = tk.Text(
            input_container,
            bg=self.colors['bg'],
            fg=self.colors['text'],
            font=('Segoe UI', 11),
            wrap=tk.WORD,
            height=3,
            relief='solid',
            bd=1,
            insertbackground=self.colors['text'],
            padx=10,
            pady=8
        )
        self.input_entry.pack(side='left', fill='both', expand=True, padx=(0, 10))
        self.input_entry.bind('<Control-Return>', self.send_message)
        self.input_entry.bind('<KeyRelease>', self.check_input)
        
        # Send button
        self.send_button = tk.Button(
            input_container,
            text="Send\n(Ctrl+Enter)",
            bg=self.colors['primary'],
            fg='white',
            font=('Segoe UI', 10, 'bold'),
            padx=15,
            pady=10,
            relief='flat',
            command=self.send_message,
            cursor='hand2',
            state='disabled'
        )
        self.send_button.pack(side='right')
        
        # Initial message
        self.add_system_message("🚀 Starting OBVIVLORUM AI with real Claude integration...")
        self.add_system_message("This version uses browser automation for 100% real Claude responses")
        
    def setup_async_loop(self):
        """Set up async loop in a separate thread."""
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.loop_thread.start()
        
    def _run_event_loop(self):
        """Run the asyncio event loop."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
        
    def run_async(self, coro):
        """Run an async function in the event loop."""
        return asyncio.run_coroutine_threadsafe(coro, self.loop)
        
    def start_authentication(self):
        """Start Claude authentication process."""
        self.update_status("🔐 Starting Claude authentication...", "warning")
        self.add_system_message("Starting real Claude.ai authentication via browser...")
        
        # Run authentication in async thread
        future = self.run_async(self._authenticate_claude())
        
        def check_auth_result():
            if future.done():
                try:
                    result = future.result()
                    if result:
                        self.authenticated = True
                        self.update_status("✅ Claude Authenticated", "success")
                        self.add_system_message("🎉 Successfully authenticated with Claude.ai!")
                        self.add_system_message("You can now chat with real Claude AI using browser automation")
                        self.enable_chat()
                    else:
                        self.update_status("❌ Authentication Failed", "error")
                        self.add_system_message("❌ Authentication failed. Please restart the application.")
                        self.show_retry_option()
                except Exception as e:
                    self.update_status("❌ Auth Error", "error")
                    self.add_system_message(f"Authentication error: {e}")
                    self.show_retry_option()
            else:
                # Check again in 1 second
                self.root.after(1000, check_auth_result)
        
        self.root.after(2000, check_auth_result)  # Start checking after 2 seconds
        
    async def _authenticate_claude(self) -> bool:
        """Authenticate with Claude (async)."""
        try:
            self.claude_client = RealClaudeClient(headless=False)  # Show browser for auth
            await self.claude_client.initialize()
            
            success = await self.claude_client.authenticate()
            return success
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return False
            
    def show_retry_option(self):
        """Show retry option for authentication."""
        retry_button = tk.Button(
            self.root,
            text="Retry Authentication",
            bg=self.colors['primary'],
            fg='white',
            font=('Segoe UI', 10, 'bold'),
            padx=20,
            pady=10,
            relief='flat',
            command=self.start_authentication,
            cursor='hand2'
        )
        retry_button.pack(pady=20)
        
    def enable_chat(self):
        """Enable chat interface."""
        self.input_entry.config(state='normal')
        self.send_button.config(state='normal')
        self.new_conversation_button.config(state='normal')
        self.input_entry.focus()
        
    def check_input(self, event=None):
        """Check if input has text and enable/disable send button."""
        content = self.input_entry.get("1.0", tk.END).strip()
        if content and self.authenticated:
            self.send_button.config(state='normal')
        else:
            self.send_button.config(state='disabled')
            
    def update_status(self, text: str, status_type: str = "info"):
        """Update status display."""
        colors = {
            "info": self.colors['text'],
            "success": self.colors['success'],
            "warning": self.colors['warning'],
            "error": self.colors['error']
        }
        
        self.status_label.config(text=f"● {text}", fg=colors.get(status_type, self.colors['text']))
        
        # Update stats
        session_time = "Active" if self.authenticated else "Connecting"
        self.stats_label.config(text=f"Messages: {self.message_count} | Session: {session_time}")
        
    def add_system_message(self, message: str, color: str = None):
        """Add system message to chat."""
        self.chat_display.config(state='normal')
        
        timestamp = datetime.now().strftime("%H:%M")
        color = color or self.colors['text_muted']
        
        # Add message with styling
        self.chat_display.insert(tk.END, f"[{timestamp}] SYSTEM: ", "system_label")
        self.chat_display.insert(tk.END, f"{message}\n\n", "system_message")
        
        # Configure tags
        self.chat_display.tag_config("system_label", foreground=self.colors['warning'], font=('Segoe UI', 9, 'bold'))
        self.chat_display.tag_config("system_message", foreground=color, font=('Segoe UI', 10))
        
        self.chat_display.config(state='disabled')
        self.chat_display.see(tk.END)
        
    def add_user_message(self, message: str):
        """Add user message to chat."""
        self.chat_display.config(state='normal')
        
        timestamp = datetime.now().strftime("%H:%M")
        
        self.chat_display.insert(tk.END, f"[{timestamp}] YOU: ", "user_label")
        self.chat_display.insert(tk.END, f"{message}\n\n", "user_message")
        
        self.chat_display.tag_config("user_label", foreground=self.colors['success'], font=('Segoe UI', 10, 'bold'))
        self.chat_display.tag_config("user_message", foreground=self.colors['text'], font=('Segoe UI', 10))
        
        self.chat_display.config(state='disabled')
        self.chat_display.see(tk.END)
        
    def add_claude_message(self, message: str):
        """Add Claude's response to chat."""
        self.chat_display.config(state='normal')
        
        timestamp = datetime.now().strftime("%H:%M")
        
        self.chat_display.insert(tk.END, f"[{timestamp}] CLAUDE: ", "claude_label")
        self.chat_display.insert(tk.END, f"{message}\n\n", "claude_message")
        
        self.chat_display.tag_config("claude_label", foreground=self.colors['primary'], font=('Segoe UI', 10, 'bold'))
        self.chat_display.tag_config("claude_message", foreground=self.colors['text'], font=('Segoe UI', 10))
        
        self.chat_display.config(state='disabled')
        self.chat_display.see(tk.END)
        
    def send_message(self, event=None):
        """Send message to Claude."""
        if not self.authenticated or not self.claude_client:
            return
            
        message = self.input_entry.get("1.0", tk.END).strip()
        if not message:
            return
            
        # Clear input
        self.input_entry.delete("1.0", tk.END)
        self.send_button.config(state='disabled')
        
        # Add user message
        self.add_user_message(message)
        self.message_count += 1
        
        # Update status
        self.update_status("🤖 Claude is thinking...", "warning")
        
        # Send to Claude asynchronously
        future = self.run_async(self._send_to_claude(message))
        
        def check_response():
            if future.done():
                try:
                    response = future.result()
                    self.add_claude_message(response)
                    self.update_status("✅ Claude Responded", "success")
                except Exception as e:
                    self.add_claude_message(f"Error: {e}")
                    self.update_status("❌ Response Error", "error")
                finally:
                    self.check_input()  # Re-enable send button if needed
            else:
                self.root.after(1000, check_response)
                
        self.root.after(1000, check_response)
        
    async def _send_to_claude(self, message: str) -> str:
        """Send message to Claude (async)."""
        try:
            response = await self.claude_client.send_message(message)
            return response
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return f"Error communicating with Claude: {e}"
            
    def new_conversation(self):
        """Start a new conversation."""
        if not self.authenticated or not self.claude_client:
            return
            
        self.update_status("🔄 Starting new conversation...", "warning")
        
        future = self.run_async(self._create_new_conversation())
        
        def check_new_conv():
            if future.done():
                try:
                    result = future.result()
                    if result:
                        self.clear_chat()
                        self.add_system_message("🆕 Started new conversation with Claude")
                        self.update_status("✅ New Conversation", "success")
                    else:
                        self.add_system_message("❌ Failed to start new conversation")
                        self.update_status("❌ Conversation Error", "error")
                except Exception as e:
                    self.add_system_message(f"Error: {e}")
            else:
                self.root.after(1000, check_new_conv)
                
        self.root.after(500, check_new_conv)
        
    async def _create_new_conversation(self) -> bool:
        """Create new conversation (async)."""
        try:
            conversation = await self.claude_client.create_conversation()
            return conversation is not None
        except Exception as e:
            logger.error(f"Error creating conversation: {e}")
            return False
            
    def clear_chat(self):
        """Clear the chat display."""
        self.chat_display.config(state='normal')
        self.chat_display.delete("1.0", tk.END)
        self.chat_display.config(state='disabled')
        self.message_count = 0
        self.update_status("🧹 Chat Cleared", "info")
        
    def run(self):
        """Run the application."""
        try:
            # Set up window close handler
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            
            # Start the GUI
            self.root.mainloop()
            
        except KeyboardInterrupt:
            logger.info("Application terminated by user")
        except Exception as e:
            logger.error(f"Application error: {e}")
            messagebox.showerror("Error", f"Application error: {e}")
        finally:
            self.cleanup()
            
    def on_closing(self):
        """Handle window closing."""
        if messagebox.askokcancel("Quit", "Do you want to quit OBVIVLORUM AI?"):
            self.cleanup()
            self.root.destroy()
            
    def cleanup(self):
        """Cleanup resources."""
        try:
            if self.claude_client:
                # Run cleanup in the async loop
                future = self.run_async(self.claude_client.close())
                try:
                    future.result(timeout=5)  # Wait up to 5 seconds
                except:
                    pass  # Ignore cleanup errors
                    
            # Stop the async loop
            if hasattr(self, 'loop') and self.loop.is_running():
                self.loop.call_soon_threadsafe(self.loop.stop)
                
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

def main():
    """Main function."""
    print("🚀 Starting OBVIVLORUM AI - Real Claude Integration")
    print("=" * 60)
    print("Features:")
    print("• Real Claude.ai integration via browser automation")
    print("• No API keys required - uses OAuth like Claude Code")
    print("• 100% real responses from Claude")
    print("• Session persistence")
    print("• Modern GUI interface")
    print("=" * 60)
    
    try:
        app = OBVIVLORUM_AI()
        app.run()
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        messagebox.showerror("Startup Error", f"Failed to start OBVIVLORUM AI:\n{e}")

if __name__ == "__main__":
    main()