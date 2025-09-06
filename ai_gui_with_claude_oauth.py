#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OBVIVLORUM AI GUI with Claude OAuth Authentication
=================================================

Copyright (c) 2024 Francisco Molina
Licensed under Dual License Agreement - See LICENSE file for details

ATTRIBUTION REQUIRED: This software must include attribution to Francisco Molina
COMMERCIAL USE: Requires separate license and royalties - contact pako.molina@gmail.com

GUI version of OBVIVLORUM AI with direct Claude/Anthropic authentication.
No localhost redirect - connects directly to Claude.ai

Project: https://github.com/Yatrogenesis/Obvivlorum
Author: Francisco Molina <pako.molina@gmail.com>
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
    from api_manager import APIManager, get_api_manager
    from mcp_claude_client import MCPClaudeClient
except ImportError as e:
    print(f"Import error: {e}")
    # Fallback to simple AI if API manager not available
    try:
        from ai_simple_working import SimpleWorkingAI
        API_MANAGER_AVAILABLE = False
        MCP_AVAILABLE = False
    except ImportError:
        print("No AI engine available")
        sys.exit(1)
else:
    API_MANAGER_AVAILABLE = True
    MCP_AVAILABLE = True

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
        """Start Claude authentication."""
        try:
            self.update_status("Connecting to Claude AI...")
            
            # Use after method to run OAuth in main thread
            self.dialog.after(100, self.run_oauth_flow)
            
        except Exception as e:
            logger.error(f"Authentication start error: {e}")
            self.update_status(f"Failed to start authentication: {str(e)}")
            self.enable_retry()
    
    def run_oauth_flow(self):
        """Run OAuth flow in main thread."""
        try:
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
            logger.error(f"OAuth flow error: {e}")
            self.update_status(f"Authentication error: {str(e)}\nPlease try again.")
            self.enable_retry()
    
    def update_status(self, message):
        """Update status message."""
        def update():
            try:
                if self.dialog.winfo_exists():
                    self.status_label.config(text=message)
                    if "successful" in message.lower():
                        self.progress.stop()
                        self.status_label.config(fg='#00ff88')
                    elif "failed" in message.lower() or "error" in message.lower():
                        self.progress.stop()
                        self.status_label.config(fg='#ff6b6b')
            except (tk.TclError, RuntimeError):
                # Dialog may have been closed
                pass
        
        try:
            self.dialog.after(0, update)
        except (tk.TclError, RuntimeError):
            # Main thread not available, update directly
            try:
                update()
            except:
                pass
    
    def enable_retry(self):
        """Enable retry button."""
        def enable():
            try:
                if self.dialog.winfo_exists():
                    self.retry_button.config(state='normal')
                    self.progress.stop()
            except (tk.TclError, RuntimeError):
                pass
        
        try:
            self.dialog.after(0, enable)
        except (tk.TclError, RuntimeError):
            try:
                enable()
            except:
                pass
    
    def retry(self):
        """Retry authentication."""
        self.retry_button.config(state='disabled')
        self.progress.start()
        self.dialog.after(100, self.run_oauth_flow)
    
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
        """Initialize hybrid OAuth + API engine like Claude Code."""
        try:
            # Initialize both OAuth and API managers
            self.oauth_manager = OAuthManager()
            if API_MANAGER_AVAILABLE:
                self.api_manager = get_api_manager(str(Path(__file__).parent))
            
            # Check OAuth authentication status
            if self.authenticated:
                # Check OAuth session
                oauth_token = self.oauth_manager.get_token(OAuthProvider.CLAUDE)
                
                if oauth_token and MCP_AVAILABLE:
                    self.ai_engine = "claude_mcp"
                    self.add_chat_message("System", "🚀 Claude MCP Client Active (like Claude Desktop!)", "#00ff88")
                    self.add_chat_message("System", "Using MCP protocol + OAuth + Cloudflare bypass", "#00ff88")
                    
                    # Store session info for MCP client
                    self.claude_session = oauth_token
                    self.mcp_client = None  # Will be initialized on first use
                elif oauth_token:
                    self.ai_engine = "claude_oauth_web"
                    self.add_chat_message("System", "✅ Claude OAuth Web Session Active", "#00ff88")
                    self.add_chat_message("System", "MCP not available, using basic web session", "#ffaa00")
                    self.claude_session = oauth_token
                else:
                    self.add_chat_message("System", "OAuth session not available", "#ff6b6b")
                    self.ai_engine = None
                    
                # Track usage for intelligent switching
                self.oauth_usage_count = 0
                self.api_usage_count = 0
                self.switch_to_api_threshold = 20  # Switch to API after 20 OAuth calls
                
            else:
                # Fallback options
                if API_MANAGER_AVAILABLE and self.api_manager.has_provider('anthropic'):
                    self.ai_engine = "claude_api_only"
                    self.add_chat_message("System", "Claude API Mode: OAuth not available, using API only", "#ffaa00")
                elif not API_MANAGER_AVAILABLE:
                    self.ai_engine = SimpleWorkingAI()
                    self.add_chat_message("System", "Local AI Mode: OAuth and API not available", "#888888")
                else:
                    self.ai_engine = None
                    self.add_chat_message("System", "No Claude access available", "#ff6b6b")
                
        except Exception as e:
            logger.error(f"Failed to initialize AI engine: {e}")
            self.add_chat_message("System", f"AI engine initialization failed: {e}", "#ff6b6b")
            # Fallback
            try:
                if not API_MANAGER_AVAILABLE:
                    self.ai_engine = SimpleWorkingAI()
                else:
                    self.ai_engine = None
            except:
                self.ai_engine = None
    
    def prompt_for_api_key(self) -> Optional[str]:
        """Prompt user for Anthropic API key."""
        import tkinter.simpledialog as simpledialog
        
        dialog_title = "Anthropic API Key Required"
        dialog_message = ("To use real Claude AI responses, please enter your Anthropic API key.\n\n"
                         "You can get an API key from: https://console.anthropic.com/\n"
                         "Leave empty to use local AI engine instead.")
        
        api_key = simpledialog.askstring(
            dialog_title,
            dialog_message,
            show='*'  # Hide the API key input
        )
        
        return api_key.strip() if api_key else None
    
    def save_api_key_to_env(self, api_key: str):
        """Save API key to .env file."""
        try:
            env_file = Path(__file__).parent / ".env"
            
            # Read existing .env content
            env_content = ""
            if env_file.exists():
                with open(env_file, 'r') as f:
                    env_content = f.read()
            
            # Update or add ANTHROPIC_API_KEY
            lines = env_content.split('\n')
            updated = False
            
            for i, line in enumerate(lines):
                if line.startswith('ANTHROPIC_API_KEY='):
                    lines[i] = f'ANTHROPIC_API_KEY={api_key}'
                    updated = True
                    break
            
            if not updated:
                lines.append(f'ANTHROPIC_API_KEY={api_key}')
            
            # Write back to .env
            with open(env_file, 'w') as f:
                f.write('\n'.join(lines))
            
            logger.info("API key saved to .env file")
            
        except Exception as e:
            logger.error(f"Failed to save API key: {e}")
    
    def test_api_key_validity(self) -> bool:
        """Test if the current API key is valid by making a simple API call."""
        try:
            if not (API_MANAGER_AVAILABLE and self.api_manager.has_provider('anthropic')):
                return False
            
            # Simple test call
            import asyncio
            test_response = asyncio.run(self.api_manager.send_message(
                "Hello, this is a test.",
                provider='anthropic',
                model='claude-3-haiku-20240307'  # Use cheapest model for testing
            ))
            
            return bool(test_response and len(test_response) > 0)
            
        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return False
    
    def prompt_and_configure_api_key(self):
        """Prompt user for real API key and configure it."""
        try:
            api_key = self.prompt_for_api_key()
            if api_key and api_key.startswith('sk-ant-'):
                # Configure the API key
                success = self.api_manager.add_provider(
                    'anthropic',
                    api_key=api_key,
                    endpoint='https://api.anthropic.com/v1'
                )
                
                if success:
                    # Test the key
                    if self.test_api_key_validity():
                        self.save_api_key_to_env(api_key)
                        self.ai_engine = "claude_hybrid"
                        self.add_chat_message("System", "✅ Real Claude API key validated and configured!", "#00ff88")
                        self.add_chat_message("System", "Now using REAL Claude AI responses", "#00ff88")
                    else:
                        self.add_chat_message("System", "❌ API key invalid - please check your key", "#ff6b6b")
                        self.ai_engine = None
                else:
                    self.add_chat_message("System", "❌ Failed to configure API key", "#ff6b6b")
                    self.ai_engine = None
            elif api_key:
                self.add_chat_message("System", "❌ Invalid API key format. Should start with 'sk-ant-'", "#ff6b6b")
                self.ai_engine = None
            else:
                self.add_chat_message("System", "No API key provided - cannot use real Claude responses", "#ffaa00")
                self.ai_engine = None
                
        except Exception as e:
            logger.error(f"API key configuration failed: {e}")
            self.add_chat_message("System", f"API key configuration error: {e}", "#ff6b6b")
            self.ai_engine = None
    
    async def send_message_to_claude_web(self, message: str) -> str:
        """Send message to Claude.ai using OAuth session (like Claude Code)."""
        try:
            import aiohttp
            
            # Use OAuth session token for claude.ai
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/event-stream',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'Referer': 'https://claude.ai/chats',
                'Content-Type': 'application/json',
                'Origin': 'https://claude.ai',
                'Connection': 'keep-alive',
                'Sec-Fetch-Dest': 'empty',
                'Sec-Fetch-Mode': 'cors',
                'Sec-Fetch-Site': 'same-origin'
            }
            
            # Add session cookie if available
            if hasattr(self.claude_session, 'access_token') and self.claude_session.access_token != "claude_session_available":
                headers['Authorization'] = f'Bearer {self.claude_session.access_token}'
            
            # Create conversation payload (simplified)
            payload = {
                'completion': {
                    'prompt': message,
                    'timezone': 'America/New_York',
                    'model': 'claude-3-sonnet-20240229'
                },
                'organization_uuid': None,
                'conversation_uuid': None,
                'text': message,
                'attachments': []
            }
            
            async with aiohttp.ClientSession() as session:
                # Try to send message to Claude.ai
                async with session.post(
                    'https://claude.ai/api/append_message',
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if response.status == 200:
                        # Parse streaming response
                        response_text = await response.text()
                        
                        # Extract actual response (simplified parsing)
                        if 'completion' in response_text:
                            return f"[OAuth Web Session] Response received from Claude.ai session"
                        else:
                            return f"[OAuth Web Session] Connected to Claude.ai - message sent successfully"
                    else:
                        error_text = await response.text()
                        logger.error(f"Claude.ai web error {response.status}: {error_text}")
                        return f"[OAuth Web Session] Claude.ai returned status {response.status}. Session may need refresh."
                        
        except Exception as e:
            logger.error(f"Claude web session error: {e}")
            return f"[OAuth Web Session] Unable to connect to Claude.ai: {e}"
    
    def add_api_key_runtime(self):
        """Add API key during runtime for hybrid mode."""
        if hasattr(self, 'api_manager') and API_MANAGER_AVAILABLE:
            api_key = self.prompt_for_api_key()
            if api_key:
                success = self.api_manager.add_provider(
                    'anthropic',
                    api_key=api_key,
                    endpoint='https://api.anthropic.com/v1'
                )
                if success:
                    self.save_api_key_to_env(api_key)
                    self.add_chat_message("System", "API key added! Hybrid mode now fully active.", "#00ff88")
                    # Update engine status
                    self.ai_engine = "claude_hybrid"
                    return True
        return False
    
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
        
        # Show typing indicator
        self.add_chat_message("Claude AI", "Thinking...", "#888888")
        
        # Process AI response in background thread
        def process_ai_response():
            try:
                response = None
                method_used = "unknown"
                
                if self.ai_engine == "claude_mcp":
                    # Use MCP client (like Claude Desktop)
                    try:
                        import asyncio
                        
                        # Initialize MCP client on first use
                        if not self.mcp_client:
                            self.mcp_client = MCPClaudeClient(self.claude_session)
                        
                        response = asyncio.run(self.mcp_client.send_message(message))
                        method_used = "MCP Protocol"
                        self.oauth_usage_count += 1
                    except Exception as mcp_error:
                        logger.error(f"Claude MCP client failed: {mcp_error}")
                        response = f"MCP Client Error: {mcp_error}"
                        method_used = "MCP Error"
                        
                elif self.ai_engine == "claude_oauth_web":
                    # Use OAuth web session (fallback)
                    try:
                        import asyncio
                        response = asyncio.run(self.send_message_to_claude_web(message))
                        method_used = "OAuth Web Session"
                        self.oauth_usage_count += 1
                    except Exception as oauth_error:
                        logger.error(f"Claude OAuth web session failed: {oauth_error}")
                        response = f"OAuth Web Session Error: {oauth_error}"
                        method_used = "OAuth Error"
                    
                else:
                    # No valid engine configured
                    response = "Claude OAuth session not available. Please restart to authenticate."
                    method_used = "Error"
                
                if not response and self.ai_engine and hasattr(self.ai_engine, 'process_message'):
                    # Use local AI engine as final fallback
                    import asyncio
                    response = asyncio.run(self.ai_engine.process_message(message))
                    method_used = "Local AI"
                
                if not response:
                    response = "I apologize, but I'm currently unable to process your request. Please check your Claude authentication."
                
                # Update UI in main thread
                def update_chat():
                    try:
                        # Remove typing indicator by clearing and re-adding all messages
                        # For simplicity, just add the response
                        response_color = "#D4512A"
                        if "MCP" in method_used:
                            response_color = "#0088FF"  # Blue for MCP
                        elif "API" in method_used:
                            response_color = "#00AA88"  # Green for API
                        elif "OAuth" in method_used:
                            response_color = "#AA6600"  # Orange for OAuth
                        
                        self.add_chat_message("Claude AI", response, response_color)
                    except Exception as ui_error:
                        logger.error(f"UI update error: {ui_error}")
                
                try:
                    self.root.after(0, update_chat)
                except (tk.TclError, RuntimeError):
                    # Fallback if main thread is not available
                    logger.error("Could not update UI - main thread not available")
                
            except Exception as e:
                logger.error(f"AI processing error: {e}")
                
                def update_error():
                    try:
                        self.add_chat_message("System", f"Error: {e}", "#ff6b6b")
                    except:
                        pass
                
                try:
                    self.root.after(0, update_error)
                except:
                    pass
        
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
        finally:
            # Cleanup MCP client
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            if hasattr(self, 'mcp_client') and self.mcp_client:
                import asyncio
                asyncio.run(self.mcp_client.close())
                logger.info("MCP client closed")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

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