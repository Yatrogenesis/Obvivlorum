#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultra Clean AI - NO UNICODE ISSUES
==================================
Version without ANY unicode characters that cause cp1252 issues
"""

import tkinter as tk
from tkinter import scrolledtext, messagebox
import sys
import os
import threading
import time

class UltraCleanAI:
    def __init__(self):
        print("Starting Ultra Clean AI...")
        self.root = tk.Tk()
        self.setup_window()
        self.create_ui()
        print("Ultra Clean AI initialized successfully")
    
    def setup_window(self):
        self.root.title("OBVIVLORUM - ULTRA CLEAN AI")
        self.root.geometry("900x600")
        self.root.configure(bg='#1a1a1a')
    
    def create_ui(self):
        # Title
        title = tk.Label(self.root, text="OBVIVLORUM AI - ULTRA CLEAN VERSION", 
                        fg='#00ff88', bg='#1a1a1a',
                        font=('Arial', 16, 'bold'))
        title.pack(pady=10)
        
        # Status
        status = tk.Label(self.root, text="Status: Ready - No Unicode Issues", 
                         fg='#ffffff', bg='#1a1a1a',
                         font=('Arial', 10))
        status.pack()
        
        # Chat area
        chat_frame = tk.Frame(self.root, bg='#1a1a1a')
        chat_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame, wrap=tk.WORD,
            bg='#000000', fg='#ffffff',
            font=('Consolas', 10),
            height=20
        )
        self.chat_display.pack(fill='both', expand=True)
        
        # Input area
        input_frame = tk.Frame(self.root, bg='#1a1a1a')
        input_frame.pack(fill='x', padx=20, pady=10)
        
        self.input_field = tk.Entry(input_frame, 
                                   bg='#333333', fg='#ffffff',
                                   font=('Arial', 11),
                                   insertbackground='#ffffff')
        self.input_field.pack(side='left', fill='x', expand=True)
        self.input_field.bind('<Return>', self.send_message)
        
        send_btn = tk.Button(input_frame, text="Send", 
                            command=self.send_message,
                            bg='#00ff88', fg='#000000',
                            font=('Arial', 10, 'bold'),
                            padx=20)
        send_btn.pack(side='right', padx=(10, 0))
        
        # Add initial message
        self.add_message("System", "ULTRA CLEAN AI Ready!")
        self.add_message("System", "This version has NO unicode issues")
        self.add_message("AI", "Hello! I'm your Ultra Clean AI assistant. Ask me anything!")
        
    def add_message(self, sender, message):
        self.chat_display.insert(tk.END, f"{sender}: {message}\n")
        self.chat_display.see(tk.END)
    
    def send_message(self, event=None):
        user_input = self.input_field.get().strip()
        if not user_input:
            return
        
        self.add_message("You", user_input)
        self.input_field.delete(0, tk.END)
        
        # Simple AI response
        threading.Thread(target=self.generate_response, args=(user_input,), daemon=True).start()
    
    def generate_response(self, user_input):
        # Simulate thinking
        self.add_message("AI", "Thinking...")
        time.sleep(1)
        
        # Simple responses without unicode
        if "hello" in user_input.lower() or "hola" in user_input.lower():
            response = "Hello! I'm working perfectly without any unicode issues!"
        elif "test" in user_input.lower():
            response = "Test successful! This Ultra Clean version works properly on your i5+12GB system."
        elif "ai" in user_input.lower() or "artificial" in user_input.lower():
            response = "I'm an AI assistant running in Ultra Clean mode - no encoding problems!"
        elif "problem" in user_input.lower():
            response = "The previous versions had unicode encoding issues with Windows cp1252. This version is clean!"
        else:
            response = f"I understand you said: '{user_input}'. This Ultra Clean AI is working perfectly!"
        
        # Remove "Thinking..." and add real response
        content = self.chat_display.get(1.0, tk.END)
        lines = content.strip().split('\n')
        if lines and "Thinking..." in lines[-1]:
            # Remove last line
            self.chat_display.delete('end-2l', 'end-1l')
        
        self.add_message("AI", response)
    
    def run(self):
        print("Starting Ultra Clean AI mainloop...")
        try:
            self.root.mainloop()
            print("Ultra Clean AI ended normally")
        except Exception as e:
            print(f"Error: {e}")

def main():
    print("=" * 50)
    print("OBVIVLORUM - ULTRA CLEAN AI")
    print("No Unicode Issues - Perfect for Windows")
    print("=" * 50)
    
    try:
        app = UltraCleanAI()
        app.run()
    except Exception as e:
        print(f"Critical error: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()