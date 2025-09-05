#!/usr/bin/env python3
"""
Brutal UI Simple - Debug Version
===============================

Version simplificada para debugging del blinking issue
"""

import tkinter as tk
import sys
import os
from pathlib import Path

print("Starting Brutal UI Simple...")

class SimpleUI:
    def __init__(self):
        print("Initializing SimpleUI...")
        self.root = tk.Tk()
        self.setup_window()
        self.create_ui()
        print("SimpleUI initialized successfully")
    
    def setup_window(self):
        print("Setting up window...")
        self.root.title("OBVIVLORUM - SIMPLE TEST")
        self.root.geometry("800x600")
        self.root.configure(bg='#0a0a0a')
        
        # No transparency to avoid issues
        print("Window setup complete")
    
    def create_ui(self):
        print("Creating UI...")
        
        # Simple header
        header = tk.Frame(self.root, bg='#1a1a1a', height=60)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        title = tk.Label(header, text="OBVIVLORUM - BRUTAL UI TEST", 
                        fg='#00ff88', bg='#1a1a1a',
                        font=('Arial', 16, 'bold'))
        title.pack(expand=True)
        
        # Main content
        main_frame = tk.Frame(self.root, bg='#0a0a0a')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Test message
        message = tk.Label(main_frame, 
                          text="If you can see this window without blinking,\nthe basic UI works correctly!",
                          fg='#ffffff', bg='#0a0a0a',
                          font=('Arial', 12),
                          justify='center')
        message.pack(expand=True)
        
        # Simple button
        test_button = tk.Button(main_frame, text="TEST BUTTON",
                               command=self.test_click,
                               bg='#00ff88', fg='#000000',
                               font=('Arial', 11, 'bold'),
                               relief='flat', padx=20, pady=10)
        test_button.pack(pady=20)
        
        # Status
        status = tk.Label(main_frame, text="Status: UI Loaded Successfully",
                         fg='#00ff88', bg='#0a0a0a',
                         font=('Arial', 10))
        status.pack()
        
        print("UI created successfully")
    
    def test_click(self):
        print("Button clicked!")
        tk.messagebox.showinfo("Test", "Button works correctly!")
    
    def run(self):
        print("Starting mainloop...")
        try:
            self.root.mainloop()
            print("Mainloop ended")
        except Exception as e:
            print(f"Error in mainloop: {e}")
            raise

def main():
    print("=" * 50)
    print("BRUTAL UI SIMPLE - DEBUG VERSION")
    print("=" * 50)
    
    try:
        app = SimpleUI()
        app.run()
    except Exception as e:
        print(f"Error creating app: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()