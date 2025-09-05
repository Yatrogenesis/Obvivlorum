#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para corregir problemas de Unicode en archivos Python
"""

import os
import re
import sys

def fix_unicode_in_file(filepath):
    """Reemplaza caracteres Unicode problematicos con equivalentes ASCII"""
    
    # Mapa de reemplazos
    replacements = {
        '[OK]': '[OK]',
        '[ERROR]': '[ERROR]',
        '[FAST]': '[FAST]',
        '[ROCKET]': '[ROCKET]',
        '[HOT]': '[HOT]',
        '[STAR]': '[STAR]',
        '[SHIELD]': '[SHIELD]',
        '[TARGET]': '[TARGET]',
        '[BRAIN]': '[BRAIN]',
        '[CHECK]': '[CHECK]',
        '[TOOL]': '[TOOL]',
        '[WARNING]': '[WARNING]',
        '[IDEA]': '[IDEA]',
        '[WEB]': '[WEB]',
        '[FOLDER]': '[FOLDER]',
        '[CHART]': '[CHART]',
        '[SEARCH]': '[SEARCH]',
        '[COMPUTER]': '[COMPUTER]',
        '[GEAR]': '[GEAR]'
    }
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Aplicar reemplazos
        for unicode_char, ascii_replacement in replacements.items():
            content = content.replace(unicode_char, ascii_replacement)
        
        # Solo escribir si hubo cambios
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"[FIXED] {filepath}")
            return True
        else:
            print(f"[SKIP] {filepath} - No unicode issues found")
            return False
            
    except Exception as e:
        print(f"[ERROR] {filepath}: {e}")
        return False

def main():
    """Buscar y corregir archivos Python con problemas de Unicode"""
    
    base_dir = r"D:\Obvivlorum"
    fixed_count = 0
    
    print("Buscando archivos Python con caracteres Unicode problematicos...")
    print(f"Directorio base: {base_dir}")
    print("-" * 50)
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                if fix_unicode_in_file(filepath):
                    fixed_count += 1
    
    print("-" * 50)
    print(f"Archivos corregidos: {fixed_count}")
    print("Correccion de Unicode completada.")

if __name__ == "__main__":
    main()