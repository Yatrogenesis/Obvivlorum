#!/usr/bin/env python3
"""
Unicode Cleanup Script - Remove ALL Unicode characters from Python files
"""

import os
import re
import shutil
import sys

def clean_unicode_from_file(filepath):
    """Remove ALL Unicode characters from a Python file."""
    try:
        # Create backup
        backup_path = filepath + ".backup_unicode"
        shutil.copy2(filepath, backup_path)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Count Unicode chars
        unicode_count = sum(1 for char in content if ord(char) > 127)
        
        if unicode_count == 0:
            os.remove(backup_path)  # Remove backup if no changes needed
            return 0, []
        
        # Common Unicode replacements for Python code
        replacements = {
            # Spanish characters
            'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ñ': 'n',
            'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U', 'Ñ': 'N',
            'ü': 'u', 'Ü': 'U',
            
            # Quotes and punctuation
            '"': '"', '"': '"', ''': "'", ''': "'",
            '–': '-', '—': '--', '…': '...',
            
            # Symbols commonly used in code comments
            '✓': '[OK]', '✗': '[FAIL]', '→': '->', '←': '<-',
            '▲': '^', '▼': 'v', '●': '*', '○': 'o',
            '★': '*', '☆': '*', '♦': '*', '♠': '*',
            '♣': '*', '♥': '*',
            
            # Math symbols
            '×': 'x', '÷': '/', '±': '+/-', '≤': '<=', '≥': '>=',
            '≠': '!=', '≈': '~=', '∞': 'inf',
            
            # Arrows and special chars
            '⇒': '=>', '⇐': '<=', '⇔': '<=>',
            '⊕': '+', '⊗': 'x', '∴': 'therefore',
            
            # Currency and units
            '€': 'EUR', '£': 'GBP', '¥': 'YEN', '°': 'deg',
            
            # Remove emojis and special Unicode blocks entirely
        }
        
        # Apply basic replacements
        cleaned_content = content
        for unicode_char, replacement in replacements.items():
            cleaned_content = cleaned_content.replace(unicode_char, replacement)
        
        # Remove any remaining Unicode characters (replace with ?)
        # But preserve newlines, tabs, and basic ASCII
        final_content = ""
        for char in cleaned_content:
            if ord(char) <= 127:  # Keep ASCII
                final_content += char
            else:
                # Replace with safe ASCII equivalent or remove
                if char.isalpha():
                    final_content += '?'  # Unknown letter -> ?
                elif char.isdigit():
                    final_content += '0'  # Unknown digit -> 0
                elif char.isspace():
                    final_content += ' '  # Unknown space -> regular space
                # Skip other Unicode chars entirely
        
        # Write cleaned file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(final_content)
        
        # Verify it's now clean
        with open(filepath, 'r', encoding='utf-8') as f:
            verify_content = f.read()
        
        remaining_unicode = [char for char in verify_content if ord(char) > 127]
        
        return unicode_count, remaining_unicode
        
    except Exception as e:
        print(f"Error cleaning {filepath}: {e}")
        return -1, []

def main():
    """Clean all Python files in directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find all Python files
    python_files = []
    for root, dirs, files in os.walk(script_dir):
        for file in files:
            if file.endswith('.py') and file != 'UNICODE_CLEANUP_SCRIPT.py':
                python_files.append(os.path.join(root, file))
    
    print(f"Found {len(python_files)} Python files to clean")
    print("=" * 60)
    
    total_cleaned = 0
    total_files_modified = 0
    
    for py_file in python_files:
        relative_path = os.path.relpath(py_file, script_dir)
        print(f"Cleaning: {relative_path}")
        
        unicode_count, remaining = clean_unicode_from_file(py_file)
        
        if unicode_count == -1:
            print(f"  ERROR: Failed to clean")
        elif unicode_count == 0:
            print(f"  CLEAN: No Unicode found")
        else:
            total_cleaned += unicode_count
            total_files_modified += 1
            if remaining:
                print(f"  PARTIAL: Cleaned {unicode_count}, {len(remaining)} remaining")
                print(f"    Remaining: {remaining[:5]}")  # Show first 5
            else:
                print(f"  SUCCESS: Cleaned {unicode_count} Unicode characters")
    
    print("=" * 60)
    print(f"SUMMARY:")
    print(f"  Files modified: {total_files_modified}")
    print(f"  Total Unicode chars cleaned: {total_cleaned}")
    print(f"  Backup files created with .backup_unicode extension")

if __name__ == "__main__":
    main()