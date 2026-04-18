#!/usr/bin/env python3
"""
Script to extract all Python code from src directory into a single file
"""

import os
import sys
from datetime import datetime

def extract_python_code():
    """Extract all Python code from src directory into a single file"""
    src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
    
    if not os.path.exists(src_dir):
        print(f"Error: src directory not found at {src_dir}")
        return
    
    print("Extracting Python code from src directory...")
    print("=" * 80)
    
    python_files = []
    
    # Walk through src directory recursively
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.py'):
                # Get relative path from src directory
                rel_path = os.path.relpath(os.path.join(root, file), src_dir)
                full_path = os.path.join(root, file)
                python_files.append((rel_path, full_path))
    
    # Sort by relative path
    python_files.sort()
    
    # Create output file
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src_code_complete.txt')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write header
        f.write("=" * 100 + "\n")
        f.write("TEMPOQR - COMPLETE SOURCE CODE EXTRACTOR\n")
        f.write("=" * 100 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Python files: {len(python_files)}\n")
        f.write("=" * 100 + "\n\n")
        
        # Write table of contents
        f.write("TABLE OF CONTENTS\n")
        f.write("-" * 50 + "\n")
        for i, (rel_path, full_path) in enumerate(python_files, 1):
            f.write(f"{i:2d}. src/{rel_path}\n")
        f.write("\n" + "=" * 100 + "\n\n")
        
        # Write each file content
        for i, (rel_path, full_path) in enumerate(python_files, 1):
            f.write(f"FILE {i}: src/{rel_path}\n")
            f.write("-" * 80 + "\n")
            
            try:
                with open(full_path, 'r', encoding='utf-8') as source_file:
                    content = source_file.read()
                    f.write(content)
                    
                    # Add a newline if file doesn't end with one
                    if content and not content.endswith('\n'):
                        f.write('\n')
                        
            except Exception as e:
                f.write(f"ERROR: Could not read file {rel_path}: {e}\n")
            
            f.write("\n" + "=" * 100 + "\n\n")
    
    print(f"Successfully extracted {len(python_files)} Python files")
    print(f"Output saved to: {output_file}")
    
    # Calculate total lines
    total_lines = 0
    for rel_path, full_path in python_files:
        try:
            with open(full_path, 'r', encoding='utf-8') as source_file:
                lines = len(source_file.readlines())
                total_lines += lines
        except:
            pass
    
    print(f"Total lines of code: {total_lines}")
    
    return output_file, total_lines

if __name__ == "__main__":
    extract_python_code()
