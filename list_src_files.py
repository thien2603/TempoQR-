#!/usr/bin/env python3
"""
Script to list all Python files in src directory with their paths
"""

import os
import sys

def list_python_files():
    """List all .py files in src directory recursively"""
    src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
    
    if not os.path.exists(src_dir):
        print(f"Error: src directory not found at {src_dir}")
        return
    
    print("Python files in src directory:")
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
    
    # Print results
    for rel_path, full_path in python_files:
        print(f"src/{rel_path}")
    
    print(f"\nTotal Python files: {len(python_files)}")
    
    # Save to file
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src_files_list.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Python files in src directory:\n")
        f.write("=" * 80 + "\n")
        for rel_path, full_path in python_files:
            f.write(f"src/{rel_path}\n")
        f.write(f"\nTotal Python files: {len(python_files)}\n")
    
    print(f"List saved to: {output_file}")
    
    return python_files

if __name__ == "__main__":
    list_python_files()
