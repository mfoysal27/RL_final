#!/usr/bin/env python
"""
Simple script to test if tkinter is installed
"""

import sys

print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

try:
    import tkinter as tk
    print("✅ Tkinter is installed!")
    print(f"Tkinter version: {tk.TkVersion}")
    
    # Create a simple window to verify it works
    root = tk.Tk()
    root.title("Tkinter Test")
    label = tk.Label(root, text="Tkinter is working!")
    label.pack(padx=20, pady=20)
    print("✅ Created a Tkinter window successfully")
    
    # Don't actually show the window - just test creation
    root.destroy()
    
except ImportError as e:
    print(f"❌ Tkinter is NOT installed: {e}")
    print("\nPossible solutions:")
    print("1. For Anaconda: conda install -c anaconda tk")
    print("2. For standard Python: Reinstall Python with tcl/tk option checked")
    
print("\nDone testing tkinter") 