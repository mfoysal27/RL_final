#!/usr/bin/env python
"""
Debug script to identify the issue with test_model.py
"""

print("🔍 Starting debug test...")

try:
    print("1. Testing basic imports...")
    import os
    import sys
    print("   ✅ os and sys")
    
    import torch
    print("   ✅ torch")
    
    import numpy as np
    print("   ✅ numpy")
    
    import tkinter as tk
    print("   ✅ tkinter")
    
    print("2. Testing tkinter GUI creation...")
    root = tk.Tk()
    root.withdraw()
    print("   ✅ tkinter root created")
    root.destroy()
    
    print("3. Testing file path...")
    current_dir = os.getcwd()
    print(f"   Current directory: {current_dir}")
    
    test_file = "test_model.py"
    if os.path.exists(test_file):
        print(f"   ✅ {test_file} exists")
    else:
        print(f"   ❌ {test_file} not found")
    
    print("4. Testing project imports...")
    
    # Test if we can import from the models directory
    models_dir = os.path.join(current_dir, "models")
    if os.path.exists(models_dir):
        print(f"   ✅ models directory exists: {models_dir}")
    else:
        print(f"   ❌ models directory not found: {models_dir}")
    
    core_dir = os.path.join(current_dir, "core") 
    if os.path.exists(core_dir):
        print(f"   ✅ core directory exists: {core_dir}")
    else:
        print(f"   ❌ core directory not found: {core_dir}")
    
    print("5. Testing sys.path modification...")
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"   Parent directory would be: {parent_dir}")
    
    # Add to path like test_model.py does
    sys.path.insert(0, parent_dir)
    print("   ✅ sys.path modified")
    
    print("6. Testing main function definition...")
    
    def test_main():
        print("   Inside test main function")
        print("   Would call select_files_gui() here")
        return "success"
    
    result = test_main()
    print(f"   ✅ test main function returned: {result}")
    
    print("🎉 All debug tests passed!")
    
except Exception as e:
    print(f"❌ Error in debug test: {e}")
    import traceback
    traceback.print_exc() 