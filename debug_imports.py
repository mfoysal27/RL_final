#!/usr/bin/env python
"""
Debug script to test the exact imports from test_model.py
"""

print("🔍 Testing exact imports from test_model.py...")

try:
    print("1. Setting up sys.path...")
    import os
    import sys
    
    # Add parent directory to path (same as test_model.py line 24)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print("   ✅ sys.path configured")
    
    print("2. Testing basic imports...")
    import torch
    import torch.nn as nn
    import numpy as np
    import argparse
    from typing import Dict, Any, Optional, List, Tuple
    import json
    import time
    from datetime import datetime
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from PIL import Image
    import cv2
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns
    import tkinter as tk
    from tkinter import filedialog, messagebox, simpledialog
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms.functional as F_transforms
    import random
    print("   ✅ All basic imports successful")
    
    print("3. Testing project imports...")
    
    # Import components (same as test_model.py lines 26-30)
    try:
        from models.model_factory import ModelFactory
        print("   ✅ ModelFactory imported")
    except Exception as e:
        print(f"   ❌ ModelFactory import failed: {e}")
    
    try:
        from models.hybrid_model import HybridUNetGNN
        print("   ✅ HybridUNetGNN imported")
    except Exception as e:
        print(f"   ❌ HybridUNetGNN import failed: {e}")
    
    try:
        from core.data_handler import create_data_loaders, nd2_to_pil, get_nd2_frame_count, SegmentationDataset
        print("   ✅ data_handler imports successful")
    except Exception as e:
        print(f"   ❌ data_handler import failed: {e}")
    
    try:
        from core.tissue_config import get_all_tissue_colors, get_tissue_name, get_num_tissue_classes
        print("   ✅ tissue_config imports successful")
    except Exception as e:
        print(f"   ❌ tissue_config import failed: {e}")
    
    try:
        from core.visualization import save_prediction_comparison, create_class_legend
        print("   ✅ visualization imports successful")
    except Exception as e:
        print(f"   ❌ visualization import failed: {e}")
    
    print("4. Testing main function call simulation...")
    
    def test_select_files_gui():
        """Simulate the GUI function"""
        print("   Simulating GUI file selection...")
        return None, None, None, False, False, False  # User cancelled simulation
    
    def test_main():
        """Simulate the main function"""
        print("   🚀 Simulating main function...")
        print("   =" * 60)
        
        # Interactive file selection
        selection_result = test_select_files_gui()
        
        if selection_result[0] is None:  # User cancelled
            print("   ❌ Selection cancelled. Exiting.")
            return
        
        print("   Would continue with model testing here...")
    
    test_main()
    print("   ✅ Main function simulation completed")
    
    print("🎉 All import tests passed!")
    
except Exception as e:
    print(f"❌ Error in import test: {e}")
    import traceback
    traceback.print_exc() 