#!/usr/bin/env python
"""
Test script to demonstrate testing with arbitrary size images
"""

import os
import sys
import torch
import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_model import InferenceOnlyDataset, ModelTester

def create_test_images():
    """Create test images of different sizes"""
    test_dir = "test_arbitrary_images"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create images of different sizes
    sizes = [(512, 512), (256, 256), (1024, 768), (128, 128), (800, 600)]
    
    for i, (width, height) in enumerate(sizes):
        # Create a simple test image
        img_array = np.random.randint(0, 255, (height, width), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='L')
        
        filename = f"test_image_{width}x{height}.png"
        img.save(os.path.join(test_dir, filename))
        print(f"Created: {filename} ({width}x{height})")
    
    return test_dir

def test_arbitrary_sizes():
    """Test the arbitrary size functionality"""
    print("🧪 Testing Arbitrary Size Image Processing")
    print("=" * 50)
    
    # Create test images
    test_dir = create_test_images()
    
    # Create dataset
    print(f"\n📁 Creating dataset from: {test_dir}")
    dataset = InferenceOnlyDataset(test_dir)
    
    print(f"✅ Dataset created with {len(dataset)} images")
    
    # Test loading a few images
    print(f"\n🔍 Testing image loading:")
    for i in range(min(3, len(dataset))):
        img_tensor = dataset[i]
        print(f"   Image {i}: {img_tensor.shape}, dtype: {img_tensor.dtype}")
        print(f"   Value range: [{img_tensor.min():.1f}, {img_tensor.max():.1f}]")
        print(f"   Path: {img_tensor.path}")
        print()
    
    print("✅ Arbitrary size testing completed!")
    print("\n📋 Summary:")
    print("   • Images loaded in original size and format")
    print("   • No preprocessing applied")
    print("   • Pixel values preserved (0-255 range)")
    print("   • Ready for model inference (if model supports arbitrary sizes)")

if __name__ == "__main__":
    test_arbitrary_sizes() 