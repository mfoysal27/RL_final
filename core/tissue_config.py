#!/usr/bin/env python
"""
Simplified Tissue Configuration for Gut Segmentation
Supports 15 tissue classes with specific color mappings
"""

import os
import numpy as np
from typing import Dict, Tuple, List

# Define the 15 tissue classes with their RGB colors
TISSUE_CLASSES = {
    0: {"name": "Villi", "color": (255, 192, 203)},
    1: {"name": "Gland", "color": (255, 255, 255)},
    2: {"name": "Submucosa_Ganglion", "color": (64, 64, 64)},
    3: {"name": "Submucosa_Fiber_tract", "color": (192, 192, 192)},
    4: {"name": "Submucosa_Blood_Vessel", "color": (128, 0, 128)},
    5: {"name": "Submucosa_Interstitial", "color": (0, 0, 128)},
    6: {"name": "Circular_muscle", "color": (0, 128, 128)},
    7: {"name": "Longitudinal_muscle", "color": (255, 165, 0)},
    8: {"name": "Myenteric_Ganglion", "color": (205, 133, 63)},
    9: {"name": "Myenteric_Fiber_tract", "color": (101, 67, 33)},
    10: {"name": "Background", "color": (0, 0, 0)},
    11: {"name": "Fat", "color": (50, 205, 50)},
    12: {"name": "Lymphoid_tissue", "color": (196, 162, 196)},
    13: {"name": "Vessel", "color": (75, 0, 130)},
    14: {"name": "Mesenteric_tissue", "color": (255, 20, 147)}
}

NUM_CLASSES = len(TISSUE_CLASSES)

def get_tissue_color(class_id: int) -> Tuple[int, int, int]:
    """Get RGB color for a tissue class"""
    if class_id in TISSUE_CLASSES:
        return TISSUE_CLASSES[class_id]["color"]
    return (128, 128, 128)  # Default gray

def get_tissue_name(class_id: int) -> str:
    """Get name for a tissue class"""
    if class_id in TISSUE_CLASSES:
        return TISSUE_CLASSES[class_id]["name"]
    return f"Unknown_{class_id}"

def get_all_tissue_colors() -> Dict[int, Tuple[int, int, int]]:
    """Get all tissue colors as dictionary"""
    return {class_id: info["color"] for class_id, info in TISSUE_CLASSES.items()}

def get_all_tissue_names() -> Dict[int, str]:
    """Get all tissue names as dictionary"""
    return {class_id: info["name"] for class_id, info in TISSUE_CLASSES.items()}

def get_num_tissue_classes() -> int:
    """Get total number of tissue classes"""
    return NUM_CLASSES

def load_custom_tissue_config(config_file: str) -> bool:
    """Load custom tissue configuration from file"""
    global TISSUE_CLASSES, NUM_CLASSES
    
    if not os.path.exists(config_file):
        print(f"Config file not found: {config_file}")
        return False
    
    try:
        new_classes = {}
        class_id = 0
        
        with open(config_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split(',')
                    if len(parts) >= 4:
                        r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
                        name = parts[3].strip()
                        new_classes[class_id] = {"name": name, "color": (r, g, b)}
                        class_id += 1
        
        if new_classes:
            TISSUE_CLASSES = new_classes
            NUM_CLASSES = len(TISSUE_CLASSES)
            print(f"Loaded {NUM_CLASSES} tissue classes from {config_file}")
            return True
    
    except Exception as e:
        print(f"Error loading config file: {e}")
    
    return False

def save_tissue_config_template(output_file: str):
    """Save current tissue configuration as template"""
    try:
        with open(output_file, 'w') as f:
            f.write("# Tissue Configuration Template\n")
            f.write("# Format: R,G,B,class_name\n\n")
            for class_id, info in TISSUE_CLASSES.items():
                color = info["color"]
                name = info["name"]
                f.write(f"{color[0]},{color[1]},{color[2]},{name}\n")
        
        print(f"Tissue configuration template saved to: {output_file}")
        return True
    
    except Exception as e:
        print(f"Error saving template: {e}")
        return False

# Initialize with simplified config on import
try:
    config_file = os.path.join(os.path.dirname(__file__), '..', 'simplified_tissue_config.txt')
    if os.path.exists(config_file):
        load_custom_tissue_config(config_file)
        print(f"✅ Loaded simplified tissue configuration with {NUM_CLASSES} classes")
    else:
        print(f"✅ Using default tissue configuration with {NUM_CLASSES} classes")
except Exception as e:
    print(f"Warning: Could not load tissue configuration: {e}")

if __name__ == "__main__":
    print("Simplified Tissue Configuration")
    print(f"Total classes: {NUM_CLASSES}")
    for class_id, info in TISSUE_CLASSES.items():
        print(f"  {class_id}: {info['name']} - RGB{info['color']}") 