"""
Centralized color configuration for segmentation visualization.
Colors are defined in RGB format.
"""

# Base class colors (RGB format) - organized by actual tissue types
CLASS_COLORS = {
    0: [220, 20, 60],      # Villi (Crimson)
    1: [50, 205, 50],      # Glands (Lime Green)
    2: [65, 105, 225],     # Submucosa (Royal Blue)
    3: [255, 140, 0],      # Circular muscle (Dark Orange)
    4: [138, 43, 226],     # Myenteric plexus (Blue Violet)
    5: [255, 20, 147],     # Longitudinal muscles (Deep Pink)
}

# Class names mapping - actual tissue types
CLASS_NAMES = {
    0: "Villi",
    1: "Glands",
    2: "Submucosa", 
    3: "Circular muscle",
    4: "Myenteric plexus",
    5: "Longitudinal muscles"
}

def get_color(class_idx):
    """Get RGB color for a class index."""
    if class_idx in CLASS_COLORS:
        return CLASS_COLORS[class_idx]
    else:
        # Generate a deterministic random color for unknown classes
        import random
        random.seed(class_idx)
        return [
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        ]

def get_class_name(class_idx):
    """Get class name for a class index."""
    return CLASS_NAMES.get(class_idx, f"Class {class_idx}")

def get_color_normalized(class_idx):
    """Get normalized RGB color (0-1 range) for a class index."""
    color = get_color(class_idx)
    return [c/255 for c in color]

def get_all_colors():
    """Get dictionary of all defined class colors."""
    return CLASS_COLORS.copy()

def get_all_class_names():
    """Get dictionary of all defined class names."""
    return CLASS_NAMES.copy() 