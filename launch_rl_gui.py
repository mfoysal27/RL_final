"""
Launcher script for Human Feedback RL GUI
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Launch the RL GUI application."""
    print("=" * 60)
    print("Human Feedback Reinforcement Learning GUI")
    print("Gut Tissue Segmentation Enhancement")
    print("=" * 60)
    print()
    
    try:
        from rl_human_feedback_gui import HumanFeedbackRLGUI
        
        print("Starting GUI application...")
        app = HumanFeedbackRLGUI()
        app.run()
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure all required modules are available.")
        print("Try running: pip install torch torchvision pillow numpy")
        
    except Exception as e:
        print(f"Error starting application: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 