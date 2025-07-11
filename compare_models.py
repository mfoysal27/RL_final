#!/usr/bin/env python
"""
Compare Original vs RL-Enhanced Model Performance
Quick tool to see if RL training helped or hurt performance
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

def load_model_simple(model_path, device):
    """Load model with minimal dependencies"""
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract model state
        if isinstance(checkpoint, dict):
            model_state = checkpoint.get('model_state_dict', checkpoint)
        else:
            model_state = checkpoint
        
        # Get model info
        param_count = sum(p.numel() for p in model_state.values())
        print(f"Model parameters: {param_count:,}")
        
        return model_state, param_count
        
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        return None, 0

def compare_predictions(original_path, rl_path, test_image_path=None):
    """Compare predictions from original vs RL model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("🔍 Model Comparison Analysis")
    print("=" * 50)
    
    # Load models
    print("📥 Loading models...")
    original_state, orig_params = load_model_simple(original_path, device)
    rl_state, rl_params = load_model_simple(rl_path, device)
    
    if original_state is None or rl_state is None:
        print("❌ Failed to load one or both models")
        return
    
    # Compare parameter counts
    print(f"\n📊 Parameter Comparison:")
    print(f"   Original model: {orig_params:,} parameters")
    print(f"   RL model: {rl_params:,} parameters")
    print(f"   Difference: {rl_params - orig_params:,} parameters")
    
    # Compare weight changes
    print(f"\n🔄 Weight Change Analysis:")
    total_change = 0
    total_params = 0
    
    for key in original_state.keys():
        if key in rl_state:
            orig_weight = original_state[key]
            rl_weight = rl_state[key]
            
            if orig_weight.shape == rl_weight.shape:
                change = torch.norm(rl_weight - orig_weight).item()
                total_change += change
                total_params += orig_weight.numel()
                
                # Show largest changes
                relative_change = change / torch.norm(orig_weight).item() * 100
                if relative_change > 5:  # Show layers with >5% change
                    print(f"   {key}: {relative_change:.1f}% change")
    
    avg_change = total_change / total_params if total_params > 0 else 0
    print(f"   Average parameter change: {avg_change:.6f}")
    
    # Interpretation
    print(f"\n💡 Interpretation:")
    if avg_change < 0.001:
        print("   ✅ Small changes - RL training was conservative")
    elif avg_change < 0.01:
        print("   ⚠️ Moderate changes - monitor performance carefully")
    else:
        print("   🚨 Large changes - possible catastrophic forgetting")
    
    print(f"\n🎯 Recommendations:")
    if avg_change > 0.01:
        print("   • Use smaller learning rate (1e-6 instead of 1e-5)")
        print("   • Increase forgetting penalty (0.05 instead of 0.01)")
        print("   • Provide more conservative feedback")
    else:
        print("   • Current RL settings seem reasonable")
        print("   • Test on validation data to confirm performance")

def main():
    """Main comparison function"""
    print("🔍 Model Performance Comparison Tool")
    print("=" * 50)
    
    # You can modify these paths to point to your actual models
    original_model = input("Enter path to original model (.pth): ").strip()
    rl_model = input("Enter path to RL-enhanced model (.pth): ").strip()
    
    if not os.path.exists(original_model):
        print(f"❌ Original model not found: {original_model}")
        return
    
    if not os.path.exists(rl_model):
        print(f"❌ RL model not found: {rl_model}")
        return
    
    compare_predictions(original_model, rl_model)

if __name__ == "__main__":
    main() 