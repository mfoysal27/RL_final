#!/usr/bin/env python
"""
Simplified Visualization Module for Gut Tissue Segmentation
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw, ImageFont
import torch
from typing import Dict, List, Tuple, Optional

# Import tissue configuration
from .tissue_config import get_all_tissue_colors, get_tissue_name, get_num_tissue_classes

def save_prediction_comparison(image: np.ndarray, 
                             ground_truth: np.ndarray, 
                             prediction: np.ndarray,
                             output_path: str,
                             title: str = "Segmentation Results"):
    """Save a comparison of image, ground truth, and prediction"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Ground truth
    gt_colored = apply_tissue_colors(ground_truth)
    axes[1].imshow(gt_colored)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Prediction
    pred_colored = apply_tissue_colors(prediction)
    axes[2].imshow(pred_colored)
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def apply_tissue_colors(mask: np.ndarray) -> np.ndarray:
    """Apply tissue-specific colors to a segmentation mask"""
    tissue_colors = get_all_tissue_colors()
    
    # Create RGB image
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Apply colors for each class
    for class_id, color in tissue_colors.items():
        mask_pixels = mask == class_id
        colored_mask[mask_pixels] = color
    
    return colored_mask

def create_class_legend(save_path: str = None) -> plt.Figure:
    """Create a legend showing all tissue classes and their colors"""
    tissue_colors = get_all_tissue_colors()
    num_classes = get_num_tissue_classes()
    
    fig, ax = plt.subplots(figsize=(8, 10))
    
    # Create legend patches
    patches = []
    labels = []
    
    for class_id in range(num_classes):
        color = tissue_colors.get(class_id, (128, 128, 128))  # Default gray
        color_normalized = [c/255.0 for c in color]  # Normalize to 0-1
        
        patch = mpatches.Rectangle((0, 0), 1, 1, facecolor=color_normalized)
        patches.append(patch)
        labels.append(f"{class_id}: {get_tissue_name(class_id)}")
    
    # Create legend
    ax.legend(patches, labels, loc='center', fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Tissue Class Legend', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Class legend saved: {save_path}")
    
    return fig

def visualize_batch_predictions(images: torch.Tensor,
                               ground_truth: torch.Tensor,
                               predictions: torch.Tensor,
                               output_dir: str,
                               batch_idx: int = 0):
    """Visualize a batch of predictions"""
    
    os.makedirs(output_dir, exist_ok=True)
    batch_size = images.shape[0]
    
    for i in range(min(batch_size, 4)):  # Visualize max 4 images per batch
        # Convert tensors to numpy
        img = images[i].cpu().numpy()
        if img.shape[0] == 1:  # Grayscale
            img = img[0]
        
        gt = ground_truth[i].cpu().numpy()
        pred = torch.argmax(predictions[i], dim=0).cpu().numpy()
        
        # Save comparison
        output_path = os.path.join(output_dir, f"batch_{batch_idx}_sample_{i}.png")
        save_prediction_comparison(img, gt, pred, output_path, 
                                 f"Batch {batch_idx}, Sample {i}")

def create_training_progress_plot(train_losses: List[float],
                                val_losses: List[float],
                                save_path: str):
    """Create a plot showing training progress"""
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Training progress plot saved: {save_path}")

def save_model_predictions(model, dataloader, device, output_dir: str, max_samples: int = 10):
    """Save model predictions for visualization"""
    
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    sample_count = 0
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(dataloader):
            if sample_count >= max_samples:
                break
                
            images = images.to(device)
            predictions = model(images)
            
            # Process each image in batch
            for i in range(images.shape[0]):
                if sample_count >= max_samples:
                    break
                    
                img = images[i].cpu().numpy()
                if img.shape[0] == 1:
                    img = img[0]
                
                mask = masks[i].cpu().numpy()
                pred = torch.argmax(predictions[i], dim=0).cpu().numpy()
                
                # Save comparison
                output_path = os.path.join(output_dir, f"prediction_{sample_count:03d}.png")
                save_prediction_comparison(img, mask, pred, output_path,
                                         f"Sample {sample_count}")
                
                sample_count += 1
    
    print(f"✅ Saved {sample_count} prediction visualizations to {output_dir}")

if __name__ == "__main__":
    # Test visualization functions
    print("Testing simplified visualization module...")
    
    # Create class legend
    create_class_legend("test_legend.png")
    
    # Test with dummy data
    dummy_image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    dummy_mask = np.random.randint(0, 15, (256, 256), dtype=np.uint8)
    dummy_pred = np.random.randint(0, 15, (256, 256), dtype=np.uint8)
    
    save_prediction_comparison(dummy_image, dummy_mask, dummy_pred, 
                             "test_comparison.png", "Test Visualization")
    
    print("✅ Visualization module test completed!") 