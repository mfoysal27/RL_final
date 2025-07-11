#!/usr/bin/env python
"""
Comprehensive Model Testing System for Gut Tissue Segmentation
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
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
print('not all module loaded')
from nd2reader import ND2Reader
print('all module loaded')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components
from models.model_factory import ModelFactory
from models.hybrid_model import HybridUNetGNN
from core.data_handler import create_data_loaders, nd2_to_pil, get_nd2_frame_count, SegmentationDataset
from core.tissue_config import get_all_tissue_colors, get_tissue_name, get_num_tissue_classes
from core.visualization import save_prediction_comparison, create_class_legend


class InferenceOnlyDataset(Dataset):
    """Simple dataset for inference-only mode (no masks required)"""
    
    def __init__(self, images_dir: str):
        """
        Args:
            images_dir: Path to directory containing images
        """
        self.images_dir = images_dir
        self.image_files = []
        
        print(f"ðŸ” Scanning for images in: {images_dir}")
        
        # Check if directory exists
        if not os.path.exists(images_dir):
            raise ValueError(f"Images directory does not exist: {images_dir}")
        
        # Collect all image files
        for root, _, files in os.walk(images_dir):
            print(f"   Checking directory: {root}")
            print(f"   Files found: {files}")
            
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    full_path = os.path.join(root, file)
                    self.image_files.append(full_path)
                    print(f"   âœ… Added image: {file}")
        
        self.image_files.sort()  # Ensure consistent ordering
        print(f"ðŸ“Š Total images found for inference: {len(self.image_files)}")
        
        # Validate that we have at least one image
        if len(self.image_files) == 0:
            raise ValueError(f"No image files found in {images_dir}. Please check that the directory contains image files with extensions: .png, .jpg, .jpeg, .tif, .tiff")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        if idx >= len(self.image_files):
            raise IndexError(f"Index {idx} out of range. Dataset has {len(self.image_files)} images.")
            
        img_path = self.image_files[idx]
        
        try:
            # Load image in original format and size - NO PREPROCESSING
            image = Image.open(img_path)
            print(f"   📐 Loading original image: {os.path.basename(img_path)} - Size: {image.size}, Mode: {image.mode}")
            
            # Convert PIL image to numpy array (keep original format)
            image_np = np.array(image)
            
            # Handle different image formats
            if len(image_np.shape) == 2:
                # Grayscale image
                print(f"      Grayscale image: {image_np.shape}")
                image_np = image_np[np.newaxis, :, :]  # Add channel dimension [1, H, W]
            elif len(image_np.shape) == 3:
                # RGB/RGBA image
                print(f"      Color image: {image_np.shape}")
                image_np = image_np.transpose(2, 0, 1)  # Convert [H, W, C] to [C, H, W]
            else:
                raise ValueError(f"Unsupported image shape: {image_np.shape}")
            
            # Convert to tensor WITHOUT normalization (keep original pixel values)
            image_tensor = torch.from_numpy(image_np.copy()).float()
            
            print(f"      Tensor shape: {image_tensor.shape}, dtype: {image_tensor.dtype}")
            print(f"      Pixel value range: [{image_tensor.min():.1f}, {image_tensor.max():.1f}]")
            
            # Extract base name and slice info from filename
            filename = os.path.basename(img_path)
            base_name = os.path.splitext(filename)[0]
            
            # Store metadata
            image_tensor.path = img_path
            image_tensor.original_size = image.size  # (width, height)
            
            # Try to extract slice information if available
            if "_slice_" in base_name:
                # Format is typically "base_name_slice_X"
                parts = base_name.split("_slice_")
                slice_base_name = parts[0]
                try:
                    slice_idx = int(parts[1]) if len(parts) > 1 else 0
                    image_tensor.base_name = slice_base_name
                    image_tensor.slice_idx = slice_idx
                except ValueError:
                    # If slice index is not a number, use full base name
                    image_tensor.base_name = base_name
                    image_tensor.slice_idx = 0
            else:
                # No slice information, use full base name
                image_tensor.base_name = base_name
                image_tensor.slice_idx = 0
            
            return image_tensor
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return minimal fallback for raw mode
            blank_image = torch.zeros((1, 100, 100))  # Minimal size
            blank_image.path = img_path
            blank_image.base_name = os.path.basename(os.path.splitext(img_path)[0])
            blank_image.slice_idx = 0
            return blank_image

class ND2TestDataset(Dataset):
    """Specialized dataset for testing ND2 files with corresponding masks"""
    
    def __init__(self, images_dir: str, masks_dir: str, has_ground_truth: bool = True):
        """
        Args:
            images_dir: Path to directory containing ND2 files
            masks_dir: Path to directory containing mask files
            has_ground_truth: Whether masks are available
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.has_ground_truth = has_ground_truth
        self.image_mask_pairs = []
        
        print(f"🔍 Processing ND2 files for testing...")
        self._process_nd2_files()
        
        print(f"📊 Total image-mask pairs found: {len(self.image_mask_pairs)}")
        
        if len(self.image_mask_pairs) == 0:
            raise ValueError("No valid ND2 slice-mask pairs found")
    
    def _process_nd2_files(self):
        """Process ND2 files and extract slices with corresponding masks"""
        # Find all ND2 files
        nd2_files = []
        for root, _, files in os.walk(self.images_dir):
            for file in files:
                if file.lower().endswith('.nd2'):
                    nd2_files.append(os.path.join(root, file))
        
        print(f"Found {len(nd2_files)} ND2 files")
        
        for nd2_path in nd2_files:
            try:
                base_name = os.path.splitext(os.path.basename(nd2_path))[0]
                print(f"Processing {base_name}.nd2...")
                
                # Get frame count
                frame_count = get_nd2_frame_count(nd2_path)
                print(f"  Found {frame_count} slices")
                
                if frame_count > 0:
                    # Extract all frames
                    all_frames = nd2_to_pil(nd2_path)
                    actual_frames = len(all_frames) if all_frames else 0
                    
                    print(f"  Successfully extracted {actual_frames} slices")
                    
                    # Process each extracted slice
                    for slice_idx in range(actual_frames):
                        slice_img = all_frames[slice_idx]
                        
                        if slice_img is not None:
                            # Look for corresponding mask file
                            mask_path = self._find_corresponding_mask(base_name, slice_idx)
                            
                            if self.has_ground_truth:
                                if mask_path and os.path.exists(mask_path):
                                    self.image_mask_pairs.append((slice_img, mask_path, f"{base_name}_slice_{slice_idx}"))
                                    print(f"   ✓ Added slice {slice_idx} with mask")
                                else:
                                    print(f"   ⚠ No mask found for slice {slice_idx}")
                            else:
                                # Inference only - no mask needed
                                self.image_mask_pairs.append((slice_img, None, f"{base_name}_slice_{slice_idx}"))
                                print(f"   ✓ Added slice {slice_idx} (inference only)")
                        
            except Exception as e:
                print(f"  ✗ Error processing {nd2_path}: {e}")
                continue
    
    def _find_corresponding_mask(self, base_name: str, slice_idx: int):
        """Find corresponding mask file for ND2 slice"""
        if not self.has_ground_truth:
            return None
            
        # Try different naming patterns for masks
        patterns = [
            f"segmented_{base_name}_slice_{slice_idx + 1}",  # 1-indexed
            f"segmented_{base_name}_slice_{slice_idx}",      # 0-indexed
            f"segmented_{base_name}_slice_{slice_idx:04d}",  # 0-indexed with padding
            f"segmented_{base_name}_slice_{slice_idx + 1:04d}"  # 1-indexed with padding
        ]
        
        extensions = ['.png', '.tif', '.tiff', '.jpg', '.jpeg']
        
        for pattern in patterns:
            for ext in extensions:
                mask_path = os.path.join(self.masks_dir, pattern + ext)
                if os.path.exists(mask_path):
                    return mask_path
        
        return None
    
    def __len__(self):
        return len(self.image_mask_pairs)
    
    def __getitem__(self, idx):
        if idx >= len(self.image_mask_pairs):
            raise IndexError(f"Index {idx} out of range. Dataset has {len(self.image_mask_pairs)} samples.")
        
        if self.has_ground_truth:
            slice_img, mask_path, slice_name = self.image_mask_pairs[idx]
            
            try:
                # Process the slice image
                image = slice_img.convert("L")  # Convert to grayscale
                image = image.resize((256, 256), Image.BILINEAR)
                
                # Load and process mask
                mask = Image.open(mask_path)
                mask = mask.resize((256, 256), Image.NEAREST)
                mask_np = np.array(mask)
                
                # Handle RGB masks if needed
                if len(mask_np.shape) == 3 and mask_np.shape[2] == 3:
                    # Convert RGB to class indices
                    tissue_colors = get_all_tissue_colors()
                    height, width = mask_np.shape[:2]
                    mask_classes = np.zeros((height, width), dtype=np.int64)
                    
                    for class_id, target_color in tissue_colors.items():
                        color_match = np.all(np.abs(mask_np - target_color) <= 10, axis=2)
                        mask_classes[color_match] = class_id
                    
                    mask_np = mask_classes
                
                # Convert to tensors
                image_tensor = F_transforms.to_tensor(image)
                mask_tensor = torch.tensor(mask_np, dtype=torch.long)
                
                # Store metadata
                image_tensor.path = slice_name
                
                return image_tensor, mask_tensor
                
            except Exception as e:
                print(f"❌ Error loading sample {idx}: {e}")
                # Return blank fallback
                blank_image = torch.zeros((1, 256, 256))
                blank_mask = torch.zeros((256, 256), dtype=torch.long)
                blank_image.path = slice_name
                return blank_image, blank_mask
        else:
            # Inference only mode - KEEP ORIGINAL SIZE
            slice_img, _, slice_name = self.image_mask_pairs[idx]
            
            try:
                # Process the slice image - convert to grayscale but KEEP ORIGINAL SIZE
                image = slice_img.convert("L")
                
                # Store original size for reference
                original_size = image.size  # (width, height)
                
                # Parse slice name to extract base name and slice index
                # Format is typically "base_name_slice_X"
                parts = slice_name.split("_slice_")
                base_name = parts[0]
                slice_idx = int(parts[1]) if len(parts) > 1 else 0
                
                # Convert to tensor without resizing
                image_tensor = F_transforms.to_tensor(image)
                
                # Store metadata
                image_tensor.path = slice_name
                image_tensor.original_size = original_size
                image_tensor.base_name = base_name
                image_tensor.slice_idx = slice_idx
                
                print(f"   📐 Loaded slice {slice_name} with original size {original_size}")
                
                return image_tensor
                
            except Exception as e:
                print(f"❌ Error loading sample {idx}: {e}")
                blank_image = torch.zeros((1, 256, 256))
                blank_image.path = slice_name
                return blank_image

class ModelTester:
    """Comprehensive model testing with evaluation metrics"""
    
    def __init__(self, model_path: str, test_data_dir: str, output_dir: str = "test_results", 
                 has_ground_truth: bool = True, use_training_split: bool = False, 
                 inference_method: str = "patch"):
        self.model_path = model_path
        self.test_data_dir = test_data_dir
        self.output_dir = output_dir
        self.has_ground_truth = has_ground_truth  # User-specified ground truth availability
        self.use_training_split = use_training_split  # Use exact training test split
        self.inference_method = inference_method  # "patch" or "resize"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create separate folders for predictions and visualizations
        self.prediction_dir = os.path.join(self.output_dir, "prediction")
        self.visualization_dir = os.path.join(self.output_dir, "visualization")
        os.makedirs(self.prediction_dir, exist_ok=True)
        os.makedirs(self.visualization_dir, exist_ok=True)
        
        # Load model and configuration
        self.model, self.config = self._load_model()
        self.num_classes = self.config.get('num_classes', get_num_tissue_classes())
        
        # Setup test data loader
        self.test_loader = self._setup_test_loader()
        
        # Initialize metrics storage
        self.predictions = []
        self.ground_truths = []
        self.test_images = []
        
        print(f"🔍 Model Tester Initialized")
        print(f"   Model: {os.path.basename(self.model_path)}")
        print(f"   Test Data: {self.test_data_dir}")
        print(f"   Output: {self.output_dir}")
        print(f"   Predictions: {self.prediction_dir}")
        print(f"   Visualizations: {self.visualization_dir}")
        print(f"   Device: {self.device}")
        print(f"   Classes: {self.num_classes}")
        print(f"   Ground Truth Expected: {'Yes' if self.has_ground_truth else 'No'}")
        print(f"   Test Split Mode: {'Training test split' if self.use_training_split else 'All available data'}")
        print(f"   Inference Method: {self.inference_method.upper()}")
    
    def _load_model(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Load trained model and configuration"""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Extract configuration
        config = checkpoint.get('config', {})
        model_type = checkpoint.get('model_type', config.get('model_type', 'unet'))
        num_classes = checkpoint.get('num_classes', config.get('num_classes', get_num_tissue_classes()))
        
        print(f"ðŸ“‹ Loading model: {model_type}")
        print(f"   Classes: {num_classes}")
        
        # Create model
        if model_type == 'hybrid_unet_gnn':
            model = HybridUNetGNN(
                in_channels=1,
                num_classes=num_classes,
                features=[64, 128, 256, 512],
                gnn_hidden_channels=64
            )
        else:
            model_config = config.copy()
            model_config['num_classes'] = num_classes
            model = ModelFactory.create_model(model_type, model_config)
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        
        print(f"âœ… Model loaded successfully")
        return model, config
    
    def _create_test_only_loader(self):
        """Create a test loader that uses ALL available data for testing"""
        try:
            print(f"ðŸŽ¯ Creating test-only loader for all available data...")
            
            # Create a dataset with ALL data for testing (no train/val splits)
            test_dataset = SegmentationDataset(
                data_dir=self.test_data_dir, 
                split='test',  # This ensures no augmentation
                augment=False  # Explicitly disable augmentation for testing
            )
            
            if len(test_dataset) == 0:
                raise ValueError("No valid test data found")
            
            print(f"âœ… Test dataset created with {len(test_dataset)} samples (100% of available data)")
            print(f"   Augmentation: {test_dataset.augment} (should be False)")
            
            # Create test loader with all data
            test_loader = DataLoader(
                test_dataset,
                batch_size=1,
                shuffle=False,  # Keep original order for testing
                num_workers=0,
                pin_memory=True
            )
            
            # Store dataset info
            self.num_classes = test_dataset.get_num_classes()
            self.class_names = test_dataset.get_class_names()
            
            print(f"   Number of classes: {self.num_classes}")
            print(f"   Class names: {self.class_names}")
            
            return test_loader
            
        except Exception as e:
            print(f"âŒ Error creating test-only loader: {e}")
            raise

    def _setup_test_loader(self):
        """Setup test data loader"""
        try:
            # Check if we should use the exact training test split
            if self.use_training_split:
                print(f"ðŸŽ¯ Using EXACT training test split mode...")
                return self._create_training_test_split_loader()
            
            # Otherwise, use the existing logic for all available data
            print(f"ðŸ“ Using ALL available data mode...")
            
            # Check if test data has proper structure
            images_dir = os.path.join(self.test_data_dir, 'images')
            masks_dir = os.path.join(self.test_data_dir, 'masks')
            
            if not os.path.exists(images_dir):
                raise ValueError(f"Test images directory not found: {images_dir}")
            
            print(f"ðŸ“ Test data structure:")
            print(f"   Images: âœ… {len(os.listdir(images_dir))} files")
            
            # Check if we have ND2 files
            nd2_files = [f for f in os.listdir(images_dir) if f.lower().endswith('.nd2')]
            regular_images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
            
            if nd2_files:
                print(f"   Detected {len(nd2_files)} ND2 files - using specialized ND2 processing")
                
                if self.has_ground_truth:
                    if not os.path.exists(masks_dir):
                        raise ValueError(f"Ground truth expected but masks directory not found: {masks_dir}")
                    print(f"   Masks: âœ… Available ({len(os.listdir(masks_dir))} files)")
                    
                    # Use specialized ND2 dataset with ground truth
                    nd2_dataset = ND2TestDataset(images_dir, masks_dir, self.has_ground_truth)
                else:
                    print(f"   Masks: âš ï¸  Not expected (inference-only mode)")
                    
                    # Use specialized ND2 dataset without ground truth - preserves original sizes
                    nd2_dataset = ND2TestDataset(images_dir, masks_dir, self.has_ground_truth)
                
                test_loader = DataLoader(
                    nd2_dataset,
                    batch_size=1,  # Always use batch size 1 for arbitrary size images
                    shuffle=False,
                    num_workers=0,
                    pin_memory=True
                )
                
                print(f"âœ… Created ND2 test loader with {len(nd2_dataset)} samples")
                
            elif regular_images:
                print(f"   Detected {len(regular_images)} regular image files")
                
                if self.has_ground_truth:
                    masks_available = os.path.exists(masks_dir) and len(os.listdir(masks_dir)) > 0
                    
                    if masks_available:
                        print(f"   Masks: âœ… Available ({len(os.listdir(masks_dir))} files)")
                        
                        # Use ALL data for testing (existing functionality)
                        print(f"ðŸš€ Using all available data approach (100% of data)")
                        test_loader = self._create_test_only_loader()
                        
                        print(f"âœ… Using ALL {len(test_loader.dataset)} samples for testing")
                        
                    else:
                        raise ValueError(f"Ground truth expected but masks directory not found or empty: {masks_dir}")
                else:
                    print(f"   Masks: âš ï¸  Not expected (inference-only mode)")
                    
                    # Use inference-only dataset for regular images
                    print(f"ðŸ”§ Creating inference-only dataset with original image sizes...")
                    inference_dataset = InferenceOnlyDataset(images_dir)
                    
                    if len(inference_dataset) == 0:
                        raise ValueError(f"No images found for inference.")
                    
                    print(f"âœ… Created inference dataset with {len(inference_dataset)} samples")
                    
                    test_loader = DataLoader(
                        inference_dataset,
                        batch_size=1,  # Always use batch size 1 for arbitrary size images
                        shuffle=False,
                        num_workers=0,
                        pin_memory=True
                    )
            else:
                raise ValueError(f"No image files found in {images_dir}. Expected ND2 files or regular image files.")
            
            return test_loader
            
        except Exception as e:
            print(f"âŒ Error setting up test loader: {e}")
            raise
    
    def run_inference(self, save_predictions: bool = True) -> Dict[str, Any]:
        """Run inference on test data"""
        print(f"\n🔍 Running inference on {len(self.test_loader)} test samples...")
        print(f"   Mode: {'Evaluation' if self.has_ground_truth else 'Inference-only'}")
        if not self.has_ground_truth:
            print(f"   Inference Method: {self.inference_method.upper()}")
        
        self.model.eval()
        total_time = 0
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(self.test_loader, desc="Testing")):
                # Handle data based on what's available and what's expected
                if self.has_ground_truth:
                    # Ground truth mode - expect (images, masks) tuple
                    if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                        images, masks = batch_data
                        has_masks_in_batch = True
                    else:
                        # Unexpected format when ground truth is expected
                        print(f"Warning: Expected (images, masks) but got different format")
                        if isinstance(batch_data, (list, tuple)):
                            images = batch_data[0]
                        else:
                            images = batch_data
                        masks = None
                        has_masks_in_batch = False
                else:
                    # Inference-only mode - expect only images (single tensor)
                    if isinstance(batch_data, (list, tuple)):
                        images = batch_data[0]  # Take first element if it's wrapped
                    else:
                        images = batch_data  # Direct tensor
                    masks = None
                    has_masks_in_batch = False
                
                # Log image size for first batch
                if batch_idx == 0:
                    print(f"   📊 First batch image shape: {images.shape}")
                    print(f"      Image size: {images.shape[2]}x{images.shape[3]}")
                
                # Store original size for resizing predictions back
                original_height = images.shape[2]
                original_width = images.shape[3]
                original_size = (original_height, original_width)
                
                # Determine inference method for inference-only mode
                if not self.has_ground_truth:
                    if self.inference_method == "patch" and (original_height > 256 or original_width > 256):
                        # Use patch-based inference for large images
                        print(f"      🧩 Using PATCH-BASED inference for image: {original_height}x{original_width}")
                        predictions_final = self._run_patch_based_inference(images)
                    elif self.inference_method == "resize":
                        # Use resize method
                        print(f"      📐 Using RESIZE inference: {original_height}x{original_width} → 256x256")
                        predictions_final = self._run_resize_inference(images, original_size)
                    else:
                        # For 256x256 or smaller images, use direct inference
                        print(f"      🔄 Using DIRECT inference for standard size image")
                        predictions_final = self._run_direct_inference(images)
                else:
                    # Ground truth mode - use existing logic
                    # Check if we need to resize for the model
                    model_requires_resize = False
                    if hasattr(self.model, 'input_size'):
                        required_size = self.model.input_size
                        model_requires_resize = True
                    elif original_height != 256 or original_width != 256:
                        # Most models are trained on 256x256, so we resize if different
                        required_size = (256, 256)
                        model_requires_resize = True
                        print(f"      ⚠️ Non-standard image size detected: {original_height}x{original_width}")
                        print(f"         Resizing to 256x256 for model inference, then resizing back")
                    
                    # Resize if needed for model compatibility
                    if model_requires_resize:
                        # Use interpolate to resize the batch
                        images_resized = F.interpolate(images, size=required_size, mode='bilinear', align_corners=False)
                        images_for_model = images_resized
                    else:
                        images_for_model = images
                    
                    # Move to device
                    images_for_model = images_for_model.to(self.device)
                    
                    try:
                        # Measure inference time
                        start_time = time.time()
                        outputs = self.model(images_for_model)
                        inference_time = time.time() - start_time
                        total_time += inference_time
                        
                        # Get predictions
                        predictions = torch.argmax(outputs, dim=1)
                        
                        # Resize predictions back to original size if needed
                        if model_requires_resize:
                            # Convert to one-hot, resize, then argmax
                            num_classes = outputs.shape[1]
                            one_hot = torch.zeros((predictions.shape[0], num_classes, *predictions.shape[1:]), 
                                                device=predictions.device)
                            one_hot.scatter_(1, predictions.unsqueeze(1), 1)
                            
                            # Resize one-hot back to original size
                            one_hot_resized = F.interpolate(one_hot.float(), size=original_size, 
                                                          mode='nearest')
                            
                            # Convert back to class indices
                            predictions_resized = torch.argmax(one_hot_resized, dim=1)
                            predictions_final = predictions_resized
                        else:
                            predictions_final = predictions
                    
                    except Exception as e:
                        print(f"❌ Error during inference on batch {batch_idx}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                
                # Store results
                for i in range(images.shape[0]):
                    img_np = images[i].cpu().numpy()
                    pred_np = predictions_final[i].cpu().numpy()
                    
                    # Keep original format for storage (don't convert single channel)
                    self.test_images.append(images[i])  # Store tensor to preserve metadata
                    self.predictions.append(pred_np)
                    
                    # Only store ground truth if we expect it and it's available
                    if self.has_ground_truth and has_masks_in_batch:
                        mask_np = masks[i].cpu().numpy()
                        self.ground_truths.append(mask_np)
                    
                    # Save individual prediction
                    if save_predictions:
                        self._save_prediction(
                            img_np, pred_np, 
                            mask_np if (self.has_ground_truth and has_masks_in_batch) else None,
                            batch_idx * images.shape[0] + i
                        )
        
        # Calculate average inference time
        avg_inference_time = total_time / len(self.test_loader) if len(self.test_loader) > 0 else 0
        
        # Return inference results
        results = {
            'num_samples': len(self.test_loader),
            'total_inference_time': total_time,
            'avg_inference_time': avg_inference_time,
            'inference_method': self.inference_method if not self.has_ground_truth else 'standard'
        }
        
        print(f"✅ Inference completed on {len(self.test_loader)} samples")
        print(f"   Total time: {total_time:.2f} seconds")
        print(f"   Average time per batch: {avg_inference_time:.4f} seconds")
        if not self.has_ground_truth:
            print(f"   Method used: {self.inference_method.upper()}")
        
        return results
    
    def _run_patch_based_inference(self, images):
        """Run patch-based inference for large images
        
        Args:
            images: Input image tensor [B, C, H, W]
            
        Returns:
            Predictions tensor with original image size
        """
        batch_size = images.shape[0]
        channels = images.shape[1]
        height = images.shape[2]
        width = images.shape[3]
        
        # Define patch size
        patch_size = 256
        
        # Calculate number of patches in each dimension
        num_patches_h = (height + patch_size - 1) // patch_size
        num_patches_w = (width + patch_size - 1) // patch_size
        
        print(f"      🧩 Dividing into {num_patches_h}x{num_patches_w} patches ({num_patches_h * num_patches_w} total)")
        
        # Create output tensor to store full-sized predictions
        predictions_final = torch.zeros((batch_size, height, width), dtype=torch.long, device=self.device)
        
        # Process each image in the batch
        for b in range(batch_size):
            image = images[b:b+1]  # Keep batch dimension [1, C, H, W]
            
            # Create empty tensor to accumulate patches
            full_prediction = torch.zeros((height, width), dtype=torch.long, device=self.device)
            
            # Track inference time for this image
            image_start_time = time.time()
            
            # Process each patch
            for h_idx in range(num_patches_h):
                for w_idx in range(num_patches_w):
                    # Calculate patch coordinates
                    h_start = h_idx * patch_size
                    w_start = w_idx * patch_size
                    h_end = min(h_start + patch_size, height)
                    w_end = min(w_start + patch_size, width)
                    
                    # Calculate actual patch size (might be smaller at edges)
                    actual_h = h_end - h_start
                    actual_w = w_end - w_start
                    
                    # Extract patch
                    patch = image[:, :, h_start:h_end, w_start:w_end]
                    
                    # Handle edge cases with padding if needed
                    if actual_h < patch_size or actual_w < patch_size:
                        # Create padded patch
                        padded_patch = torch.zeros((1, channels, patch_size, patch_size), 
                                                  dtype=patch.dtype, device=patch.device)
                        padded_patch[:, :, :actual_h, :actual_w] = patch
                        patch = padded_patch
                    
                    # Move to device
                    patch = patch.to(self.device)
                    
                    # Run inference on patch
                    with torch.no_grad():
                        outputs = self.model(patch)
                        patch_pred = torch.argmax(outputs, dim=1)[0]  # Remove batch dimension
                    
                    # Place prediction in the full image
                    if actual_h < patch_size or actual_w < patch_size:
                        # Only copy the valid part from padded prediction
                        full_prediction[h_start:h_end, w_start:w_end] = patch_pred[:actual_h, :actual_w]
                    else:
                        full_prediction[h_start:h_end, w_start:w_end] = patch_pred
            
            # Store the full prediction for this image
            predictions_final[b] = full_prediction
            
            # Log time for this image
            image_time = time.time() - image_start_time
            print(f"      ⏱️ Image processed in {image_time:.2f}s with {num_patches_h * num_patches_w} patches")
        
        return predictions_final
    
    def _run_resize_inference(self, images, original_size):
        """Run inference using resize method"""
        # Resize to 256x256
        images_resized = F.interpolate(images, size=(256, 256), mode='bilinear', align_corners=False)
        images_resized = images_resized.to(self.device)
        
        # Run inference
        outputs = self.model(images_resized)
        predictions = torch.argmax(outputs, dim=1)
        
        # Resize predictions back to original size
        num_classes = outputs.shape[1]
        one_hot = torch.zeros((predictions.shape[0], num_classes, *predictions.shape[1:]), 
                             device=predictions.device)
        one_hot.scatter_(1, predictions.unsqueeze(1), 1)
        
        # Resize one-hot back to original size
        one_hot_resized = F.interpolate(one_hot.float(), size=original_size, mode='nearest')
        
        # Convert back to class indices
        predictions_final = torch.argmax(one_hot_resized, dim=1)
        
        return predictions_final
    
    def _run_direct_inference(self, images):
        """Run direct inference on images (for 256x256 or smaller)"""
        images = images.to(self.device)
        
        # Run inference
        outputs = self.model(images)
        predictions = torch.argmax(outputs, dim=1)
        
        return predictions

    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics"""
        if not self.ground_truths or len(self.ground_truths) == 0:
            print("âš ï¸  No ground truth available - cannot calculate metrics")
            return {}
        
        print(f"\nðŸ“Š Calculating evaluation metrics...")
        
        # Flatten predictions and ground truths
        pred_flat = np.concatenate([pred.flatten() for pred in self.predictions])
        gt_flat = np.concatenate([gt.flatten() for gt in self.ground_truths])
        
        # Overall accuracy
        overall_accuracy = np.mean(pred_flat == gt_flat)
        
        # Per-class metrics
        class_metrics = {}
        for class_id in range(self.num_classes):
            class_mask_pred = (pred_flat == class_id)
            class_mask_gt = (gt_flat == class_id)
            
            if np.sum(class_mask_gt) == 0:
                continue
            
            # Precision, Recall, F1
            tp = np.sum(class_mask_pred & class_mask_gt)
            fp = np.sum(class_mask_pred & ~class_mask_gt)
            fn = np.sum(~class_mask_pred & class_mask_gt)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # IoU
            intersection = tp
            union = tp + fp + fn
            iou = intersection / union if union > 0 else 0
            
            class_metrics[class_id] = {
                'name': get_tissue_name(class_id),
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'iou': iou,
                'pixel_count': np.sum(class_mask_gt)
            }
        
        # Mean IoU
        mean_iou = np.mean([metrics['iou'] for metrics in class_metrics.values()])
        
        # Dice coefficient
        dice_scores = []
        for class_id in range(self.num_classes):
            pred_mask = (pred_flat == class_id)
            gt_mask = (gt_flat == class_id)
            
            intersection = np.sum(pred_mask & gt_mask)
            dice = 2 * intersection / (np.sum(pred_mask) + np.sum(gt_mask)) if (np.sum(pred_mask) + np.sum(gt_mask)) > 0 else 0
            dice_scores.append(dice)
        
        mean_dice = np.mean(dice_scores)
        
        metrics = {
            'overall_accuracy': overall_accuracy,
            'mean_iou': mean_iou,
            'mean_dice': mean_dice,
            'class_metrics': class_metrics,
            'confusion_matrix': confusion_matrix(gt_flat, pred_flat, labels=range(self.num_classes))
        }
        
        print(f"âœ… Metrics calculated!")
        print(f"   Overall Accuracy: {overall_accuracy:.4f}")
        print(f"   Mean IoU: {mean_iou:.4f}")
        print(f"   Mean Dice: {mean_dice:.4f}")
        
        return metrics
    
    def _save_prediction(self, image: np.ndarray, prediction: np.ndarray, 
                        ground_truth: Optional[np.ndarray], sample_idx: int):
        """Save individual prediction visualization"""
        # Get image metadata if available (for inference-only mode with original names)
        image_path = getattr(self.test_images[sample_idx], 'path', f"sample_{sample_idx}")
        base_name = getattr(self.test_images[sample_idx], 'base_name', None)
        slice_idx = getattr(self.test_images[sample_idx], 'slice_idx', None)
        
        # Use original ND2 naming if available, otherwise use generic name
        if base_name is not None and slice_idx is not None:
            # Use original ND2 file and slice names
            prediction_filename = f"predicted_{base_name}_slice_{slice_idx}.png"
            title = f"{base_name} - Slice {slice_idx}"
        else:
            # Use generic naming
            prediction_filename = f"prediction_{sample_idx:04d}.png"
            title = f"Test Sample {sample_idx}"
        
        # Paths for visualization and prediction
        visualization_path = os.path.join(self.visualization_dir, prediction_filename)
        prediction_path = os.path.join(self.prediction_dir, prediction_filename)
        
        # Handle different image formats for visualization
        if len(image.shape) == 3 and image.shape[0] in [1, 3, 4]:
            # Convert from [C, H, W] to [H, W, C] for visualization
            if image.shape[0] == 1:
                # Grayscale: [1, H, W] -> [H, W]
                vis_image = image[0]
                cmap = 'gray'
            else:
                # Color: [C, H, W] -> [H, W, C]
                vis_image = image.transpose(1, 2, 0)
                cmap = None
        else:
            # Already in [H, W] or [H, W, C] format
            vis_image = image
            cmap = 'gray' if len(vis_image.shape) == 2 else None
        
        # Normalize image for visualization if needed
        if vis_image.dtype != np.uint8:
            if vis_image.max() > 1.0:
                # Scale from 0-255 range to 0-1
                vis_image = vis_image.astype(np.float32) / 255.0
            elif vis_image.max() <= 1.0:
                # Already in 0-1 range
                pass
            else:
                # Normalize to 0-1 range
                vis_image = (vis_image - vis_image.min()) / (vis_image.max() - vis_image.min())
        
        # Apply tissue colors to prediction
        tissue_colors = get_all_tissue_colors()
        colored_pred = np.zeros((*prediction.shape, 3), dtype=np.uint8)
        for class_id, color in tissue_colors.items():
            mask = prediction == class_id
            colored_pred[mask] = color
        
        # Save colored prediction to prediction folder
        colored_pred_img = Image.fromarray(colored_pred)
        colored_pred_img.save(prediction_path)
        print(f"   💾 Saved prediction mask to {prediction_path}")
        
        if ground_truth is not None:
            # Save comparison visualization to visualization folder
            save_prediction_comparison(vis_image, ground_truth, prediction, visualization_path, title)
            print(f"   🖼️ Saved comparison to {visualization_path}")
        else:
            # Save visualization to visualization folder
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # Input image
            axes[0].imshow(vis_image, cmap=cmap)
            axes[0].set_title(f'Input Image ({vis_image.shape[1]}x{vis_image.shape[0]})')
            axes[0].axis('off')
            
            # Prediction
            axes[1].imshow(colored_pred)
            axes[1].set_title(f'Prediction ({prediction.shape[1]}x{prediction.shape[0]})')
            axes[1].axis('off')
            
            plt.suptitle(title)
            plt.tight_layout()
            plt.savefig(visualization_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"   🖼️ Saved visualization to {visualization_path}")

    def generate_report(self, metrics: Dict[str, Any], inference_results: Dict[str, Any]):
        """Generate comprehensive test report"""
        print(f"\n📋 Generating test report...")
        
        report_path = os.path.join(self.output_dir, "test_report.json")
        
        # Create comprehensive report
        report = {
            'test_info': {
                'model_path': self.model_path,
                'test_data_dir': self.test_data_dir,
                'output_dir': self.output_dir,
                'prediction_dir': self.prediction_dir,
                'visualization_dir': self.visualization_dir,
                'test_date': datetime.now().isoformat(),
                'device': str(self.device),
                'num_classes': self.num_classes,
                'has_ground_truth': self.has_ground_truth
            },
            'inference_results': inference_results,
            'metrics': metrics,
            'model_config': self.config
        }
        
        # Save JSON report
        with open(report_path, 'w') as f:
            # Convert numpy types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj
            
            json.dump(convert_numpy(report), f, indent=4)
        
        # Generate visualizations only if metrics are available
        if metrics and len(metrics) > 0:
            print(f"📊 Generating metric visualizations...")
            self._generate_metric_visualizations(metrics)
        else:
            print(f"⚠️ Skipping metric visualizations (no ground truth available)")
        
        # Create class legend
        legend_path = os.path.join(self.visualization_dir, "class_legend.png")
        create_class_legend(legend_path)
        
        print(f"✅ Test report generated!")
        print(f"   Report: {report_path}")
        print(f"   Predictions: {self.prediction_dir}")
        print(f"   Visualizations: {self.visualization_dir}")
        if not self.has_ground_truth:
            print(f"   Note: Metric visualizations skipped (no ground truth)")
        
        return report_path
    
    def _generate_metric_visualizations(self, metrics: Dict[str, Any]):
        """Generate metric visualization plots"""
        
        # 1. Confusion Matrix
        if 'confusion_matrix' in metrics:
            plt.figure(figsize=(10, 8))
            cm = metrics['confusion_matrix']
            
            # Create class labels
            class_labels = [get_tissue_name(i) for i in range(self.num_classes)]
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_labels, yticklabels=class_labels)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(os.path.join(self.visualization_dir, 'confusion_matrix.png'), dpi=150)
            plt.close()
        
        # 2. Per-class metrics
        if 'class_metrics' in metrics:
            class_metrics = metrics['class_metrics']
            
            classes = list(class_metrics.keys())
            precisions = [class_metrics[c]['precision'] for c in classes]
            recalls = [class_metrics[c]['recall'] for c in classes]
            f1s = [class_metrics[c]['f1'] for c in classes]
            ious = [class_metrics[c]['iou'] for c in classes]
            
            class_names = [get_tissue_name(c) for c in classes]
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Per-Class Metrics', fontsize=16)
            
            # Precision
            axes[0, 0].bar(range(len(classes)), precisions)
            axes[0, 0].set_title('Precision')
            axes[0, 0].set_xticks(range(len(classes)))
            axes[0, 0].set_xticklabels(class_names, rotation=45)
            axes[0, 0].set_ylim(0, 1)
            
            # Recall
            axes[0, 1].bar(range(len(classes)), recalls)
            axes[0, 1].set_title('Recall')
            axes[0, 1].set_xticks(range(len(classes)))
            axes[0, 1].set_xticklabels(class_names, rotation=45)
            axes[0, 1].set_ylim(0, 1)
            
            # F1 Score
            axes[1, 0].bar(range(len(classes)), f1s)
            axes[1, 0].set_title('F1 Score')
            axes[1, 0].set_xticks(range(len(classes)))
            axes[1, 0].set_xticklabels(class_names, rotation=45)
            axes[1, 0].set_ylim(0, 1)
            
            # IoU
            axes[1, 1].bar(range(len(classes)), ious)
            axes[1, 1].set_title('IoU')
            axes[1, 1].set_xticks(range(len(classes)))
            axes[1, 1].set_xticklabels(class_names, rotation=45)
            axes[1, 1].set_ylim(0, 1)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.visualization_dir, 'class_metrics.png'), dpi=150)
            plt.close()
    
    def run_complete_test(self, save_predictions: bool = True) -> str:
        """Run complete testing pipeline"""
        print(f"\nðŸš€ Starting Complete Model Test")
        print("=" * 50)
        
        # Run inference
        inference_results = self.run_inference(save_predictions=save_predictions)
        
        # Calculate metrics only if ground truth is available
        if self.has_ground_truth and len(self.ground_truths) > 0:
            print(f"\nðŸ“Š Ground truth available - calculating metrics...")
            metrics = self.calculate_metrics()
        else:
            print(f"\nâš ï¸  No ground truth available - skipping metrics calculation")
            metrics = {}
        
        # Generate report
        report_path = self.generate_report(metrics, inference_results)
        
        print(f"\nâœ… Complete test finished!")
        print(f"ðŸ“ Results saved to: {self.output_dir}")
        
        return report_path

    def _create_training_test_split_loader(self, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1):
        """Create a test loader using the EXACT same test split from training"""
        try:
            print(f"ðŸŽ¯ Creating test loader with EXACT training test split...")
            
            # Use the same function logic but integrated into the class
            test_loader, num_classes, class_names = extract_training_test_split(
                data_dir=self.test_data_dir,
                train_ratio=train_ratio,
                val_ratio=val_ratio, 
                test_ratio=test_ratio,
                batch_size=1
            )
            
            # Store dataset info
            self.num_classes = num_classes
            self.class_names = class_names
            
            print(f"âœ… Training test split loader created with {len(test_loader.dataset)} samples")
            
            return test_loader
            
        except Exception as e:
            print(f"âŒ Error creating training test split loader: {e}")
            raise

def select_files_gui() -> Tuple[Optional[str], Optional[str], Optional[str], bool, bool, bool, str]:
    """
    Interactive GUI for selecting model file and test data directory
    Returns: (model_path, test_data_dir, output_dir, save_predictions, has_ground_truth, use_training_split, inference_method)
    """
    # Hide main tkinter window
    root = tk.Tk()
    root.withdraw()
    
    print("🖱️ Please select files using the GUI dialogs...")
    
    # Select model file
    print("   Step 1: Select trained model file (.pth)")
    model_path = filedialog.askopenfilename(
        title="Select Trained Model File",
        filetypes=[
            ("PyTorch Models", "*.pth"),
            ("All Files", "*.*")
        ]
    )
    
    if not model_path:
        messagebox.showwarning("Selection Cancelled", "Model selection cancelled. Exiting.")
        return None, None, None, False, False, False, "patch"
    
    print(f"   ✅ Model selected: {os.path.basename(model_path)}")
    
    # Select test data directory
    print("   Step 2: Select test data directory")
    test_data_dir = filedialog.askdirectory(
        title="Select Test Data Directory"
    )
    
    if not test_data_dir:
        messagebox.showwarning("Selection Cancelled", "Test data directory selection cancelled. Exiting.")
        return None, None, None, False, False, False, "patch"
    
    print(f"   ✅ Test data directory selected: {os.path.basename(test_data_dir)}")
    
    # Ask about test split type - NEW OPTION!
    print("   Step 3: Test split type")
    use_training_split = messagebox.askyesno(
        "Test Split Type", 
        "Which test data do you want to use?\n\n"
        "• Select 'Yes' to use the EXACT same test split from training\n"
        "  (Uses deterministic splitting with same ratios to extract\n"
        "   the exact test data that was held out during training)\n\n"
        "• Select 'No' to use ALL available data for testing\n"
        "  (Uses all data in the directory for evaluation)\n\n"
        "Recommended: 'Yes' for proper evaluation on unseen test data"
    )
    
    if use_training_split:
        print(f"   ✅ Using EXACT training test split (recommended)")
        
        # Ask for training ratios if using training split
        ratios_msg = (
            "Please confirm the ratios used during training:\n\n"
            "Default ratios:\n"
            "• Training: 80%\n"
            "• Validation: 10%\n" 
            "• Test: 10%\n\n"
            "Are these the ratios you used during training?"
        )
        
        use_default_ratios = messagebox.askyesno("Training Ratios", ratios_msg)
        
        if not use_default_ratios:
            messagebox.showinfo("Custom Ratios", 
                              "You can modify the ratios in the code by calling:\n"
                              "extract_training_test_split(data_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)\n\n"
                              "For now, using default ratios (0.8, 0.1, 0.1)")
        
        print(f"   ✅ Using training ratios: 80% train, 10% val, 10% test")
    else:
        print(f"   ✅ Using ALL available data for testing")
    
    # Ask about ground truth availability
    print("   Step 4: Ground truth availability")
    has_ground_truth = messagebox.askyesno(
        "Ground Truth Availability", 
        "Does your test data include ground truth masks?\n\n"
        "• Select 'Yes' if you have masks for evaluation\n"
        "• Select 'No' for inference-only mode\n\n"
        "Note: Performance metrics will only be calculated if ground truth is available."
    )
    
    print(f"   ✅ Ground truth available: {'Yes' if has_ground_truth else 'No'}")
    
    # Ask about inference method for inference-only mode
    inference_method = "patch"  # Default
    if not has_ground_truth:
        print("   Step 5: Inference method selection")
        inference_method_choice = messagebox.askyesnocancel(
            "Inference Method", 
            "Choose inference method for processing images:\n\n"
            "🧩 PATCH-BASED (Recommended for high-resolution)\n"
            "• Divides large images into 256x256 patches\n"
            "• Processes each patch separately\n"
            "• Stitches results back to original size\n"
            "• Preserves fine details and original resolution\n"
            "• Best for ND2 files and large images\n\n"
            "📐 RESIZE METHOD\n"
            "• Resizes entire image to 256x256\n"
            "• Single inference pass\n"
            "• Faster but may lose detail\n"
            "• May distort aspect ratio\n\n"
            "Select 'Yes' for PATCH-BASED\n"
            "Select 'No' for RESIZE\n"
            "Select 'Cancel' to exit"
        )
        
        if inference_method_choice is None:  # User clicked Cancel
            return None, None, None, False, False, False, "patch"
        elif inference_method_choice:  # User clicked Yes
            inference_method = "patch"
            print(f"   ✅ Using PATCH-BASED inference (preserves original resolution)")
        else:  # User clicked No
            inference_method = "resize"
            print(f"   ✅ Using RESIZE inference (256x256 processing)")
    else:
        print(f"   ✅ Inference method: Not applicable (ground truth mode)")
    
    # Select output directory (optional)
    print("   Step 6: Select output directory (optional)")
    use_custom_output = messagebox.askyesno(
        "Output Directory", 
        "Do you want to select a custom output directory?\n\n"
        "Select 'No' to use default 'test_results' folder."
    )
    
    if use_custom_output:
        output_dir = filedialog.askdirectory(
            title="Select Output Directory"
        )
        if not output_dir:
            output_dir = "test_results"
            print(f"   ⚠️ No output directory selected, using default: {output_dir}")
        else:
            print(f"   ✅ Output directory selected: {os.path.basename(output_dir)}")
    else:
        output_dir = "test_results"
        print(f"   ✅ Using default output directory: {output_dir}")
    
    # Ask about saving predictions
    save_predictions = messagebox.askyesno(
        "Save Predictions", 
        "Do you want to save individual prediction images?\n\n"
        "This will create visualization images for each test sample."
    )
    
    print(f"   ✅ Save predictions: {'Yes' if save_predictions else 'No'}")
    
    root.destroy()
    
    return model_path, test_data_dir, output_dir, save_predictions, has_ground_truth, use_training_split, inference_method

def main():
    """Main testing function with interactive GUI"""
    print("🧠 Gut Tissue Segmentation - Model Testing System")
    print("=" * 60)
    
    # Interactive file selection
    selection_result = select_files_gui()
    
    if selection_result[0] is None:  # User cancelled
        print("❌ Error selection cancelled. Exiting.")
        return
    
    model_path, test_data_dir, output_dir, save_predictions, has_ground_truth, use_training_split, inference_method = selection_result
    
    # Validate selections
    print(f"\n🔍 Validating selections...")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        messagebox.showerror("File Error", f"Model file not found:\n{model_path}")
        return
    
    if not os.path.exists(test_data_dir):
        print(f"❌ Test data directory not found: {test_data_dir}")
        messagebox.showerror("Directory Error", f"Test data directory not found:\n{test_data_dir}")
        return
    
    # Check test data structure
    images_dir = os.path.join(test_data_dir, 'images')
    if not os.path.exists(images_dir):
        print(f"❌ Test images directory not found: {images_dir}")
        # images_dir=test_data_dir
        messagebox.showerror("Structure Error", 
                           f"Test data directory should contain an 'images' subdirectory.\n"
                           f"Expected: {images_dir}")
        return
    
    # Only check for masks if ground truth is expected
    if has_ground_truth:
        masks_dir = os.path.join(test_data_dir, 'masks')
        if not os.path.exists(masks_dir) or len(os.listdir(masks_dir)) == 0:
            print(f"❌ Masks directory not found or empty: {masks_dir}")
            messagebox.showerror("Ground Truth Error", 
                               f"Ground truth expected but masks directory not found or empty.\n"
                               f"Expected: {masks_dir}\n\n"
                               f"Either:\n"
                               f"• Add masks to the directory, or\n"
                               f"• Run again and select 'No' for ground truth availability")
            return
    
    print(f"✅ All selections validated!")
    print(f"   Model: {os.path.basename(model_path)}")
    print(f"   Test Data: {os.path.basename(test_data_dir)}")
    print(f"   Output: {output_dir}")
    print(f"   Save Predictions: {save_predictions}")
    print(f"   Ground Truth: {'Expected' if has_ground_truth else 'Not expected'}")
    print(f"   Test Split: {'Training test split' if use_training_split else 'All available data'}")
    if not has_ground_truth:
        print(f"   Inference Method: {inference_method.upper()}")
    
    try:
        # Create tester
        tester = ModelTester(model_path, test_data_dir, output_dir, has_ground_truth, use_training_split, inference_method)
        
        # Run complete test
        print(f"\n🚀 Starting model testing...")
        report_path = tester.run_complete_test(save_predictions=save_predictions)
        
        print(f"\n✅ Testing completed successfully!")
        print(f"📊 Test report: {report_path}")
        
        # Show completion message
        completion_msg = f"Model testing completed successfully!\n\nResults saved to:\n{output_dir}\n\nReport: {os.path.basename(report_path)}"
        
        if use_training_split:
            completion_msg += f"\n\n🎯 Used EXACT training test split for evaluation"
        else:
            completion_msg += f"\n\n📊 Used all available data for evaluation"
        
        if has_ground_truth:
            completion_msg += f"\n\n✅ Metrics calculated (ground truth available)"
        else:
            completion_msg += f"\n\n⚠️ Metrics skipped (no ground truth available)"
            completion_msg += f"\n\n🔧 Inference method: {inference_method.upper()}"
        
        messagebox.showinfo("Testing Complete", completion_msg)
        
    except Exception as e:
        error_msg = f"Error during testing: {str(e)}"
        print(f"❌ {error_msg}")
        messagebox.showerror("Testing Error", error_msg)

def extract_test_dataset(test_data_dir: str, batch_size: int = 1) -> Tuple[DataLoader, int, List[str]]:
    """
    Convenience function to extract test dataset directly
    
    Args:
        test_data_dir: Path to test data directory (should have images/ and masks/ subdirs)
        batch_size: Batch size for the DataLoader
    
    Returns:
        tuple: (test_loader, num_classes, class_names)
        
    Example:
        # Extract test dataset for evaluation
        test_loader, num_classes, class_names = extract_test_dataset("data/test")
        
        # Use in your own testing loop
        for batch_idx, (images, masks) in enumerate(test_loader):
            # Your testing code here
            pass
    """
    try:
        print(f"ðŸŽ¯ Extracting test dataset from: {test_data_dir}")
        
        # Create dataset with ALL data for testing
        test_dataset = SegmentationDataset(
            data_dir=test_data_dir, 
            split='test',     # No augmentation
            augment=False     # Explicitly disable augmentation
        )
        
        if len(test_dataset) == 0:
            raise ValueError(f"No valid test data found in {test_data_dir}")
        
        print(f"âœ… Test dataset: {len(test_dataset)} samples")
        print(f"   Augmentation: {test_dataset.augment} (should be False)")
        
        # Create test loader
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,  # Keep original order for testing
            num_workers=0,
            pin_memory=True
        )
        
        num_classes = test_dataset.get_num_classes()
        class_names = test_dataset.get_class_names()
        
        print(f"   Classes: {num_classes}")
        print(f"   Names: {class_names}")
        
        return test_loader, num_classes, class_names
        
    except Exception as e:
        print(f"âŒ Error extracting test dataset: {e}")
        raise

def extract_training_test_split(data_dir: str, train_ratio: float = 0.8, val_ratio: float = 0.1, 
                               test_ratio: float = 0.1, batch_size: int = 1) -> Tuple[DataLoader, int, List[str]]:
    """
    Extract the EXACT same test split that was used during training
    
    This function recreates the same deterministic split used during training to ensure
    you're evaluating on the exact test data that was held out during training.
    
    Args:
        data_dir: Path to the same data directory used for training
        train_ratio: Same ratio used during training (default: 0.8)
        val_ratio: Same ratio used during training (default: 0.1) 
        test_ratio: Same ratio used during training (default: 0.1)
        batch_size: Batch size for the test DataLoader
    
    Returns:
        tuple: (test_loader, num_classes, class_names)
        
    Example:
        # Extract the exact test split used during training
        test_loader, num_classes, class_names = extract_training_test_split("data/training")
        
        # Evaluate your model on the exact same test data
        model.eval()
        with torch.no_grad():
            for images, masks in test_loader:
                outputs = model(images)
                # Calculate metrics...
    """
    try:
        print(f"ðŸŽ¯ Extracting EXACT training test split from: {data_dir}")
        print(f"   Using training ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")
        
        # âœ… STEP 1: Validate and normalize ratios (same as training)
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-5:
            print(f"âš ï¸  Warning: Ratios sum to {total_ratio:.4f}, normalizing to 1.0")
            train_ratio /= total_ratio
            val_ratio /= total_ratio
            test_ratio /= total_ratio
            print(f"âœ… Normalized ratios: train={train_ratio:.3f}, val={val_ratio:.3f}, test={test_ratio:.3f}")
        
        # âœ… STEP 2: Create temp dataset to get file lists (same as training)
        print("ðŸ” Analyzing available data (same as training)...")
        temp_dataset = SegmentationDataset(data_dir=data_dir, split='temp', augment=False)
        
        if len(temp_dataset) == 0:
            raise ValueError(f"No valid datasets found in {data_dir}")
        
        # âœ… STEP 3: Calculate split sizes (same logic as training)
        total_size = len(temp_dataset)
        train_size = max(1, int(total_size * train_ratio))
        val_size = max(1, int(total_size * val_ratio))
        test_size = max(1, total_size - train_size - val_size)
        
        # Validate split sizes (same as training)
        if train_size + val_size + test_size > total_size:
            if total_size >= 3:
                train_size = max(1, total_size - 2)
                val_size = 1
                test_size = 1
            else:
                train_size = total_size
                val_size = 0
                test_size = 0
                print(f"âš ï¸  Warning: Very small dataset ({total_size} samples). Using all for training.")
        
        print(f"ðŸ“Š Dataset split sizes: {train_size} train, {val_size} val, {test_size} test")
        
        # âœ… STEP 4: Create EXACT same split using same random seed
        print("ðŸŽ² Recreating exact training split (seed=42)...")
        torch.manual_seed(42)  # SAME SEED AS TRAINING!
        indices = list(range(total_size))
        random.shuffle(indices)  # SAME SHUFFLE AS TRAINING!
        
        # Extract the SAME test indices that were used during training
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size+val_size]
        test_indices = indices[train_size+val_size:train_size+val_size+test_size]
        
        print(f"âœ… Recreated exact split:")
        print(f"   Train indices: {len(train_indices)} samples (indices {train_indices[0]}-{train_indices[-1]})")
        print(f"   Val indices: {len(val_indices)} samples")
        print(f"   Test indices: {len(test_indices)} samples (indices {test_indices[0] if test_indices else 'N/A'}-{test_indices[-1] if test_indices else 'N/A'})")
        
        if test_size == 0:
            raise ValueError("No test samples in this split configuration!")
        
        # âœ… STEP 5: Extract test files using exact same indices
        test_image_files = [temp_dataset.image_files[i] for i in test_indices]
        test_mask_files = [temp_dataset.mask_files[i] for i in test_indices]
        
        print(f"ðŸŽ¯ Extracted test files:")
        print(f"   Test images: {len(test_image_files)}")
        print(f"   Test masks: {len(test_mask_files)}")
        
        # Clean up temp dataset
        num_classes = temp_dataset.get_num_classes()
        class_names = temp_dataset.get_class_names()
        temp_dataset.cleanup_temp_dirs()
        del temp_dataset
        
        # âœ… STEP 6: Create test-only dataset with exact split data
        print("ðŸ”¨ Creating test dataset with exact training split data...")
        test_dataset = SegmentationDataset(
            data_dir=data_dir, 
            split='test',     # No augmentation
            augment=False     # Explicitly disable augmentation
        )
        
        # Override with the exact test split files
        test_dataset.image_files = test_image_files
        test_dataset.mask_files = test_mask_files
        
        print(f"âœ… Test dataset created with {len(test_dataset)} samples (EXACT training test split)")
        print(f"   Augmentation: {test_dataset.augment} (should be False)")
        
        # âœ… STEP 7: Create test loader
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,  # Keep original order for testing
            num_workers=0,
            pin_memory=True
        )
        
        print(f"   Classes: {num_classes}")
        print(f"   Names: {class_names}")
        
        # âœ… STEP 8: Verify this is the exact same split
        print(f"ðŸ” Verification:")
        print(f"   Total original data: {total_size} samples")
        print(f"   Training split used: {len(train_indices)} samples")
        print(f"   Test split extracted: {len(test_dataset)} samples")
        print(f"   Test percentage: {len(test_dataset)/total_size*100:.1f}%")
        
        return test_loader, num_classes, class_names
        
    except Exception as e:
        print(f"âŒ Error extracting training test split: {e}")
        raise

if __name__ == "__main__":
    main() 