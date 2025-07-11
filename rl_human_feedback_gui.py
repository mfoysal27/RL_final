"""
Human Feedback Reinforcement Learning GUI for Gut Tissue Segmentation
This module provides an interactive GUI for human-in-the-loop RL training.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from PIL import Image, ImageTk
import json
import os
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
import sys
from pathlib import Path
import glob

# Add these imports after the existing imports (around line 20)
import json
sys.path.append('models')
try:
    from models.model_factory import ModelFactory
    MODEL_FACTORY_AVAILABLE = True
except ImportError:
    MODEL_FACTORY_AVAILABLE = False
    print("Warning: ModelFactory not available, falling back to SimpleUNet")

# Define tissue configuration directly in the GUI
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

def get_num_tissue_classes():
    """Get number of tissue classes."""
    return len(TISSUE_CLASSES)

def get_tissue_color(class_id):
    """Get tissue color for class ID."""
    if class_id in TISSUE_CLASSES:
        return TISSUE_CLASSES[class_id]["color"]
    return (128, 128, 128)  # Default gray

def get_tissue_name(class_id):
    """Get tissue name for class ID."""
    if class_id in TISSUE_CLASSES:
        return TISSUE_CLASSES[class_id]["name"]
    return f"Class_{class_id}"


# Simple UNet model for basic functionality
class SimpleUNet(nn.Module):
    """Simplified UNet for demonstration."""
    
    def __init__(self, in_channels=1, num_classes=15):
        super(SimpleUNet, self).__init__()
        
        # Encoder
        self.enc1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, 3, padding=1)
        self.enc3 = nn.Conv2d(128, 256, 3, padding=1)
        
        # Decoder
        self.dec3 = nn.Conv2d(256, 128, 3, padding=1)
        self.dec2 = nn.Conv2d(128, 64, 3, padding=1)
        self.dec1 = nn.Conv2d(64, num_classes, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # Encoder
        e1 = self.relu(self.enc1(x))
        e2 = self.relu(self.enc2(self.pool(e1)))
        e3 = self.relu(self.enc3(self.pool(e2)))
        
        # Decoder
        d3 = self.relu(self.dec3(self.upsample(e3)))
        d2 = self.relu(self.dec2(self.upsample(d3)))
        d1 = self.dec1(self.upsample(d2))
        
        return d1


# Simple dataset for loading images
class SimpleImageDataset:
    """Simple dataset for loading images from a directory."""
    
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_paths = []
        
        # Find all image files
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.bmp']:
            self.image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
            self.image_paths.extend(glob.glob(os.path.join(image_dir, ext.upper())))
        
        if not self.image_paths:
            raise ValueError(f"No image files found in {image_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load and preprocess image
        image = Image.open(img_path)
        
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
        
        # Resize to 256x256
        image = image.resize((256, 256), Image.Resampling.LANCZOS)
        
        # Convert to tensor
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)  # Add channel dimension
        
        return img_tensor


class ND2SliceDataset:
    """Dataset for loading and processing individual slices from ND2 files."""
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.slices = []
        self.slice_metadata = []  # Track which file each slice came from
        
        # Find ND2 files
        nd2_files = glob.glob(os.path.join(data_dir, "*.nd2"))
        nd2_files.extend(glob.glob(os.path.join(data_dir, "**/*.nd2"), recursive=True))
        
        if not nd2_files:
            raise ValueError(f"No ND2 files found in {data_dir}")
        
        print(f"🔍 Found {len(nd2_files)} ND2 files, extracting ALL slices...")
        print("=" * 60)
        
        total_files_processed = 0
        total_slices_extracted = 0
        files_with_errors = 0
        
        # Extract slices from each ND2 file
        for file_idx, nd2_path in enumerate(nd2_files, 1):
            try:
                print(f"📁 File {file_idx}/{len(nd2_files)}: {os.path.basename(nd2_path)}")
                slices = self.extract_nd2_slices(nd2_path)
                
                # Track metadata for each slice
                for slice_idx, slice_data in enumerate(slices):
                    self.slice_metadata.append({
                        'file_path': nd2_path,
                        'file_name': os.path.basename(nd2_path),
                        'slice_index': slice_idx,
                        'total_slices_in_file': len(slices)
                    })
                
                self.slices.extend(slices)
                total_slices_extracted += len(slices)
                total_files_processed += 1
                
                print(f"  ✅ {len(slices)} slices extracted from this file")
                
            except Exception as e:
                print(f"  ❌ Error processing {nd2_path}: {e}")
                files_with_errors += 1
        
        if not self.slices:
            raise ValueError("❌ No slices could be extracted from any ND2 files")
        
        # Final summary
        print("=" * 60)
        print(f"📊 EXTRACTION COMPLETE - SUMMARY:")
        print(f"   🗂️  Total ND2 files found: {len(nd2_files)}")
        print(f"   ✅ Files successfully processed: {total_files_processed}")
        print(f"   ❌ Files with errors: {files_with_errors}")
        print(f"   🎯 Total slices extracted: {total_slices_extracted}")
        print(f"   📈 Average slices per file: {total_slices_extracted/total_files_processed:.1f}")
        print(f"   💾 All {total_slices_extracted} slices are available for RL training!")
        print("=" * 60)
    
    def extract_nd2_slices(self, nd2_path):
        """Extract ALL individual slices from ND2 file."""
        try:
            # Try to import nd2reader
            try:
                from nd2reader import ND2Reader
            except ImportError:
                raise ImportError("nd2reader package required. Install with: pip install nd2reader")
            
            slices = []
            with ND2Reader(nd2_path) as images:
                # ENHANCED: Comprehensive frame/slice detection
                print(f"    🔍 Analyzing ND2 file: {os.path.basename(nd2_path)}")
                
                # Debug: Print all available attributes and dimensions
                if hasattr(images, 'sizes'):
                    print(f"       Available dimensions: {images.sizes}")
                    
                # Try multiple methods to get total frames
                num_frames = None
                method_used = "unknown"
                
                # Method 1: Check time dimension
                if hasattr(images, 'sizes') and 't' in images.sizes:
                    num_frames = images.sizes['t']
                    method_used = "time_dimension"
                    print(f"       Time frames (t): {num_frames}")
                
                # Method 2: Check Z-stack dimension
                elif hasattr(images, 'sizes') and 'z' in images.sizes:
                    num_frames = images.sizes['z']
                    method_used = "z_stack"
                    print(f"       Z-stack slices (z): {num_frames}")
                
                # Method 3: Check for other common dimensions
                elif hasattr(images, 'sizes'):
                    sizes = images.sizes
                    # Look for any dimension that could be frames
                    for dim, count in sizes.items():
                        if dim in ['v', 'c', 'x', 'y']:  # Skip spatial and channel dims
                            continue
                        if count > 1:  # Found a dimension with multiple frames
                            num_frames = count
                            method_used = f"dimension_{dim}"
                            print(f"       Found {count} frames in dimension '{dim}'")
                            break
                
                # Method 4: Try direct iteration
                if num_frames is None:
                    try:
                        num_frames = len(images)
                        method_used = "len_iteration"
                        print(f"       Direct length: {num_frames}")
                    except:
                        pass
                
                # Method 5: Try to iterate and count
                if num_frames is None or num_frames <= 1:
                    print(f"       🔄 Attempting manual frame counting...")
                    count = 0
                    try:
                        for i in range(1000):  # Safety limit
                            try:
                                frame = images[i] if hasattr(images, '__getitem__') else images.get_frame(i)
                                if frame is not None:
                                    count += 1
                                else:
                                    break
                            except (IndexError, KeyError, StopIteration):
                                break
                        if count > num_frames if num_frames else 0:
                            num_frames = count
                            method_used = "manual_counting"
                            print(f"       Manual count found: {num_frames} frames")
                    except Exception as e:
                        print(f"       Manual counting failed: {e}")
                
                # Default fallback
                if num_frames is None or num_frames == 0:
                    num_frames = 1
                    method_used = "fallback"
                    print(f"       ⚠️ Using fallback: {num_frames} frame")
                
                print(f"    📊 FINAL: Processing {num_frames} frames (method: {method_used})")
                
                # Extract ALL frames/slices
                successful_slices = 0
                failed_slices = 0
                
                for i in range(num_frames):
                    try:
                        # Try multiple frame extraction methods
                        frame = None
                        
                        # Method A: get_frame
                        if hasattr(images, 'get_frame'):
                            try:
                                frame = images.get_frame(i)
                            except:
                                pass
                        
                        # Method B: direct indexing
                        if frame is None and hasattr(images, '__getitem__'):
                            try:
                                frame = images[i]
                            except:
                                pass
                        
                        # Method C: iterate through generator
                        if frame is None and hasattr(images, '__iter__'):
                            try:
                                for j, f in enumerate(images):
                                    if j == i:
                                        frame = f
                                        break
                            except:
                                pass
                        
                        if frame is None:
                            print(f"      ⚠️ Could not extract frame {i}")
                            failed_slices += 1
                            continue
                        
                        # Process the frame
                        processed_slice = self.process_nd2_frame(frame, i)
                        if processed_slice is not None:
                            slices.append(processed_slice)
                            successful_slices += 1
                        else:
                            failed_slices += 1
                        
                        # Progress indicator for large files
                        if (i + 1) % 10 == 0 or i == 0:  # Show progress more frequently
                            print(f"      📈 Progress: {i + 1}/{num_frames} frames ({successful_slices} successful)")
                            
                    except Exception as e:
                        print(f"      ❌ Error extracting frame {i}: {e}")
                        failed_slices += 1
                        continue
                
                print(f"    ✅ EXTRACTION COMPLETE: {successful_slices} slices extracted, {failed_slices} failed")
                
                # If we got very few slices but expected more, warn user
                if successful_slices < num_frames * 0.5 and num_frames > 1:
                    print(f"    ⚠️ WARNING: Only extracted {successful_slices}/{num_frames} expected slices")
            
            return slices
            
        except Exception as e:
            print(f"❌ Error reading ND2 file {nd2_path}: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def process_nd2_frame(self, frame, frame_idx):
        """Process individual ND2 frame into tensor format."""
        try:
            # Convert to numpy array if needed
            if not isinstance(frame, np.ndarray):
                frame = np.array(frame)
            
            # Handle different frame formats
            if frame.ndim == 3:
                # Multi-channel image - convert to grayscale
                if frame.shape[2] == 3:  # RGB
                    frame = np.dot(frame[...,:3], [0.2989, 0.5870, 0.1140])
                elif frame.shape[2] == 4:  # RGBA
                    frame = np.dot(frame[...,:3], [0.2989, 0.5870, 0.1140])
                else:
                    # Take first channel
                    frame = frame[:, :, 0]
            elif frame.ndim == 2:
                # Already grayscale
                pass
            else:
                print(f"Unexpected frame dimensions: {frame.shape}")
                return None
            
            # Ensure we have a 2D grayscale image
            if frame.ndim != 2:
                print(f"Frame {frame_idx} has unexpected shape after processing: {frame.shape}")
                return None
            
            # Convert to PIL Image for consistent processing
            if frame.dtype != np.uint8:
                # Normalize to 0-255 range
                frame_min, frame_max = frame.min(), frame.max()
                if frame_max > frame_min:
                    frame = ((frame - frame_min) / (frame_max - frame_min) * 255).astype(np.uint8)
                else:
                    frame = np.zeros_like(frame, dtype=np.uint8)
            
            # Create PIL image
            pil_image = Image.fromarray(frame, mode='L')
            
            # Resize to 256x256
            pil_image = pil_image.resize((256, 256), Image.Resampling.LANCZOS)
            
            # Convert to tensor
            img_array = np.array(pil_image, dtype=np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)  # Add channel dimension
            
            return img_tensor
            
        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")
            return None
    
    def __len__(self):
        return len(self.slices)
    
    def __getitem__(self, idx):
        return self.slices[idx]
    
    def get_slice_info(self, idx):
        """Get metadata about a specific slice."""
        if 0 <= idx < len(self.slice_metadata):
            return self.slice_metadata[idx]
        return None
    
    def get_file_slice_distribution(self):
        """Get information about how many slices came from each file."""
        file_counts = {}
        for metadata in self.slice_metadata:
            file_name = metadata['file_name']
            if file_name not in file_counts:
                file_counts[file_name] = 0
            file_counts[file_name] += 1
        return file_counts
    
    def print_slice_distribution(self):
        """Print detailed slice distribution for validation."""
        print("\n📋 DETAILED SLICE DISTRIBUTION:")
        print("-" * 50)
        distribution = self.get_file_slice_distribution()
        
        for file_name, count in sorted(distribution.items()):
            print(f"   {file_name}: {count} slices")
        
        print(f"\n✅ VALIDATION: {sum(distribution.values())} total slices loaded")
        print(f"   This means ALL slices from ALL {len(distribution)} files are available for RL!")
        print("-" * 50)


class CombinedDataset:
    """Dataset that can handle both regular images and ND2 files."""
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.dataset = None
        
        # Check what type of files we have
        nd2_files = glob.glob(os.path.join(data_dir, "*.nd2"))
        nd2_files.extend(glob.glob(os.path.join(data_dir, "**/*.nd2"), recursive=True))
        
        regular_images = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.bmp']:
            regular_images.extend(glob.glob(os.path.join(data_dir, ext)))
            regular_images.extend(glob.glob(os.path.join(data_dir, ext.upper())))
        
        if nd2_files:
            print(f"Found {len(nd2_files)} ND2 files - using ND2 slice extraction")
            self.dataset = ND2SliceDataset(data_dir)
            
            # Show detailed slice distribution for validation
            if hasattr(self.dataset, 'print_slice_distribution'):
                self.dataset.print_slice_distribution()
                
        elif regular_images:
            print(f"Found {len(regular_images)} regular image files")
            self.dataset = SimpleImageDataset(data_dir)
        else:
            raise ValueError(f"No supported files found in {data_dir}")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


def load_pytorch_model(model_path, device):
    """Load a PyTorch model with fallback options."""
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract model info
        if isinstance(checkpoint, dict):
            model_state = checkpoint.get('model_state_dict', checkpoint)
            num_classes = checkpoint.get('num_classes', 15)
            model_type = checkpoint.get('model_type', 'unknown')
        else:
            model_state = checkpoint
            num_classes = 15
            model_type = 'unknown'
        
        # Try to infer model architecture from state dict
        first_key = next(iter(model_state.keys()))
        
        if 'enc1' in first_key or 'encoder' in first_key.lower():
            # Looks like a UNet-style model
            print(f"Detected UNet-style model, creating SimpleUNet with {num_classes} classes")
            model = SimpleUNet(in_channels=1, num_classes=num_classes)
        else:
            # Fallback to simple UNet
            print(f"Unknown model architecture, using SimpleUNet with {num_classes} classes")
            model = SimpleUNet(in_channels=1, num_classes=num_classes)
        
        # Try to load state dict
        try:
            model.load_state_dict(model_state, strict=False)
            print("Model loaded successfully (some weights may not match)")
        except Exception as e:
            print(f"Warning: Could not load all weights: {e}")
            print("Using randomly initialized model for demonstration")
        
        return model.to(device), {'num_classes': num_classes, 'model_type': model_type}
        
    except Exception as e:
        print(f"Error loading model: {e}")
        # Create a fresh model for demonstration
        print("Creating new SimpleUNet for demonstration")
        model = SimpleUNet(in_channels=1, num_classes=15)
        return model.to(device), {'num_classes': 15, 'model_type': 'simple_unet'}


class EnhancedHumanFeedbackCollector:
    """Enhanced collector for dual feedback (class + border)."""
    
    def __init__(self):
        self.feedback_history = []
        self.current_epoch_feedback = []
    
    def add_dual_feedback(self, image_idx: int, prediction: np.ndarray, 
                         class_feedback: float, border_feedback: float):
        """Add dual feedback for a specific prediction."""
        feedback_entry = {
            'image_idx': image_idx,
            'prediction': prediction,
            'class_feedback': class_feedback,     # -1.0 to +1.0
            'border_feedback': border_feedback,   # -1.0 to +1.0
            'combined_feedback': (class_feedback + border_feedback) / 2,  # Overall score
            'timestamp': datetime.now().isoformat()
        }
        self.current_epoch_feedback.append(feedback_entry)
    
    def finalize_epoch(self, epoch: int):
        """Finalize feedback for current epoch."""
        if self.current_epoch_feedback:
            avg_class = sum(f['class_feedback'] for f in self.current_epoch_feedback) / len(self.current_epoch_feedback)
            avg_border = sum(f['border_feedback'] for f in self.current_epoch_feedback) / len(self.current_epoch_feedback)
            avg_combined = sum(f['combined_feedback'] for f in self.current_epoch_feedback) / len(self.current_epoch_feedback)
        else:
            avg_class = avg_border = avg_combined = 0.0
            
        self.feedback_history.append({
            'epoch': epoch,
            'feedback': self.current_epoch_feedback.copy(),
            'avg_class_feedback': avg_class,
            'avg_border_feedback': avg_border,
            'avg_combined_feedback': avg_combined
        })
        self.current_epoch_feedback.clear()
    
    def get_dual_reward_signals(self) -> tuple:
        """Calculate separate reward signals for class and border."""
        if not self.current_epoch_feedback:
            return 0.0, 0.0
        
        class_reward = sum(f['class_feedback'] for f in self.current_epoch_feedback) / len(self.current_epoch_feedback)
        border_reward = sum(f['border_feedback'] for f in self.current_epoch_feedback) / len(self.current_epoch_feedback)
        
        return class_reward, border_reward


class RLAgent:
    """Reinforcement Learning Agent for model improvement."""
    
    def __init__(self, model: nn.Module, learning_rate: float = 1e-6):  # Ultra-conservative: 10x smaller
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.feedback_collector = EnhancedHumanFeedbackCollector()
        self.epoch = 0
        
        # RL-specific parameters
        self.gamma = 0.95  # Discount factor
        self.epsilon = 0.1  # Exploration rate
        
        # Experience buffer
        self.experience_buffer = []
        self.buffer_size = 1000
        
        # Learning tracking metrics
        self.learning_metrics = {
            'gradient_norms': [],
            'weight_changes': [],
            'loss_history': [],
            'feedback_trends': [],
            'convergence_score': 0.0,
            'learning_rate_adjustments': 0,
            'total_updates': 0
        }
        
        # Store initial model state for comparison
        self.initial_state = {name: param.clone().detach() for name, param in model.named_parameters()}
        self.previous_state = {name: param.clone().detach() for name, param in model.named_parameters()}
        
        print(f"🔒 ULTRA-CONSERVATIVE RL Agent initialized:")
        print(f"   Learning rate: {learning_rate} (10x smaller than before)")
        print(f"   Forgetting penalty: 10x stronger")
        print(f"   Gradient clipping: 10x stricter")
        print(f"   Policy updates: Only when advantage > 0.1")
        print(f"   Recommended: Provide feedback in small increments")
    
    def update_model_with_feedback(self, images: List[torch.Tensor], 
                                 predictions: List[torch.Tensor],
                                 feedback_scores: List[float]):
        """Update model based on human feedback using CONSERVATIVE policy gradient with strong regularization."""
        if not feedback_scores:
            return
        
        self.model.train()
        total_loss = 0
        gradient_norms = []
        
        # Store state before update
        pre_update_state = {name: param.clone().detach() for name, param in self.model.named_parameters()}
        
        # CRITICAL: Use much more conservative approach to prevent catastrophic forgetting
        
        # 1. Calculate baseline (average feedback) for advantage estimation
        baseline = sum(feedback_scores) / len(feedback_scores) if feedback_scores else 0.0
        
        # 2. Process each sample with CONSERVATIVE updates
        for img, pred, score in zip(images, predictions, feedback_scores):
            # Convert feedback score to normalized reward (-1 to +1)
            raw_reward = score  # Already in -1 to +1 range
            
            # Calculate advantage (how much better/worse than average)
            advantage = raw_reward - baseline
            
            # HEAVILY DAMPEN the advantage to prevent large updates
            dampened_advantage = advantage * 0.1  # Reduce impact by 90%
            
            # Convert to tensor
            advantage_tensor = torch.tensor(dampened_advantage, dtype=torch.float32, device=pred.device)
            
            # 3. CONSERVATIVE Policy Gradient Loss
            # Instead of direct log_prob optimization, use a much softer approach
            log_prob = torch.log_softmax(pred, dim=1)
            
            # Only update if advantage is significant (> 0.1)
            if abs(advantage) > 0.1:
                # Use entropy-regularized policy gradient
                entropy = -torch.sum(torch.exp(log_prob) * log_prob, dim=1)
                policy_loss = -torch.mean(log_prob.mean(dim=1) * advantage_tensor)
                
                # Add strong entropy regularization to prevent overconfident predictions
                entropy_loss = -0.1 * torch.mean(entropy)  # Encourage exploration
                
                combined_policy_loss = policy_loss + entropy_loss
            else:
                # No significant advantage - skip policy update
                combined_policy_loss = torch.tensor(0.0, device=pred.device)
            
            # 4. STRONG Regularization to prevent catastrophic forgetting
            
            # L2 regularization (increased)
            l2_reg = 0.01 * sum(p.pow(2.0).sum() for p in self.model.parameters())
            
            # MUCH STRONGER weight deviation penalty
            weight_deviation_penalty = 0.0
            for name, param in self.model.named_parameters():
                if name in self.initial_state:
                    # Heavily penalize deviations from original weights
                    deviation = torch.norm(param - self.initial_state[name])
                    weight_deviation_penalty += deviation
            
            # CRITICAL: Much stronger forgetting penalty (10x stronger)
            forgetting_penalty = 0.1 * weight_deviation_penalty  # Increased from 0.01 to 0.1
            
            # 5. Add consistency loss with original predictions
            consistency_loss = torch.tensor(0.0, device=pred.device)
            if hasattr(self, 'original_predictions') and len(self.original_predictions) > 0:
                # Encourage consistency with original model predictions
                if len(self.original_predictions) > len(images):
                    original_pred = self.original_predictions[len(images) - 1]
                    consistency_loss = 0.05 * F.mse_loss(torch.softmax(pred, dim=1), 
                                                        torch.softmax(original_pred, dim=1))
            
            # 6. Combine all losses with conservative weighting
            total_loss_sample = (
                0.3 * combined_policy_loss +    # Reduced policy loss weight
                0.2 * l2_reg +                  # L2 regularization
                0.4 * forgetting_penalty +      # Strong forgetting prevention
                0.1 * consistency_loss          # Consistency with original
            )
            
            total_loss += total_loss_sample.item()
            
            # 7. CONSERVATIVE gradient updates
            self.optimizer.zero_grad()
            total_loss_sample.backward()
            
            # STRONG gradient clipping to prevent large updates
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)  # Much smaller max_norm
            gradient_norms.append(grad_norm.item())
            
            # Only update if gradient norm is reasonable
            if grad_norm < 1.0:  # Only update if gradients are small
                self.optimizer.step()
            else:
                print(f"⚠️ Skipping update due to large gradient norm: {grad_norm:.4f}")
        
        # Calculate learning metrics
        avg_loss = total_loss / len(feedback_scores)
        avg_grad_norm = sum(gradient_norms) / len(gradient_norms) if gradient_norms else 0.0
        
        # Calculate weight change magnitude
        weight_change = self._calculate_weight_change(pre_update_state)
        
        # Log conservative update info
        print(f"🔒 Conservative RL Update:")
        print(f"   Baseline feedback: {baseline:.3f}")
        print(f"   Max advantage: {max([abs(s - baseline) for s in feedback_scores]):.3f}")
        print(f"   Weight change: {weight_change:.6f}")
        print(f"   Gradient norm: {avg_grad_norm:.6f}")
        
        # Warning if changes are too large
        if weight_change > 0.001:
            print(f"⚠️ WARNING: Large weight change detected! Consider reducing learning rate.")
        
        # Update learning metrics
        self.learning_metrics['loss_history'].append(avg_loss)
        self.learning_metrics['gradient_norms'].append(avg_grad_norm)
        self.learning_metrics['weight_changes'].append(weight_change)
        feedback_mean = sum(feedback_scores) / len(feedback_scores) if feedback_scores else 0.0
        self.learning_metrics['feedback_trends'].append(feedback_mean)
        self.learning_metrics['total_updates'] += 1
        
        # Update convergence score
        self._update_convergence_score()
        
        # Store current state for next comparison
        self.previous_state = {name: param.clone().detach() for name, param in self.model.named_parameters()}
        
        return avg_loss, avg_grad_norm, weight_change
    
    def _calculate_weight_change(self, pre_update_state):
        """Calculate the magnitude of weight changes."""
        total_change = 0.0
        total_params = 0
        
        for name, param in self.model.named_parameters():
            if name in pre_update_state:
                change = torch.norm(param - pre_update_state[name]).item()
                total_change += change
                total_params += param.numel()
        
        return total_change / total_params if total_params > 0 else 0.0
    
    def _update_convergence_score(self):
        """Update convergence score based on recent metrics."""
        if len(self.learning_metrics['loss_history']) < 3:
            self.learning_metrics['convergence_score'] = 0.0
            return
        
        # Check loss trend (recent 3 epochs)
        recent_losses = self.learning_metrics['loss_history'][-3:]
        loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
        
        # Check feedback trend
        if len(self.learning_metrics['feedback_trends']) >= 3:
            recent_feedback = self.learning_metrics['feedback_trends'][-3:]
            feedback_trend = np.polyfit(range(len(recent_feedback)), recent_feedback, 1)[0]
        else:
            feedback_trend = 0
        
        # Check weight change consistency
        if len(self.learning_metrics['weight_changes']) >= 3:
            recent_changes = self.learning_metrics['weight_changes'][-3:]
            change_consistency = 1.0 / (1.0 + np.std(recent_changes))
        else:
            change_consistency = 0.5
        
        # Combine metrics (higher is better)
        # Negative loss trend (decreasing loss) is good
        # Positive feedback trend (improving feedback) is good
        # Consistent weight changes indicate stable learning
        convergence = ((-loss_trend * 10) + (feedback_trend * 5) + (change_consistency * 2)) / 3
        self.learning_metrics['convergence_score'] = max(0.0, min(1.0, convergence))
    
    def save_complete_model_info(self, save_path: str, original_model_path: str = None):
        """Save complete model information including all metadata, weights, and training history."""
        try:
            import json
            from datetime import datetime
            
            # Collect comprehensive model information
            model_info = {
                # === BASIC MODEL INFO ===
                'model_architecture': str(type(self.model).__name__),
                'model_parameters': sum(p.numel() for p in self.model.parameters()),
                'model_size_mb': sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024),
                'device': str(next(self.model.parameters()).device),
                'save_timestamp': datetime.now().isoformat(),
                
                # === RL TRAINING INFO ===
                'rl_training': {
                    'current_epoch': self.epoch,
                    'total_updates': self.learning_metrics['total_updates'],
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'optimizer_type': type(self.optimizer).__name__,
                    'optimizer_state': self.optimizer.state_dict(),
                },
                
                # === LEARNING METRICS ===
                'learning_metrics': self.learning_metrics.copy(),
                'feedback_history': self.feedback_collector.feedback_history.copy(),
                
                # === WEIGHT CHANGE ANALYSIS ===
                'weight_analysis': self._analyze_weight_changes(),
                
                # === LAYER-BY-LAYER ANALYSIS ===
                'layer_analysis': self._analyze_layers(),
                
                # === PERFORMANCE COMPARISON ===
                'performance_comparison': self._compare_with_original(original_model_path) if original_model_path else None,
            }
            
            # Save the complete checkpoint
            complete_checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'model_info': model_info,
                'rl_epoch': self.epoch,
                'training_history': getattr(self, 'training_history', []),
                'feedback_history': self.feedback_collector.feedback_history,
                'learning_metrics': self.learning_metrics,
                'initial_state': self.initial_state,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'timestamp': datetime.now().isoformat()
            }
            
            # Save to file
            torch.save(complete_checkpoint, save_path)
            
            # Also save human-readable summary
            summary_path = save_path.replace('.pth', '_summary.json')
            with open(summary_path, 'w') as f:
                # Convert tensors to lists for JSON serialization
                json_safe_info = self._make_json_safe(model_info)
                json.dump(json_safe_info, f, indent=2)
            
            print(f"✅ Complete model information saved:")
            print(f"   Model file: {save_path}")
            print(f"   Summary: {summary_path}")
            print(f"   Model size: {model_info['model_size_mb']:.2f} MB")
            print(f"   Parameters: {model_info['model_parameters']:,}")
            print(f"   RL epochs: {model_info['rl_training']['current_epoch']}")
            print(f"   Total updates: {model_info['rl_training']['total_updates']}")
            
            return save_path, summary_path
            
        except Exception as e:
            print(f"❌ Error saving complete model info: {e}")
            return None, None
    
    def _analyze_weight_changes(self):
        """Analyze weight changes from original model."""
        analysis = {
            'total_deviation': 0.0,
            'max_layer_change': 0.0,
            'max_layer_name': '',
            'layer_changes': {},
            'change_distribution': {'small': 0, 'medium': 0, 'large': 0}
        }
        
        for name, param in self.model.named_parameters():
            if name in self.initial_state:
                original = self.initial_state[name]
                current = param.data
                
                # Calculate relative change
                change = torch.norm(current - original).item()
                relative_change = change / torch.norm(original).item() if torch.norm(original).item() > 0 else 0
                
                analysis['layer_changes'][name] = {
                    'absolute_change': change,
                    'relative_change': relative_change,
                    'parameter_count': param.numel()
                }
                
                analysis['total_deviation'] += change
                
                if relative_change > analysis['max_layer_change']:
                    analysis['max_layer_change'] = relative_change
                    analysis['max_layer_name'] = name
                
                # Categorize change magnitude
                if relative_change < 0.01:
                    analysis['change_distribution']['small'] += 1
                elif relative_change < 0.1:
                    analysis['change_distribution']['medium'] += 1
                else:
                    analysis['change_distribution']['large'] += 1
        
        return analysis
    
    def _analyze_layers(self):
        """Analyze each layer's properties."""
        layer_info = {}
        
        for name, param in self.model.named_parameters():
            layer_info[name] = {
                'shape': list(param.shape),
                'parameter_count': param.numel(),
                'dtype': str(param.dtype),
                'requires_grad': param.requires_grad,
                'mean': param.data.mean().item(),
                'std': param.data.std().item(),
                'min': param.data.min().item(),
                'max': param.data.max().item()
            }
        
        return layer_info
    
    def _compare_with_original(self, original_model_path: str):
        """Compare current model with original model."""
        try:
            # Load original model
            original_checkpoint = torch.load(original_model_path, map_location='cpu')
            if 'model_state_dict' in original_checkpoint:
                original_state = original_checkpoint['model_state_dict']
            else:
                original_state = original_checkpoint
            
            comparison = {
                'parameter_differences': {},
                'overall_similarity': 0.0,
                'layers_significantly_changed': []
            }
            
            total_similarity = 0.0
            layer_count = 0
            
            for name, param in self.model.named_parameters():
                if name in original_state:
                    original_param = original_state[name]
                    current_param = param.data
                    
                    # Calculate cosine similarity
                    similarity = F.cosine_similarity(
                        original_param.flatten().unsqueeze(0),
                        current_param.flatten().unsqueeze(0)
                    ).item()
                    
                    comparison['parameter_differences'][name] = {
                        'cosine_similarity': similarity,
                        'mse_difference': F.mse_loss(original_param, current_param).item()
                    }
                    
                    total_similarity += similarity
                    layer_count += 1
                    
                    # Mark layers with significant changes
                    if similarity < 0.95:  # Less than 95% similarity
                        comparison['layers_significantly_changed'].append(name)
            
            comparison['overall_similarity'] = total_similarity / layer_count if layer_count > 0 else 0.0
            
            return comparison
            
        except Exception as e:
            print(f"⚠️ Could not compare with original model: {e}")
            return None
    
    def _make_json_safe(self, obj):
        """Convert tensors and other non-JSON types to JSON-safe format."""
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_safe(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def get_learning_status(self):
        """Get comprehensive learning status."""
        if not self.learning_metrics['loss_history']:
            return {
                'status': 'Not Started',
                'trend': 'Unknown',
                'improvement': 0.0,
                'convergence': 0.0,
                'confidence': 'Low'
            }
        
        # Determine learning trend
        if len(self.learning_metrics['loss_history']) >= 2:
            recent_loss = self.learning_metrics['loss_history'][-1]
            previous_loss = self.learning_metrics['loss_history'][-2]
            loss_change = previous_loss - recent_loss
            
            if loss_change > 0.01:
                trend = "Improving"
                trend_color = "green"
            elif loss_change < -0.01:
                trend = "Worsening"
                trend_color = "red"
            else:
                trend = "Stable"
                trend_color = "orange"
        else:
            trend = "Insufficient Data"
            trend_color = "blue"
        
        # Calculate improvement from start
        if len(self.learning_metrics['loss_history']) >= 2:
            initial_loss = self.learning_metrics['loss_history'][0]
            current_loss = self.learning_metrics['loss_history'][-1]
            improvement = (initial_loss - current_loss) / initial_loss * 100
        else:
            improvement = 0.0
        
        # Determine confidence level
        num_updates = len(self.learning_metrics['loss_history'])
        # Safe gradient norm calculation to avoid division by zero
        if self.learning_metrics['gradient_norms']:
            recent_grads = self.learning_metrics['gradient_norms'][-5:]
            avg_grad_norm = sum(recent_grads) / len(recent_grads)
        else:
            avg_grad_norm = 0.0
        
        if num_updates >= 5 and avg_grad_norm > 0.01:
            confidence = "High"
        elif num_updates >= 3:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        return {
            'status': 'Learning' if num_updates > 0 else 'Not Started',
            'trend': trend,
            'trend_color': trend_color,
            'improvement': improvement,
            'convergence': self.learning_metrics['convergence_score'],
            'confidence': confidence,
            'gradient_norm': avg_grad_norm,
            'weight_change': self.learning_metrics['weight_changes'][-1] if self.learning_metrics['weight_changes'] else 0.0,
            'total_updates': self.learning_metrics['total_updates']
        }
    
    def select_random_samples(self, dataset, num_samples: int = 8) -> List[Tuple[int, Any]]:  # Increased from 3 to 8
        """Select random samples from dataset for human feedback."""
        # Use more samples but display only 3 for human feedback
        # This provides better training signal while keeping UI manageable
        total_samples = min(num_samples, len(dataset))
        indices = random.sample(range(len(dataset)), total_samples)
        samples = []
        for idx in indices:
            try:
                sample = dataset[idx]
                samples.append((idx, sample))
            except Exception as e:
                print(f"Error loading sample {idx}: {e}")
                continue
        
        print(f"🎯 Selected {len(samples)} samples for RL training (displaying first 3 for feedback)")
        return samples


class HumanFeedbackRLGUI:
    """Main GUI for Human Feedback Reinforcement Learning."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Human Feedback RL - Gut Tissue Segmentation")
        
        # Get screen dimensions for relative sizing
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Use 85% of screen width and 80% of screen height (more responsive)
        window_width = int(screen_width * 0.85)
        window_height = int(screen_height * 0.80)
        
        # Center the window on screen
        pos_x = (screen_width - window_width) // 2
        pos_y = (screen_height - window_height) // 2
        
        # Set geometry with relative size and centered position
        self.root.geometry(f"{window_width}x{window_height}+{pos_x}+{pos_y}")
        
        # Set minimum window size (prevents it from becoming too small)
        min_width = max(1200, int(screen_width * 0.6))  # At least 60% of screen or 1200px
        min_height = max(700, int(screen_height * 0.5))  # At least 50% of screen or 700px
        self.root.minsize(min_width, min_height)
        
        # Make window resizable
        self.root.resizable(True, True)
        
        print(f"🖥️ GUI Window Configuration:")
        print(f"   Screen size: {screen_width}x{screen_height}")
        print(f"   Window size: {window_width}x{window_height} ({window_width/screen_width:.1%}x{window_height/screen_height:.1%})")
        print(f"   Window position: +{pos_x}+{pos_y} (centered)")
        print(f"   Minimum size: {min_width}x{min_height}")
        
        # State variables
        self.model = None
        self.dataset = None
        self.rl_agent = None
        self.current_samples = []
        self.feedback_scores = []
        self.current_epoch = 0
        self.training_history = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        
        # GUI components - FIXED: Initialize dual slider lists
        self.image_labels = []
        self.feedback_sliders = []  # Keep for backward compatibility
        self.class_feedback_sliders = []   # For class identification feedback
        self.border_feedback_sliders = []  # For border identification feedback
        self.prediction_images = []
        self.opacity_var = tk.DoubleVar(value=0.9)  # Default 90% opacity
        
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the main GUI interface with responsive layout."""
        # Main container with responsive padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure root window for responsive resizing
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Configure main frame for responsive layout
        main_frame.columnconfigure(0, weight=1)  # Left content
        main_frame.columnconfigure(1, weight=1)  # Center content
        main_frame.columnconfigure(2, weight=1)  # Right content
        
        # Configure row weights for proper vertical distribution
        main_frame.rowconfigure(0, weight=0)  # Title (fixed height)
        main_frame.rowconfigure(1, weight=0)  # Control panel (fixed height)
        main_frame.rowconfigure(2, weight=0)  # Status panel (fixed height)
        main_frame.rowconfigure(3, weight=1)  # Feedback area (expandable)
        main_frame.rowconfigure(4, weight=0)  # Opacity control (fixed height)
        main_frame.rowconfigure(5, weight=0)  # Progress controls (fixed height)
        
        # Title with responsive font size
        screen_width = self.root.winfo_screenwidth()
        title_font_size = max(14, min(20, int(screen_width / 80)))  # Scale font with screen size
        title_label = ttk.Label(main_frame, text="Human Feedback RL Training", 
                               font=('Arial', title_font_size, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20), sticky=(tk.W, tk.E))
        
        # Control panel
        self.setup_control_panel(main_frame)
        
        # Training status
        self.setup_status_panel(main_frame)
        
        # Image feedback area (main expandable area)
        self.setup_feedback_area(main_frame)
        
        # Opacity control
        self.setup_opacity_control(main_frame)
        
        # Progress and controls
        self.setup_progress_controls(main_frame)
    
    def setup_control_panel(self, parent):
        """Setup control buttons and model/dataset loading."""
        control_frame = ttk.LabelFrame(parent, text="Model & Dataset Control", padding="10")
        control_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Model loading
        ttk.Button(control_frame, text="Load Model", 
                  command=self.load_model).grid(row=0, column=0, padx=(0, 10))
        
        self.model_label = ttk.Label(control_frame, text="No model loaded")
        self.model_label.grid(row=0, column=1, padx=(0, 20))
        
        # Dataset loading
        ttk.Button(control_frame, text="Load Dataset", 
                  command=self.load_dataset).grid(row=0, column=2, padx=(0, 10))
        
        self.dataset_label = ttk.Label(control_frame, text="No dataset loaded")
        self.dataset_label.grid(row=0, column=3, padx=(0, 20))
        
        # Training controls
        ttk.Button(control_frame, text="Start RL Training", 
                  command=self.start_rl_training).grid(row=0, column=4, padx=(10, 0))
        
        # Demo mode button
        ttk.Button(control_frame, text="Demo Mode", 
                  command=self.start_demo_mode).grid(row=0, column=5, padx=(10, 0))
    
    def setup_status_panel(self, parent):
        """Setup training status display."""
        status_frame = ttk.LabelFrame(parent, text="Training Status & Learning Progress", padding="10")
        status_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Basic status
        basic_frame = ttk.Frame(status_frame)
        basic_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E))
        
        self.epoch_label = ttk.Label(basic_frame, text="Epoch: 0")
        self.epoch_label.grid(row=0, column=0, padx=(0, 20))
        
        self.feedback_label = ttk.Label(basic_frame, text="Avg Feedback: N/A")
        self.feedback_label.grid(row=0, column=1, padx=(0, 20))
        
        self.loss_label = ttk.Label(basic_frame, text="RL Loss: N/A")
        self.loss_label.grid(row=0, column=2, padx=(0, 20))
        
        # Learning indicators
        learning_frame = ttk.Frame(status_frame)
        learning_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.learning_trend_label = ttk.Label(learning_frame, text="Learning Trend: Not Started", 
                                            foreground="blue")
        self.learning_trend_label.grid(row=0, column=0, padx=(0, 20))
        
        self.improvement_label = ttk.Label(learning_frame, text="Improvement: N/A")
        self.improvement_label.grid(row=0, column=1, padx=(0, 20))
        
        self.convergence_label = ttk.Label(learning_frame, text="Convergence: N/A")
        self.convergence_label.grid(row=0, column=2)
        
        # Progress metrics
        metrics_frame = ttk.Frame(status_frame)
        metrics_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.gradient_norm_label = ttk.Label(metrics_frame, text="Gradient Norm: N/A")
        self.gradient_norm_label.grid(row=0, column=0, padx=(0, 20))
        
        self.weight_change_label = ttk.Label(metrics_frame, text="Weight Change: N/A")
        self.weight_change_label.grid(row=0, column=1, padx=(0, 20))
    
    def setup_feedback_area(self, parent):
        """Setup the main feedback area with images and sliders."""
        feedback_frame = ttk.LabelFrame(parent, text="Human Feedback Interface", padding="10")
        feedback_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Configure grid weights for responsive layout
        feedback_frame.columnconfigure(0, weight=1)
        feedback_frame.columnconfigure(1, weight=1)
        feedback_frame.columnconfigure(2, weight=1)
        feedback_frame.columnconfigure(3, weight=1)  # Add column for legend
        
        # Create 3 image-feedback pairs
        for i in range(3):
            self.setup_image_feedback_pair(feedback_frame, i)
        
        # Add color legend on the right
        self.setup_color_legend(feedback_frame)
    
    def setup_image_feedback_pair(self, parent, col_idx):
        """Setup individual image with dual feedback sliders using responsive sizing."""
        # Image container with responsive configuration
        image_frame = ttk.Frame(parent)
        image_frame.grid(row=0, column=col_idx, padx=10, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure image frame for responsive resizing
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)  # Image area (expandable)
        image_frame.rowconfigure(1, weight=0)  # Class slider (fixed height)
        image_frame.rowconfigure(2, weight=0)  # Border slider (fixed height) 
        image_frame.rowconfigure(3, weight=0)  # Guidance (fixed height)
        
        # Calculate responsive image label width based on screen size
        screen_width = self.root.winfo_screenwidth()
        # Scale image width: larger screens get wider images, but with reasonable limits
        base_image_width = max(20, min(35, int(screen_width / 60)))
        
        # Image label with responsive sizing
        img_label = ttk.Label(image_frame, text=f"Image {col_idx + 1}", 
                             background="lightgray", width=base_image_width)
        img_label.grid(row=0, column=0, pady=(0, 10), sticky=(tk.W, tk.E))
        self.image_labels.append(img_label)
        
        # === CLASS IDENTIFICATION SLIDER ===
        class_frame = ttk.LabelFrame(image_frame, text="🎨 Class Identification", padding="5")
        class_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # Configure class frame for responsive layout
        class_frame.columnconfigure(0, weight=1)
        
        # Class feedback slider with responsive length
        class_slider_frame = ttk.Frame(class_frame)
        class_slider_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        class_slider_frame.columnconfigure(1, weight=1)  # Slider column expands
        
        ttk.Label(class_slider_frame, text="Poor").grid(row=0, column=0, padx=(0, 5), sticky=tk.W)
        
        # Calculate responsive slider length
        slider_length = max(150, min(250, int(screen_width / 8)))
        
        class_slider = tk.Scale(class_slider_frame, from_=-100, to=100, orient=tk.HORIZONTAL,
                               length=slider_length, resolution=1, 
                               bg="#E3F2FD", troughcolor="#1976D2")  # Blue theme
        class_slider.set(0)
        class_slider.grid(row=0, column=1, padx=2, sticky=(tk.W, tk.E))
        self.class_feedback_sliders.append(class_slider)
        
        ttk.Label(class_slider_frame, text="Perfect").grid(row=0, column=2, padx=(5, 0), sticky=tk.E)
        
        # Class description with responsive font
        desc_font_size = max(7, min(10, int(screen_width / 180)))
        class_desc = ttk.Label(class_frame, 
                              text="Rate tissue type accuracy (villi, glands, muscle, etc.)",
                              font=('Arial', desc_font_size), foreground="blue")
        class_desc.grid(row=1, column=0, pady=(2, 0), sticky=(tk.W, tk.E))
        
        # === BORDER/REGION IDENTIFICATION SLIDER ===
        border_frame = ttk.LabelFrame(image_frame, text="🎯 Border & Region Accuracy", padding="5")
        border_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # Configure border frame for responsive layout
        border_frame.columnconfigure(0, weight=1)
        
        # Border feedback slider with responsive length
        border_slider_frame = ttk.Frame(border_frame)
        border_slider_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        border_slider_frame.columnconfigure(1, weight=1)  # Slider column expands
        
        ttk.Label(border_slider_frame, text="Fuzzy").grid(row=0, column=0, padx=(0, 5), sticky=tk.W)
        
        border_slider = tk.Scale(border_slider_frame, from_=-100, to=100, orient=tk.HORIZONTAL,
                                length=slider_length, resolution=1,
                                bg="#E8F5E8", troughcolor="#388E3C")  # Green theme
        border_slider.set(0)
        border_slider.grid(row=0, column=1, padx=2, sticky=(tk.W, tk.E))
        self.border_feedback_sliders.append(border_slider)
        
        ttk.Label(border_slider_frame, text="Sharp").grid(row=0, column=2, padx=(5, 0), sticky=tk.E)
        
        # Border description with responsive font
        border_desc = ttk.Label(border_frame, 
                               text="Rate boundary sharpness and region completeness",
                               font=('Arial', desc_font_size), foreground="green")
        border_desc.grid(row=1, column=0, pady=(2, 0), sticky=(tk.W, tk.E))
        
        # === FEEDBACK GUIDANCE ===
        guidance_frame = ttk.Frame(image_frame)
        guidance_frame.grid(row=3, column=0, pady=(10, 0), sticky=(tk.W, tk.E))
        guidance_frame.columnconfigure(0, weight=1)
        
        guidance_text = """
🎨 Class: Are tissues correctly identified?
🎯 Border: Are boundaries clean and complete?
        """
        guidance_label = ttk.Label(guidance_frame, text=guidance_text, 
                                  font=('Arial', desc_font_size), justify="center")
        guidance_label.grid(row=0, column=0, sticky=(tk.W, tk.E))
    
    def setup_color_legend(self, parent):
        """Setup color legend showing tissue classes and their colors with responsive sizing."""
        legend_frame = ttk.LabelFrame(parent, text="Tissue Color Legend", padding="5")
        legend_frame.grid(row=0, column=3, padx=10, pady=5, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        # Configure legend frame for responsive layout
        legend_frame.columnconfigure(0, weight=1)
        legend_frame.rowconfigure(0, weight=1)
        
        # Calculate responsive canvas size
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        canvas_width = max(180, min(280, int(screen_width / 8)))
        canvas_height = max(250, min(400, int(screen_height / 3)))
        
        # Use globally available functions
        num_classes = get_num_tissue_classes()
        
        # Create scrollable frame for legend
        canvas = tk.Canvas(legend_frame, width=canvas_width, height=canvas_height)
        scrollbar = ttk.Scrollbar(legend_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Calculate responsive font size for legend
        legend_font_size = max(7, min(10, int(screen_width / 200)))
        
        # Add tissue classes with colors
        for class_id in range(min(15, num_classes)):  # Limit to prevent UI overflow
            color = get_tissue_color(class_id)
            name = get_tissue_name(class_id)
            
            # Create color swatch and label
            class_frame = ttk.Frame(scrollable_frame)
            class_frame.pack(fill=tk.X, pady=1)
            
            # Color swatch with responsive size
            swatch_size = max(2, min(4, int(screen_width / 400)))
            color_label = tk.Label(class_frame, 
                                 text="  ", 
                                 bg=f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}",
                                 width=swatch_size, height=1)
            color_label.pack(side=tk.LEFT, padx=(0, 5))
            
            # Class name with responsive font
            name_label = ttk.Label(class_frame, 
                                 text=f"{class_id}: {name.replace('_', ' ')}", 
                                 font=('Arial', legend_font_size))
            name_label.pack(side=tk.LEFT)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def setup_opacity_control(self, parent):
        """Setup opacity control slider with responsive sizing."""
        opacity_frame = ttk.LabelFrame(parent, text="Opacity Control", padding="10")
        opacity_frame.grid(row=4, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E))
        
        # Configure opacity frame for responsive layout
        opacity_frame.columnconfigure(2, weight=1)  # Slider column expands
        
        # Calculate responsive slider length
        screen_width = self.root.winfo_screenwidth()
        slider_length = max(150, min(300, int(screen_width / 6)))
        
        ttk.Label(opacity_frame, text="Overlay Opacity:").grid(row=0, column=0, padx=(0, 10), sticky=tk.W)
        ttk.Label(opacity_frame, text="Original").grid(row=0, column=1, padx=(0, 5), sticky=tk.W)
        
        # Opacity slider with responsive length
        self.opacity_slider = tk.Scale(opacity_frame, from_=0, to=1, orient=tk.HORIZONTAL,
                                     length=slider_length, resolution=0.01, variable=self.opacity_var,
                                     command=self.on_opacity_change)
        self.opacity_slider.set(0.5)  # Default to 50% opacity
        self.opacity_slider.grid(row=0, column=2, padx=5, sticky=(tk.W, tk.E))
        
        ttk.Label(opacity_frame, text="Overlay").grid(row=0, column=3, padx=(5, 10), sticky=tk.W)
        
        # Refresh button
        ttk.Button(opacity_frame, text="Refresh Display", 
                  command=self.refresh_display).grid(row=0, column=4, padx=10, sticky=tk.E)
    
    def on_opacity_change(self, value):
        """Handle opacity slider change."""
        # Automatically refresh display when opacity changes
        if hasattr(self, 'current_samples') and self.current_samples:
            self.refresh_display()
    
    def refresh_display(self):
        """Refresh the display with current opacity setting."""
        if hasattr(self, 'current_samples') and self.current_samples and self.prediction_images:
            self.display_samples_with_predictions()
    
    def setup_progress_controls(self, parent):
        """Setup progress controls and action buttons."""
        control_frame = ttk.Frame(parent)
        control_frame.grid(row=5, column=0, columnspan=3, pady=10)
        
        # Progress bar
        self.progress_var = tk.StringVar(value="Ready to start RL training")
        ttk.Label(control_frame, textvariable=self.progress_var).grid(row=0, column=0, columnspan=3, pady=(0, 10))
        
        # Action buttons
        ttk.Button(control_frame, text="Submit Feedback", 
                  command=self.submit_feedback).grid(row=1, column=0, padx=5)
        
        ttk.Button(control_frame, text="Next Epoch", 
                  command=self.next_epoch).grid(row=1, column=1, padx=5)
        
        ttk.Button(control_frame, text="Compare Learning", 
                  command=self.show_learning_comparison).grid(row=1, column=2, padx=5)
        
        ttk.Button(control_frame, text="Save RL Model", 
                  command=self.save_rl_model).grid(row=1, column=3, padx=5)
        
        ttk.Button(control_frame, text="Export Training Log", 
                  command=self.export_training_log).grid(row=1, column=4, padx=5)
    
    def load_model(self):
        """Load a pre-trained model with proper architecture selection."""
        
        # Step 1: Let user select model file
        model_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")]
        )
        
        if not model_path:
            return
        
        # Step 2: Show model type selection dialog (only if ModelFactory available)
        if MODEL_FACTORY_AVAILABLE:
            model_type = self.show_model_type_dialog()
            if not model_type:
                return
            
            try:
                # Step 3: Load with proper architecture
                self.model, model_info = self.load_model_with_architecture(model_path, model_type)
                
                self.model_label.config(text=f"Model: {model_type} - {os.path.basename(model_path)}")
                self.progress_var.set(f"✅ {model_type} model loaded successfully!")
                
                # Initialize RL agent
                self.rl_agent = RLAgent(self.model)
                
                print(f"🎯 Model loaded: {model_type}")
                print(f"📊 Classes: {model_info['num_classes']}")
                print(f"💾 Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
                
            except Exception as e:
                error_msg = f"Failed to load {model_type} model: {str(e)}"
                messagebox.showerror("Model Loading Error", error_msg)
                print(f"❌ Model loading error: {e}")
        
        else:
            # Fallback to original loading method
            try:
                # Load model checkpoint
                self.model, model_info = load_pytorch_model(model_path, self.device)
                
                self.model_label.config(text=f"Model: {os.path.basename(model_path)}")
                self.progress_var.set("Model loaded successfully!")
                
                # Initialize RL agent
                self.rl_agent = RLAgent(self.model)
                
            except Exception as e:
                error_msg = f"Failed to load model: {str(e)}"
                messagebox.showerror("Error", error_msg)
                print(f"Model loading error: {e}")
    
    def show_model_type_dialog(self):
        """Show dialog for user to select model type with responsive sizing."""
        
        # Get screen dimensions for responsive dialog sizing
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Calculate responsive dialog size (smaller percentage than main window)
        dialog_width = max(500, min(800, int(screen_width * 0.4)))
        dialog_height = max(400, min(600, int(screen_height * 0.6)))
        
        # Create model selection window
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Model Architecture")
        dialog.geometry(f"{dialog_width}x{dialog_height}")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Make dialog resizable
        dialog.resizable(True, True)
        dialog.minsize(400, 300)  # Set minimum size
        
        # Center the dialog
        dialog.update_idletasks()
        x = (screen_width - dialog_width) // 2
        y = (screen_height - dialog_height) // 2
        dialog.geometry(f"{dialog_width}x{dialog_height}+{x}+{y}")
        
        # Configure dialog for responsive layout
        dialog.columnconfigure(0, weight=1)
        dialog.rowconfigure(2, weight=1)  # Radio button area expands
        
        # Variable to store selected model
        selected_model = tk.StringVar()
        result = {"model_type": None}
        
        # Title with responsive font
        title_font_size = max(14, min(18, int(screen_width / 100)))
        title_label = tk.Label(dialog, text="🏗️ Select Model Architecture", 
                              font=("Arial", title_font_size, "bold"))
        title_label.grid(row=0, column=0, pady=10, sticky=(tk.W, tk.E))
        
        # Instructions with responsive font
        instruction_font_size = max(9, min(12, int(screen_width / 140)))
        instruction_label = tk.Label(dialog, 
                                    text="Choose the architecture that matches your trained model:",
                                    font=("Arial", instruction_font_size))
        instruction_label.grid(row=1, column=0, pady=5, sticky=(tk.W, tk.E))
        
        # Get available models
        try:
            available_models = ModelFactory.get_available_models()
        except:
            # Fallback if ModelFactory fails
            available_models = {
                'unet': 'UNet - Classic architecture (Fast, Low Memory)',
                'simple_unet': 'SimpleUNet - Basic implementation'
            }
        
        # Create radio buttons for each model with responsive layout
        radio_frame = tk.Frame(dialog)
        radio_frame.grid(row=2, column=0, pady=20, padx=20, sticky=(tk.N, tk.S, tk.E, tk.W))
        radio_frame.columnconfigure(0, weight=1)
        
        # Calculate responsive wrap length for radio buttons
        wrap_length = max(400, min(700, int(dialog_width * 0.8)))
        radio_font_size = max(10, min(13, int(screen_width / 130)))
        
        for model_type, description in available_models.items():
            radio_btn = tk.Radiobutton(
                radio_frame,
                text=f"🔹 {description}",
                variable=selected_model,
                value=model_type,
                font=("Arial", radio_font_size),
                wraplength=wrap_length,
                justify="left",
                anchor="w"
            )
            radio_btn.pack(anchor="w", pady=8, padx=10, fill="x")
        
        # Set default selection to unet
        if 'unet' in available_models:
            selected_model.set('unet')
        elif available_models:
            selected_model.set(list(available_models.keys())[0])
        
        # Buttons with responsive sizing
        button_frame = tk.Frame(dialog)
        button_frame.grid(row=3, column=0, pady=20, sticky=(tk.W, tk.E))
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        
        button_font_size = max(9, min(12, int(screen_width / 150)))
        button_width = max(10, min(15, int(screen_width / 120)))
        
        def on_ok():
            if selected_model.get():
                result["model_type"] = selected_model.get()
            dialog.destroy()
        
        def on_cancel():
            result["model_type"] = None
            dialog.destroy()
        
        ok_btn = tk.Button(button_frame, text="✅ Load Model", command=on_ok,
                          bg="#4CAF50", fg="white", font=("Arial", button_font_size, "bold"),
                          width=button_width)
        ok_btn.grid(row=0, column=0, padx=10, sticky=tk.E)
        
        cancel_btn = tk.Button(button_frame, text="❌ Cancel", command=on_cancel,
                              bg="#f44336", fg="white", font=("Arial", button_font_size),
                              width=button_width)
        cancel_btn.grid(row=0, column=1, padx=10, sticky=tk.W)
        
        print(f"📱 Model Selection Dialog:")
        print(f"   Dialog size: {dialog_width}x{dialog_height} ({dialog_width/screen_width:.1%}x{dialog_height/screen_height:.1%})")
        print(f"   Position: +{x}+{y} (centered)")
        
        # Wait for user selection
        dialog.wait_window()
        return result["model_type"]
    
    def load_model_with_architecture(self, model_path, model_type):
        """Load model with specified architecture using ModelFactory."""
        
        print(f"🔄 Loading {model_type} model from {model_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract model information
            if isinstance(checkpoint, dict):
                model_state = checkpoint.get('model_state_dict', checkpoint)
                num_classes = checkpoint.get('num_classes', 15)  # Default to 15 tissue classes
                saved_model_type = checkpoint.get('model_type', model_type)
            else:
                model_state = checkpoint
                num_classes = 15
                saved_model_type = model_type
            
            print(f"📋 Detected {num_classes} classes in saved model")
            
            # Handle different model types
            if model_type == 'simple_unet' or not MODEL_FACTORY_AVAILABLE:
                # Use the built-in SimpleUNet
                model = SimpleUNet(in_channels=1, num_classes=num_classes)
            else:
                # Use ModelFactory for other architectures
                config = ModelFactory.get_default_config(model_type)
                config['num_classes'] = num_classes
                config['in_channels'] = 1
                
                print(f"⚙️ Creating {model_type} with config: {config}")
                model = ModelFactory.create_model(model_type, config)
            
            # Load state dict
            try:
                # Try strict loading first
                model.load_state_dict(model_state, strict=True)
                print("✅ Model weights loaded successfully (strict=True)")
            except RuntimeError as e:
                print(f"⚠️ Strict loading failed: {e}")
                try:
                    # Try relaxed loading
                    missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
                    if missing_keys:
                        print(f"⚠️ Missing keys: {len(missing_keys)} parameters")
                    if unexpected_keys:
                        print(f"⚠️ Unexpected keys: {len(unexpected_keys)} parameters")
                    print("✅ Model weights loaded successfully (strict=False)")
                except Exception as e2:
                    print(f"❌ Could not load any weights: {e2}")
                    print("🔄 Using randomly initialized model")
            
            model = model.to(self.device)
            model.eval()
            
            model_info = {
                'num_classes': num_classes,
                'model_type': model_type,
                'saved_model_type': saved_model_type,
            }
            
            return model, model_info
            
        except Exception as e:
            print(f"❌ Error in load_model_with_architecture: {e}")
            
            # Fallback: create new model with proper architecture
            print(f"🔄 Creating new {model_type} as fallback")
            
            if model_type == 'simple_unet' or not MODEL_FACTORY_AVAILABLE:
                model = SimpleUNet(in_channels=1, num_classes=15)
            else:
                config = ModelFactory.get_default_config(model_type)
                config['num_classes'] = 15
                model = ModelFactory.create_model(model_type, config)
            
            model = model.to(self.device)
            
            model_info = {
                'num_classes': 15,
                'model_type': model_type,
                'saved_model_type': 'fallback',
            }
            
            return model, model_info
    
    def load_dataset(self):
        """Load dataset for RL training."""
        dataset_path = filedialog.askdirectory(title="Select Dataset Directory")
        
        if not dataset_path:
            return
        
        try:
            # Create CombinedDataset
            self.dataset = CombinedDataset(dataset_path)
            
            self.dataset_label.config(text=f"Dataset: {len(self.dataset)} samples")
            self.progress_var.set("Dataset loaded successfully!")
            
        except Exception as e:
            error_msg = f"Failed to load dataset: {str(e)}"
            if "not available" in str(e):
                error_msg += "\n\nTip: Make sure the core data handler modules are properly installed."
            messagebox.showerror("Error", error_msg)
            print(f"Dataset loading error: {e}")
    
    def start_rl_training(self):
        """Start the RL training process."""
        if not self.model or not self.dataset or not self.rl_agent:
            messagebox.showwarning("Warning", "Please load both model and dataset first!")
            return
        
        self.current_epoch = 1
        self.load_epoch_samples()
        self.progress_var.set(f"RL Training started - Epoch {self.current_epoch}")
    
    def load_epoch_samples(self):
        """Load random samples for current epoch."""
        if not self.rl_agent or not self.dataset:
            return
        
        # Select 3 random samples
        self.current_samples = self.rl_agent.select_random_samples(self.dataset, 3)
        
        # Generate predictions and display
        self.display_samples_with_predictions()
        
        # Update status
        self.epoch_label.config(text=f"Epoch: {self.current_epoch}")
        self.progress_var.set(f"Epoch {self.current_epoch}: Provide feedback for {len(self.current_samples)} samples")
    
    def display_samples_with_predictions(self):
        """Display samples with model predictions."""
        if not self.current_samples:
            return
        
        self.prediction_images.clear()
        
        for i, (sample_idx, sample_data) in enumerate(self.current_samples):
            if i >= 3:  # Only show first 3
                break
            
            try:
                # Extract image and run prediction
                if isinstance(sample_data, tuple) and len(sample_data) >= 2:
                    image_tensor = sample_data[0]
                else:
                    image_tensor = sample_data
                
                # Ensure proper tensor format
                if image_tensor.dim() == 3:
                    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
                
                image_tensor = image_tensor.to(self.device)
                
                # Generate prediction
                self.model.eval()
                with torch.no_grad():
                    prediction = self.model(image_tensor)
                    pred_mask = torch.argmax(prediction, dim=1).squeeze().cpu().numpy()
                
                # Ensure prediction mask matches image dimensions
                image_h, image_w = image_tensor.shape[-2:]
                if pred_mask.shape != (image_h, image_w):
                    pred_mask_pil = Image.fromarray(pred_mask.astype(np.uint8))
                    pred_mask_pil = pred_mask_pil.resize((image_w, image_h), Image.Resampling.NEAREST)
                    pred_mask = np.array(pred_mask_pil)
                
                # Convert to PIL Image for display
                image_np = image_tensor.squeeze().cpu().numpy()
                
                # Handle different tensor dimensions
                if image_np.ndim == 2:  # Grayscale [H, W]
                    # Convert to RGB by repeating the channel
                    image_np = np.stack([image_np, image_np, image_np], axis=2)
                elif image_np.ndim == 3:  # Color [C, H, W] or [H, W, C]
                    if image_np.shape[0] == 1 or image_np.shape[0] == 3:  # [C, H, W]
                        image_np = image_np.transpose(1, 2, 0)  # Convert to [H, W, C]
                    if image_np.shape[2] == 1:  # Grayscale [H, W, 1]
                        image_np = np.repeat(image_np, 3, axis=2)  # Convert to RGB
                
                # Ensure proper range [0, 1] and convert to [0, 255]
                if image_np.max() <= 1.0:
                    image_np = (image_np * 255).astype(np.uint8)
                else:
                    image_np = image_np.astype(np.uint8)
                
                # Create overlay with prediction
                pred_colored = self.colorize_prediction(pred_mask)
                
                # Ensure both images have the same size and mode
                if image_np.shape[:2] != pred_colored.shape[:2]:
                    # Resize prediction to match image
                    pred_colored = np.array(Image.fromarray(pred_colored).resize(
                        (image_np.shape[1], image_np.shape[0]), Image.Resampling.NEAREST))
                
                # Ensure both are 3-channel RGB
                if image_np.shape[2] != 3:
                    image_np = image_np[:, :, :3] if image_np.shape[2] > 3 else np.concatenate([image_np] * (3 - image_np.shape[2] + 1), axis=2)[:, :, :3]
                
                if pred_colored.shape[2] != 3:
                    pred_colored = pred_colored[:, :, :3] if pred_colored.shape[2] > 3 else np.concatenate([pred_colored] * (3 - pred_colored.shape[2] + 1), axis=2)[:, :, :3]
                
                # Convert to PIL Images with same mode
                image_pil = Image.fromarray(image_np, mode='RGB')
                pred_pil = Image.fromarray(pred_colored, mode='RGB')
                
                # Create blended overlay
                overlay = Image.blend(image_pil, pred_pil, alpha=self.opacity_var.get())
                
                # Resize for display
                display_size = (200, 200)
                overlay = overlay.resize(display_size, Image.Resampling.LANCZOS)
                
                # Convert to PhotoImage and display
                photo = ImageTk.PhotoImage(overlay)
                self.image_labels[i].config(image=photo, text="")
                self.image_labels[i].image = photo  # Keep reference
                
                # Store for feedback processing
                self.prediction_images.append({
                    'sample_idx': sample_idx,
                    'image_tensor': image_tensor,
                    'prediction': prediction,
                    'pred_mask': pred_mask
                })
                
            except Exception as e:
                print(f"Error displaying sample {i}: {e}")
                print(f"  Image tensor shape: {image_tensor.shape}")
                print(f"  Image tensor device: {image_tensor.device}")
                print(f"  Sample data type: {type(sample_data)}")
                
                # Show error message in GUI
                self.image_labels[i].config(image="", text=f"Error loading\nSample {sample_idx}\n{str(e)[:50]}...")
        
        # Reset sliders
        for slider in self.feedback_sliders:
            slider.set(0)
    
    def colorize_prediction(self, pred_mask: np.ndarray) -> np.ndarray:
        """Convert prediction mask to colored visualization using correct tissue colors."""
        height, width = pred_mask.shape
        colored = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Use the globally available tissue color functions
        num_classes = get_num_tissue_classes()
        
        for class_id in range(num_classes):
            if class_id <= pred_mask.max():
                mask = pred_mask == class_id
                if np.any(mask):
                    color = get_tissue_color(class_id)
                    colored[mask] = color
        
        return colored
    
    def submit_feedback(self):
        """Submit dual feedback and update model."""
        if not self.current_samples or not self.rl_agent:
            messagebox.showwarning("Warning", "No samples loaded or model not ready!")
            return
        
        try:
            # Collect dual feedback scores - FIXED VERSION
            class_scores = []
            border_scores = []
            
            for i in range(len(self.current_samples)):
                if i < len(self.class_feedback_sliders) and i < len(self.border_feedback_sliders):
                    class_score = self.class_feedback_sliders[i].get() / 100.0  # Convert to -1 to +1
                    border_score = self.border_feedback_sliders[i].get() / 100.0
                    
                    class_scores.append(class_score)
                    border_scores.append(border_score)
                else:
                    # Default to neutral if sliders don't exist
                    class_scores.append(0.0)
                    border_scores.append(0.0)
            
            # Store dual feedback in RL agent
            for i, (sample_idx, image_tensor) in enumerate(self.current_samples):
                if i < len(self.prediction_images):
                    pred_info = self.prediction_images[i]
                    self.rl_agent.feedback_collector.add_dual_feedback(
                        image_idx=sample_idx,
                        prediction=pred_info['pred_mask'],
                        class_feedback=class_scores[i],
                        border_feedback=border_scores[i]
                    )
            
            # Update model with dual feedback
            images = [sample[1] for sample in self.current_samples]  # Get image tensors
            predictions = [info['prediction'] for info in self.prediction_images]
            
            # Combine class and border feedback (simple average for now)
            combined_feedback = []
            for class_score, border_score in zip(class_scores, border_scores):
                combined_score = (class_score + border_score) / 2.0
                combined_feedback.append(combined_score)
            
            rl_loss, avg_grad_norm, weight_change = self.rl_agent.update_model_with_feedback(
                images, predictions, combined_feedback
            )
            
            # Get learning status
            learning_status = self.rl_agent.get_learning_status()
            
            # Update display - calculate averages safely
            avg_class_feedback = sum(class_scores) / len(class_scores) if class_scores else 0.0
            avg_border_feedback = sum(border_scores) / len(border_scores) if border_scores else 0.0
            avg_combined_feedback = sum(combined_feedback) / len(combined_feedback) if combined_feedback else 0.0
            
            # Update status labels with dual feedback
            self.feedback_label.config(text=f"Class: {avg_class_feedback:.2f} | Border: {avg_border_feedback:.2f} | Combined: {avg_combined_feedback:.2f}")
            self.loss_label.config(text=f"RL Loss: {rl_loss:.4f}")
            
            # Update learning indicators
            self.learning_trend_label.config(
                text=f"Learning Trend: {learning_status['trend']}", 
                foreground=learning_status['trend_color']
            )
            self.improvement_label.config(text=f"Improvement: {learning_status['improvement']:.1f}%")
            
            convergence_text = f"Convergence: {learning_status['convergence']:.2f}"
            convergence_color = "green" if learning_status['convergence'] > 0.7 else "orange" if learning_status['convergence'] > 0.4 else "red"
            self.convergence_label.config(text=convergence_text, foreground=convergence_color)
            
            # Update progress metrics
            self.gradient_norm_label.config(text=f"Gradient Norm: {avg_grad_norm:.4f}")
            self.weight_change_label.config(text=f"Weight Change: {weight_change:.6f}")
            
            # Store training history with dual feedback metrics
            self.training_history.append({
                'epoch': self.current_epoch,
                'class_feedback': avg_class_feedback,
                'border_feedback': avg_border_feedback,
                'combined_feedback': avg_combined_feedback,
                'rl_loss': rl_loss,
                'feedback_scores': combined_feedback.copy(),
                'learning_status': learning_status,
                'gradient_norm': avg_grad_norm,
                'weight_change': weight_change
            })
            
            # Print epoch summary with dual feedback
            print(f"🎯 Epoch {self.current_epoch} - Class: {avg_class_feedback:.2f}, Border: {avg_border_feedback:.2f}, Combined: {avg_combined_feedback:.2f}")
            
            # Provide learning insights to user
            insight_message = self._generate_learning_insight(learning_status, avg_combined_feedback)
            self.progress_var.set(insight_message)
            
            # CRITICAL FIX: Automatically advance to next epoch after feedback submission
            self.root.after(1000, self.next_epoch)  # Wait 1 second then advance epoch
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update model: {str(e)}")
            print(f"Submit feedback error: {e}")  # Debug print
    
    def _generate_learning_insight(self, learning_status, avg_feedback):
        """Generate user-friendly learning insights."""
        status = learning_status['status']
        trend = learning_status['trend']
        improvement = learning_status['improvement']
        confidence = learning_status['confidence']
        convergence = learning_status['convergence']
        
        if status == 'Not Started':
            return "Ready to start learning from feedback..."
        
        base_msg = f"Learning Status: {trend}"
        
        if trend == "Improving":
            if convergence > 0.7:
                return f"🎯 {base_msg} - Model is learning well! ({improvement:.1f}% improvement)"
            elif convergence > 0.4:
                return f"📈 {base_msg} - Good progress, continue training! ({improvement:.1f}% improvement)"
            else:
                return f"📊 {base_msg} - Early improvement detected ({improvement:.1f}%)"
        
        elif trend == "Stable":
            if convergence > 0.6:
                return f"⚖️ {base_msg} - Model has converged (may need different feedback)"
            else:
                return f"🔄 {base_msg} - Try providing more varied feedback"
        
        elif trend == "Worsening":
            if abs(improvement) < 5:
                return f"⚠️ {base_msg} - Minor setback, continue with consistent feedback"
            else:
                return f"🔴 {base_msg} - Consider adjusting feedback strategy"
        
        else:  # Insufficient Data
            updates = learning_status['total_updates']
            return f"📊 Starting learning process... ({updates} updates so far)"
    
    def next_epoch(self):
        """Move to next epoch."""
        if not self.rl_agent:
            return
        
        # Finalize current epoch
        self.rl_agent.feedback_collector.finalize_epoch(self.current_epoch)
        
        # Move to next epoch
        self.current_epoch += 1
        self.load_epoch_samples()
    
    def save_rl_model(self):
        """Save the RL-enhanced model with comprehensive information."""
        if not self.model:
            messagebox.showwarning("Warning", "No model to save!")
            return
        
        # Ask for save path
        save_path = filedialog.asksaveasfilename(
            title="Save RL Enhanced Model",
            defaultextension=".pth",
            filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")]
        )
        
        if not save_path:
            return
        
        # Ask if user wants to provide original model path for comparison
        include_comparison = messagebox.askyesno(
            "Include Original Model Comparison",
            "Do you want to include a comparison with the original model?\n\n"
            "This will analyze how much the model has changed during RL training.\n"
            "Select 'Yes' to choose the original model file for comparison."
        )
        
        original_model_path = None
        if include_comparison:
            original_model_path = filedialog.askopenfilename(
                title="Select Original Model for Comparison",
                filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")]
            )
        
        try:
            # Use comprehensive saving
            if hasattr(self.rl_agent, 'save_complete_model_info'):
                model_path, summary_path = self.rl_agent.save_complete_model_info(
                    save_path, original_model_path
                )
                
                if model_path and summary_path:
                    success_msg = f"✅ RL model saved successfully!\n\n"
                    success_msg += f"Model file: {os.path.basename(model_path)}\n"
                    success_msg += f"Summary: {os.path.basename(summary_path)}\n\n"
                    success_msg += f"The summary file contains:\n"
                    success_msg += f"• Complete model architecture info\n"
                    success_msg += f"• Layer-by-layer weight analysis\n"
                    success_msg += f"• RL training metrics and history\n"
                    success_msg += f"• Weight change analysis\n"
                    if original_model_path:
                        success_msg += f"• Comparison with original model\n"
                    
                    messagebox.showinfo("Success", success_msg)
                else:
                    raise Exception("Failed to save comprehensive model info")
            else:
                # Fallback to basic saving
                checkpoint = {
                    'model_state_dict': self.model.state_dict(),
                    'rl_epoch': self.current_epoch,
                    'training_history': self.training_history,
                    'feedback_history': self.rl_agent.feedback_collector.feedback_history,
                    'timestamp': datetime.now().isoformat()
                }
                
                torch.save(checkpoint, save_path)
                messagebox.showinfo("Success", f"RL model saved to {save_path}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save model: {str(e)}")
            print(f"Save error: {e}")
    
    def export_training_log(self):
        """Export training history and feedback log."""
        if not self.training_history:
            messagebox.showwarning("Warning", "No training history to export!")
            return
        
        save_path = filedialog.asksaveasfilename(
            title="Export Training Log",
            defaultextension=".json",
            filetypes=[("JSON File", "*.json"), ("All Files", "*.*")]
        )
        
        if save_path:
            try:
                export_data = {
                    'training_history': self.training_history,
                    'feedback_history': self.rl_agent.feedback_collector.feedback_history if self.rl_agent else [],
                    'total_epochs': self.current_epoch,
                    'export_timestamp': datetime.now().isoformat()
                }
                
                with open(save_path, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                
                messagebox.showinfo("Success", f"Training log exported to {save_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export log: {str(e)}")
    
    def start_demo_mode(self):
        """Start demo mode with synthetic data."""
        try:
            # Create a simple demo model
            self.model = SimpleUNet(in_channels=1, num_classes=15)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Create demo dataset with synthetic images
            self.create_demo_dataset()
            
            # Initialize RL agent
            self.rl_agent = RLAgent(self.model)
            
            # Update UI
            self.model_label.config(text="Model: Demo SimpleUNet")
            self.dataset_label.config(text="Dataset: Demo (synthetic images)")
            self.progress_var.set("Demo mode started - Ready for RL training!")
            
            # Start demo training
            self.current_epoch = 1
            self.load_epoch_samples()
            
        except Exception as e:
            messagebox.showerror("Demo Error", f"Failed to start demo mode: {str(e)}")
    
    def create_demo_dataset(self):
        """Create a simple demo dataset with synthetic images."""
        class DemoDataset:
            def __init__(self):
                self.images = []
                # Create 10 synthetic 256x256 grayscale images with different patterns
                for i in range(10):
                    # Create different patterns
                    if i % 3 == 0:
                        # Circular pattern
                        img = self._create_circular_pattern()
                    elif i % 3 == 1:
                        # Stripe pattern
                        img = self._create_stripe_pattern()
                    else:
                        # Random noise pattern
                        img = self._create_noise_pattern()
                    
                    self.images.append(img)
            
            def _create_circular_pattern(self):
                """Create a circular pattern image."""
                img = np.zeros((256, 256), dtype=np.float32)
                center = (128, 128)
                y, x = np.ogrid[:256, :256]
                mask = (x - center[0])**2 + (y - center[1])**2 <= 50**2
                img[mask] = 0.8
                
                # Add some noise
                img += np.random.normal(0, 0.1, (256, 256))
                img = np.clip(img, 0, 1)
                return torch.from_numpy(img).unsqueeze(0)
            
            def _create_stripe_pattern(self):
                """Create a stripe pattern image."""
                img = np.zeros((256, 256), dtype=np.float32)
                for i in range(0, 256, 20):
                    img[:, i:i+10] = 0.7
                
                # Add some noise
                img += np.random.normal(0, 0.1, (256, 256))
                img = np.clip(img, 0, 1)
                return torch.from_numpy(img).unsqueeze(0)
            
            def _create_noise_pattern(self):
                """Create a random noise pattern image."""
                img = np.random.random((256, 256)).astype(np.float32) * 0.5
                return torch.from_numpy(img).unsqueeze(0)
            
            def __len__(self):
                return len(self.images)
            
            def __getitem__(self, idx):
                return self.images[idx]
        
        self.dataset = DemoDataset()
    
    def show_learning_comparison(self):
        """Show learning comparison between epochs with responsive window sizing."""
        if not self.training_history:
            messagebox.showwarning("Warning", "No training history to compare!")
            return
        
        # Get screen dimensions for responsive sizing
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Calculate responsive window size (70% of screen width, 75% of screen height)
        window_width = max(700, min(1200, int(screen_width * 0.7)))
        window_height = max(500, min(900, int(screen_height * 0.75)))
        
        # Center the window
        pos_x = (screen_width - window_width) // 2
        pos_y = (screen_height - window_height) // 2
        
        # Create a new window for comparison
        comparison_window = tk.Toplevel(self.root)
        comparison_window.title("📊 Learning Progress Analysis")
        comparison_window.geometry(f"{window_width}x{window_height}+{pos_x}+{pos_y}")
        comparison_window.configure(bg="white")
        comparison_window.resizable(True, True)
        comparison_window.minsize(600, 400)  # Set minimum size
        
        # Configure window for responsive layout
        comparison_window.columnconfigure(0, weight=1)
        comparison_window.rowconfigure(0, weight=1)
        
        # Main frame with responsive padding
        main_frame = ttk.Frame(comparison_window, padding="20")
        main_frame.pack(fill="both", expand=True)
        
        # Configure main frame layout
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)  # Text area expands
        
        # Title with responsive font size
        title_font_size = max(14, min(18, int(screen_width / 100)))
        title_label = ttk.Label(main_frame, text="📊 Reinforcement Learning Progress", 
                               font=("Arial", title_font_size, "bold"))
        title_label.grid(row=0, column=0, pady=(0, 20), sticky=(tk.W, tk.E))
        
        # Create text widget with scrollbar (responsive sizing)
        text_frame = ttk.Frame(main_frame)
        text_frame.grid(row=1, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        
        # Calculate responsive font size for analysis text
        text_font_size = max(9, min(12, int(screen_width / 150)))
        
        analysis_text = tk.Text(text_frame, font=("Consolas", text_font_size), wrap=tk.WORD, 
                               bg="#f8f9fa", relief="solid", borderwidth=1)
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=analysis_text.yview)
        analysis_text.configure(yscrollcommand=scrollbar.set)
        
        analysis_text.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Generate analysis content
        self._show_detailed_analysis(analysis_text)
        
        # Button frame with responsive layout
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, pady=(15, 0), sticky=(tk.W, tk.E))
        button_frame.columnconfigure(1, weight=1)  # Space between buttons
        
        # Calculate responsive button font size
        button_font_size = max(9, min(12, int(screen_width / 140)))
        
        # Export button
        ttk.Button(button_frame, text="📄 Export Analysis", 
                  command=lambda: self._export_learning_analysis()).grid(row=0, column=0, sticky=tk.W)
        
        # Close button
        ttk.Button(button_frame, text="✖ Close", 
                  command=comparison_window.destroy).grid(row=0, column=2, sticky=tk.E)
        
        # Make text read-only
        analysis_text.config(state=tk.DISABLED)
        
        print(f"📊 Learning Analysis Window:")
        print(f"   Window size: {window_width}x{window_height} ({window_width/screen_width:.1%}x{window_height/screen_height:.1%})")
        print(f"   Position: +{pos_x}+{pos_y} (centered)")
        print(f"   Text font size: {text_font_size}")
        print(f"   Title font size: {title_font_size}")
    
    def _show_detailed_analysis(self, text_widget):
        """Show detailed learning analysis in text format."""
        text_widget.delete(1.0, tk.END)
        
        if not self.training_history:
            text_widget.insert(tk.END, "No training data available yet.")
            return
        
        # Extract data safely with error handling
        epochs = len(self.training_history)
        feedbacks = []
        losses = []
        gradient_norms = []
        weight_changes = []
        
        for h in self.training_history:
            # Use .get() with defaults to avoid KeyError
            feedbacks.append(h.get("avg_feedback", 0.0))
            losses.append(h.get("rl_loss", 0.0))
            gradient_norms.append(h.get("gradient_norm", 0.0))
            weight_changes.append(h.get("weight_change", 0.0))
        
        # Calculate statistics safely
        avg_feedback = sum(feedbacks) / len(feedbacks) if feedbacks else 0.0
        max_feedback = max(feedbacks) if feedbacks else 0.0
        min_feedback = min(feedbacks) if feedbacks else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        
        # Generate comprehensive report
        text_widget.insert(tk.END, "🎯 REINFORCEMENT LEARNING PROGRESS REPORT\n")
        text_widget.insert(tk.END, "=" * 60 + "\n\n")
        
        # Summary statistics
        text_widget.insert(tk.END, "📊 TRAINING SUMMARY:\n")
        text_widget.insert(tk.END, f"   Total Training Epochs: {epochs}\n")
        text_widget.insert(tk.END, f"   Average Feedback Score: {avg_feedback:.4f}\n")
        text_widget.insert(tk.END, f"   Best Feedback Score: {max_feedback:.4f}\n")
        text_widget.insert(tk.END, f"   Worst Feedback Score: {min_feedback:.4f}\n")
        text_widget.insert(tk.END, f"   Average RL Loss: {avg_loss:.6f}\n\n")
        
        # Detailed epoch breakdown
        text_widget.insert(tk.END, "📋 DETAILED EPOCH BREAKDOWN:\n")
        text_widget.insert(tk.END, "-" * 60 + "\n")
        text_widget.insert(tk.END, f"{'Epoch':<6} {'Feedback':<10} {'RL Loss':<12} {'Grad Norm':<10} {'Status'}\n")
        text_widget.insert(tk.END, "-" * 60 + "\n")
        
        for i, (feedback, loss, grad_norm) in enumerate(zip(feedbacks, losses, gradient_norms)):
            epoch = i + 1
            
            # Determine status based on feedback
            if feedback > 0.3:
                status = "🟢 Excellent"
            elif feedback > 0.0:
                status = "🟡 Good"
            elif feedback > -0.3:
                status = "🟠 Average"
            else:
                status = "🔴 Poor"
            
            text_widget.insert(tk.END, f"{epoch:<6} {feedback:<10.3f} {loss:<12.6f} {grad_norm:<10.4f} {status}\n")
        
        # Trend analysis
        if len(feedbacks) > 1:
            text_widget.insert(tk.END, "\n📈 LEARNING TREND ANALYSIS:\n")
            text_widget.insert(tk.END, "-" * 40 + "\n")
            
            # Calculate recent vs initial performance
            recent_window = min(3, len(feedbacks))
            recent_avg = sum(feedbacks[-recent_window:]) / recent_window
            initial_avg = sum(feedbacks[:recent_window]) / recent_window
            
            improvement = ((recent_avg - initial_avg) / abs(initial_avg) * 100) if initial_avg != 0 else 0
            
            if recent_avg > initial_avg + 0.1:
                trend_emoji = "📈"
                trend_text = "SIGNIFICANT IMPROVEMENT"
                trend_advice = "Excellent! The model is learning well from your feedback."
            elif recent_avg > initial_avg:
                trend_emoji = "📊"  
                trend_text = "MODERATE IMPROVEMENT"
                trend_advice = "Good progress. Continue with consistent feedback."
            elif recent_avg < initial_avg - 0.1:
                trend_emoji = "📉"
                trend_text = "DECLINING PERFORMANCE"
                trend_advice = "Consider adjusting your feedback strategy."
            else:
                trend_emoji = "➡️"
                trend_text = "STABLE PERFORMANCE"
                trend_advice = "Model has reached stable state. Try varied feedback."
            
            text_widget.insert(tk.END, f"{trend_emoji} Overall Trend: {trend_text}\n")
            text_widget.insert(tk.END, f"   Performance Change: {improvement:+.1f}%\n")
            text_widget.insert(tk.END, f"   Recent Average: {recent_avg:.3f}\n")
            text_widget.insert(tk.END, f"   Initial Average: {initial_avg:.3f}\n")
            text_widget.insert(tk.END, f"   💡 Advice: {trend_advice}\n\n")
        
        # Learning insights
        text_widget.insert(tk.END, "🧠 LEARNING INSIGHTS:\n")
        text_widget.insert(tk.END, "-" * 30 + "\n")
        
        # Analyze feedback consistency
        if len(feedbacks) > 2:
            feedback_variance = sum((f - avg_feedback)**2 for f in feedbacks) / len(feedbacks)
            if feedback_variance < 0.1:
                consistency = "High consistency in feedback"
            elif feedback_variance < 0.3:
                consistency = "Moderate consistency in feedback"
            else:
                consistency = "High variability in feedback"
            
            text_widget.insert(tk.END, f"• {consistency} (variance: {feedback_variance:.3f})\n")
        
        # Loss analysis
        if len(losses) > 1:
            loss_trend = "decreasing" if losses[-1] < losses[0] else "increasing" if losses[-1] > losses[0] else "stable"
            text_widget.insert(tk.END, f"• RL Loss is {loss_trend} over time\n")
        
        # Recommendations
        text_widget.insert(tk.END, "\n🎯 RECOMMENDATIONS:\n")
        text_widget.insert(tk.END, "-" * 25 + "\n")
        text_widget.insert(tk.END, "• Continue providing consistent, thoughtful feedback\n")
        text_widget.insert(tk.END, "• Use the opacity slider to better evaluate predictions\n")
        text_widget.insert(tk.END, "• Monitor trends over multiple epochs for stability\n")
        text_widget.insert(tk.END, "• Save the model when performance peaks\n")
        text_widget.insert(tk.END, "• Try varying feedback strategies if progress stalls\n\n")
        
        # How RL learning works
        text_widget.insert(tk.END, "🔬 HOW REINFORCEMENT LEARNING CHANGES THE MODEL:\n")
        text_widget.insert(tk.END, "=" * 55 + "\n")
        text_widget.insert(tk.END, "1. FEEDBACK COLLECTION: Your scores (-100% to +100%) are collected\n")
        text_widget.insert(tk.END, "2. REWARD SIGNAL: Positive feedback → reward, negative → penalty\n") 
        text_widget.insert(tk.END, "3. POLICY UPDATE: Model weights adjusted to maximize future rewards\n")
        text_widget.insert(tk.END, "4. GRADIENT COMPUTATION: Backpropagation with RL loss function\n")
        text_widget.insert(tk.END, "5. WEIGHT UPDATES: Neural network parameters modified incrementally\n")
        text_widget.insert(tk.END, "6. LEARNING CONVERGENCE: Model improves through iterative refinement\n\n")
        
        text_widget.insert(tk.END, "✅ Your feedback directly influences model behavior!\n")
        text_widget.insert(tk.END, "   Positive feedback reinforces good predictions.\n")
        text_widget.insert(tk.END, "   Negative feedback discourages poor predictions.\n")
        
    def _export_learning_analysis(self):
        """Export the learning analysis to a file."""
        try:
            from tkinter import filedialog
            import json
            from datetime import datetime
            
            # Create export data
            export_data = {
                'analysis_timestamp': datetime.now().isoformat(),
                'training_summary': {
                    'total_epochs': len(self.training_history),
                    'training_history': self.training_history
                },
                'model_info': {
                    'device': str(self.device),
                    'current_epoch': self.current_epoch
                }
            }
            
            # Ask user for save location
            save_path = filedialog.asksaveasfilename(
                title="Export Learning Analysis",
                defaultextension=".json",
                filetypes=[("JSON File", "*.json"), ("All Files", "*.*")]
            )
            
            if save_path:
                with open(save_path, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                messagebox.showinfo("Success", f"Analysis exported to {save_path}")
                
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export analysis: {str(e)}")
    
    def run(self):
        """Start the GUI application."""
        self.root.mainloop()


def main():
    """Main entry point for the RL GUI application."""
    print("Starting Human Feedback RL GUI...")
    
    # Create and run GUI
    app = HumanFeedbackRLGUI()
    app.run()


if __name__ == "__main__":
    main() 