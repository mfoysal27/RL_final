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
    
    def __init__(self, model: nn.Module, learning_rate: float = 1e-4):
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
    
    def update_model_with_feedback(self, images: List[torch.Tensor], 
                                 predictions: List[torch.Tensor],
                                 feedback_scores: List[float]):
        """Update model based on human feedback using policy gradient."""
        if not feedback_scores:
            return
        
        self.model.train()
        total_loss = 0
        gradient_norms = []
        
        # Store state before update
        pre_update_state = {name: param.clone().detach() for name, param in self.model.named_parameters()}
        
        for img, pred, score in zip(images, predictions, feedback_scores):
            # Convert feedback score to reward (-1 to +1)
            reward = torch.tensor(score, dtype=torch.float32, device=pred.device)
            
            # Calculate policy gradient loss
            # Higher rewards should increase probability of current actions
            log_prob = torch.log_softmax(pred, dim=1)
            policy_loss = -torch.mean(log_prob * reward)
            
            # Add regularization to prevent overfitting to human feedback
            l2_reg = 0.001 * sum(p.pow(2.0).sum() for p in self.model.parameters())
            
            loss = policy_loss + l2_reg
            total_loss += loss.item()
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Calculate gradient norm before clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            gradient_norms.append(grad_norm.item())
            
            self.optimizer.step()
        
        # Calculate learning metrics
        avg_loss = total_loss / len(feedback_scores)
        # Safe gradient norm calculation
        avg_grad_norm = sum(gradient_norms) / len(gradient_norms) if gradient_norms else 0.0
        
        # Calculate weight change magnitude
        weight_change = self._calculate_weight_change(pre_update_state)
        
        # Update learning metrics
        self.learning_metrics['loss_history'].append(avg_loss)
        self.learning_metrics['gradient_norms'].append(avg_grad_norm)
        self.learning_metrics['weight_changes'].append(weight_change)
        # Safe feedback mean calculation
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
    
    def select_random_samples(self, dataset, num_samples: int = 3) -> List[Tuple[int, Any]]:
        """Select random samples from dataset for human feedback."""
        indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
        samples = []
        for idx in indices:
            try:
                sample = dataset[idx]
                samples.append((idx, sample))
            except Exception as e:
                print(f"Error loading sample {idx}: {e}")
                continue
        return samples


class HumanFeedbackRLGUI:
    """Main GUI for Human Feedback Reinforcement Learning."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Human Feedback RL - Gut Tissue Segmentation")
        self.root.geometry("1400x700")
        
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
        """Setup the main GUI interface."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Human Feedback RL Training", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Control panel
        self.setup_control_panel(main_frame)
        
        # Training status
        self.setup_status_panel(main_frame)
        
        # Image feedback area
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
        """Setup individual image with dual feedback sliders."""
        # Image container
        image_frame = ttk.Frame(parent)
        image_frame.grid(row=0, column=col_idx, padx=10, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Image label
        img_label = ttk.Label(image_frame, text=f"Image {col_idx + 1}", 
                             background="lightgray", width=25)
        img_label.grid(row=0, column=0, pady=(0, 10))
        self.image_labels.append(img_label)
        
        # === CLASS IDENTIFICATION SLIDER ===
        class_frame = ttk.LabelFrame(image_frame, text="🎨 Class Identification", padding="5")
        class_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # Class feedback slider
        class_slider_frame = ttk.Frame(class_frame)
        class_slider_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        ttk.Label(class_slider_frame, text="Poor").grid(row=0, column=0, padx=(0, 5))
        
        class_slider = tk.Scale(class_slider_frame, from_=-100, to=100, orient=tk.HORIZONTAL,
                               length=180, resolution=1, 
                               bg="#E3F2FD", troughcolor="#1976D2")  # Blue theme
        class_slider.set(0)
        class_slider.grid(row=0, column=1, padx=2)
        self.class_feedback_sliders.append(class_slider)
        
        ttk.Label(class_slider_frame, text="Perfect").grid(row=0, column=2, padx=(5, 0))
        
        # Class description
        class_desc = ttk.Label(class_frame, 
                              text="Rate tissue type accuracy (villi, glands, muscle, etc.)",
                              font=('Arial', 8), foreground="blue")
        class_desc.grid(row=1, column=0, pady=(2, 0))
        
        # === BORDER/REGION IDENTIFICATION SLIDER ===
        border_frame = ttk.LabelFrame(image_frame, text="🎯 Border & Region Accuracy", padding="5")
        border_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # Border feedback slider
        border_slider_frame = ttk.Frame(border_frame)
        border_slider_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        ttk.Label(border_slider_frame, text="Fuzzy").grid(row=0, column=0, padx=(0, 5))
        
        border_slider = tk.Scale(border_slider_frame, from_=-100, to=100, orient=tk.HORIZONTAL,
                                length=180, resolution=1,
                                bg="#E8F5E8", troughcolor="#388E3C")  # Green theme
        border_slider.set(0)
        border_slider.grid(row=0, column=1, padx=2)
        self.border_feedback_sliders.append(border_slider)
        
        ttk.Label(border_slider_frame, text="Sharp").grid(row=0, column=2, padx=(5, 0))
        
        # Border description
        border_desc = ttk.Label(border_frame, 
                               text="Rate boundary sharpness and region completeness",
                               font=('Arial', 8), foreground="green")
        border_desc.grid(row=1, column=0, pady=(2, 0))
        
        # === FEEDBACK GUIDANCE ===
        guidance_frame = ttk.Frame(image_frame)
        guidance_frame.grid(row=3, column=0, pady=(10, 0))
        
        guidance_text = """
🎨 Class: Are tissues correctly identified?
🎯 Border: Are boundaries clean and complete?
        """
        guidance_label = ttk.Label(guidance_frame, text=guidance_text, 
                                  font=('Arial', 7), justify="center")
        guidance_label.grid(row=0, column=0)
    
    def setup_color_legend(self, parent):
        """Setup color legend showing tissue classes and their colors."""
        legend_frame = ttk.LabelFrame(parent, text="Tissue Color Legend", padding="5")
        legend_frame.grid(row=0, column=3, padx=10, pady=5, sticky=(tk.N, tk.S))
        
        # Use globally available functions
        num_classes = get_num_tissue_classes()
        
        # Create scrollable frame for legend
        canvas = tk.Canvas(legend_frame, width=200, height=300)
        scrollbar = ttk.Scrollbar(legend_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Add tissue classes with colors
        for class_id in range(min(15, num_classes)):  # Limit to prevent UI overflow
            color = get_tissue_color(class_id)
            name = get_tissue_name(class_id)
            
            # Create color swatch and label
            class_frame = ttk.Frame(scrollable_frame)
            class_frame.pack(fill=tk.X, pady=1)
            
            # Color swatch
            color_label = tk.Label(class_frame, 
                                 text="  ", 
                                 bg=f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}",
                                 width=3, height=1)
            color_label.pack(side=tk.LEFT, padx=(0, 5))
            
            # Class name
            name_label = ttk.Label(class_frame, 
                                 text=f"{class_id}: {name.replace('_', ' ')}", 
                                 font=('Arial', 8))
            name_label.pack(side=tk.LEFT)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def setup_opacity_control(self, parent):
        """Setup opacity control slider."""
        opacity_frame = ttk.LabelFrame(parent, text="Opacity Control", padding="10")
        opacity_frame.grid(row=4, column=0, columnspan=3, pady=10)
        
        ttk.Label(opacity_frame, text="Overlay Opacity:").grid(row=0, column=0, padx=(0, 10))
        ttk.Label(opacity_frame, text="Original").grid(row=0, column=1, padx=(0, 5))
        
        # Opacity slider
        self.opacity_slider = tk.Scale(opacity_frame, from_=0, to=1, orient=tk.HORIZONTAL,
                                     length=200, resolution=0.01, variable=self.opacity_var,
                                     command=self.on_opacity_change)
        self.opacity_slider.set(0.5)  # Default to 50% opacity
        self.opacity_slider.grid(row=0, column=2, padx=5)
        
        ttk.Label(opacity_frame, text="Overlay").grid(row=0, column=3, padx=(5, 10))
        
        # Refresh button
        ttk.Button(opacity_frame, text="Refresh Display", 
                  command=self.refresh_display).grid(row=0, column=4, padx=10)
    
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
        """Show dialog for user to select model type."""
        
        # Create model selection window
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Model Architecture")
        dialog.geometry("600x500")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (600 // 2)
        y = (dialog.winfo_screenheight() // 2) - (500 // 2)
        dialog.geometry(f"600x500+{x}+{y}")
        
        # Variable to store selected model
        selected_model = tk.StringVar()
        result = {"model_type": None}
        
        # Title
        title_label = tk.Label(dialog, text="🏗️ Select Model Architecture", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Instructions
        instruction_label = tk.Label(dialog, 
                                    text="Choose the architecture that matches your trained model:",
                                    font=("Arial", 10))
        instruction_label.pack(pady=5)
        
        # Get available models
        try:
            available_models = ModelFactory.get_available_models()
        except:
            # Fallback if ModelFactory fails
            available_models = {
                'unet': 'UNet - Classic architecture (Fast, Low Memory)',
                'simple_unet': 'SimpleUNet - Basic implementation'
            }
        
        # Create radio buttons for each model
        radio_frame = tk.Frame(dialog)
        radio_frame.pack(pady=20, padx=20, fill="both", expand=True)
        
        for model_type, description in available_models.items():
            radio_btn = tk.Radiobutton(
                radio_frame,
                text=f"🔹 {description}",
                variable=selected_model,
                value=model_type,
                font=("Arial", 11),
                wraplength=550,
                justify="left",
                anchor="w"
            )
            radio_btn.pack(anchor="w", pady=8, padx=10)
        
        # Set default selection to unet
        if 'unet' in available_models:
            selected_model.set('unet')
        elif available_models:
            selected_model.set(list(available_models.keys())[0])
        
        # Buttons
        button_frame = tk.Frame(dialog)
        button_frame.pack(pady=20)
        
        def on_ok():
            if selected_model.get():
                result["model_type"] = selected_model.get()
            dialog.destroy()
        
        def on_cancel():
            result["model_type"] = None
            dialog.destroy()
        
        ok_btn = tk.Button(button_frame, text="✅ Load Model", command=on_ok,
                          bg="#4CAF50", fg="white", font=("Arial", 10, "bold"),
                          width=12)
        ok_btn.pack(side="left", padx=10)
        
        cancel_btn = tk.Button(button_frame, text="❌ Cancel", command=on_cancel,
                              bg="#f44336", fg="white", font=("Arial", 10),
                              width=12)
        cancel_btn.pack(side="left", padx=10)
        
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
        """Save the RL-enhanced model."""
        if not self.model:
            messagebox.showwarning("Warning", "No model to save!")
            return
        
        save_path = filedialog.asksaveasfilename(
            title="Save RL Enhanced Model",
            defaultextension=".pth",
            filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")]
        )
        
        if save_path:
            try:
                # Create enhanced checkpoint
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
        """Show learning comparison between epochs - FIXED VERSION."""
        if not self.training_history:
            messagebox.showwarning("Warning", "No training history to compare!")
            return
        
        # Create a new window for comparison
        comparison_window = tk.Toplevel(self.root)
        comparison_window.title("📊 Learning Progress Analysis")
        comparison_window.geometry("900x600")
        comparison_window.configure(bg="white")
        
        # Main frame
        main_frame = ttk.Frame(comparison_window, padding="20")
        main_frame.pack(fill="both", expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="📊 Reinforcement Learning Progress", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Create text widget with scrollbar
        text_frame = ttk.Frame(main_frame)
        text_frame.pack(fill="both", expand=True)
        
        analysis_text = tk.Text(text_frame, font=("Consolas", 11), wrap=tk.WORD, 
                               bg="#f8f9fa", relief="solid", borderwidth=1)
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=analysis_text.yview)
        analysis_text.configure(yscrollcommand=scrollbar.set)
        
        analysis_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Generate analysis content
        self._show_detailed_analysis(analysis_text)
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x", pady=(15, 0))
        
        # Export button
        ttk.Button(button_frame, text="📄 Export Analysis", 
                  command=lambda: self._export_learning_analysis()).pack(side="left")
        
        # Close button
        ttk.Button(button_frame, text="✖ Close", 
                  command=comparison_window.destroy).pack(side="right")
        
        # Make text read-only
        analysis_text.config(state=tk.DISABLED)
    
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