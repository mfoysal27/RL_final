#!/usr/bin/env python
"""
Headless Model Testing Script for Gut Tissue Segmentation
This version fixes the ND2 inference-only issue
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import json
import time
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F_transforms
import random
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components
from models.model_factory import ModelFactory
from models.hybrid_model import HybridUNetGNN
from core.data_handler import create_data_loaders, nd2_to_pil, get_nd2_frame_count
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
        
        print(f"🔍 Scanning for images in: {images_dir}")
        
        # Check if directory exists
        if not os.path.exists(images_dir):
            raise ValueError(f"Images directory does not exist: {images_dir}")
        
        # Collect all image files
        for root, _, files in os.walk(images_dir):
            print(f"   Checking directory: {root}")
            
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    full_path = os.path.join(root, file)
                    self.image_files.append(full_path)
                    print(f"   ✅ Added image: {file}")
        
        self.image_files.sort()  # Ensure consistent ordering
        print(f"📊 Total images found for inference: {len(self.image_files)}")
        
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
            
            # Store path for reference
            image_tensor.path = img_path
            
            return image_tensor
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return minimal fallback for raw mode
            blank_image = torch.zeros((1, 100, 100))  # Minimal size
            blank_image.path = img_path
            return blank_image

class ND2InferenceDataset(Dataset):
    """Specialized dataset for ND2 files in inference-only mode (no masks needed)"""
    
    def __init__(self, images_dir: str):
        """
        Args:
            images_dir: Path to directory containing ND2 files
        """
        self.images_dir = images_dir
        self.image_slices = []
        
        print(f"🔍 Processing ND2 files for inference-only mode...")
        self._process_nd2_files()
        
        print(f"📊 Total ND2 slices found: {len(self.image_slices)}")
        
        if len(self.image_slices) == 0:
            raise ValueError("No valid ND2 slices found")
    
    def _process_nd2_files(self):
        """Process ND2 files and extract slices without requiring masks"""
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
                    
                    # Process each extracted slice - NO MASK REQUIRED
                    for slice_idx in range(actual_frames):
                        slice_img = all_frames[slice_idx]
                        
                        if slice_img is not None:
                            # Add slice with metadata (no mask needed)
                            self.image_slices.append((slice_img, f"{base_name}_slice_{slice_idx}"))
                            print(f"   ✅ Added slice {slice_idx} (inference only)")
                        
            except Exception as e:
                print(f"  ❌ Error processing {nd2_path}: {e}")
                continue
    
    def __len__(self):
        return len(self.image_slices)
    
    def __getitem__(self, idx):
        if idx >= len(self.image_slices):
            raise IndexError(f"Index {idx} out of range. Dataset has {len(self.image_slices)} samples.")
        
        # Inference only mode
        slice_img, slice_name = self.image_slices[idx]
        
        try:
            # Process the slice image
            image = slice_img.convert("L")
            image = image.resize((256, 256), Image.BILINEAR)
            
            # Convert to tensor
            image_tensor = F_transforms.to_tensor(image)
            image_tensor.path = slice_name
            
            return image_tensor
            
        except Exception as e:
            print(f"❌ Error loading sample {idx}: {e}")
            blank_image = torch.zeros((1, 256, 256))
            blank_image.path = slice_name
            return blank_image

class HeadlessModelTester:
    """Headless model testing with no GUI dependencies"""
    
    def __init__(self, model_path: str, test_data_dir: str, output_dir: str = "test_results"):
        self.model_path = model_path
        self.test_data_dir = test_data_dir
        self.output_dir = output_dir
        self.has_ground_truth = False  # Always inference-only in headless mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load model and configuration
        self.model, self.config = self._load_model()
        self.num_classes = self.config.get('num_classes', get_num_tissue_classes())
        
        # Setup test data loader
        self.test_loader = self._setup_test_loader()
        
        # Initialize metrics storage
        self.predictions = []
        self.test_images = []
        
        print(f"📋 Model Tester Initialized")
        print(f"   Model: {os.path.basename(self.model_path)}")
        print(f"   Test Data: {self.test_data_dir}")
        print(f"   Output: {self.output_dir}")
        print(f"   Device: {self.device}")
        print(f"   Classes: {self.num_classes}")
        print(f"   Mode: Inference-only (no ground truth)")
    
    def _load_model(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Load trained model and configuration"""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Extract configuration
        config = checkpoint.get('config', {})
        model_type = checkpoint.get('model_type', config.get('model_type', 'unet'))
        num_classes = checkpoint.get('num_classes', config.get('num_classes', get_num_tissue_classes()))
        
        print(f"📋 Loading model: {model_type}")
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
        
        print(f"✅ Model loaded successfully")
        return model, config
    
    def _setup_test_loader(self):
        """Setup test data loader"""
        try:
            # Check if test data has proper structure
            images_dir = os.path.join(self.test_data_dir, 'images')
            
            if not os.path.exists(images_dir):
                # Try using the directory directly
                if os.path.exists(self.test_data_dir):
                    images_dir = self.test_data_dir
                    print(f"Using provided directory directly: {images_dir}")
                else:
                    raise ValueError(f"Test images directory not found: {images_dir}")
            
            print(f"📂 Test data structure:")
            print(f"   Images: ✅ {len(os.listdir(images_dir))} files")
            
            # Check if we have ND2 files
            nd2_files = [f for f in os.listdir(images_dir) if f.lower().endswith('.nd2')]
            regular_images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
            
            if nd2_files:
                print(f"   Detected {len(nd2_files)} ND2 files - using specialized ND2 processing")
                
                # Use specialized ND2 inference dataset - NO MASK REQUIRED
                nd2_dataset = ND2InferenceDataset(images_dir)
                
                test_loader = DataLoader(
                    nd2_dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=True
                )
                
                print(f"✅ Created ND2 test loader with {len(nd2_dataset)} samples")
                
            elif regular_images:
                print(f"   Detected {len(regular_images)} regular image files")
                
                # Use inference-only dataset for regular images
                print(f"🔧 Creating inference-only dataset...")
                inference_dataset = InferenceOnlyDataset(images_dir)
                
                if len(inference_dataset) == 0:
                    raise ValueError(f"No images found for inference.")
                
                print(f"✅ Created inference dataset with {len(inference_dataset)} samples")
                
                test_loader = DataLoader(
                    inference_dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=True
                )
            else:
                raise ValueError(f"No image files found in {images_dir}. Expected ND2 files or regular image files.")
            
            return test_loader
            
        except Exception as e:
            print(f"❌ Error setting up test loader: {e}")
            raise
    
    def run_inference(self, save_predictions: bool = True) -> Dict[str, Any]:
        """Run inference on test data"""
        print(f"\n🔍 Running inference on {len(self.test_loader)} test samples...")
        print(f"   Mode: Inference-only")
        
        self.model.eval()
        total_time = 0
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(self.test_loader, desc="Testing")):
                # Handle data
                if isinstance(batch_data, (list, tuple)):
                    images = batch_data[0]  # Take first element if it's wrapped
                else:
                    images = batch_data  # Direct tensor
                
                # Log image size for first batch
                if batch_idx == 0:
                    print(f"   📊 First batch image shape: {images.shape}")
                    print(f"      Image size: {images.shape[2]}x{images.shape[3]}")
                    if images.shape[2] != 256 or images.shape[3] != 256:
                        print(f"      ⚠️  Warning: Non-standard image size detected!")
                        print(f"         Some models may require specific input sizes")
                        print(f"         If inference fails, consider resizing images to 256x256")
                
                images = images.to(self.device)
                
                try:
                    # Measure inference time
                    start_time = time.time()
                    outputs = self.model(images)
                    inference_time = time.time() - start_time
                    total_time += inference_time
                    
                    # Get predictions
                    predictions = torch.argmax(outputs, dim=1)
                    
                    # Store results
                    for i in range(images.shape[0]):
                        img_np = images[i].cpu().numpy()
                        pred_np = predictions[i].cpu().numpy()
                        
                        # Keep original format for storage (don't convert single channel)
                        self.test_images.append(img_np)
                        self.predictions.append(pred_np)
                        
                        # Save individual prediction
                        if save_predictions:
                            self._save_prediction(
                                img_np, pred_np, None,
                                batch_idx * images.shape[0] + i
                            )
                            
                except RuntimeError as e:
                    if "size" in str(e).lower() or "dimension" in str(e).lower():
                        print(f"❌ Model inference failed due to image size incompatibility!")
                        print(f"   Error: {e}")
                        print(f"   💡 Solution: Resize your images to 256x256 or check model requirements")
                        raise
                    else:
                        print(f"❌ Model inference failed: {e}")
                        raise
        
        avg_inference_time = total_time / len(self.test_loader)
        
        results = {
            'total_samples': len(self.predictions),
            'has_ground_truth': False,
            'avg_inference_time': avg_inference_time,
            'total_time': total_time
        }
        
        print(f"✅ Inference completed!")
        print(f"   Samples processed: {results['total_samples']}")
        print(f"   Average inference time: {avg_inference_time:.4f}s")
        print(f"   Total time: {total_time:.2f}s")
        
        return results
    
    def _save_prediction(self, image: np.ndarray, prediction: np.ndarray, 
                        ground_truth: Optional[np.ndarray], sample_idx: int):
        """Save individual prediction visualization"""
        output_path = os.path.join(self.output_dir, f"prediction_{sample_idx:04d}.png")
        
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
        
        # Save prediction only
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Input image
        axes[0].imshow(vis_image, cmap=cmap)
        axes[0].set_title(f'Input Image ({vis_image.shape[1]}x{vis_image.shape[0]})')
        axes[0].axis('off')
        
        # Apply tissue colors to prediction
        tissue_colors = get_all_tissue_colors()
        colored_pred = np.zeros((*prediction.shape, 3), dtype=np.uint8)
        for class_id, color in tissue_colors.items():
            mask = prediction == class_id
            colored_pred[mask] = color
        
        axes[1].imshow(colored_pred)
        axes[1].set_title(f'Prediction ({prediction.shape[1]}x{prediction.shape[0]})')
        axes[1].axis('off')
        
        plt.suptitle(f"Test Sample {sample_idx}")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, inference_results: Dict[str, Any]):
        """Generate inference report"""
        print(f"\n📋 Generating inference report...")
        
        report_path = os.path.join(self.output_dir, "inference_report.json")
        
        # Create report
        report = {
            'test_info': {
                'model_path': self.model_path,
                'test_data_dir': self.test_data_dir,
                'output_dir': self.output_dir,
                'test_date': datetime.now().isoformat(),
                'device': str(self.device),
                'num_classes': self.num_classes,
                'has_ground_truth': False
            },
            'inference_results': inference_results,
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
        
        # Create class legend
        legend_path = os.path.join(self.output_dir, "class_legend.png")
        create_class_legend(legend_path)
        
        print(f"✅ Inference report generated!")
        print(f"   Report: {report_path}")
        print(f"   Visualizations: {self.output_dir}")
        
        return report_path
    
    def run_complete_test(self, save_predictions: bool = True) -> str:
        """Run complete inference pipeline"""
        print(f"\n🚀 Starting Headless Inference")
        print("=" * 50)
        
        # Run inference
        inference_results = self.run_inference(save_predictions=save_predictions)
        
        # Generate report
        report_path = self.generate_report(inference_results)
        
        print(f"\n✅ Complete inference finished!")
        print(f"📂 Results saved to: {self.output_dir}")
        
        return report_path

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Headless Model Testing')
    parser.add_argument('--model', type=str, required=True, help='Path to model file (.pth)')
    parser.add_argument('--data', type=str, required=True, help='Path to test data directory')
    parser.add_argument('--output', type=str, default='test_results', help='Output directory')
    parser.add_argument('--no-save', action='store_true', help='Do not save individual predictions')
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    # Validate arguments
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    if not os.path.exists(args.data):
        print(f"Error: Test data directory not found: {args.data}")
        return
    
    # Create tester
    tester = HeadlessModelTester(args.model, args.data, args.output)
    
    # Run test
    tester.run_complete_test(save_predictions=not args.no_save)

if __name__ == "__main__":
    main() 