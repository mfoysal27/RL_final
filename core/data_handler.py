import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import random
from sklearn.cluster import KMeans
import warnings
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import shutil

# Add proper error handling for nd2reader import
try:
    from nd2reader import ND2Reader
    print('✅ nd2reader imported successfully in data_handler')
    ND2_AVAILABLE = True
except ImportError as e:
    print(f'⚠️ nd2reader import failed in data_handler: {e}')
    print('   Trying alternative nd2 library...')
    try:
        import nd2
        print('✅ Using nd2 library as alternative in data_handler')
        ND2_AVAILABLE = True
        # Create a wrapper class to maintain compatibility
        class ND2Reader:
            def __init__(self, path):
                self.path = path
                self._nd2_file = None
            def __enter__(self):
                self._nd2_file = nd2.imread(self.path)
                # Create a mock object with sizes attribute for compatibility
                self.sizes = {'z': len(self._nd2_file) if hasattr(self._nd2_file, '__len__') else 1}
                return self
            def __exit__(self, *args):
                pass
            def __len__(self):
                return len(self._nd2_file) if hasattr(self._nd2_file, '__len__') else 1
            def __iter__(self):
                if hasattr(self._nd2_file, '__iter__'):
                    return iter(self._nd2_file)
                else:
                    return iter([self._nd2_file])
            def __getitem__(self, idx):
                if hasattr(self._nd2_file, '__getitem__'):
                    return self._nd2_file[idx]
                elif idx == 0:
                    return self._nd2_file
                else:
                    raise IndexError("Index out of range")
    except ImportError:
        print('❌ No ND2 libraries available in data_handler. ND2 file support disabled.')
        ND2_AVAILABLE = False
        ND2Reader = None
except Exception as e:
    print(f'❌ Unexpected error importing nd2reader in data_handler: {e}')
    ND2_AVAILABLE = False
    ND2Reader = None

import io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import rgb2hex
from PIL import ImageDraw, ImageFont
from core.tissue_config import (
    get_tissue_color, get_tissue_name, get_all_tissue_colors, 
    get_all_tissue_names, get_num_tissue_classes, 
    load_custom_tissue_config, save_tissue_config_template
)
import atexit

# ============================================================================
# TEMPORARY DIRECTORY CLEANUP UTILITIES
# ============================================================================

class TempConvertedCleaner:
    """Utility class to track and clean up temp_converted directories"""
    
    temp_dirs = set()  # Class variable to track all temp directories
    
    @classmethod
    def register_temp_dir(cls, temp_dir):
        """Register a temp directory for cleanup"""
        cls.temp_dirs.add(temp_dir)
        print(f"📁 Registered temp directory for cleanup: {temp_dir}")
    
    @classmethod
    def cleanup_temp_dir(cls, temp_dir):
        """Clean up a specific temp directory"""
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"🗑️  Cleaned up temp directory: {temp_dir}")
                cls.temp_dirs.discard(temp_dir)
                return True
            except Exception as e:
                print(f"❌ Failed to cleanup temp directory {temp_dir}: {e}")
                return False
        return True
    
    @classmethod
    def cleanup_all_temp_dirs(cls):
        """Clean up all registered temp directories"""
        print(f"🧹 Cleaning up {len(cls.temp_dirs)} temp directories...")
        cleaned_count = 0
        for temp_dir in list(cls.temp_dirs):  # Use list() to avoid modification during iteration
            if cls.cleanup_temp_dir(temp_dir):
                cleaned_count += 1
        print(f"✅ Successfully cleaned {cleaned_count} temp directories")
    
    @classmethod
    def get_temp_dir_info(cls):
        """Get information about current temp directories"""
        if not cls.temp_dirs:
            return "No temp directories currently registered"
        
        info = f"Current temp directories ({len(cls.temp_dirs)}):\n"
        for temp_dir in cls.temp_dirs:
            if os.path.exists(temp_dir):
                try:
                    size = sum(os.path.getsize(os.path.join(temp_dir, f)) 
                             for f in os.listdir(temp_dir) if os.path.isfile(os.path.join(temp_dir, f)))
                    file_count = len([f for f in os.listdir(temp_dir) if os.path.isfile(os.path.join(temp_dir, f))])
                    info += f"  📁 {temp_dir}: {file_count} files, {size/1024/1024:.1f} MB\n"
                except:
                    info += f"  📁 {temp_dir}: (access error)\n"
            else:
                info += f"  📁 {temp_dir}: (already deleted)\n"
                cls.temp_dirs.discard(temp_dir)
        
        return info

# Register cleanup function to run at program exit
atexit.register(TempConvertedCleaner.cleanup_all_temp_dirs)

class SegmentationDataset(Dataset):
    def __init__(self, data_dir, split='train', num_classes=None, class_names=None, augment=True):
        """
        Args:
            data_dir (str): Path to the data directory
            split (str): 'train', 'val', or 'test'
            num_classes (int): Number of classes for segmentation (if None, auto-detect)
            class_names (list): List of class names (if None, auto-detect)
            augment (bool): Whether to apply data augmentation (for training only)
        """
        self.data_dir = data_dir
        self.split = split
        self.num_classes = num_classes
        self.class_names = class_names
        self.augment = augment and split == 'train'  # Only augment training data
        
        # Always use image-mask structure
        self.using_class_folders = False
        self._setup_from_images_masks(data_dir, split, num_classes)
            
        print(f"Found {len(self.image_files)} images for {split} split")
    
    def _setup_from_images_masks(self, data_dir, split, num_classes):
        """Setup dataset from images/masks structure with ND2 support"""
        # Use simple structure: data_dir/images/ and data_dir/masks/
        images_dir = os.path.join(data_dir, 'images')
        masks_dir = os.path.join(data_dir, 'masks')
        
        print(f"Using simple structure: {data_dir}/{{images,masks}}/")
        
        # Create directories if they don't exist
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)
        
        # Check if directories exist
        if not os.path.exists(images_dir):
            raise ValueError(f"Images directory not found: {images_dir}")
        
        if not os.path.exists(masks_dir):
            raise ValueError(f"Masks directory not found: {masks_dir}")
        
        # Collect image files and their corresponding masks
        self.image_files = []
        self.mask_files = []
        
        # First, process ND2 files and extract ALL slices
        self._process_nd2_files_with_masks(images_dir, masks_dir, split)
        
        # Then, process regular image files
        self._process_regular_image_files(images_dir, masks_dir)
        
        # Sort for deterministic ordering
        combined_files = list(zip(self.image_files, self.mask_files))
        combined_files.sort(key=lambda x: x[0])  # Sort by image path
        self.image_files, self.mask_files = zip(*combined_files) if combined_files else ([], [])
        self.image_files = list(self.image_files)
        self.mask_files = list(self.mask_files)
        
        # If no images found, create placeholders
        if not self.image_files:
            print(f"Warning: No images found in {images_dir}. Creating placeholder images for demonstration.")
            self._create_placeholder_data(images_dir, masks_dir)
        
        # Determine the number of classes from the masks
        if num_classes is None:
            print("Determining number of classes from masks...")
            self.num_classes = self._detect_num_classes_from_files(self.mask_files)
            print(f"Detected {self.num_classes} classes from masks")
        else:
            self.num_classes = num_classes
        
        # Create a color map for the classes
        self.color_map = self._create_color_map(self.num_classes)
        
        # Use tissue configuration names instead of generic names
        self.class_names = [get_tissue_name(i) for i in range(self.num_classes)]
        
        print(f"Using {self.num_classes} tissue classes: {', '.join(self.class_names)}")
    
    def _process_nd2_files_with_masks(self, images_dir, masks_dir, split):
        """Process ND2 files and extract ALL slices with corresponding segmented masks"""
        print(f"Processing ND2 files in {images_dir}...")
        
        # Create temp directory for converted ND2 slices
        temp_converted_dir = os.path.join(os.path.dirname(images_dir), "temp_converted")
        os.makedirs(temp_converted_dir, exist_ok=True)
        
        # Register for cleanup
        TempConvertedCleaner.register_temp_dir(temp_converted_dir)
        
        # Find all ND2 files in the images directory
        nd2_files = []
        for root, _, files in os.walk(images_dir):
            for file in files:
                if file.lower().endswith('.nd2'):
                    nd2_files.append(os.path.join(root, file))
        
        print(f"Found {len(nd2_files)} ND2 files")
        
        # Process each ND2 file
        for nd2_path in nd2_files:
            try:
                base_name = os.path.splitext(os.path.basename(nd2_path))[0]
                print(f"Processing {base_name}.nd2...")
                
                # Get frame count from ND2 file
                frame_count = get_nd2_frame_count(nd2_path)
                print(f"  Found {frame_count} slices in {base_name}.nd2")
                
                if frame_count > 0:
                    # Extract ALL frames (no limits)
                    selected_slices = list(range(frame_count))
                    print(f"  Extracting ALL {len(selected_slices)} slices from {base_name}.nd2")
                    
                    # Extract each slice
                    for slice_idx in selected_slices:
                        try:
                            # Extract slice from ND2 using the modified nd2_to_pil function
                            # Since nd2_to_pil now returns all frames, we need to get specific frame
                            all_frames = nd2_to_pil(nd2_path)
                            if all_frames and slice_idx < len(all_frames):
                                slice_img = all_frames[slice_idx]
                            else:
                                print(f"    ✗ Could not extract slice {slice_idx}")
                                continue
                            
                            if slice_img is not None:
                                # Save converted slice
                                slice_name = f"{base_name}_slice_{slice_idx:04d}.png"
                                slice_path = os.path.join(temp_converted_dir, slice_name)
                                slice_img.save(slice_path)
                                
                                # Look for corresponding segmented mask using the specific naming pattern
                                # Pattern: segmented_{base_name}_slice_{slice_idx} (1-indexed, no leading zeros, no extension in pattern)
                                mask_name = f"segmented_{base_name}_slice_{slice_idx + 1}"  # 1-indexed
                                
                                # Try different extensions for the mask
                                mask_path = None
                                for ext in ['.png', '.tif', '.tiff', '.jpg', '.jpeg']:
                                    potential_mask_path = os.path.join(masks_dir, mask_name + ext)
                                    if os.path.exists(potential_mask_path):
                                        mask_path = potential_mask_path
                                        break
                                
                                # Also try 0-indexed version as fallback
                                if mask_path is None:
                                    mask_name_alt = f"segmented_{base_name}_slice_{slice_idx}"  # 0-indexed
                                    for ext in ['.png', '.tif', '.tiff', '.jpg', '.jpeg']:
                                        potential_mask_path = os.path.join(masks_dir, mask_name_alt + ext)
                                        if os.path.exists(potential_mask_path):
                                            mask_path = potential_mask_path
                                            break
                                
                                # Also try with leading zeros as fallback
                                if mask_path is None:
                                    mask_name_zeros = f"segmented_{base_name}_slice_{slice_idx:04d}"
                                    for ext in ['.png', '.tif', '.tiff', '.jpg', '.jpeg']:
                                        potential_mask_path = os.path.join(masks_dir, mask_name_zeros + ext)
                                        if os.path.exists(potential_mask_path):
                                            mask_path = potential_mask_path
                                            break
                                
                                if mask_path and os.path.exists(mask_path):
                                    # Verify mask is valid
                                    try:
                                        with Image.open(mask_path) as test_mask:
                                            self.image_files.append(slice_path)
                                            self.mask_files.append(mask_path)
                                            print(f"    ✓ Added slice {slice_idx} (frame {slice_idx + 1}) with mask: {os.path.basename(mask_path)}")
                                    except Exception as e:
                                        print(f"    ✗ Invalid mask for slice {slice_idx}: {e}")
                                else:
                                    print(f"    ⚠ No corresponding mask found for slice {slice_idx}")
                                    print(f"      Expected patterns tried:")
                                    print(f"        - segmented_{base_name}_slice_{slice_idx + 1}.[png|tif|jpg]")
                                    print(f"        - segmented_{base_name}_slice_{slice_idx}.[png|tif|jpg]")
                                    print(f"        - segmented_{base_name}_slice_{slice_idx:04d}.[png|tif|jpg]")
                                    
                        except Exception as e:
                            print(f"    ✗ Error extracting slice {slice_idx}: {e}")
                            continue
                            
                else:
                    print(f"  ⚠ No frames detected in {base_name}.nd2")
                    
            except Exception as e:
                print(f"  ✗ Error processing {nd2_path}: {e}")
                continue
    
    def _process_regular_image_files(self, images_dir, masks_dir):
        """Process regular image files (PNG, JPEG, etc.) with corresponding masks"""
        print(f"Processing regular image files in {images_dir}...")
        
        regular_files = []
        for root, _, files in os.walk(images_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    regular_files.append(os.path.join(root, file))
        
        print(f"Found {len(regular_files)} regular image files")
        
        for img_path in regular_files:
            try:
                img_name = os.path.basename(img_path)
                base_name = os.path.splitext(img_name)[0]
                
                # Look for corresponding mask
                mask_path = self._find_corresponding_mask(masks_dir, base_name, img_name)
                
                if mask_path and os.path.exists(mask_path):
                    # Verify both image and mask are valid
                    try:
                        with Image.open(img_path) as test_img, Image.open(mask_path) as test_mask:
                            self.image_files.append(img_path)
                            self.mask_files.append(mask_path)
                            print(f"  ✓ Added {img_name} with mask")
                    except Exception as e:
                        print(f"  ✗ Invalid image/mask pair {img_name}: {e}")
                else:
                    print(f"  ⚠ No corresponding mask found for {img_name}")
                    
            except Exception as e:
                print(f"  ✗ Error processing {img_path}: {e}")
                continue
    
    def _find_corresponding_mask(self, masks_dir, base_name, img_name):
        """Find corresponding mask file for a given image"""
        # Try direct match first
        for ext in ['.png', '.tif', '.tiff', '.jpg', '.jpeg']:
            direct_mask_path = os.path.join(masks_dir, base_name + ext)
            if os.path.exists(direct_mask_path):
                return direct_mask_path
            
            # Try with original extension
            direct_mask_path = os.path.join(masks_dir, img_name)
            if os.path.exists(direct_mask_path):
                return direct_mask_path
        
        # Try segmented_ prefix patterns
        for ext in ['.png', '.tif', '.tiff', '.jpg', '.jpeg']:
            segmented_mask_path = os.path.join(masks_dir, f"segmented_{base_name}{ext}")
            if os.path.exists(segmented_mask_path):
                return segmented_mask_path
        
        # Handle ND2-style naming: image_slice_0000.png -> segmented_image_slice_1.png
        if "_slice_" in base_name:
            try:
                # Extract the parts: "name_slice_0000" -> ["name", "slice", "0000"]
                parts = base_name.split("_slice_")
                if len(parts) == 2:
                    image_base = parts[0]  # e.g., "simulated_nd2"
                    slice_num_str = parts[1]  # e.g., "0000"
                    
                    # Convert to integer and try different indexing schemes
                    slice_num = int(slice_num_str)
                    
                    # Try 1-indexed (most common)
                    mask_name_1indexed = f"segmented_{image_base}_slice_{slice_num + 1}"
                    for ext in ['.png', '.tif', '.tiff', '.jpg', '.jpeg']:
                        mask_path = os.path.join(masks_dir, mask_name_1indexed + ext)
                        if os.path.exists(mask_path):
                            return mask_path
                    
                    # Try 0-indexed
                    mask_name_0indexed = f"segmented_{image_base}_slice_{slice_num}"
                    for ext in ['.png', '.tif', '.tiff', '.jpg', '.jpeg']:
                        mask_path = os.path.join(masks_dir, mask_name_0indexed + ext)
                        if os.path.exists(mask_path):
                            return mask_path
                    
                    # Try with zero-padding maintained
                    mask_name_padded = f"segmented_{image_base}_slice_{slice_num_str}"
                    for ext in ['.png', '.tif', '.tiff', '.jpg', '.jpeg']:
                        mask_path = os.path.join(masks_dir, mask_name_padded + ext)
                        if os.path.exists(mask_path):
                            return mask_path
                    
                    # Try 1-indexed with zero-padding
                    mask_name_1indexed_padded = f"segmented_{image_base}_slice_{slice_num + 1:04d}"
                    for ext in ['.png', '.tif', '.tiff', '.jpg', '.jpeg']:
                        mask_path = os.path.join(masks_dir, mask_name_1indexed_padded + ext)
                        if os.path.exists(mask_path):
                            return mask_path
                            
            except (ValueError, IndexError):
                # If parsing fails, continue with other methods
                pass
        
        return None
    
    def _create_placeholder_data(self, images_dir, masks_dir):
        """Create placeholder images and masks for demonstration"""
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)
        
        # Create a simple placeholder ND2-style dataset
        for i in range(1, 4):  # Create 3 sample images
            for slice_idx in range(3):  # 3 slices each
                # Create image
                image = np.zeros((100, 100), dtype=np.uint8)
                mask = np.zeros((100, 100), dtype=np.uint8)
                
                # Set some random pixels in the image
                for _ in range(30):
                    x, y = np.random.randint(0, 100, 2)
                    val = np.random.randint(50, 255)
                    image[y, x] = val
                
                # Create a simple segmentation mask
                mask[20+slice_idx*5:40+slice_idx*5, 20+slice_idx*5:40+slice_idx*5] = 1  # Class 1
                mask[60-slice_idx*5:80-slice_idx*5, 60-slice_idx*5:80-slice_idx*5] = 2  # Class 2
                
                # Save the image and mask with ND2-style naming
                img_name = f'placeholder_{i}_slice_{slice_idx:04d}.png'
                mask_name = f'segmented_placeholder_{i}_slice_{slice_idx:04d}.png'
                
                img_path = os.path.join(images_dir, img_name)
                mask_path = os.path.join(masks_dir, mask_name)
                
                Image.fromarray(image).save(img_path)
                Image.fromarray(mask).save(mask_path)
                
                self.image_files.append(img_path)
                self.mask_files.append(mask_path)
        
        print(f"Created {len(self.image_files)} placeholder image-mask pairs")
    
    def _detect_num_classes_from_files(self, mask_files):
        """Use the configured number of tissue classes instead of detecting from pixel values"""
        # Use the centralized tissue configuration
        configured_num_classes = get_num_tissue_classes()
        print(f"Using configured number of tissue classes: {configured_num_classes}")
        return configured_num_classes
    
    def _create_color_map(self, num_classes):
        """Create a color map for visualization"""
        # Use centralized color configuration
        self.color_map = get_all_tissue_colors()
        self.class_names = get_all_tissue_names()
        
        # Add any missing classes with deterministic random colors
        for i in range(num_classes):
            if i not in self.color_map:
                self.color_map[i] = get_tissue_color(i)
        
        return self.color_map
    
    def __getitem__(self, idx):
        """Get an image and its mask using the images/masks directory structure with ND2 support"""
        return self._getitem_images_masks(idx)
    
    def _getitem_images_masks(self, idx):
        """Get an image and its mask using the images/masks directory structure with ND2 support"""
        img_path = self.image_files[idx]
        mask_path = self.mask_files[idx]
        
        # Load image and mask
        try:
            image = Image.open(img_path)
            mask = Image.open(mask_path)
        except Exception as e:
            print(f"Error loading image/mask pair {idx}: {e}")
            # Create fallback blank images
            image = Image.new('L', (256, 256), 0)
            mask = Image.new('L', (256, 256), 0)
        
        # Convert to the appropriate number of channels based on the model input
        # In this case, we're using a 1-channel (grayscale) model
        image = image.convert("L")  # 'L' mode is grayscale
        
        # Resize both image and mask to ensure consistent size
        image = image.resize((256, 256), Image.BILINEAR)
        mask = mask.resize((256, 256), Image.NEAREST)
        
        # Apply data augmentation if enabled (training mode)
        if self.augment:
            image, mask = apply_augmentation(image, mask)
            
            # Ensure images are still 256x256 after augmentation
            image = image.resize((256, 256), Image.BILINEAR)
            mask = mask.resize((256, 256), Image.NEAREST)
        
        # Convert mask to tensor of class indices
        mask_np = np.array(mask)
        
        # If mask is RGB, convert to class indices using tissue configuration
        if len(mask_np.shape) == 3 and mask_np.shape[2] == 3:
            print(f"Processing RGB mask with shape: {mask_np.shape}")
            
            # Get tissue color mapping from configuration
            tissue_colors = get_all_tissue_colors()
            height, width = mask_np.shape[:2]
            mask_classes = np.zeros((height, width), dtype=np.int64)
            
            print(f"Mapping RGB colors to {len(tissue_colors)} tissue classes...")
            
            # Map each tissue color to its class index
            for class_id, target_color in tissue_colors.items():
                # Find pixels matching this tissue color (with tolerance for compression artifacts)
                color_match = np.all(np.abs(mask_np - target_color) <= 10, axis=2)
                mask_classes[color_match] = class_id
                pixel_count = np.sum(color_match)
                if pixel_count > 0:
                    print(f"  Class {class_id} ({get_tissue_name(class_id)}): {pixel_count} pixels matching RGB{target_color}")
            
            mask_np = mask_classes
            print(f"Converted RGB mask to tissue class indices. Unique classes found: {np.unique(mask_np)}")
        
        # Ensure mask has values in the correct range
        if mask_np.max() >= self.num_classes:
            print(f"Warning: Mask has values greater than num_classes ({self.num_classes}). Clamping.")
            mask_np = np.clip(mask_np, 0, self.num_classes - 1)
        
        # Convert to tensors
        image_tensor = F.to_tensor(image)  # This will be [1, H, W] for grayscale
        mask_tensor = torch.tensor(mask_np, dtype=torch.long)
        
        # Store original image path for reference
        image_tensor.path = img_path
        
        return image_tensor, mask_tensor

    def __len__(self):
        return len(self.image_files)

    def get_num_classes(self):
        """Get the number of classes"""
        return self.num_classes
    
    def get_class_names(self):
        """Return the class names"""
        return self.class_names
    
    def cleanup_temp_dirs(self):
        """Clean up any temp_converted directories created during processing"""
        TempConvertedCleaner.cleanup_all_temp_dirs()
    
    def __del__(self):
        """Destructor to clean up temp directories when dataset is destroyed"""
        # Note: This may not always be called reliably in Python, 
        # so we also rely on the atexit cleanup
        pass

class RLDataset(Dataset):
    """
    Dataset class specifically for Reinforcement Learning with human feedback.
    This dataset doesn't require predefined labels as they'll be determined through human feedback.
    """
    def __init__(self, data_dir, transform=None, in_channels=1, image_size=(256, 256), split='train'):
        """
        Args:
            data_dir (str): Path to the data directory containing images
            transform (callable, optional): Optional transform to be applied on the images
            in_channels (int): Number of input channels (1 for grayscale, 3 for RGB)
            image_size (tuple): Size to resize images to (height, width)
            split (str): Dataset split - 'train' uses random frame selection, 'test'/'val' use sequential
        """
        self.data_dir = data_dir
        self.transform = transform
        self.in_channels = in_channels
        self.image_size = image_size
        self.split = split
        
        # Collect all image files
        self.image_files = []
        self.collect_images(data_dir)
        
        # Default transforms if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
            ])
        
        print(f"RLDataset: Found {len(self.image_files)} images in {data_dir} for {split} split")
    
    def collect_images(self, directory):
        """Collect all valid image files from the directory and subdirectories"""
        # Directories to exclude
        exclude_dirs = ['__pycache__', 'output', 'temp_converted', '.ipynb_checkpoints', '.git', 'models', 'results']
        
        # Determine if we're in training mode (random selection) or testing mode (sequential)
        is_training = self.split == 'train'
        mode = 'random_frames' if is_training else 'all_frames'
        print(f"RLDataset: Using {'random selection' if is_training else 'sequential processing'} for frames")
        
        # Walk through directory
        for root, dirs, files in os.walk(directory):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                # Check standard image formats
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    try:
                        img_path = os.path.join(root, file)
                        # Verify it's a valid image
                        with Image.open(img_path) as img:
                            self.image_files.append(img_path)
                    except Exception as e:
                        print(f"RLDataset: Skipping invalid image {file}: {e}")
                
                # Check for ND2 files
                elif file.lower().endswith('.nd2'):
                    try:
                        file_path = os.path.join(root, file)
                        # Get frame count from ND2 file
                        frame_count = get_nd2_frame_count(file_path)
                        
                        if frame_count > 0:
                            # Create temp directory for converted images
                            temp_dir = os.path.join(directory, "temp_converted")
                            os.makedirs(temp_dir, exist_ok=True)
                            
                            # Register for cleanup
                            TempConvertedCleaner.register_temp_dir(temp_dir)
                            
                            # Determine how many frames to extract based on training/testing mode
                            if is_training:
                                # For training: Random selection of frames, limited to prevent memory issues
                                max_frames = min(20, frame_count)
                                if frame_count <= max_frames:
                                    # If fewer frames than max, use all of them
                                    selected_frames = list(range(frame_count))
                                else:
                                    # Randomly select frames without replacement
                                    selected_frames = random.sample(range(frame_count), max_frames)
                                print(f"RLDataset: Selected {len(selected_frames)} random frames from {file}")
                            else:
                                # For validation/testing: Use sequential frames
                                selected_frames = list(range(frame_count))
                                print(f"RLDataset: Using all {frame_count} frames from {file}")
                            
                            # Process each frame
                            for frame_idx in selected_frames:
                                # Convert ND2 to PNG
                                img = nd2_to_pil(file_path, frame=frame_idx, mode='specific_frame')
                                if img is not None:
                                    # Save to temporary file
                                    base_name = os.path.splitext(os.path.basename(file_path))[0]
                                    temp_path = os.path.join(temp_dir, f"{base_name}_frame{frame_idx}.png")
                                    img.save(temp_path)
                                    self.image_files.append(temp_path)
                                    print(f"RLDataset: Converted {file_path} frame {frame_idx} to {temp_path}")
                        else:
                            # Fallback to single frame if frame count detection fails
                            img = nd2_to_pil(file_path)
                            if img is not None:
                                # Save to temporary file
                                base_name = os.path.splitext(os.path.basename(file_path))[0]
                                temp_dir = os.path.join(directory, "temp_converted")
                                os.makedirs(temp_dir, exist_ok=True)
                                
                                # Register for cleanup
                                TempConvertedCleaner.register_temp_dir(temp_dir)
                                
                                temp_path = os.path.join(temp_dir, f"{base_name}.png")
                                img.save(temp_path)
                                self.image_files.append(temp_path)
                                print(f"RLDataset: Converted {file_path} to {temp_path}")
                    except Exception as e:
                        print(f"RLDataset: Skipping invalid ND2 file {file}: {e}")
        
        # Sort image files for deterministic ordering (especially important for testing)
        self.image_files.sort()
    
    def __getitem__(self, idx):
        """Get an image without a predefined label"""
        img_path = self.image_files[idx]
        
        try:
            # Load image
            if img_path.lower().endswith('.nd2'):
                image = nd2_to_pil(img_path)
            else:
                image = Image.open(img_path)
            
            # Convert to grayscale if needed
            if self.in_channels == 1 and image.mode != 'L':
                image = image.convert('L')
            elif self.in_channels == 3 and image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            else:
                # Resize and convert to tensor
                image = image.resize(self.image_size)
                image = F.to_tensor(image)
            
            # For RL, we don't have a predefined label/mask
            # Return the image and its path for reference
            return image, img_path
        
        except Exception as e:
            print(f"RLDataset: Error loading image {img_path}: {e}")
            # Return a blank image as fallback
            if self.in_channels == 1:
                blank = torch.zeros((1, *self.image_size))
            else:
                blank = torch.zeros((3, *self.image_size))
            return blank, img_path
    
    def __len__(self):
        """Return the number of images in the dataset"""
        return len(self.image_files)
    
    def get_image_path(self, idx):
        """Get the path of an image by index"""
        return self.image_files[idx]
    
    def cleanup_temp_dirs(self):
        """Clean up any temp_converted directories created during processing"""
        TempConvertedCleaner.cleanup_all_temp_dirs()
    
    def get_temp_dir_info(self):
        """Get information about current temp directories"""
        return TempConvertedCleaner.get_temp_dir_info()
    
    def __del__(self):
        """Destructor to clean up temp directories when dataset is destroyed"""
        # Note: This may not always be called reliably in Python, 
        # so we also rely on the atexit cleanup
        pass


def select_dataset_location():
    """Open folder dialog to select dataset location"""
    root = tk.Tk()
    root.withdraw()
    
    folder_path = filedialog.askdirectory(
        title="Select Dataset Location"
    )
    
    if not folder_path:
        print("No folder selected.")
        return None
    
    print(f"Selected dataset location: {folder_path}")
    return folder_path

def prepare_dataset_splits(dataset_path, output_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Prepare dataset splits from a folder structure organized by class
    
    Args:
        dataset_path: Path to the dataset with class folders
        output_path: Path to save the organized dataset
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        test_ratio: Ratio of data for testing
    
    Returns:
        Dictionary with class names and counts
    """
    # Ensure ratios sum to 1
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-5:
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Create split directories
    train_dir = os.path.join(output_path, 'train')
    val_dir = os.path.join(output_path, 'val')
    test_dir = os.path.join(output_path, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Get class folders
    class_folders = [d for d in os.listdir(dataset_path) 
                    if os.path.isdir(os.path.join(dataset_path, d))]
    
    if not class_folders:
        raise ValueError(f"No class folders found in {dataset_path}")
    
    print(f"Found {len(class_folders)} class folders: {', '.join(class_folders)}")
    
    class_info = {}
    
    # Process each class folder
    for class_name in class_folders:
        class_path = os.path.join(dataset_path, class_name)
        
        # Create class folder in each split
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)
        
        # Get all image files in this class
        image_files = []
        for root, _, files in os.walk(class_path):
            for file in files:
                file_path = os.path.join(root, file)
                
                # Check if it's a supported image file
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    try:
                        with Image.open(file_path) as img:
                            # If we can open it, it's a valid image
                            image_files.append((file_path, None))  # (path, conversion_function)
                    except Exception as e:
                        print(f"Skipping invalid image file {file_path}: {e}")
                
                # Handle ND2 files
                elif file.lower().endswith('.nd2'):
                    try:
                        # Get frame count
                        frame_count = get_nd2_frame_count(file_path)
                        
                        if frame_count > 0:
                            # Determine frames to extract based on the split
                            for split_type, split_dir in [('train', train_class_dir), ('val', val_class_dir), ('test', test_class_dir)]:
                                # Determine number of frames to extract based on split type
                                if split_type == 'train':
                                    # For training: use random selection
                                    max_frames_per_split = min(10, frame_count)  # Limit frames per split for training
                                    if frame_count <= max_frames_per_split:
                                        # If fewer frames than max, use all of them
                                        frames_for_this_split = list(range(frame_count))
                                    else:
                                        # Randomly select frames without replacement
                                        random.seed(42 + hash(file_path + split_type))  # Ensure deterministic selection
                                        frames_for_this_split = random.sample(range(frame_count), max_frames_per_split)
                                    print(f"Selected {len(frames_for_this_split)} random frames from {file} for {split_type}")
                                else:
                                    # For validation and test: use sequential frames
                                    # Divide remaining frames evenly between val and test
                                    if split_type == 'val':
                                        start_idx = 0
                                        end_idx = frame_count // 2
                                    else:  # test
                                        start_idx = frame_count // 2
                                        end_idx = frame_count
                                    
                                    frames_for_this_split = list(range(start_idx, end_idx))
                                    print(f"Using {len(frames_for_this_split)} sequential frames ({start_idx}-{end_idx-1}) from {file} for {split_type}")
                                
                                # Process each frame for this split
                                for frame_idx in frames_for_this_split:
                                    # Convert ND2 to PNG
                                    img = nd2_to_pil(file_path, frame=frame_idx, mode='specific_frame')
                                    if img is not None:
                                        # Save to the appropriate split directory
                                        base_name = os.path.splitext(os.path.basename(file_path))[0]
                                        dest_file = os.path.join(split_dir, f"{base_name}_frame{frame_idx}.png")
                                        print(f"Converting {file_path} frame {frame_idx} to {dest_file}")
                                        img.save(dest_file)
                    except Exception as e:
                        print(f"Skipping invalid ND2 file {file_path}: {e}")
        
        print(f"Found {len(image_files)} images in class {class_name}")
        
        if not image_files:
            print(f"Warning: No valid image files found in class folder {class_name}")
            continue
        
        # Shuffle files
        random.shuffle(image_files)
        
        # Calculate split indices
        n_files = len(image_files)
        n_train = max(1, int(n_files * train_ratio))
        n_val = max(1, int(n_files * val_ratio))
        
        # Ensure we don't exceed the number of files
        if n_train + n_val > n_files:
            n_train = max(1, n_files - n_val)
        
        train_files = image_files[:n_train]
        val_files = image_files[n_train:n_train+n_val]
        test_files = image_files[n_train+n_val:]
        
        # Copy files to respective directories
        for item in train_files:
            if len(item) == 2:
                file_path, conversion_func = item
                frame_idx = 0  # Default frame
            else:
                file_path, conversion_func, frame_idx = item
                
            # Generate output filename
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            if conversion_func:
                # For ND2 files, convert to PNG
                if frame_idx > 0:
                    dest_file = os.path.join(train_class_dir, f"{base_name}_frame{frame_idx}.png")
                else:
                    dest_file = os.path.join(train_class_dir, f"{base_name}.png")
                print(f"Converting {file_path} frame {frame_idx} to {dest_file}")
                img = conversion_func(file_path, frame=frame_idx)
                if img:
                    img.save(dest_file)
            else:
                # For regular image files, just copy
                dest_file = os.path.join(train_class_dir, os.path.basename(file_path))
                print(f"Copying {file_path} to {dest_file}")
                shutil.copy2(file_path, dest_file)
        
        for item in val_files:
            if len(item) == 2:
                file_path, conversion_func = item
                frame_idx = 0  # Default frame
            else:
                file_path, conversion_func, frame_idx = item
                
            # Generate output filename
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            if conversion_func:
                # For ND2 files, convert to PNG
                if frame_idx > 0:
                    dest_file = os.path.join(val_class_dir, f"{base_name}_frame{frame_idx}.png")
                else:
                    dest_file = os.path.join(val_class_dir, f"{base_name}.png")
                print(f"Converting {file_path} frame {frame_idx} to {dest_file}")
                img = conversion_func(file_path, frame=frame_idx)
                if img:
                    img.save(dest_file)
            else:
                # For regular image files, just copy
                dest_file = os.path.join(val_class_dir, os.path.basename(file_path))
                print(f"Copying {file_path} to {dest_file}")
                shutil.copy2(file_path, dest_file)
        
        for item in test_files:
            if len(item) == 2:
                file_path, conversion_func = item
                frame_idx = 0  # Default frame
            else:
                file_path, conversion_func, frame_idx = item
                
            # Generate output filename
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            if conversion_func:
                # For ND2 files, convert to PNG
                if frame_idx > 0:
                    dest_file = os.path.join(test_class_dir, f"{base_name}_frame{frame_idx}.png")
                else:
                    dest_file = os.path.join(test_class_dir, f"{base_name}.png")
                print(f"Converting {file_path} frame {frame_idx} to {dest_file}")
                img = conversion_func(file_path, frame=frame_idx)
                if img:
                    img.save(dest_file)
            else:
                # For regular image files, just copy
                dest_file = os.path.join(test_class_dir, os.path.basename(file_path))
                print(f"Copying {file_path} to {dest_file}")
                shutil.copy2(file_path, dest_file)
        
        # Store class information
        class_info[class_name] = {
            'total': n_files,
            'train': len(train_files),
            'val': len(val_files),
            'test': len(test_files)
        }
        
        print(f"Class {class_name}: {n_files} images split into {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
    
    return class_info

def create_data_loaders(data_dir, batch_size=8, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Create data loaders for training, validation and testing from simple structure
    
    Args:
        data_dir: Path to data directory (images/ and masks/ folders)
        batch_size: Batch size for data loaders
        train_ratio: Ratio of data for training (default: 0.8)
        val_ratio: Ratio of data for validation (default: 0.1)
        test_ratio: Ratio of data for testing (default: 0.1)
    
    Returns:
        train_loader, val_loader, test_loader: DataLoader objects
    """
    # Check if data directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    print(f"Using simple image-mask structure from {data_dir}")
    
    # ✅ FIX 1: Validate and normalize ratios to sum to 1.0
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-5:
        print(f"⚠️ Warning: Ratios sum to {total_ratio:.4f}, normalizing to 1.0")
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio
        print(f"✅ Normalized ratios: train={train_ratio:.3f}, val={val_ratio:.3f}, test={test_ratio:.3f}")
    
    # ✅ FIX 2: Create a temporary dataset to get file lists, then split BEFORE creating real datasets
    print("🔍 Analyzing available data...")
    temp_dataset = SegmentationDataset(data_dir=data_dir, split='temp', augment=False)
    
    if len(temp_dataset) == 0:
        raise ValueError("No valid datasets found. Please check your data directory structure.")
    
    # Get total size and calculate split sizes
    total_size = len(temp_dataset)
    train_size = max(1, int(total_size * train_ratio))  # ✅ FIX 3: Ensure minimum size of 1
    val_size = max(1, int(total_size * val_ratio))
    test_size = max(1, total_size - train_size - val_size)  # Remaining data goes to test
    
    # ✅ FIX 4: Validate split sizes
    if train_size + val_size + test_size > total_size:
        # Adjust splits if they exceed total (can happen with very small datasets)
        if total_size >= 3:
            train_size = max(1, total_size - 2)
            val_size = 1
            test_size = 1
        else:
            train_size = total_size
            val_size = 0
            test_size = 0
            print(f"⚠️ Warning: Very small dataset ({total_size} samples). Using all for training.")
    
    print(f"Dataset split: {train_size} train, {val_size} validation, {test_size} test samples")
    
    # ✅ FIX 5: Split file indices, then create separate datasets with correct augmentation
    torch.manual_seed(42)  # For reproducible splits
    indices = list(range(total_size))
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:train_size+val_size+test_size]
    
    # Get the file lists from temp dataset
    train_image_files = [temp_dataset.image_files[i] for i in train_indices]
    train_mask_files = [temp_dataset.mask_files[i] for i in train_indices]
    
    val_image_files = [temp_dataset.image_files[i] for i in val_indices] if val_size > 0 else []
    val_mask_files = [temp_dataset.mask_files[i] for i in val_indices] if val_size > 0 else []
    
    test_image_files = [temp_dataset.image_files[i] for i in test_indices] if test_size > 0 else []
    test_mask_files = [temp_dataset.mask_files[i] for i in test_indices] if test_size > 0 else []
    
    # Clean up temp dataset
    temp_dataset.cleanup_temp_dirs()
    del temp_dataset
    
    # ✅ FIX 6: Create separate datasets with correct split labels and augmentation
    print("🔨 Creating split-specific datasets...")
    
    # Training dataset with augmentation
    train_dataset = SegmentationDataset(data_dir=data_dir, split='train', augment=True)
    train_dataset.image_files = train_image_files
    train_dataset.mask_files = train_mask_files
    print(f"✅ Training dataset: {len(train_dataset)} samples with augmentation=True")
    
    # Validation dataset without augmentation
    val_dataset = None
    if val_size > 0:
        val_dataset = SegmentationDataset(data_dir=data_dir, split='val', augment=False)
        val_dataset.image_files = val_image_files
        val_dataset.mask_files = val_mask_files
        print(f"✅ Validation dataset: {len(val_dataset)} samples with augmentation=False")
    
    # Test dataset without augmentation
    test_dataset = None
    if test_size > 0:
        test_dataset = SegmentationDataset(data_dir=data_dir, split='test', augment=False)
        test_dataset.image_files = test_image_files
        test_dataset.mask_files = test_mask_files
        print(f"✅ Test dataset: {len(test_dataset)} samples with augmentation=False")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 workers to avoid Windows permission issues
        pin_memory=True
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Use 0 workers to avoid Windows permission issues
            pin_memory=True
        )
    
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Use 0 workers to avoid Windows permission issues
            pin_memory=True
        )
    
    # Get class information from the training dataset
    num_classes = train_dataset.get_num_classes()
    class_names = train_dataset.get_class_names()
    
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    
    # ✅ FIX 7: Verify augmentation settings
    print(f"🔍 Augmentation verification:")
    print(f"  Train augmentation: {train_dataset.augment}")
    if val_dataset:
        print(f"  Val augmentation: {val_dataset.augment}")
    if test_dataset:
        print(f"  Test augmentation: {test_dataset.augment}")
    
    return train_loader, val_loader, test_loader, num_classes, class_names

def get_nd2_frame_count(nd2_path):
    """
    Get the number of frames/slices in an ND2 file - Enhanced version
    
    Args:
        nd2_path: Path to the ND2 file
        
    Returns:
        int: Number of frames/slices in the file
    """
    # Check if ND2 support is available
    if not ND2_AVAILABLE or ND2Reader is None:
        print(f"⚠️ ND2 support not available. Cannot process {nd2_path}")
        print("   Install nd2reader with: pip install nd2reader")
        return 0
    
    try:
        print('trying nd2')
        with ND2Reader(nd2_path) as images:
            print('working nd2')
            # Check which axes exist
            available_axes = list(images.sizes.keys())
            print(f"ND2 frame count - Available axes: {available_axes}")
            print(f"ND2 frame count - Sizes: {images.sizes}")
            
            # Strategy 1: Count by direct iteration (most reliable)
            print("🔍 Counting frames by direct iteration...")
            actual_frame_count = 0
            try:
                for i, frame in enumerate(images):
                    if frame is not None and hasattr(frame, 'shape'):
                        actual_frame_count += 1
                        if i < 5:  # Log first few for debugging
                            print(f"   Frame {i}: {frame.shape}")
                    else:
                        print(f"   Frame {i}: Invalid or null frame - stopping count")
                        break  # Stop if we hit a null frame
                
                # Successful completion of iteration
                if actual_frame_count > 0:
                    print(f"✅ Direct iteration found {actual_frame_count} actual frames")
                    return actual_frame_count
                else:
                    print("⚠️ No valid frames found by direct iteration")
                    
            except Exception as iter_error:
                # If we got some frames before the error, that's still success
                if actual_frame_count > 0:
                    print(f"✅ Direct iteration found {actual_frame_count} actual frames (stopped at iterator end)")
                    return actual_frame_count
                else:
                    print(f"⚠️ Direct iteration failed: {iter_error}")
            
            # Strategy 2: Use metadata with sanity checks
            print("📊 Falling back to metadata analysis...")
            
            # Get z-stacks count (most common for biological imaging)
            n_z_stacks = images.sizes.get('z', 1)
            
            # If no z-axis or only 1 z-stack, check for time series
            if 'z' not in available_axes or n_z_stacks <= 1:
                n_time_points = images.sizes.get('t', 1)
                if 't' in available_axes and n_time_points > 1:
                    print(f"ND2 file metadata suggests {n_time_points} time points")
                    return n_time_points
            
            # Sanity check: if metadata suggests > 100 frames, be suspicious
            if n_z_stacks > 100:
                print(f"⚠️ Metadata suggests {n_z_stacks} frames - this seems high, using conservative estimate")
                return min(50, n_z_stacks)  # Cap at 50
            
            print(f"ND2 file metadata suggests {n_z_stacks} z-stacks")
            return n_z_stacks
            
    except Exception as e:
        print(f"Error reading ND2 file frame count {nd2_path}: {e}")
        print(f"Assuming 1 frame as fallback")
        return 1

def nd2_to_pil(nd2_path):
    """
    Convert an ND2 file to a list of PIL images (all frames) - Enhanced version
    
    Args:
        nd2_path: Path to the ND2 file
    
    Returns:
        list of PIL.Image: All frames extracted from the ND2 file
    """
    # Check if ND2 support is available
    if not ND2_AVAILABLE or ND2Reader is None:
        print(f"⚠️ ND2 support not available. Cannot process {nd2_path}")
        print("   Install nd2reader with: pip install nd2reader")
        return []
    
    try:
        print(f"🔍 Opening ND2 file: {nd2_path}")
        
        # Open the ND2 file
        with ND2Reader(nd2_path) as images:
            # Get basic information
            available_axes = list(images.sizes.keys())
            n_z_stacks = images.sizes.get('z', 1)
            n_channels = images.sizes.get('c', 1)
            n_time_points = images.sizes.get('t', 1)
            
            print(f"📊 ND2 file analysis:")
            print(f"   Available axes: {available_axes}")
            print(f"   File sizes: {images.sizes}")
            print(f"   Z-stacks: {n_z_stacks}, Channels: {n_channels}, Time: {n_time_points}")
            print(f"   Total length: {len(images)}")
            
            # Determine extraction strategy
            total_frames = len(images)
            if total_frames == 0:
                print("❌ No frames found in ND2 file")
                return []
            
            print(f"🎯 Attempting to extract {total_frames} frames...")
            
            # List to hold all extracted images
            all_images = []
            successful_extractions = 0
            
            # Strategy 1: Direct iteration (most reliable)
            print("📋 Strategy 1: Direct iteration through ND2 file")
            try:
                for i, frame in enumerate(images):
                    try:
                        # Ensure the frame is a valid numpy array
                        if frame is not None and hasattr(frame, 'shape'):
                            # Convert to 8-bit if necessary
                            processed_frame = _process_nd2_frame(frame, i)
                            if processed_frame is not None:
                                all_images.append(processed_frame)
                                successful_extractions += 1
                                if i % 5 == 0 or i < 10:  # Log first 10 and every 5th
                                    print(f"   ✅ Frame {i}: {processed_frame.size}")
                            else:
                                print(f"   ⚠️ Frame {i}: Processing failed")
                        else:
                            print(f"   ❌ Frame {i}: Invalid frame data - stopping iteration")
                            break  # Stop if we hit invalid frame
                    except Exception as frame_error:
                        print(f"   ❌ Frame {i}: {frame_error} - stopping iteration")
                        break  # Stop on any error to avoid infinite loops
                
                if successful_extractions > 0:
                    print(f"✅ Strategy 1 SUCCESS: {successful_extractions} actual frames extracted")
                    print(f"   Note: Metadata suggested {total_frames} frames, but actual count is {successful_extractions}")
                    return all_images
                else:
                    print("❌ Strategy 1 FAILED: No frames extracted")
                    
            except Exception as strategy1_error:
                # If we got some frames before the error, that's still success
                if successful_extractions > 0:
                    print(f"✅ Strategy 1 SUCCESS: {successful_extractions} actual frames extracted (stopped at iterator end)")
                    print(f"   Note: Metadata suggested {total_frames} frames, but actual count is {successful_extractions}")
                    return all_images
                else:
                    print(f"❌ Strategy 1 FAILED: {strategy1_error}")
            
            # Strategy 2: Index-based extraction with coordinate management
            print("📋 Strategy 2: Index-based extraction with coordinates")
            all_images.clear()
            successful_extractions = 0
            
            try:
                # Reset coordinates to defaults
                if 'z' in available_axes:
                    images.default_coords['z'] = 0
                if 'c' in available_axes:
                    images.default_coords['c'] = 0
                if 't' in available_axes:
                    images.default_coords['t'] = 0
                
                for frame_idx in range(total_frames):
                    frame_data = None
                    
                    # Try different indexing approaches
                    for approach in ['direct', 'z_coord', 't_coord', 'iteration']:
                        try:
                            if approach == 'direct':
                                frame_data = images[frame_idx]
                            elif approach == 'z_coord' and 'z' in available_axes:
                                images.default_coords['z'] = frame_idx % n_z_stacks
                                frame_data = images[0]
                            elif approach == 't_coord' and 't' in available_axes:
                                images.default_coords['t'] = frame_idx % n_time_points
                                frame_data = images[0]
                            elif approach == 'iteration':
                                # Manual iteration to specific index
                                for i, frame in enumerate(images):
                                    if i == frame_idx:
                                        frame_data = frame
                                        break
                            
                            if frame_data is not None:
                                processed_frame = _process_nd2_frame(frame_data, frame_idx)
                                if processed_frame is not None:
                                    all_images.append(processed_frame)
                                    successful_extractions += 1
                                    if frame_idx < 5:  # Log first few
                                        print(f"   ✅ Frame {frame_idx} ({approach}): {processed_frame.size}")
                                    break
                        except Exception as approach_error:
                            if frame_idx < 5:  # Only log errors for first few frames
                                print(f"   ⚠️ Frame {frame_idx} ({approach}): {approach_error}")
                            continue
                    
                    if frame_data is None:
                        if frame_idx < 10:  # Only log for first 10 frames
                            print(f"   ❌ Frame {frame_idx}: All approaches failed")
                
                if successful_extractions > 0:
                    print(f"✅ Strategy 2 SUCCESS: {successful_extractions}/{total_frames} frames extracted")
                    return all_images
                else:
                    print("❌ Strategy 2 FAILED: No frames extracted")
                    
            except Exception as strategy2_error:
                print(f"❌ Strategy 2 FAILED: {strategy2_error}")
            
            # Strategy 3: Single frame fallback
            print("📋 Strategy 3: Single frame fallback")
            try:
                # Try to get at least one frame
                first_frame = images[0]
                if first_frame is not None:
                    processed_frame = _process_nd2_frame(first_frame, 0)
                    if processed_frame is not None:
                        print(f"✅ Strategy 3 SUCCESS: Single frame extracted {processed_frame.size}")
                        return [processed_frame]
                    
            except Exception as strategy3_error:
                print(f"❌ Strategy 3 FAILED: {strategy3_error}")
            
            print("❌ ALL STRATEGIES FAILED: Could not extract any frames")
            return []
            
    except Exception as e:
        print(f"❌ CRITICAL ERROR reading ND2 file {nd2_path}: {e}")
        import traceback
        traceback.print_exc()
        return []

def _process_nd2_frame(frame, frame_idx):
    """Process a single ND2 frame into a PIL image"""
    try:
        if frame is None:
            return None
            
        # Ensure it's a numpy array
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)
        
        # Handle different data types
        if frame.dtype == np.uint8:
            # Already 8-bit, good to go
            processed_data = frame
        else:
            # Convert to 8-bit
            if frame.dtype in [np.uint16, np.uint32, np.float32, np.float64]:
                # Normalize to 0-255 range
                frame_min = np.min(frame)
                frame_max = np.max(frame)
                
                if frame_max > frame_min:
                    normalized = (frame - frame_min) / (frame_max - frame_min)
                    processed_data = (normalized * 255).astype(np.uint8)
                else:
                    # All values are the same
                    processed_data = np.full_like(frame, 128, dtype=np.uint8)
            else:
                # Unknown dtype, try direct conversion
                processed_data = frame.astype(np.uint8)
        
        # Handle multi-dimensional arrays
        if len(processed_data.shape) > 2:
            # Take first channel if multi-channel
            if processed_data.shape[2] > 1:  # RGB or multi-channel
                processed_data = processed_data[:, :, 0]
            elif len(processed_data.shape) == 3 and processed_data.shape[0] == 1:
                processed_data = processed_data[0, :, :]
            elif len(processed_data.shape) == 3:
                processed_data = processed_data[:, :, 0]
        
        # Ensure 2D
        if len(processed_data.shape) != 2:
            print(f"⚠️ Frame {frame_idx}: Unexpected shape {processed_data.shape}, attempting reshape")
            # Try to reshape to 2D
            if processed_data.size > 0:
                side = int(np.sqrt(processed_data.size))
                if side * side == processed_data.size:
                    processed_data = processed_data.reshape((side, side))
                else:
                    # Fallback: create a square image
                    side = 256
                    processed_data = np.resize(processed_data, (side, side))
            else:
                # Create a blank image
                processed_data = np.zeros((256, 256), dtype=np.uint8)
        
        # Create PIL image
        pil_img = Image.fromarray(processed_data, mode='L')
        return pil_img
        
    except Exception as e:
        print(f"❌ Error processing frame {frame_idx}: {e}")
        # Return a blank image as fallback
        return Image.new('L', (256, 256), 0)

def visualize_class_legends(dataset):
    """Create visualization of class legends for the training phase"""
    # Get class names and color map from centralized configuration
    class_names = get_all_tissue_names()
    color_map = get_all_tissue_colors()
    
    # Create a legend image
    width, height = 400, 50 * len(class_names)
    legend_img = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(legend_img)
    
    # Try to load a font
    try:
        # Try to load a system font
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        # Fallback to default font
        font = ImageFont.load_default()
    
    # Draw each class with its color
    for i, (idx, class_name) in enumerate(class_names.items()):
        y_pos = i * 50 + 10
        
        # Get color for this class
        color = tuple(get_tissue_color(idx))
        
        # Draw color box
        draw.rectangle([(20, y_pos), (60, y_pos + 30)], fill=color)
        
        # Draw class name
        draw.text((80, y_pos + 5), class_name, fill=(0, 0, 0), font=font)
    
    return legend_img

def save_class_legends(dataset, output_path):
    """
    Save class legends visualization to a file
    
    Args:
        dataset: SegmentationDataset instance
        output_path: Path to save the legend image
    """
    legend_img = visualize_class_legends(dataset)
    legend_img.save(output_path)
    print(f"Saved class legends to {output_path}")
    
    return legend_img

def create_rl_data_loader(data_dir, batch_size=1, in_channels=1, image_size=(256, 256), shuffle=True, split='train'):
    """
    Create a DataLoader for reinforcement learning with human feedback
    
    Args:
        data_dir (str): Path to data directory
        batch_size (int): Batch size for DataLoader
        in_channels (int): Number of input channels
        image_size (tuple): Size to resize images to
        shuffle (bool): Whether to shuffle the data
        split (str): Dataset split - 'train' uses random frame selection, 'test'/'val' use sequential
    
    Returns:
        DataLoader: DataLoader for RL training
        RLDataset: The dataset instance
    """
    # Create dataset
    dataset = RLDataset(data_dir, in_channels=in_channels, image_size=image_size, split=split)
    
    # Create DataLoader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Using 0 for easier debugging
        pin_memory=True
    )
    
    return data_loader, dataset

def generate_synthetic_data(data_dir, target_class, num_samples=10):
    """
    Generate synthetic data for underrepresented classes to balance the dataset
    
    Args:
        data_dir: Path to the data directory
        target_class: Name of the underrepresented class
        num_samples: Number of synthetic samples to generate
    """
    # Create class directories if they don't exist
    train_dir = os.path.join(data_dir, 'train', target_class)
    val_dir = os.path.join(data_dir, 'val', target_class)
    test_dir = os.path.join(data_dir, 'test', target_class)
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Generate synthetic data for training
    for i in range(num_samples):
        # Create a synthetic image with random patterns
        image_size = (256, 256)
        # Create a blank image with random noise
        synthetic_img = np.random.randint(50, 200, image_size, dtype=np.uint8)
        
        # Add some structures - random shapes that might represent the class features
        num_shapes = np.random.randint(3, 8)
        
        for _ in range(num_shapes):
            # Random shape center
            center_x = np.random.randint(20, image_size[0] - 20)
            center_y = np.random.randint(20, image_size[1] - 20)
            
            # Random shape size
            shape_size = np.random.randint(10, 40)
            
            # Random intensity
            intensity = np.random.randint(150, 250)
            
            # Draw shape (either a filled circle or rectangle)
            if np.random.random() < 0.5:
                # Draw filled circle
                y, x = np.ogrid[:image_size[0], :image_size[1]]
                dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                mask = dist <= shape_size
                synthetic_img[mask] = intensity
            else:
                # Draw filled rectangle
                x1 = max(0, center_x - shape_size)
                y1 = max(0, center_y - shape_size)
                x2 = min(image_size[1], center_x + shape_size)
                y2 = min(image_size[0], center_y + shape_size)
                synthetic_img[y1:y2, x1:x2] = intensity
        
        # Save the synthetic image
        img_pil = Image.fromarray(synthetic_img)
        img_path = os.path.join(train_dir, f"synthetic_{target_class}_{i+1}.png")
        img_pil.save(img_path)
        
        # Generate corresponding mask (all pixels belong to the target class)
        mask = np.ones(image_size, dtype=np.uint8)
        # Save the mask
        mask_dir = os.path.join(data_dir, 'train', 'masks')
        os.makedirs(mask_dir, exist_ok=True)
        mask_pil = Image.fromarray(mask)
        mask_path = os.path.join(mask_dir, f"synthetic_{target_class}_{i+1}.png")
        mask_pil.save(mask_path)
        
        print(f"Generated synthetic image for class {target_class}: {img_path}")
    
    # Copy a subset to validation and test sets
    val_samples = min(2, num_samples)
    test_samples = min(2, num_samples)
    
    for i in range(val_samples):
        # Copy to validation
        src_img = os.path.join(train_dir, f"synthetic_{target_class}_{i+1}.png")
        dst_img = os.path.join(val_dir, f"synthetic_{target_class}_{i+1}.png")
        shutil.copy2(src_img, dst_img)
        
        # Copy mask
        src_mask = os.path.join(data_dir, 'train', 'masks', f"synthetic_{target_class}_{i+1}.png")
        dst_mask_dir = os.path.join(data_dir, 'val', 'masks')
        os.makedirs(dst_mask_dir, exist_ok=True)
        dst_mask = os.path.join(dst_mask_dir, f"synthetic_{target_class}_{i+1}.png")
        shutil.copy2(src_mask, dst_mask)
    
    for i in range(val_samples, val_samples + test_samples):
        if i < num_samples:
            # Copy to test
            src_img = os.path.join(train_dir, f"synthetic_{target_class}_{i+1}.png")
            dst_img = os.path.join(test_dir, f"synthetic_{target_class}_{i+1}.png")
            shutil.copy2(src_img, dst_img)
            
            # Copy mask
            src_mask = os.path.join(data_dir, 'train', 'masks', f"synthetic_{target_class}_{i+1}.png")
            dst_mask_dir = os.path.join(data_dir, 'test', 'masks')
            os.makedirs(dst_mask_dir, exist_ok=True)
            dst_mask = os.path.join(dst_mask_dir, f"synthetic_{target_class}_{i+1}.png")
            shutil.copy2(src_mask, dst_mask)
    
    print(f"Generated and distributed {num_samples} synthetic samples for class {target_class}")


def apply_augmentation(img, mask=None, target_size=(256, 256)):
    """
    Apply data augmentation techniques to an image and its mask    
    Args:
        img: PIL Image to augment
        mask: Optional mask to apply the same transformations
        target_size: Target image size (width, height) to ensure consistency
    
    Returns:
        Augmented image and mask
    """
    # Ensure input images are the right size
    if img.size != target_size:
        img = img.resize(target_size, Image.BILINEAR)
    if mask is not None and mask.size != target_size:
        mask = mask.resize(target_size, Image.NEAREST)
    
    # Apply horizontal flip with 50% probability
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if mask is not None:
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    
    # Adjust brightness (50% probability)
    if random.random() < 0.5:
        brightness_factor = random.uniform(0.5, 3)
        img = transforms.functional.adjust_brightness(img, brightness_factor)
    
    # Adjust contrast (50% probability)
    if random.random() < 0.5:
        contrast_factor = random.uniform(0.5, 3)
        img = transforms.functional.adjust_contrast(img, contrast_factor)
    
    # Add random noise (30% probability)
    if random.random() < 0.3:
        # Convert to numpy array
        img_array = np.array(img)
        noise_level = 10
        
        # Create noise with the same shape as the image
        if len(img_array.shape) == 2:  # Grayscale
            noise = np.random.randint(-noise_level, noise_level, img_array.shape, dtype=np.int16)
        else:  # RGB
            noise = np.random.randint(-noise_level, noise_level, img_array.shape, dtype=np.int16)
        
        # Add noise to the image
        noisy_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(noisy_array)
    
    # Random crop and resize (40% probability)
    if random.random() < 0.4:
        # Get image dimensions
        width, height = img.size
        
        # Randomly select a crop area between 60% and 90% of original size
        scale = random.uniform(0.6, 0.9)
        crop_width = int(scale * width)
        crop_height = int(scale * height)
        
        # Select a random position for the crop
        left = random.randint(0, width - crop_width) if width > crop_width else 0
        top = random.randint(0, height - crop_height) if height > crop_height else 0
        right = left + crop_width
        bottom = top + crop_height
        
        # Crop the image
        img = img.crop((left, top, right, bottom))
        img = img.resize(target_size, Image.BILINEAR)
        
        # Apply same crop to mask if provided
        if mask is not None:
            mask = mask.crop((left, top, right, bottom))
            mask = mask.resize(target_size, Image.NEAREST)
    
    # Final check to ensure images are the correct size
    if img.size != target_size:
        img = img.resize(target_size, Image.BILINEAR)
    if mask is not None and mask.size != target_size:
        mask = mask.resize(target_size, Image.NEAREST)
    
    return img, mask


def random_resized_crop(img, mask=None, target_size=(256, 256)):
    """Perform a random resized crop transformation"""
    width, height = img.size
    
    # Randomly select a crop area between 60% and 100% of original size
    scale = random.uniform(0.6, 1.0)
    crop_width = int(scale * width)
    crop_height = int(scale * height)
    
    # Select a random position for the crop
    left = random.randint(0, width - crop_width) if width > crop_width else 0
    top = random.randint(0, height - crop_height) if height > crop_height else 0
    right = left + crop_width
    bottom = top + crop_height
    
    # Crop the image
    cropped_img = img.crop((left, top, right, bottom))
    
    # Resize back to target size
    resized_img = cropped_img.resize(target_size, Image.BICUBIC)
    
    # Apply the same transformation to mask if provided
    resized_mask = None
    if mask is not None:
        cropped_mask = mask.crop((left, top, right, bottom))
        resized_mask = cropped_mask.resize(target_size, Image.NEAREST)
    
    return resized_img, resized_mask


def add_random_noise(img, noise_level=10):
    """Add random noise to an image"""
    # Convert to numpy array
    img_array = np.array(img)
    
    # Create noise with the same shape as the image
    if img_array.ndim == 2:  # Grayscale
        noise = np.random.randint(-noise_level, noise_level, img_array.shape, dtype=np.int16)
    else:  # RGB
        noise = np.random.randint(-noise_level, noise_level, img_array.shape, dtype=np.int16)
    
    # Add noise to the image
    noisy_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Convert back to PIL Image
    noisy_img = Image.fromarray(noisy_array)
    return noisy_img


def balance_dataset(data_dir, class_counts):
    """
    Balance the dataset by adding synthetic data for underrepresented classes
    
    Args:
        data_dir: Path to the data directory
        class_counts: Dictionary with class name as key and count as value
    """
    if not class_counts:
        print("No class counts provided, skipping dataset balancing")
        return
    
    # Find the class with the most samples
    max_samples = max(class_counts.values())
    
    # For each underrepresented class, generate synthetic data
    for class_name, count in class_counts.items():
        if count < max_samples:
            samples_to_add = max_samples - count
            print(f"Class {class_name} is underrepresented with {count} samples (max: {max_samples})")
            print(f"Adding {samples_to_add} synthetic samples to balance the dataset")
            generate_synthetic_data(data_dir, class_name, samples_to_add)

def create_nd2_dataset_structure(base_dir, splits=['train', 'val', 'test']):
    """
    Create the proper directory structure for ND2-based datasets with train/val/test splits
    
    Args:
        base_dir (str): Base directory for the dataset
        splits (list): List of split names to create
    
    Returns:
        dict: Dictionary with paths to created directories
    """
    structure = {}
    
    # Create base directory
    os.makedirs(base_dir, exist_ok=True)
    
    # Create split directories
    for split in splits:
        split_dir = os.path.join(base_dir, split)
        images_dir = os.path.join(split_dir, 'images')
        masks_dir = os.path.join(split_dir, 'masks')
        
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)
        
        structure[split] = {
            'split_dir': split_dir,
            'images_dir': images_dir,
            'masks_dir': masks_dir
        }
    
    print(f"Created ND2 dataset structure at {base_dir}")
    for split in splits:
        print(f"  {split}:")
        print(f"    Images: {structure[split]['images_dir']}")
        print(f"    Masks:  {structure[split]['masks_dir']}")
    
    return structure

def validate_nd2_dataset(data_dir, check_masks=True):
    """
    Validate an ND2-based dataset structure and report statistics
    
    Args:
        data_dir (str): Path to the dataset directory
        check_masks (bool): Whether to check for corresponding masks
    
    Returns:
        dict: Validation results and statistics
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'statistics': {}
    }
    
    if not os.path.exists(data_dir):
        results['valid'] = False
        results['errors'].append(f"Dataset directory not found: {data_dir}")
        return results
    
    # Check for split directories
    splits = ['train', 'val', 'test']
    found_splits = []
    
    for split in splits:
        split_dir = os.path.join(data_dir, split)
        if os.path.exists(split_dir):
            found_splits.append(split)
            
            # Check images and masks directories
            images_dir = os.path.join(split_dir, 'images')
            masks_dir = os.path.join(split_dir, 'masks')
            
            if not os.path.exists(images_dir):
                results['errors'].append(f"Images directory not found: {images_dir}")
                results['valid'] = False
            
            if check_masks and not os.path.exists(masks_dir):
                results['errors'].append(f"Masks directory not found: {masks_dir}")
                results['valid'] = False
            
            # Count files in each directory
            if os.path.exists(images_dir):
                nd2_files = [f for f in os.listdir(images_dir) if f.lower().endswith('.nd2')]
                png_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
                
                results['statistics'][split] = {
                    'nd2_files': len(nd2_files),
                    'regular_images': len(png_files),
                    'total_source_files': len(nd2_files) + len(png_files)
                }
                
                # Estimate total slices from ND2 files
                total_estimated_slices = 0
                for nd2_file in nd2_files:
                    try:
                        nd2_path = os.path.join(images_dir, nd2_file)
                        frame_count = get_nd2_frame_count(nd2_path)
                        total_estimated_slices += frame_count
                    except Exception as e:
                        results['warnings'].append(f"Could not read {nd2_file}: {e}")
                
                results['statistics'][split]['estimated_total_slices'] = total_estimated_slices
                
                # Check for corresponding masks if requested
                if check_masks and os.path.exists(masks_dir):
                    mask_files = [f for f in os.listdir(masks_dir) if f.startswith('segmented_')]
                    results['statistics'][split]['segmented_masks'] = len(mask_files)
                    
                    # Check mask naming convention
                    valid_mask_names = 0
                    for mask_file in mask_files:
                        if '_slice_' in mask_file:
                            valid_mask_names += 1
                        else:
                            results['warnings'].append(f"Mask file doesn't follow naming convention: {mask_file}")
                    
                    results['statistics'][split]['valid_mask_names'] = valid_mask_names
    
    if not found_splits:
        results['errors'].append(f"No split directories found in {data_dir}")
        results['valid'] = False
    else:
        results['statistics']['found_splits'] = found_splits
    
    return results

def print_dataset_validation_report(validation_results):
    """
    Print a formatted validation report for an ND2 dataset
    
    Args:
        validation_results (dict): Results from validate_nd2_dataset()
    """
    print("\n" + "="*60)
    print("ND2 DATASET VALIDATION REPORT")
    print("="*60)
    
    if validation_results['valid']:
        print("✅ Dataset structure is VALID")
    else:
        print("❌ Dataset structure has ERRORS")
    
    # Print errors
    if validation_results['errors']:
        print("\n🚨 ERRORS:")
        for error in validation_results['errors']:
            print(f"  • {error}")
    
    # Print warnings
    if validation_results['warnings']:
        print("\n⚠️  WARNINGS:")
        for warning in validation_results['warnings']:
            print(f"  • {warning}")
    
    # Print statistics
    if validation_results['statistics']:
        print("\n📊 DATASET STATISTICS:")
        stats = validation_results['statistics']
        
        if 'found_splits' in stats:
            print(f"  Found splits: {', '.join(stats['found_splits'])}")
        
        for split, split_stats in stats.items():
            if split != 'found_splits':
                print(f"\n  {split.upper()} Split:")
                print(f"    ND2 files: {split_stats.get('nd2_files', 0)}")
                print(f"    Regular images: {split_stats.get('regular_images', 0)}")
                print(f"    Estimated total slices: {split_stats.get('estimated_total_slices', 0)}")
                if 'segmented_masks' in split_stats:
                    print(f"    Segmented masks: {split_stats['segmented_masks']}")
                    print(f"    Valid mask names: {split_stats.get('valid_mask_names', 0)}")
    
    print("\n" + "="*60)

def extract_nd2_slices_preview(nd2_path, output_dir, max_slices=5):
    """
    Extract a preview of slices from an ND2 file to help understand the data
    
    Args:
        nd2_path (str): Path to the ND2 file
        output_dir (str): Directory to save preview slices
        max_slices (int): Maximum number of slices to extract for preview
    
    Returns:
        list: List of extracted slice paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(nd2_path))[0]
    extracted_paths = []
    
    try:
        frame_count = get_nd2_frame_count(nd2_path)
        print(f"ND2 file {base_name} has {frame_count} frames")
        
        if frame_count > 0:
            # Select slices to extract (evenly distributed)
            if frame_count <= max_slices:
                selected_slices = list(range(frame_count))
            else:
                step = frame_count // max_slices
                selected_slices = [i * step for i in range(max_slices)]
            
            print(f"Extracting {len(selected_slices)} preview slices...")
            
            for i, slice_idx in enumerate(selected_slices):
                try:
                    slice_img = nd2_to_pil(nd2_path, frame=slice_idx, mode='specific_frame')
                    
                    if slice_img is not None:
                        slice_name = f"{base_name}_slice_{slice_idx:04d}_preview.png"
                        slice_path = os.path.join(output_dir, slice_name)
                        slice_img.save(slice_path)
                        extracted_paths.append(slice_path)
                        print(f"  ✓ Extracted slice {slice_idx} -> {slice_name}")
                    
                except Exception as e:
                    print(f"  ✗ Error extracting slice {slice_idx}: {e}")
                    continue
        
        print(f"Preview extraction complete: {len(extracted_paths)} slices saved to {output_dir}")
        return extracted_paths
        
    except Exception as e:
        print(f"Error processing ND2 file {nd2_path}: {e}")
        return []

def create_training_slice_folders(data_dir, output_base_dir="training_output"):
    """
    Create organized folders with training slices and prediction slices for all training images
    
    Args:
        data_dir: Path to data directory (images/ and masks/ folders)
        output_base_dir: Base directory for output folders
        
    Returns:
        dict: Dictionary with paths to created folders
    """
    # Create output directory structure
    slice_dirs = {
        'training_slices': os.path.join(output_base_dir, 'training_slices'),
        'prediction_slices': os.path.join(output_base_dir, 'prediction_slices'),
        'masks_slices': os.path.join(output_base_dir, 'masks_slices'),
        'visualization': os.path.join(output_base_dir, 'visualization')
    }
    
    # Create all directories
    for dir_name, dir_path in slice_dirs.items():
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    # Load the dataset to get all image-mask pairs
    dataset = SegmentationDataset(data_dir=data_dir, split='train')
    
    print(f"Processing {len(dataset)} image-mask pairs...")
    
    slice_count = 0
    for idx in range(len(dataset)):
        try:
            # Get image and mask
            image_tensor, mask_tensor = dataset[idx]
            image_path = dataset.image_files[idx]
            mask_path = dataset.mask_files[idx]
            
            # Convert tensors back to PIL images for saving
            if image_tensor.shape[0] == 1:  # Grayscale
                image_pil = transforms.ToPILImage()(image_tensor.squeeze(0))
            else:  # RGB
                image_pil = transforms.ToPILImage()(image_tensor)
            
            # Convert mask tensor to PIL
            mask_array = mask_tensor.numpy().astype(np.uint8)
            mask_pil = Image.fromarray(mask_array)
            
            # Generate file names
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            slice_name = f"slice_{slice_count:06d}_{base_name}"
            
            # Save training slice (original image)
            training_slice_path = os.path.join(slice_dirs['training_slices'], f"{slice_name}.png")
            image_pil.save(training_slice_path)
            
            # Save mask slice
            mask_slice_path = os.path.join(slice_dirs['masks_slices'], f"{slice_name}_mask.png")
            mask_pil.save(mask_slice_path)
            
            # Create blank prediction slice (to be filled during inference)
            prediction_slice_path = os.path.join(slice_dirs['prediction_slices'], f"{slice_name}_prediction.png")
            blank_prediction = Image.new('L', image_pil.size, 0)  # Black image
            blank_prediction.save(prediction_slice_path)
            
            # Create visualization (side-by-side image and mask)
            create_training_visualization(image_pil, mask_pil, slice_dirs['visualization'], slice_name)
            
            slice_count += 1
            
            if slice_count % 10 == 0:
                print(f"Processed {slice_count} slices...")
                
        except Exception as e:
            print(f"Error processing slice {idx}: {e}")
            continue
    
    print(f"✅ Created {slice_count} training slices in organized folders")
    print(f"📁 Training slices: {slice_dirs['training_slices']}")
    print(f"📁 Prediction slices: {slice_dirs['prediction_slices']}")
    print(f"📁 Mask slices: {slice_dirs['masks_slices']}")
    print(f"📁 Visualizations: {slice_dirs['visualization']}")
    
    return slice_dirs

def create_training_visualization(image_pil, mask_pil, viz_dir, slice_name):
    """
    Create a side-by-side visualization of image and colored mask
    
    Args:
        image_pil: PIL Image of the original image
        mask_pil: PIL Image of the mask
        viz_dir: Directory to save visualization
        slice_name: Base name for the slice
    """
    try:
        # Convert grayscale image to RGB for consistency
        if image_pil.mode == 'L':
            image_rgb = image_pil.convert('RGB')
        else:
            image_rgb = image_pil
        
        # Convert mask to colored visualization
        mask_array = np.array(mask_pil)
        colored_mask = np.zeros((*mask_array.shape, 3), dtype=np.uint8)
        
        # Apply tissue colors to mask
        for class_id in np.unique(mask_array):
            if class_id < len(get_all_tissue_colors()):
                color = get_tissue_color(class_id)
                colored_mask[mask_array == class_id] = color
        
        colored_mask_pil = Image.fromarray(colored_mask)
        
        # Create side-by-side visualization
        width, height = image_rgb.size
        combined = Image.new('RGB', (width * 2, height))
        combined.paste(image_rgb, (0, 0))
        combined.paste(colored_mask_pil, (width, 0))
        
        # Add labels
        draw = ImageDraw.Draw(combined)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        draw.text((10, 10), "Original", fill=(255, 255, 255), font=font)
        draw.text((width + 10, 10), "Mask", fill=(255, 255, 255), font=font)
        
        # Save visualization
        viz_path = os.path.join(viz_dir, f"{slice_name}_visualization.png")
        combined.save(viz_path)
        
    except Exception as e:
        print(f"Error creating visualization for {slice_name}: {e}")

def create_prediction_folders_from_nd2(nd2_files, output_base_dir="prediction_output"):
    """
    Create prediction folders structure from ND2 files for inference
    
    Args:
        nd2_files: List of ND2 file paths
        output_base_dir: Base directory for output folders
        
    Returns:
        dict: Dictionary with paths to created folders and slice info
    """
    # Create output directory structure
    pred_dirs = {
        'input_slices': os.path.join(output_base_dir, 'input_slices'),
        'predictions': os.path.join(output_base_dir, 'predictions'),
        'colored_predictions': os.path.join(output_base_dir, 'colored_predictions'),
        'overlay_results': os.path.join(output_base_dir, 'overlay_results')
    }
    
    # Create all directories
    for dir_name, dir_path in pred_dirs.items():
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    slice_info = []
    total_slices = 0
    
    for nd2_path in nd2_files:
        try:
            base_name = os.path.splitext(os.path.basename(nd2_path))[0]
            print(f"Processing {base_name}.nd2...")
            
            # Extract all frames from ND2
            frames = nd2_to_pil(nd2_path)
            
            if frames:
                for frame_idx, frame_img in enumerate(frames):
                    # Generate slice name
                    slice_name = f"{base_name}_slice_{frame_idx:04d}"
                    
                    # Save input slice
                    input_slice_path = os.path.join(pred_dirs['input_slices'], f"{slice_name}.png")
                    frame_img.save(input_slice_path)
                    
                    # Create blank prediction files (to be filled during inference)
                    pred_path = os.path.join(pred_dirs['predictions'], f"{slice_name}_prediction.png")
                    colored_pred_path = os.path.join(pred_dirs['colored_predictions'], f"{slice_name}_colored.png")
                    overlay_path = os.path.join(pred_dirs['overlay_results'], f"{slice_name}_overlay.png")
                    
                    # Create blank files
                    blank_img = Image.new('L', frame_img.size, 0)
                    blank_img.save(pred_path)
                    
                    blank_colored = Image.new('RGB', frame_img.size, (0, 0, 0))
                    blank_colored.save(colored_pred_path)
                    blank_colored.save(overlay_path)
                    
                    # Store slice information
                    slice_info.append({
                        'nd2_file': nd2_path,
                        'frame_idx': frame_idx,
                        'slice_name': slice_name,
                        'input_path': input_slice_path,
                        'prediction_path': pred_path,
                        'colored_path': colored_pred_path,
                        'overlay_path': overlay_path
                    })
                    
                    total_slices += 1
                
                print(f"  Extracted {len(frames)} slices from {base_name}.nd2")
            else:
                print(f"  ⚠️ No frames extracted from {base_name}.nd2")
                
        except Exception as e:
            print(f"Error processing {nd2_path}: {e}")
            continue
    
    print(f"✅ Created prediction folders for {total_slices} slices from {len(nd2_files)} ND2 files")
    print(f"📁 Input slices: {pred_dirs['input_slices']}")
    print(f"📁 Predictions: {pred_dirs['predictions']}")
    print(f"📁 Colored predictions: {pred_dirs['colored_predictions']}")
    print(f"📁 Overlay results: {pred_dirs['overlay_results']}")
    
    return pred_dirs, slice_info

def save_training_slice_summary(slice_dirs, data_dir, summary_file="training_slice_summary.txt"):
    """
    Save a summary of the training slice creation process
    
    Args:
        slice_dirs: Dictionary of slice directories
        data_dir: Original data directory
        summary_file: Path to save summary file
    """
    try:
        # Count files in each directory
        counts = {}
        for dir_name, dir_path in slice_dirs.items():
            if os.path.exists(dir_path):
                counts[dir_name] = len([f for f in os.listdir(dir_path) if f.endswith('.png')])
            else:
                counts[dir_name] = 0
        
        # Write summary
        summary_path = os.path.join(os.path.dirname(slice_dirs['training_slices']), summary_file)
        with open(summary_path, 'w') as f:
            f.write("Training Slice Creation Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Source data directory: {data_dir}\n")
            f.write(f"Output base directory: {os.path.dirname(slice_dirs['training_slices'])}\n\n")
            
            f.write("Created directories and file counts:\n")
            for dir_name, count in counts.items():
                f.write(f"  {dir_name}: {count} files\n")
            
            f.write(f"\nTotal training slices: {counts.get('training_slices', 0)}\n")
            f.write(f"Total mask slices: {counts.get('masks_slices', 0)}\n")
            f.write(f"Total prediction placeholders: {counts.get('prediction_slices', 0)}\n")
            f.write(f"Total visualizations: {counts.get('visualization', 0)}\n")
            
            f.write(f"\nGenerated on: {__import__('datetime').datetime.now()}\n")
            
            # Add tissue class information
            f.write(f"\nConfigured tissue classes ({get_num_tissue_classes()}):\n")
            for class_id, name in get_all_tissue_names().items():
                color = get_tissue_color(class_id)
                f.write(f"  {class_id}: {name} - RGB{color}\n")
        
        print(f"✅ Training slice summary saved to: {summary_path}")
        return summary_path
        
    except Exception as e:
        print(f"Error saving training slice summary: {e}")
        return None

def debug_nd2_file(nd2_path):
    """
    Debug function to analyze ND2 file structure and available axes
    
    Args:
        nd2_path: Path to the ND2 file
        
    Returns:
        dict: Information about the ND2 file structure
    """
    try:
        print(f"\n🔍 DEBUGGING ND2 FILE: {nd2_path}")
        print("=" * 60)
        
        with ND2(nd2_path) as images:
            # Get basic information
            info = {
                'file_path': nd2_path,
                'sizes': dict(images.sizes),
                'axes': list(images.sizes.keys()),
                'metadata': {}
            }
            
            print(f"📊 File sizes: {info['sizes']}")
            print(f"📐 Available axes: {info['axes']}")
            print(f"📏 Total length: {len(images)}")
            
            # Check each possible axis
            axes_info = {}
            for axis in ['x', 'y', 'z', 't', 'c']:
                value = images.sizes.get(axis, None)
                axes_info[axis] = value
                if value is not None:
                    print(f"  {axis.upper()}: {value}")
                else:
                    print(f"  {axis.upper()}: Not present")
            
            info['axes_info'] = axes_info
            
            # Try to access metadata
            try:
                if hasattr(images, 'metadata'):
                    info['metadata'] = str(images.metadata)[:200] + "..." if len(str(images.metadata)) > 200 else str(images.metadata)
                    print(f"📋 Metadata preview: {info['metadata']}")
            except Exception as meta_e:
                print(f"⚠️ Could not access metadata: {meta_e}")
            
            # Try to access the first frame to test
            try:
                print("\n🧪 Testing frame access...")
                
                # Reset coordinates to defaults
                if 'z' in info['axes']:
                    images.default_coords['z'] = 0
                if 'c' in info['axes']:
                    images.default_coords['c'] = 0
                if 't' in info['axes']:
                    images.default_coords['t'] = 0
                
                first_frame = images[0]
                print(f"✅ First frame shape: {first_frame.shape}")
                print(f"✅ First frame dtype: {first_frame.dtype}")
                print(f"✅ First frame min/max: {first_frame.min()}/{first_frame.max()}")
                
                info['first_frame_shape'] = first_frame.shape
                info['first_frame_dtype'] = str(first_frame.dtype)
                info['test_successful'] = True
                
            except Exception as frame_e:
                print(f"❌ Error accessing first frame: {frame_e}")
                info['test_successful'] = False
                info['frame_error'] = str(frame_e)
            
            print("=" * 60)
            return info
            
    except Exception as e:
        print(f"❌ CRITICAL ERROR debugging ND2 file: {e}")
        return {'error': str(e), 'file_path': nd2_path}

# Add this function at the end of the file for easy testing
