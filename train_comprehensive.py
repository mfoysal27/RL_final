#!/usr/bin/env python
"""
Comprehensive Training System for Biological Image Segmentation
Supports multiple architectures, fine-tuning, and RL enhancement
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from typing import Dict, Any, Optional, Tuple, List
import json
import time
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# Additional imports for testing functionality
try:
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns
    SKLEARN_AVAILABLE = True
except ImportError:
    print("âš ï¸  Warning: sklearn/seaborn not available. Some testing features will be limited.")
    SKLEARN_AVAILABLE = False

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all components
from models.model_factory import ModelFactory
from core.losses import IoULoss
import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog
from models.hybrid_model import HybridUNetGNN
from core.tissue_config import get_all_tissue_colors

class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

class ComprehensiveTrainer:
    """Comprehensive trainer for all model architectures"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.get('use_cuda', True) else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize model type first (but not the model itself yet)
        self.model_type = config['model_type']
        
        # Setup data loaders FIRST to determine the actual number of classes
        self.train_loader, self.val_loader, self.num_classes, self.class_names = self._setup_data_loaders()
        
        # Update config with the detected number of classes
        self.config['num_classes'] = self.num_classes
        print(f"Updated config with detected num_classes: {self.num_classes}")
        
        # NOW create the model with the correct number of classes
        self.model = self._create_model()
        
        # Setup optimizers and schedulers
        self.optimizers = self._setup_optimizers()
        self.schedulers = self._setup_schedulers()
        
        # Setup loss functions
        self.criterion = self._setup_loss_functions()
        
        # Setup training utilities
        self.early_stopping = EarlyStopping(
            patience=config.get('patience', 15),
            min_delta=config.get('min_delta', 0.001)
        )
        
        # Setup logging
        self.setup_logging()
        

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_dice': [],
            'val_dice': [],
            'learning_rates': {}
        }
        
        # New attribute for epoch visualizations
        self.save_epoch_visualizations = config.get('save_epoch_visualizations', False)
        
    def _create_model(self) -> nn.Module:
        """Create model using factory"""
        model_config = ModelFactory.validate_config(self.model_type, self.config)
        if self.model_type == 'hybrid_unet_gnn':
            model = HybridUNetGNN(
                in_channels=1,
                num_classes=self.num_classes, 
                features=[64, 128, 256, 512],
                gnn_hidden_channels=64
            )
        elif self.model_type == 'cellpose_sam':
            params = {
                'num_classes': self.config.get('num_classes', 15),  # Only supported params
                'sam_model_type': self.config.get('sam_model_type', 'vit_b'),
                'freeze_sam': self.config.get('freeze_sam', True),
                'gnn_hidden_channels': self.config.get('gnn_hidden_channels', 128),
                'device': self.config.get('device', None)
            }
            model = ModelFactory.create_model(self.model_type, params)
        else:
            model = ModelFactory.create_model(self.model_type, model_config)
        model = model.to(self.device)
        
        # Load pre-trained weights if specified
        if self.config.get('pretrained_path'):
            self._load_pretrained_weights(model)
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        return model
    
    def _load_pretrained_weights(self, model: nn.Module):
        """Load pre-trained weights"""
        pretrained_path = self.config['pretrained_path']
        if os.path.exists(pretrained_path):
            try:
                checkpoint = torch.load(pretrained_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                # Load with partial matching
                model_dict = model.state_dict()
                pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
                
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
                
                print(f"Loaded pre-trained weights from {pretrained_path}")
                print(f"Matched {len(pretrained_dict)}/{len(model_dict)} parameters")
                
            except Exception as e:
                print(f"Warning: Could not load pre-trained weights: {e}")
        else:
            print(f"Warning: Pre-trained weights file not found: {pretrained_path}")
    
    def _setup_data_loaders(self):
        """Setup data loaders with image-mask structure only"""
        data_dir = self.config['data_dir']
        batch_size = self.config.get('batch_size', 4)
        color_config_file = self.config.get('color_config_file', None)
        img_size = self.config.get('img_size', 256)
        
        print(f"Setting up data loaders:")
        print(f"  Data directory: {data_dir}")
        print(f"  Using image-mask pair structure")
        print(f"  Batch size: {batch_size}")
        print(f"  Image size: {img_size}")
        if color_config_file:
            print(f"  Color config: {color_config_file}")
        
        # Validate data directory exists
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
        
        print(f"ðŸ“ Loading image-mask datasets from {data_dir}...")
        
        # Try enhanced data handler first
        try:
            print(f"ðŸ”„ Trying enhanced data handler...")
            from core.enhanced_data_handler import create_streamlined_data_loaders
            
            train_loader, val_loader, test_loader, num_classes, class_names, color_config = create_streamlined_data_loaders(
                data_dir=data_dir,
                color_config_file=color_config_file,
                mode='image_mask',
                batch_size=batch_size,
                image_size=(img_size, img_size)
            )
            
            print(f"Enhanced data loader created successfully")
            print(f"   Number of classes: {num_classes}")
            print(f"   Class names: {class_names}")
            
            # Store color config for later use
            self.color_config = color_config
            
        except Exception as e:
            print(f"Enhanced data loader failed: {e}")
            
            # Try original data handler
            try:
                print(f"ðŸ”„ Trying original data handler...")
                from core.data_handler import create_data_loaders
                
                train_loader, val_loader, test_loader, num_classes, class_names = create_data_loaders(
                    data_dir=data_dir,
                    batch_size=batch_size
                )
                print(f"Original data handler created data loaders successfully")
                self.color_config = None
                
            except Exception as e2:
                error_msg = f"Error setting up data loaders: {str(e2)}\n\n"
                error_msg += "Common solutions:\n"
                error_msg += "1. Check that your data directory exists\n"
                error_msg += "2. Ensure proper image-mask structure:\n"
                error_msg += "   data_dir/\n"
                error_msg += "   â”œâ”€â”€ train/\n"
                error_msg += "   â”‚   â”œâ”€â”€ images/\n"
                error_msg += "   â”‚   â””â”€â”€ masks/\n"
                error_msg += "   â”œâ”€â”€ val/\n"
                error_msg += "   â”‚   â”œâ”€â”€ images/\n"
                error_msg += "   â”‚   â””â”€â”€ masks/\n"
                error_msg += "   â””â”€â”€ test/\n"
                error_msg += "       â”œâ”€â”€ images/\n"
                error_msg += "       â””â”€â”€ masks/\n"
                error_msg += "3. Check file permissions\n"
                error_msg += "4. Ensure you have valid image-mask pairs\n"
                error_msg += "5. Supported formats: .png, .jpg, .jpeg, .tif, .tiff, .bmp, .nd2\n"
                
                print(error_msg)
                raise Exception(error_msg)
        
        # Final validation
        if train_loader is None:
            raise Exception("No training data found! Please check your data directory structure.")
        
        print(f"Data loading successful!")
        print(f"   Training samples: {len(train_loader.dataset) if train_loader else 0}")
        print(f"   Validation samples: {len(val_loader.dataset) if val_loader else 0}")
        print(f"   Number of classes: {num_classes}")
        print(f"   Class names: {class_names}")
        
        return train_loader, val_loader, num_classes, class_names
    
    def _setup_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        """Setup optimizers for different model components"""
        return ModelFactory.setup_optimizers(self.model, self.model_type, self.config)
    
    def _setup_schedulers(self) -> Dict[str, Any]:
        """Setup learning rate schedulers"""
        return ModelFactory.setup_schedulers(self.optimizers, self.config)
    
    def _setup_loss_functions(self) -> Dict[str, nn.Module]:
        """Setup loss functions"""
        criterion = {}
        
        # Primary loss
        if self.config.get('loss_type', 'cross_entropy') == 'cross_entropy':
            criterion['ce'] = nn.CrossEntropyLoss(
                label_smoothing=self.config.get('label_smoothing', 0.0)
            )
        
        # Additional losses
        criterion['iou'] = IoULoss()
        
        # Dice loss
        criterion['dice'] = self._dice_loss
        
        return criterion
    
    def _dice_loss(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Dice loss implementation"""
        smooth = 1e-5
        
        # Convert to one-hot if needed
        if target.dim() == 3:
            target = F.one_hot(target.long(), num_classes=predicted.shape[1]).permute(0, 3, 1, 2).float()
        
        predicted = F.softmax(predicted, dim=1)
        
        intersection = (predicted * target).sum(dim=(2, 3))
        dice = (2. * intersection + smooth) / (predicted.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth)
        
        return 1 - dice.mean()
    
    def setup_logging(self):
        """Setup logging and visualization"""
        # Default save directory
        save_dir = self.config.get('save_dir', 'logs')  # Default: 'logs'

        # But your GUI overrides this to 'training_output'
        self.log_dir = os.path.join(save_dir, f"{self.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.writer = SummaryWriter(self.log_dir)
        
        # Convert numpy types in config to JSON-serializable types
        def convert_numpy_types(obj):
            """Convert numpy types to Python native types for JSON serialization"""
            import numpy as np
            
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Save config with numpy types converted
        config_path = os.path.join(self.log_dir, 'config.json')
        with open(config_path, 'w') as f:
            json_config = convert_numpy_types(self.config)
            json.dump(json_config, f, indent=4)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_dice = 0.0
        num_batches = 0
        
        # Setup progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images, targets = images.to(self.device), targets.to(self.device)
            
            # Zero gradients
            for optimizer in self.optimizers.values():
                optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            
            # Compute losses
            losses = self._compute_losses(outputs, targets)
            total_loss_batch = losses['total']
            
            # Backward pass
            total_loss_batch.backward()
            
            # Gradient clipping
            if self.config.get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip'])
            
            # Optimizer steps
            for optimizer in self.optimizers.values():
                optimizer.step()
            
            # Update metrics
            total_loss += total_loss_batch.item()
            total_dice += self._compute_dice_score(outputs, targets).item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{total_loss_batch.item():.4f}',
                'Dice': f'{self._compute_dice_score(outputs, targets).item():.4f}'
            })
            
            # Log batch metrics
            if batch_idx % self.config.get('log_frequency', 50) == 0:
                self.writer.add_scalar('Train/BatchLoss', total_loss_batch.item(), 
                                     self.current_epoch * len(self.train_loader) + batch_idx)
        
        # Compute epoch metrics
        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches
        
        return {'loss': avg_loss, 'dice': avg_dice}
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        
        total_loss = 0.0
        total_dice = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]")
            
            for images, targets in pbar:
                images, targets = images.to(self.device), targets.to(self.device)
                
                outputs = self.model(images)
                losses = self._compute_losses(outputs, targets)
                
                total_loss += losses['total'].item()
                total_dice += self._compute_dice_score(outputs, targets).item()
                num_batches += 1
                
                pbar.set_postfix({
                    'Loss': f'{losses["total"].item():.4f}',
                    'Dice': f'{self._compute_dice_score(outputs, targets).item():.4f}'
                })
        
        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches
        
        return {'loss': avg_loss, 'dice': avg_dice}
    
    def _compute_losses(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute all losses"""
        losses = {}
        
        # Cross-entropy loss
        losses['ce'] = self.criterion['ce'](outputs, targets)
        
        # IoU loss
        losses['iou'] = self.criterion['iou'](outputs, targets)
        
        # Dice loss
        losses['dice'] = self.criterion['dice'](outputs, targets)
        
        # Combine losses
        loss_weights = {
            'ce': self.config.get('ce_weight', 0.1),
            'iou': self.config.get('iou_weight', 0.4),
            'dice': self.config.get('dice_weight', 0.5)
        }
        
        losses['total'] = sum(losses[key] * loss_weights[key] for key in losses.keys())
        
        return losses
    
    def _compute_dice_score(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Dice score"""
        smooth = 1e-5
        
        predicted = torch.softmax(outputs, dim=1)
        predicted = torch.argmax(predicted, dim=1)
        
        dice_scores = []
        for cls in range(self.num_classes):
            pred_cls = (predicted == cls).float()
            target_cls = (targets == cls).float()
            
            intersection = (pred_cls * target_cls).sum()
            dice = (2. * intersection + smooth) / (pred_cls.sum() + target_cls.sum() + smooth)
            dice_scores.append(dice)
        
        return torch.stack(dice_scores).mean()
    
    def update_learning_rates(self, val_loss: float):
        """Update learning rates using schedulers"""
        for name, scheduler in self.schedulers.items():
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
            
            # Log current learning rate
            current_lr = self.optimizers[name].param_groups[0]['lr']
            self.training_history['learning_rates'][name] = current_lr
            self.writer.add_scalar(f'LearningRate/{name}', current_lr, self.current_epoch)
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_type': self.model_type,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dicts': {name: opt.state_dict() for name, opt in self.optimizers.items()},
            'scheduler_state_dicts': {name: sch.state_dict() for name, sch in self.schedulers.items()},
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'training_history': self.training_history,
            'num_classes': self.num_classes,
            'class_names': self.class_names
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.log_dir, f'checkpoint_epoch_{self.current_epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.log_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"New best model saved with validation loss: {self.best_val_loss:.6f}")
    
    def save_epoch_visualization(self, epoch: int, train_metrics: Dict[str, float], 
                               val_metrics: Dict[str, float]):
        """
        Create and save comprehensive epoch visualization as PNG
        
        Args:
            epoch: Current epoch number
            train_metrics: Training metrics for the epoch
            val_metrics: Validation metrics for the epoch
        """
        import matplotlib.patches as patches
        from matplotlib.gridspec import GridSpec
        
        # Create visualization directory
        vis_dir = os.path.join(self.log_dir, 'epoch_visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Set up the figure with custom layout
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(4, 6, figure=fig, hspace=0.3, wspace=0.4)
        
        # === TOP SECTION: TRAINING METRICS ===
        
        # Training Loss Plot
        ax_train_loss = fig.add_subplot(gs[0, 0:2])
        if len(self.training_history['train_loss']) > 0:
            epochs_range = range(1, len(self.training_history['train_loss']) + 1)
            ax_train_loss.plot(epochs_range, self.training_history['train_loss'], 'b-', label='Train Loss', linewidth=2)
            ax_train_loss.plot(epochs_range, self.training_history['val_loss'], 'r-', label='Val Loss', linewidth=2)
            ax_train_loss.axvline(x=epoch+1, color='green', linestyle='--', alpha=0.7, label=f'Current Epoch {epoch+1}')
            ax_train_loss.set_xlabel('Epoch')
            ax_train_loss.set_ylabel('Loss')
            ax_train_loss.set_title('Training & Validation Loss')
            ax_train_loss.legend()
            ax_train_loss.grid(True, alpha=0.3)
        
        # Dice Score Plot
        ax_dice = fig.add_subplot(gs[0, 2:4])
        if len(self.training_history['train_dice']) > 0:
            epochs_range = range(1, len(self.training_history['train_dice']) + 1)
            ax_dice.plot(epochs_range, self.training_history['train_dice'], 'g-', label='Train Dice', linewidth=2)
            ax_dice.plot(epochs_range, self.training_history['val_dice'], 'orange', label='Val Dice', linewidth=2)
            ax_dice.axvline(x=epoch+1, color='green', linestyle='--', alpha=0.7, label=f'Current Epoch {epoch+1}')
            ax_dice.set_xlabel('Epoch')
            ax_dice.set_ylabel('Dice Score')
            ax_dice.set_title('Training & Validation Dice Score')
            ax_dice.legend()
            ax_dice.grid(True, alpha=0.3)
            ax_dice.set_ylim(0, 1)
        
        # Current Epoch Metrics Box
        ax_metrics = fig.add_subplot(gs[0, 4:6])
        ax_metrics.axis('off')
        
        metrics_text = f"""EPOCH {epoch + 1} METRICS

Training:
â€¢ Loss: {train_metrics['loss']:.6f}
â€¢ Dice: {train_metrics['dice']:.4f}

Validation:
â€¢ Loss: {val_metrics['loss']:.6f}
â€¢ Dice: {val_metrics['dice']:.4f}

Best Val Loss: {self.best_val_loss:.6f}

Learning Rates:"""
        
        # Add learning rates
        for name, optimizer in self.optimizers.items():
            current_lr = optimizer.param_groups[0]['lr']
            metrics_text += f"\nâ€¢ {name}: {current_lr:.2e}"
        
        ax_metrics.text(0.05, 0.95, metrics_text, transform=ax_metrics.transAxes, 
                       fontsize=12, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        # === MIDDLE SECTION: SAMPLE VISUALIZATIONS ===
        
        # Collect sample data for visualization
        sample_data = self._collect_sample_data()
        
        if sample_data and 'images' in sample_data:
            images = sample_data['images']
            ground_truth = sample_data['ground_truth'] 
            predictions = sample_data['predictions']
            
            # Display up to 3 samples
            num_samples = min(3, len(images))
            
            for i in range(num_samples):
                # Original Image
                ax_img = fig.add_subplot(gs[1, i*2])
                img_tensor = images[i]
                
                if img_tensor.dim() == 3 and img_tensor.shape[0] == 1:
                    img_np = img_tensor.squeeze(0).cpu().numpy()
                    ax_img.imshow(img_np, cmap='gray')
                else:
                    img_np = img_tensor.cpu().numpy()
                    if img_np.shape[0] <= 3:  # Channels first
                        img_np = np.transpose(img_np, (1, 2, 0))
                    if img_np.shape[2] == 1:
                        ax_img.imshow(img_np[:,:,0], cmap='gray')
                    else:
                        ax_img.imshow(img_np)
                
                ax_img.set_title(f'Sample {i+1}: Input', fontsize=10)
                ax_img.axis('off')
                
                # Ground Truth
                ax_gt = fig.add_subplot(gs[1, i*2+1])
                gt_np = ground_truth[i].cpu().numpy()
                
                # Create color visualization for ground truth
                gt_colored = self._create_colored_mask(gt_np)
                ax_gt.imshow(gt_colored)
                ax_gt.set_title(f'Sample {i+1}: Ground Truth', fontsize=10)
                ax_gt.axis('off')
                
                # Prediction
                ax_pred = fig.add_subplot(gs[2, i*2])
                pred_np = predictions[i].cpu().numpy()
                
                # Create color visualization for prediction
                pred_colored = self._create_colored_mask(pred_np)
                ax_pred.imshow(pred_colored)
                ax_pred.set_title(f'Sample {i+1}: Prediction', fontsize=10)
                ax_pred.axis('off')
                
                # Difference/Error Map
                ax_diff = fig.add_subplot(gs[2, i*2+1])
                diff_map = (pred_np != gt_np).astype(np.float32)
                ax_diff.imshow(diff_map, cmap='Reds', alpha=0.7)
                ax_diff.imshow(gt_colored, alpha=0.3)  # Show ground truth as background
                ax_diff.set_title(f'Sample {i+1}: Errors', fontsize=10)
                ax_diff.axis('off')
        
        # === BOTTOM SECTION: CLASS LEGEND AND TRAINING INFO ===
        
        # Class Legend
        ax_legend = fig.add_subplot(gs[3, 0:3])
        ax_legend.axis('off')
        
        if hasattr(self, 'class_names') and self.class_names:
            legend_text = "CLASS LEGEND:\n\n"
            colors = self._get_class_colors()
            
            # Create colored legend
            legend_elements = []
            for i, class_name in enumerate(self.class_names[:self.num_classes]):
                color = colors[i] if i < len(colors) else [128, 128, 128]
                color_normalized = [c/255.0 for c in color]
                legend_elements.append(patches.Rectangle((0,0),1,1, facecolor=color_normalized, label=f'Class {i}: {class_name}'))
            
            ax_legend.legend(handles=legend_elements, loc='center', ncol=2, fontsize=10)
        
        # Training Progress Info
        ax_info = fig.add_subplot(gs[3, 3:6])
        ax_info.axis('off')
        
        # Calculate training progress
        progress_pct = ((epoch + 1) / self.config['num_epochs']) * 100
        
        info_text = f"""TRAINING PROGRESS

Epoch: {epoch + 1} / {self.config['num_epochs']} ({progress_pct:.1f}%)
Model: {self.model_type}
Device: {self.device}
Batch Size: {self.config.get('batch_size', 'N/A')}

Dataset Info:
â€¢ Training samples: {len(self.train_loader.dataset) if self.train_loader else 0}
â€¢ Validation samples: {len(self.val_loader.dataset) if self.val_loader else 0}
â€¢ Classes: {self.num_classes}

Data Directory: {os.path.basename(self.config.get('data_dir', 'N/A'))}"""
        
        ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        
        # Main title
        fig.suptitle(f'Training Epoch {epoch + 1} - {self.model_type} - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                     fontsize=16, fontweight='bold')
        
        # Save the figure
        vis_path = os.path.join(vis_dir, f'epoch_{epoch+1:03d}_visualization.png')
        plt.savefig(vis_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)  # Important: close to free memory
        
        print(f"ðŸ“Š Epoch {epoch+1} visualization saved: {vis_path}")
        
        return vis_path
    
    def _create_colored_mask(self, mask_np: np.ndarray) -> np.ndarray:
        """Create a colored visualization of a segmentation mask"""
        colors = self._get_class_colors()
        
        height, width = mask_np.shape
        colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
        
        for class_idx in range(self.num_classes):
            if class_idx < len(colors):
                color = colors[class_idx]
            else:
                # Generate a random color for unknown classes
                color = [np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)]
            
            mask = (mask_np == class_idx)
            colored_mask[mask] = color
        
        return colored_mask
    
    def _get_class_colors(self):
        """Get colors for each class"""
        # Use the proper tissue colors from configuration
        tissue_colors = get_all_tissue_colors()
        colors = []
        
        for i in range(self.num_classes):
            if i in tissue_colors:
                colors.append(list(tissue_colors[i]))
            else:
                # Fallback to gray for missing classes
                colors.append([128, 128, 128])
        
        return colors
    
    def _collect_sample_data(self):
        """Collect sample data for visualization"""
        self.model.eval()
        
        sample_data = {
            'images': [],
            'ground_truth': [],
            'predictions': []
        }
        
        with torch.no_grad():
            # Get a few samples from validation loader
            for batch_idx, (images, targets) in enumerate(self.val_loader):
                if batch_idx >= 1:  # Only take first batch
                    break
                    
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(images)
                predictions = torch.argmax(outputs, dim=1)
                
                # Take up to 3 samples from the batch
                num_samples = min(3, images.shape[0])
                for i in range(num_samples):
                    sample_data['images'].append(images[i])
                    sample_data['ground_truth'].append(targets[i])
                    sample_data['predictions'].append(predictions[i])
        
        return sample_data if sample_data['images'] else None
    
    def train(self) -> Dict[str, Any]:
        """Main training loop"""
        print(f"\nStarting training with {self.model_type}")
        print(f"Training for {self.config['num_epochs']} epochs")
        print(f"Device: {self.device}")
        
        # Check if data loaders are valid
        if self.train_loader is None:
            raise ValueError(
                "Training failed: No training data found!\n"
                f"Please check that your data directory '{self.config['data_dir']}' contains:\n"
                "  - Image files in class subfolders (for class_folders mode)\n"
                "  - Valid image formats: .png, .jpg, .jpeg, .tif, .tiff, .bmp, .nd2\n"
                "  - At least one class folder with image files\n\n"
                "Current data directory structure should be:\n"
                "  data_dir/\n"
                "    â”œâ”€â”€ class1_name/\n"
                "    â”‚   â”œâ”€â”€ image1.jpg\n"
                "    â”‚   â””â”€â”€ image2.jpg\n"
                "    â””â”€â”€ class2_name/\n"
                "        â”œâ”€â”€ image3.jpg\n"
                "        â””â”€â”€ image4.jpg"
            )
        
        if self.val_loader is None:
            print("âš ï¸  Warning: No validation data found. Using training data for validation.")
            self.val_loader = self.train_loader
        
        start_time = time.time()
        
        for epoch in range(self.config['num_epochs']):
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self.train_epoch()
            
            # Validation phase
            val_metrics = self.validate_epoch()
            
            # Update learning rates
            self.update_learning_rates(val_metrics['loss'])
            
            # Update training history
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['train_dice'].append(train_metrics['dice'])
            self.training_history['val_dice'].append(val_metrics['dice'])
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/Validation', val_metrics['loss'], epoch)
            self.writer.add_scalar('Dice/Train', train_metrics['dice'], epoch)
            self.writer.add_scalar('Dice/Validation', val_metrics['dice'], epoch)
            
            # Check for best model
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
            
            # Save checkpoint
            if (epoch + 1) % self.config.get('save_frequency', 10) == 0 or is_best:
                self.save_checkpoint(is_best)
            
            # Early stopping
            if self.early_stopping(val_metrics['loss'], self.model):
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break
            
            # Print epoch summary
            print(f"Epoch {epoch + 1}/{self.config['num_epochs']}")
            print(f"Train Loss: {train_metrics['loss']:.6f}, Train Dice: {train_metrics['dice']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.6f}, Val Dice: {val_metrics['dice']:.4f}")
            print("-" * 50)
            
            # Save epoch visualization if enabled
            if self.config.get('save_epoch_visualizations', True):
                self.save_epoch_visualization(epoch, train_metrics, val_metrics)
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time/3600:.2f} hours")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        
        self.writer.close()
        
        # ðŸ†• NEW: Automatic testing after training completion
        print("\n" + "="*60)
        print("ðŸ§ª STARTING AUTOMATIC MODEL TESTING")
        print("="*60)
        
        # Run comprehensive testing
        test_results = self.run_comprehensive_testing()
        
        training_results = {
            'best_val_loss': self.best_val_loss,
            'training_time': training_time,
            'final_epoch': self.current_epoch + 1,
            'training_history': self.training_history,
            'test_results': test_results  # Include test results
        }
        
        return training_results
    
    def run_comprehensive_testing(self) -> Dict[str, Any]:
        """
        Run comprehensive testing after training completion
        
        Returns:
            Dictionary with test results and metrics
        """
        print("ðŸ”„ Setting up test environment...")
        
        # Create test output directory
        test_output_dir = os.path.join(self.log_dir, 'test_results')
        os.makedirs(test_output_dir, exist_ok=True)
        
        # Load best model
        best_model_path = os.path.join(self.log_dir, 'best_model.pth')
        if not os.path.exists(best_model_path):
            print("âš ï¸  Warning: Best model not found, using current model state")
            best_model_path = None
        else:
            print(f"ðŸ“¦ Loading best model from: {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Setup test data loader
        test_loader = self._setup_test_loader()
        
        if test_loader is None:
            print("âŒ No test data available. Skipping testing phase.")
            return {'status': 'skipped', 'reason': 'no_test_data'}
        
        print(f"âœ… Test data loaded: {len(test_loader.dataset)} samples")
        
        # Run testing
        test_metrics = self._evaluate_model(test_loader, test_output_dir)
        
        # Generate comprehensive test report
        report_path = self._generate_test_report(test_metrics, test_output_dir)
        
        print(f"\nðŸ“Š Testing completed!")
        print(f"ðŸ“ Test results saved to: {test_output_dir}")
        print(f"ðŸ“„ Test report: {report_path}")
        
        return test_metrics
    
    def _setup_test_loader(self):
        """Setup test data loader from the same data directory"""
        try:
            # Try to create test loader using the same data setup
            from core.enhanced_data_handler import create_streamlined_data_loaders
            
            _, _, test_loader, _, _, _ = create_streamlined_data_loaders(
                data_dir=self.config['data_dir'],
                color_config_file=self.config.get('color_config_file'),
                mode='image_mask',
                batch_size=1,  # Use batch size 1 for testing to save individual images
                image_size=(self.config.get('img_size', 256), self.config.get('img_size', 256))
            )
            
            return test_loader
            
        except Exception as e:
            print(f"âš ï¸  Enhanced data loader failed for testing: {e}")
            
            # Try original data handler
            try:
                from core.data_handler import create_data_loaders
                
                _, _, test_loader, _, _ = create_data_loaders(
                    data_dir=self.config['data_dir'],
                    batch_size=1
                )
                
                return test_loader
                
            except Exception as e2:
                print(f"âŒ Could not create test loader: {e2}")
                return None
    
    def _evaluate_model(self, test_loader, output_dir) -> Dict[str, Any]:
        """
        Evaluate model on test data and save visualizations
        
        Args:
            test_loader: DataLoader for test data
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary with test metrics
        """
        print("ðŸ” Running model evaluation...")
        
        # Create subdirectories for organized output
        predictions_dir = os.path.join(output_dir, 'predictions')
        visualizations_dir = os.path.join(output_dir, 'visualizations')
        raw_outputs_dir = os.path.join(output_dir, 'raw_outputs')
        
        os.makedirs(predictions_dir, exist_ok=True)
        os.makedirs(visualizations_dir, exist_ok=True)
        os.makedirs(raw_outputs_dir, exist_ok=True)
        
        # Metrics tracking
        total_loss = 0.0
        total_dice = 0.0
        class_dice_scores = np.zeros(self.num_classes)
        class_sample_counts = np.zeros(self.num_classes)
        all_predictions = []
        all_ground_truth = []
        
        # Process each test sample
        sample_count = 0
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Testing")
            
            for batch_idx, (images, targets) in enumerate(pbar):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Compute losses
                losses = self._compute_losses(outputs, targets)
                total_loss += losses['total'].item()
                
                # Get predictions
                predictions = torch.argmax(outputs, dim=1)
                
                # Compute metrics
                dice_score = self._compute_dice_score(outputs, targets)
                total_dice += dice_score.item()
                
                # Process each sample in batch (should be 1 for testing)
                for i in range(images.shape[0]):
                    sample_count += 1
                    
                    # Extract individual sample data
                    sample_image = images[i]
                    sample_target = targets[i]
                    sample_prediction = predictions[i]
                    sample_output = outputs[i]
                    
                    # Convert to numpy for saving
                    img_np = sample_image.cpu().numpy()
                    target_np = sample_target.cpu().numpy()
                    pred_np = sample_prediction.cpu().numpy()
                    output_np = sample_output.cpu().numpy()
                    
                    # Save individual outputs
                    sample_name = f"test_sample_{sample_count:04d}"
                    
                    # Save raw image
                    if img_np.shape[0] == 1:  # Grayscale
                        img_to_save = (img_np[0] * 255).astype(np.uint8)
                    else:
                        img_to_save = np.transpose(img_np, (1, 2, 0))
                        img_to_save = (img_to_save * 255).astype(np.uint8)
                    
                    plt.imsave(
                        os.path.join(raw_outputs_dir, f"{sample_name}_image.png"),
                        img_to_save,
                        cmap='gray' if img_np.shape[0] == 1 else None
                    )
                    
                    # Save ground truth mask
                    gt_colored = self._create_colored_mask(target_np)
                    plt.imsave(
                        os.path.join(raw_outputs_dir, f"{sample_name}_ground_truth.png"),
                        gt_colored
                    )
                    
                    # Save prediction mask
                    pred_colored = self._create_colored_mask(pred_np)
                    plt.imsave(
                        os.path.join(predictions_dir, f"{sample_name}_prediction.png"),
                        pred_colored
                    )
                    
                    # Create comprehensive visualization
                    self._create_test_visualization(
                        img_np, target_np, pred_np, output_np,
                        os.path.join(visualizations_dir, f"{sample_name}_comparison.png"),
                        sample_name
                    )
                    
                    # Compute per-class metrics
                    for cls in range(self.num_classes):
                        if (target_np == cls).sum() > 0:  # Class present in ground truth
                            pred_cls = (pred_np == cls).astype(np.float32)
                            target_cls = (target_np == cls).astype(np.float32)
                            
                            intersection = (pred_cls * target_cls).sum()
                            dice = (2.0 * intersection + 1e-5) / (pred_cls.sum() + target_cls.sum() + 1e-5)
                            
                            class_dice_scores[cls] += dice
                            class_sample_counts[cls] += 1
                    
                    # Store for confusion matrix
                    all_predictions.extend(pred_np.flatten())
                    all_ground_truth.extend(target_np.flatten())
                
                # Update progress
                pbar.set_postfix({
                    'Loss': f'{losses["total"].item():.4f}',
                    'Dice': f'{dice_score.item():.4f}',
                    'Samples': sample_count
                })
        
        # Compute final metrics
        avg_test_loss = total_loss / len(test_loader)
        avg_test_dice = total_dice / len(test_loader)
        
        # Compute per-class dice scores
        per_class_dice = {}
        for cls in range(self.num_classes):
            if class_sample_counts[cls] > 0:
                per_class_dice[cls] = class_dice_scores[cls] / class_sample_counts[cls]
                class_name = self.class_names[cls] if cls < len(self.class_names) else f"Class_{cls}"
                print(f"   {class_name}: {per_class_dice[cls]:.4f}")
            else:
                per_class_dice[cls] = 0.0
        
        # Create confusion matrix
        confusion_matrix = self._create_confusion_matrix(all_ground_truth, all_predictions, output_dir)
        
        print(f"\nðŸ“Š Test Results Summary:")
        print(f"   Average Loss: {avg_test_loss:.6f}")
        print(f"   Average Dice: {avg_test_dice:.4f}")
        print(f"   Samples Processed: {sample_count}")
        print(f"   Outputs Saved: {output_dir}")
        
        return {
            'status': 'completed',
            'avg_loss': avg_test_loss,
            'avg_dice': avg_test_dice,
            'per_class_dice': per_class_dice,
            'samples_processed': sample_count,
            'confusion_matrix': confusion_matrix,
            'output_directory': output_dir
        }
    
    def _create_test_visualization(self, image, ground_truth, prediction, raw_output, save_path, sample_name):
        """
        Create comprehensive test visualization showing image, ground truth, prediction, and metrics
        
        Args:
            image: Input image array
            ground_truth: Ground truth mask
            prediction: Predicted mask
            raw_output: Raw model output (logits)
            save_path: Path to save visualization
            sample_name: Name of the sample
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Test Results: {sample_name}', fontsize=16, fontweight='bold')
        
        # Original Image
        ax = axes[0, 0]
        if image.shape[0] == 1:  # Grayscale
            ax.imshow(image[0], cmap='gray')
        else:
            img_display = np.transpose(image, (1, 2, 0))
            ax.imshow(img_display)
        ax.set_title('Input Image')
        ax.axis('off')
        
        # Ground Truth
        ax = axes[0, 1]
        gt_colored = self._create_colored_mask(ground_truth)
        ax.imshow(gt_colored)
        ax.set_title('Ground Truth')
        ax.axis('off')
        
        # Prediction
        ax = axes[0, 2]
        pred_colored = self._create_colored_mask(prediction)
        ax.imshow(pred_colored)
        ax.set_title('Prediction')
        ax.axis('off')
        
        # Error Map
        ax = axes[1, 0]
        error_map = (prediction != ground_truth).astype(np.float32)
        ax.imshow(error_map, cmap='Reds')
        ax.set_title('Error Map (Red = Wrong)')
        ax.axis('off')
        
        # Confidence Map (using max probability from softmax)
        ax = axes[1, 1]
        softmax_output = torch.softmax(torch.from_numpy(raw_output), dim=0)
        confidence_map = torch.max(softmax_output, dim=0)[0].numpy()
        im = ax.imshow(confidence_map, cmap='viridis')
        ax.set_title('Confidence Map')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Per-class Dice Scores
        ax = axes[1, 2]
        ax.axis('off')
        
        # Compute per-class dice for this sample
        dice_text = "Per-Class Dice Scores:\n\n"
        overall_dice = 0
        valid_classes = 0
        
        for cls in range(min(self.num_classes, 10)):  # Limit to 10 classes for display
            pred_cls = (prediction == cls).astype(np.float32)
            gt_cls = (ground_truth == cls).astype(np.float32)
            
            if gt_cls.sum() > 0:  # Class present in ground truth
                intersection = (pred_cls * gt_cls).sum()
                dice = (2.0 * intersection + 1e-5) / (pred_cls.sum() + gt_cls.sum() + 1e-5)
                
                class_name = self.class_names[cls] if cls < len(self.class_names) else f"Class {cls}"
                dice_text += f"{class_name}: {dice:.3f}\n"
                overall_dice += dice
                valid_classes += 1
        
        if valid_classes > 0:
            overall_dice /= valid_classes
            dice_text += f"\nOverall: {overall_dice:.3f}"
        
        ax.text(0.05, 0.95, dice_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_confusion_matrix(self, ground_truth, predictions, output_dir):
        """Create and save confusion matrix"""
        if not SKLEARN_AVAILABLE:
            print("âš ï¸  Warning: sklearn/seaborn not available. Cannot create confusion matrix.")
            return None
        
        # Convert to numpy arrays
        gt_array = np.array(ground_truth)
        pred_array = np.array(predictions)
        
        # Create confusion matrix
        cm = confusion_matrix(gt_array, pred_array, labels=range(self.num_classes))
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=[f'Class {i}' for i in range(self.num_classes)],
                   yticklabels=[f'Class {i}' for i in range(self.num_classes)])
        plt.title('Confusion Matrix - Test Results')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Generate classification report
        try:
            class_names = [self.class_names[i] if i < len(self.class_names) else f'Class_{i}' 
                          for i in range(self.num_classes)]
            report = classification_report(gt_array, pred_array, 
                                         target_names=class_names, 
                                         output_dict=True)
            
            # Save classification report
            report_path = os.path.join(output_dir, 'classification_report.json')
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=4)
                
        except Exception as e:
            print(f"âš ï¸  Could not generate classification report: {e}")
            report = None
        
        print(f"ðŸ“Š Confusion matrix saved: {cm_path}")
        
        return cm.tolist()  # Return as list for JSON serialization
    
    def _generate_test_report(self, test_metrics, output_dir):
        """Generate comprehensive test report"""
        report_path = os.path.join(output_dir, 'test_report.json')
        
        # Compile comprehensive report
        report = {
            'test_summary': {
                'model_type': self.model_type,
                'test_date': datetime.now().isoformat(),
                'samples_processed': test_metrics.get('samples_processed', 0),
                'avg_loss': test_metrics.get('avg_loss', 0.0),
                'avg_dice': test_metrics.get('avg_dice', 0.0)
            },
            'model_info': {
                'num_classes': self.num_classes,
                'class_names': self.class_names,
                'best_val_loss': self.best_val_loss,
                'total_parameters': sum(p.numel() for p in self.model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            },
            'training_config': self.config,
            'per_class_performance': test_metrics.get('per_class_dice', {}),
            'file_outputs': {
                'predictions_dir': 'predictions/',
                'visualizations_dir': 'visualizations/',
                'raw_outputs_dir': 'raw_outputs/',
                'confusion_matrix': 'confusion_matrix.png',
                'classification_report': 'classification_report.json'
            },
            'training_history': self.training_history
        }
        
        # Save report
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4, default=str)  # default=str handles numpy types
        
        print(f"ðŸ“„ Comprehensive test report saved: {report_path}")
        
        return report_path

def get_hyperparameters_interactive(model_type: str) -> Dict[str, Any]:
    """Get hyperparameters through interactive GUI"""
    
    # Create root window (hidden)
    root = tk.Tk()
    root.withdraw()
    
    # Force root window to front and focus
    root.lift()
    root.attributes('-topmost', True)
    root.after_idle(root.attributes, '-topmost', False)
    
    config = ModelFactory.get_default_config(model_type)
    model_info = ModelFactory.get_model_info(model_type)
    
    print(f"ðŸŽ›ï¸ Starting interactive hyperparameter configuration for {model_type}")
    
    # Show model information
    messagebox.showinfo(
        "Model Information",
        f"Model: {model_info['name']}\n"
        f"Description: {model_info['description']}\n"
        f"Complexity: {model_info['complexity']}\n"
        f"Memory Usage: {model_info['memory_usage']}\n"
        f"Training Time: {model_info['training_time']}\n"
        f"Best For: {model_info['best_for']}"
    )
    
    # Get training hyperparameters with improved error handling
    print(f"ðŸ“ Collecting training hyperparameters...")
    
    # Number of epochs
    try:
        user_epochs = simpledialog.askinteger(
            "Training Parameters",
            "Number of epochs (1-500):",
            initialvalue=50,
            minvalue=1,
            maxvalue=500,
            parent=root
        )
        
        if user_epochs is not None:
            config['num_epochs'] = user_epochs
            print(f"âœ… Epochs set to: {user_epochs}")
        else:
            print(f"âš ï¸ No epochs specified, using default: {config.get('num_epochs', 50)}")
            config['num_epochs'] = 50
            
    except Exception as e:
        print(f"âŒ Error getting epochs: {e}")
        config['num_epochs'] = 50
    
    # Batch size
    try:
        user_batch = simpledialog.askinteger(
            "Training Parameters", 
            "Batch size (1-32):",
            initialvalue=4,
            minvalue=1,
            maxvalue=32,
            parent=root
        )
        
        if user_batch is not None:
            config['batch_size'] = user_batch
            print(f"âœ… Batch size set to: {user_batch}")
        else:
            print(f"âš ï¸ No batch size specified, using default: {config.get('batch_size', 4)}")
            config['batch_size'] = 4
            
    except Exception as e:
        print(f"âŒ Error getting batch size: {e}")
        config['batch_size'] = 4
    
    # Learning rate configuration
    try:
        if model_type in ['transunet', 'swin_unetr']:
            encoder_lr = simpledialog.askfloat(
                "Learning Rates",
                "Encoder learning rate (for transformers):",
                initialvalue=1e-4,
                minvalue=1e-6,
                maxvalue=1e-2,
                parent=root
            )
            config['encoder_lr'] = encoder_lr if encoder_lr is not None else 1e-4
            print(f"âœ… Encoder LR set to: {config['encoder_lr']}")
            
            decoder_lr = simpledialog.askfloat(
                "Learning Rates",
                "Decoder learning rate:",
                initialvalue=1e-3,
                minvalue=1e-6,
                maxvalue=1e-2,
                parent=root
            )
            config['decoder_lr'] = decoder_lr if decoder_lr is not None else 1e-3
            print(f"âœ… Decoder LR set to: {config['decoder_lr']}")
        
        elif model_type == 'hybrid_unet_gnn':
            unet_lr = simpledialog.askfloat(
                "Learning Rates",
                "UNet learning rate:",
                initialvalue=1e-3,
                minvalue=1e-6,
                maxvalue=1e-2,
                parent=root
            )
            config['unet_lr'] = unet_lr if unet_lr is not None else 1e-3
            print(f"âœ… UNet LR set to: {config['unet_lr']}")
            
            gnn_lr = simpledialog.askfloat(
                "Learning Rates",
                "GNN learning rate:",
                initialvalue=5e-4,
                minvalue=1e-6,
                maxvalue=1e-2,
                parent=root
            )
            config['gnn_lr'] = gnn_lr if gnn_lr is not None else 5e-4
            print(f"âœ… GNN LR set to: {config['gnn_lr']}")
        
        else:
            lr = simpledialog.askfloat(
                "Learning Rates",
                "Learning rate:",
                initialvalue=1e-3,
                minvalue=1e-6,
                maxvalue=1e-2,
                parent=root
            )
            config['lr'] = lr if lr is not None else 1e-3
            print(f"âœ… Learning rate set to: {config['lr']}")
            
    except Exception as e:
        print(f"âŒ Error getting learning rates: {e}")
        # Set default learning rates based on model type
        if model_type in ['transunet', 'swin_unetr']:
            config['encoder_lr'] = 1e-4
            config['decoder_lr'] = 1e-3
        elif model_type == 'hybrid_unet_gnn':
            config['unet_lr'] = 1e-3
            config['gnn_lr'] = 5e-4
        else:
            config['lr'] = 1e-3
    
    # Advanced options
    try:
        use_advanced = messagebox.askyesno(
            "Advanced Options",
            "Configure advanced training options?"
        )
        
        if use_advanced:
            weight_decay = simpledialog.askfloat(
                "Regularization",
                "Weight decay:",
                initialvalue=1e-4,
                minvalue=0,
                maxvalue=1e-2,
                parent=root
            )
            config['weight_decay'] = weight_decay if weight_decay is not None else 1e-4
            print(f"âœ… Weight decay set to: {config['weight_decay']}")
            
            gradient_clip = simpledialog.askfloat(
                "Regularization",
                "Gradient clipping (0 for none):",
                initialvalue=1.0,
                minvalue=0,
                maxvalue=10.0,
                parent=root
            )
            config['gradient_clip'] = gradient_clip if gradient_clip is not None else 1.0
            print(f"âœ… Gradient clip set to: {config['gradient_clip']}")
            
            patience = simpledialog.askinteger(
                "Early Stopping",
                "Early stopping patience:",
                initialvalue=15,
                minvalue=5,
                maxvalue=50,
                parent=root
            )
            config['patience'] = patience if patience is not None else 15
            print(f"âœ… Patience set to: {config['patience']}")
        
    except Exception as e:
        print(f"âŒ Error getting advanced options: {e}")
    
    root.destroy()
    
    print(f"ðŸŽ¯ Interactive configuration complete!")
    print(f"ðŸ“‹ Captured parameters:")
    for key, value in config.items():
        if key in ['num_epochs', 'batch_size', 'lr', 'unet_lr', 'gnn_lr', 'encoder_lr', 'decoder_lr', 'weight_decay', 'gradient_clip', 'patience']:
            print(f"   {key}: {value}")
    
    return config

def main():
    """Main training function with enhanced argument support"""
    parser = argparse.ArgumentParser(description="Comprehensive Training for Biological Image Segmentation")
    
    # Model and data arguments
    parser.add_argument('--model_type', type=str, default='unet',
                       choices=['unet', 'hybrid_unet_gnn', 'transunet', 'cellpose_sam'],
                       help='Model architecture to use')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Path to training data directory')
    parser.add_argument('--data_mode', type=str, default='class_folders',
                       choices=['class_folders', 'image_mask'],
                       help='Data organization mode: class_folders or image_mask')
    parser.add_argument('--color_config', type=str, default=None,
                       help='Path to color configuration file for image_mask mode')
    
    # Training hyperparameters
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--img_size', type=int, default=256,
                       help='Input image size (square)')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    
    # Model loading
    parser.add_argument('--pretrained_path', type=str, default=None,
                       help='Path to pretrained model weights')
    parser.add_argument('--config_file', type=str, default=None,
                       help='Path to JSON configuration file')
    
    # Training options
    parser.add_argument('--interactive', action='store_true',
                       help='Use interactive hyperparameter selection')
    parser.add_argument('--save_dir', type=str, default='logs',
                       help='Directory to save training logs and models')
    
    args = parser.parse_args()
    
    # Load configuration from file if provided
    file_config = {}
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, 'r') as f:
            file_config = json.load(f)
        print(f"Loaded configuration from: {args.config_file}")

    # Store user GUI input separately to preserve it
    gui_config = {}
    
    # Use interactive parameter selection if requested
    if args.interactive:
        print("ðŸŽ›ï¸ Starting interactive parameter selection...")
        try:
            gui_config = get_hyperparameters_interactive(args.model_type)
            print(f"ðŸŽ¯ GUI configuration captured: {list(gui_config.keys())}")
            print("âœ… Interactive parameter selection completed")
        except Exception as e:
            print(f"âŒ Interactive parameter selection failed: {e}")
            print("Using default parameters instead")

    # Core configuration (non-conflicting settings) - start with file config as base
    config = file_config.copy()  # Start with file config as base
    config.update({
        'model_type': args.model_type if args.model_type != 'unet' else file_config.get('model_type', args.model_type),
        'data_dir': args.data_dir,
        'data_mode': args.data_mode,
        'color_config_file': args.color_config,
        'save_dir': args.save_dir,
        'save_epoch_visualizations': True
    })

    # Handle data directory - prompt if not provided
    if not config['data_dir']:
        print("ðŸ“ Data directory not specified, prompting user...")
        try:
            # Create a simple tkinter dialog for directory selection
            import tkinter as tk
            from tkinter import filedialog
            
            root = tk.Tk()
            root.withdraw()  # Hide the main window
            root.lift()
            root.attributes('-topmost', True)
            root.after_idle(root.attributes, '-topmost', False)
            
            data_dir = filedialog.askdirectory(
                title="Select Training Data Directory",
                mustexist=True
            )
            root.destroy()
            
            if not data_dir:
                print("âŒ No data directory selected. Cannot proceed without training data.")
                return
            
            config['data_dir'] = data_dir
            print(f"âœ… Selected data directory: {data_dir}")
            
        except Exception as e:
            print(f"âŒ Failed to prompt for data directory: {e}")
            print("Please provide --data_dir argument when running from command line")
            return

    # Validate data directory exists and contains data
    if not os.path.exists(config['data_dir']):
        print(f"âŒ Data directory does not exist: {config['data_dir']}")
        return
    
    # Basic validation - check if directory has subdirectories (for class_folders mode) or files
    try:
        dir_contents = os.listdir(config['data_dir'])
        has_subdirs = any(os.path.isdir(os.path.join(config['data_dir'], item)) for item in dir_contents)
        has_images = any(item.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.nd2')) 
                        for item in dir_contents)
        
        if not has_subdirs and not has_images:
            print(f"âš ï¸  Warning: Data directory appears to be empty or contains no recognizable image files")
            print(f"   Directory: {config['data_dir']}")
            print(f"   Contents: {dir_contents[:5]}{'...' if len(dir_contents) > 5 else ''}")
        else:
            print(f"âœ… Data directory validated: {config['data_dir']}")
            if has_subdirs:
                subdirs = [item for item in dir_contents if os.path.isdir(os.path.join(config['data_dir'], item))]
                print(f"   Found {len(subdirs)} subdirectories: {subdirs[:3]}{'...' if len(subdirs) > 3 else ''}")
            if has_images:
                image_count = sum(1 for item in dir_contents 
                                if item.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.nd2')))
                print(f"   Found {image_count} image files")
                
    except Exception as e:
        print(f"âš ï¸  Warning: Could not validate data directory: {e}")
        print("Proceeding anyway - validation will occur during data loading")

    # Handle training hyperparameters with priority: GUI > Config File > Command Line > Defaults
    def get_param_with_priority(param_name, gui_val, config_val, cmd_val, default_val):
        """Get parameter value with proper priority: GUI > Config > Command Line > Default"""
        if gui_val is not None:
            print(f"ðŸŽ¯ Using GUI value for {param_name}: {gui_val}")
            return gui_val
        elif config_val is not None:
            print(f"ðŸ“„ Using config file value for {param_name}: {config_val}")
            return config_val
        elif cmd_val != default_val:  # Command line was explicitly changed
            print(f"âš¡ Using command line value for {param_name}: {cmd_val}")
            return cmd_val
        else:
            print(f"ðŸ”§ Using default value for {param_name}: {default_val}")
            return default_val

    # Apply hyperparameters with proper priority
    config['num_epochs'] = get_param_with_priority(
        'num_epochs', 
        gui_config.get('num_epochs'), 
        file_config.get('num_epochs'),  # Use file_config instead of mixed config
        args.num_epochs, 
        50
    )
    
    config['batch_size'] = get_param_with_priority(
        'batch_size', 
        gui_config.get('batch_size'), 
        file_config.get('batch_size'),  # Use file_config instead of mixed config
        args.batch_size, 
        4
    )
    
    config['img_size'] = get_param_with_priority(
        'img_size', 
        gui_config.get('img_size'), 
        file_config.get('img_size'),  # Use file_config instead of mixed config
        args.img_size, 
        256
    )

    # Handle learning rates (different models have different structures)
    if args.model_type == 'hybrid_unet_gnn':
        config['unet_lr'] = get_param_with_priority(
            'unet_lr', 
            gui_config.get('unet_lr'), 
            file_config.get('unet_lr'),  # Use file_config instead of mixed config
            args.lr if args.lr != 1e-3 else None, 
            1e-3
        )
        config['gnn_lr'] = get_param_with_priority(
            'gnn_lr', 
            gui_config.get('gnn_lr'), 
            file_config.get('gnn_lr'),  # Use file_config instead of mixed config
            None,  # No command line equivalent
            5e-4
        )
    elif args.model_type in ['transunet', 'swin_unetr']:
        config['encoder_lr'] = get_param_with_priority(
            'encoder_lr', 
            gui_config.get('encoder_lr'), 
            file_config.get('encoder_lr'),  # Use file_config instead of mixed config
            None, 
            1e-4
        )
        config['decoder_lr'] = get_param_with_priority(
            'decoder_lr', 
            gui_config.get('decoder_lr'), 
            file_config.get('decoder_lr'),  # Use file_config instead of mixed config
            args.lr if args.lr != 1e-3 else None, 
            1e-3
        )
    else:
        config['lr'] = get_param_with_priority(
            'lr', 
            gui_config.get('lr'), 
            file_config.get('lr'),  # Use file_config instead of mixed config
            args.lr, 
            1e-3
        )

    # Handle advanced parameters if provided by GUI
    for param in ['weight_decay', 'gradient_clip', 'patience']:
        if param in gui_config:
            config[param] = gui_config[param]
            print(f"ðŸŽ¯ Using GUI value for {param}: {gui_config[param]}")

    # Always respect pretrained path if provided
    if args.pretrained_path:
        config['pretrained_path'] = args.pretrained_path

    print(f"\nðŸ” FINAL CONFIGURATION:")
    print(f"  Model: {config['model_type']}")
    print(f"  Epochs: {config['num_epochs']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Image size: {config['img_size']}")
    if 'lr' in config:
        print(f"  Learning rate: {config['lr']}")
    if 'unet_lr' in config:
        print(f"  UNet LR: {config['unet_lr']}, GNN LR: {config['gnn_lr']}")
    if 'encoder_lr' in config:
        print(f"  Encoder LR: {config['encoder_lr']}, Decoder LR: {config['decoder_lr']}")
    print(f"  Interactive mode: {args.interactive}")

    print("ðŸš€ Starting Comprehensive Training")
    print("=" * 50)
    print(f"Model: {config['model_type']}")
    print(f"Data: {config['data_dir']} ({config['data_mode']})")
    if config.get('color_config_file'):
        print(f"Color config: {config['color_config_file']}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print("=" * 50)
    
    # Initialize trainer
    trainer = ComprehensiveTrainer(config)
    
    # Start training
    trainer.train()
    
    print("âœ… Training completed!")

if __name__ == "__main__":
    main() 