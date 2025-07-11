#!/usr/bin/env python
"""
SAM-based model for tissue segmentation
Using Segment Anything Model (SAM) for powerful feature extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple
import warnings

# Try to import SAM
try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
    print("SAM (Segment Anything Model) imported successfully")
except ImportError:
    SAM_AVAILABLE = False
    warnings.warn("SAM not available. Install with: pip install git+https://github.com/facebookresearch/segment-anything.git")

# Import existing GNN components
from core.gnn_module import SegmentationGNN

class SAMFeatureExtractor(nn.Module):
    """SAM-based feature extractor for tissue segmentation"""
    
    def __init__(
        self,
        sam_model_type: str = 'vit_b',
        freeze_sam: bool = True,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        self.sam_model_type = sam_model_type
        self.freeze_sam = freeze_sam
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize SAM model or create dummy
        self._init_sam_model()
        
        # Feature adaptation layers
        sam_embed_dim = self._get_sam_embed_dim()
        self.feature_adapter = nn.Sequential(
            nn.Conv2d(sam_embed_dim, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
    
    def _init_sam_model(self):
        """Initialize SAM model or create dummy for testing"""
        self.using_dummy_sam = False  # Track if we're using dummy SAM
        
        if SAM_AVAILABLE:
            try:
                print(f"Loading SAM model: {self.sam_model_type}")
                # Try to load actual SAM model
                checkpoint_path = f"sam_{self.sam_model_type}.pth"
                self.sam = sam_model_registry[self.sam_model_type](checkpoint=checkpoint_path)
                print("SAM model loaded successfully")
            except:
                print("SAM weights not found, creating dummy model for testing")
                self.sam = self._create_dummy_sam()
                self.using_dummy_sam = True  # Mark as using dummy
        else:
            print("SAM not available, creating dummy model")
            self.sam = self._create_dummy_sam()
            self.using_dummy_sam = True  # Mark as using dummy
        
        self.sam.to(self.device)
        
        if self.freeze_sam:
            for param in self.sam.parameters():
                param.requires_grad = False
    
    def _create_dummy_sam(self):
        """Create a dummy SAM-like model for testing without actual SAM weights"""
        class DummySAM(nn.Module):
            def __init__(self):
                super().__init__()
                self.image_encoder = nn.Sequential(
                    # Reduced downsampling to maintain spatial resolution
                    nn.Conv2d(3, 64, 7, stride=1, padding=3),  # Changed stride from 2 to 1
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(2, stride=2, padding=0),  # Only 2x downsampling
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU()
                )
            
            def forward(self, x):
                return self.image_encoder(x)
        
        return DummySAM()
    
    def _get_sam_embed_dim(self):
        """Get SAM embedding dimension based on model type"""
        # If using dummy SAM, return dummy output channels
        if hasattr(self, 'using_dummy_sam') and self.using_dummy_sam:
            return 256  # Dummy SAM output channels
        
        if SAM_AVAILABLE:
            embed_dims = {
                'vit_b': 768,
                'vit_l': 1024, 
                'vit_h': 1280
            }
            return embed_dims.get(self.sam_model_type, 768)
        else:
            return 256  # For dummy model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features using SAM"""
        # Convert single channel to RGB if needed
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # Extract SAM features
        with torch.set_grad_enabled(not self.freeze_sam):
            if hasattr(self.sam, 'image_encoder'):
                sam_features = self.sam.image_encoder(x)
            else:
                sam_features = self.sam(x)
        
        # Adapt features for our task
        adapted_features = self.feature_adapter(sam_features)
        
        return adapted_features

class CellposeSAMModel(nn.Module):
    """
    SAM-based model for tissue segmentation
    
    This model uses SAM (Segment Anything Model) for powerful feature extraction
    combined with GNN for spatial relationship modeling.
    """
    
    def __init__(
        self,
        num_classes: int = 6,
        sam_model_type: str = 'vit_b',
        freeze_sam: bool = True,
        gnn_hidden_channels: int = 128,
        device: Optional[torch.device] = None
    ):
        """
        Initialize CellposeSAMModel
        
        Args:
            num_classes: Number of output classes
            sam_model_type: SAM model type ('vit_b', 'vit_l', 'vit_h')
            freeze_sam: Whether to freeze SAM weights during training
            gnn_hidden_channels: Hidden channels for GNN
            device: Device to load model on
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.sam_model_type = sam_model_type
        self.freeze_sam = freeze_sam
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # SAM feature extractor
        self.sam_extractor = SAMFeatureExtractor(
            sam_model_type=sam_model_type,
            freeze_sam=freeze_sam,
            device=self.device
        )
        
        # Feature fusion and projection
        sam_features = 128  # From SAM adapter
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(sam_features, gnn_hidden_channels, 3, padding=1),
            nn.BatchNorm2d(gnn_hidden_channels),
            nn.ReLU(),
            nn.Conv2d(gnn_hidden_channels, gnn_hidden_channels, 3, padding=1),
            nn.BatchNorm2d(gnn_hidden_channels),
            nn.ReLU()
        )
        
        # GNN layers for spatial relationships
        self.gnn_layers = nn.ModuleList([
            SegmentationGNN(gnn_hidden_channels, gnn_hidden_channels)
            for _ in range(3)
        ])
        
        # Multi-scale pooling
        self.pooling_blocks = nn.ModuleList([
            self._create_pooling_block(gnn_hidden_channels, gnn_hidden_channels // 2),
            self._create_pooling_block(gnn_hidden_channels // 2, gnn_hidden_channels // 4)
        ])
        
        # Final classifier
        total_features = gnn_hidden_channels + gnn_hidden_channels // 2 + gnn_hidden_channels // 4
        self.classifier = nn.Sequential(
            nn.Conv2d(total_features, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, num_classes, 1)
        )
        
        # Initialize weights
        self._init_weights()
        
        # Print model info
        self._print_model_info()
    
    def _create_pooling_block(self, in_channels: int, out_channels: int):
        """Create a pooling block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def _init_weights(self):
        """Initialize weights for new layers"""
        for module in [self.feature_fusion, self.classifier]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
    
    def _print_model_info(self):
        """Print model information"""
        print("\n" + "="*50)
        print("SAM-BASED SEGMENTATION MODEL")
        print("="*50)
        print(f"SAM Model Type: {self.sam_model_type}")
        print(f"SAM Available: {SAM_AVAILABLE}")
        print(f"Using Dummy SAM: {getattr(self.sam_extractor, 'using_dummy_sam', False)}")
        print(f"Freeze SAM: {self.freeze_sam}")
        print(f"Output Classes: {self.num_classes}")
        print(f"Device: {self.device}")
        print("="*50)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Store original input size for final upsampling
        original_size = x.shape[2:]  # (H, W)
        
        # Extract SAM features
        sam_features = self.sam_extractor(x)
        
        # Feature fusion
        fused_features = self.feature_fusion(sam_features)
        
        # Apply GNN layers
        gnn_features = fused_features
        for gnn_layer in self.gnn_layers:
            gnn_features = gnn_layer(gnn_features)
        
        # Multi-scale features
        scale_features = [gnn_features]
        current_features = gnn_features
        
        for pooling_block in self.pooling_blocks:
            # Apply pooling
            current_features = F.avg_pool2d(current_features, kernel_size=2, stride=2)
            current_features = pooling_block(current_features)
            
            # Upsample back to original size
            upsampled = F.interpolate(
                current_features, 
                size=gnn_features.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
            scale_features.append(upsampled)
        
        # Concatenate multi-scale features
        multi_scale_features = torch.cat(scale_features, dim=1)
        
        # Final classification
        output = self.classifier(multi_scale_features)
        
        # Ensure output matches original input size
        if output.shape[2:] != original_size:
            output = F.interpolate(
                output, 
                size=original_size, 
                mode='bilinear', 
                align_corners=False
            )
        
        return output
    
    def get_sam_parameters(self):
        """Get SAM parameters"""
        return list(self.sam_extractor.parameters())
    
    def get_gnn_parameters(self):
        """Get GNN and other trainable parameters"""
        params = []
        for module in [self.feature_fusion, self.gnn_layers, self.pooling_blocks, self.classifier]:
            params.extend(list(module.parameters()))
        return params
    
    def freeze_sam_layers(self):
        """Freeze SAM components"""
        for param in self.sam_extractor.parameters():
            param.requires_grad = False
    
    def unfreeze_sam_layers(self):
        """Unfreeze SAM components"""
        for param in self.sam_extractor.parameters():
            param.requires_grad = True

def create_cellpose_sam_model(
    num_classes: int = 6,
    sam_model_type: str = 'vit_b',
    freeze_sam: bool = True,
    **kwargs
) -> CellposeSAMModel:
    """
    Create CellposeSAMModel with specified configuration
    
    Args:
        num_classes: Number of output classes
        sam_model_type: SAM model type ('vit_b', 'vit_l', 'vit_h')
        freeze_sam: Whether to freeze SAM weights
        **kwargs: Additional arguments
        
    Returns:
        Configured CellposeSAMModel
    """
    return CellposeSAMModel(
        num_classes=num_classes,
        sam_model_type=sam_model_type,
        freeze_sam=freeze_sam,
        **kwargs
    ) 