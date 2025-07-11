import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet import UNet
from core.gnn_module import SegmentationGNN

class HybridUNetGNN(nn.Module):
    """Hybrid model combining UNet with GNN for improved segmentation"""
    def __init__(self, in_channels=1, num_classes=6, features=[64, 128, 256, 512], gnn_hidden_channels=64):
        super(HybridUNetGNN, self).__init__()
        
        # Store configuration
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.features = features
        self.gnn_hidden_channels = gnn_hidden_channels
        
        # Create UNet components directly instead of using UNet class
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Downsampling part
        in_channels_temp = in_channels
        for feature in features:
            self.downs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels_temp, feature, kernel_size=3, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(inplace=True)
                )
            )
            in_channels_temp = feature
            
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[-1], features[-1]*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(features[-1]*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[-1]*2, features[-1]*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(features[-1]*2),
            nn.ReLU(inplace=True)
        )
        
        # Upsampling part
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(feature*2, feature, kernel_size=3, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(inplace=True)
                )
            )
            
        # Final convolution
        self.final_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)
        
        # GNN modules for feature refinement at different scales
        self.gnn_modules = nn.ModuleList()
        
        # Add GNN modules for each scale in the decoder path
        for feature_channels in reversed(features):
            self.gnn_modules.append(
                SegmentationGNN(
                    in_channels=feature_channels,
                    hidden_channels=gnn_hidden_channels
                )
            )
        
        # Final GNN for refining logits
        self.final_gnn = SegmentationGNN(
            in_channels=num_classes,
            hidden_channels=gnn_hidden_channels
        )
        
    def forward(self, x):
        """
        Forward pass through the hybrid model
        
        Args:
            x: Input image tensor (B, C, H, W)
            
        Returns:
            Refined segmentation logits (B, num_classes, H, W)
        """
        # Encoder Path
        skip_connections = []
        
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder Path with GNN refinement
        skip_connections = skip_connections[::-1]
        
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=True)
            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
            
            # Apply GNN refinement at each decoder stage
            x = self.gnn_modules[idx//2](x)
        
        # Final convolution
        x = self.final_conv(x)
        
        # Final GNN refinement
        x = self.final_gnn(x)
        
        return x
    
    def get_unet_parameters(self):
        """Get parameters of the UNet backbone"""
        params = []
        params.extend([p for down in self.downs for p in down.parameters()])
        params.extend([p for up in self.ups for p in up.parameters()])
        params.extend(list(self.bottleneck.parameters()))
        params.extend(list(self.final_conv.parameters()))
        return params
    
    def get_gnn_parameters(self):
        """Get parameters of the GNN modules"""
        params = []
        for gnn_module in self.gnn_modules:
            params.extend(list(gnn_module.parameters()))
        params.extend(list(self.final_gnn.parameters()))
        return params
        
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Override state_dict to ensure proper key naming"""
        state_dict = super().state_dict(destination, prefix, keep_vars)
        
        # Add model configuration to state dict
        state_dict['config'] = {
            'num_classes': self.num_classes,
            'in_channels': self.in_channels,
            'features': self.features,
            'gnn_hidden_channels': self.gnn_hidden_channels
        }
        
        return state_dict
        
    def load_state_dict(self, state_dict, strict=True):
        """Override load_state_dict to handle configuration and ensure proper loading"""
        # Extract configuration if present
        if 'config' in state_dict:
            config = state_dict.pop('config')
            # Update model configuration if needed
            if hasattr(self, 'num_classes'):
                self.num_classes = config.get('num_classes', self.num_classes)
            if hasattr(self, 'in_channels'):
                self.in_channels = config.get('in_channels', self.in_channels)
            if hasattr(self, 'features'):
                self.features = config.get('features', self.features)
            if hasattr(self, 'gnn_hidden_channels'):
                self.gnn_hidden_channels = config.get('gnn_hidden_channels', self.gnn_hidden_channels)
        
        # Call parent class load_state_dict
        return super().load_state_dict(state_dict, strict) 