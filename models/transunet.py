"""
TransUNet: Transformers for Medical Image Segmentation
Adapted for biological image segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention computation
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x

class TransformerBlock(nn.Module):
    """Transformer encoder block"""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual connection
        x = x + self.attn(self.norm1(x))
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        return x

class PatchEmbedding(nn.Module):
    """Image to patch embedding for transformer"""
    
    def __init__(self, img_size: int = 256, patch_size: int = 16, in_channels: int = 1, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Project to patches
        x = self.proj(x)  # [B, embed_dim, H//patch_size, W//patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        x = self.norm(x)
        
        return x

class TransformerEncoder(nn.Module):
    """Transformer encoder with multiple blocks"""
    
    def __init__(self, embed_dim: int = 768, depth: int = 12, num_heads: int = 12, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

class CNNEncoder(nn.Module):
    """CNN encoder for feature extraction"""
    
    def __init__(self, in_channels: int = 1, features: list = [64, 128, 256, 512]):
        super().__init__()
        self.features = features
        
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        in_ch = in_channels
        for feature in features:
            self.encoder_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, feature, 3, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feature, feature, 3, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(inplace=True)
                )
            )
            self.pools.append(nn.MaxPool2d(2, 2))
            in_ch = feature
            
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        skip_connections = []
        
        for encoder, pool in zip(self.encoder_blocks, self.pools):
            x = encoder(x)
            skip_connections.append(x)
            x = pool(x)
            
        return x, skip_connections

class TransUNet(nn.Module):
    """
    TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation
    Adapted for biological image segmentation
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 6,
        img_size: int = 256,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        cnn_features: list = [64, 128, 256, 512]
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # CNN encoder for multi-scale features
        self.cnn_encoder = CNNEncoder(in_channels, cnn_features)
        
        # Patch embedding for transformer
        self.patch_embed = PatchEmbedding(img_size // 4, patch_size, cnn_features[-1], embed_dim)  # After 4x downsampling
        
        # Positional embedding
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        
        # Transformer encoder
        self.transformer = TransformerEncoder(embed_dim, depth, num_heads, mlp_ratio, dropout)
        
        # Decoder with skip connections
        self.decoder = self._build_decoder(cnn_features, embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
        
    def _build_decoder(self, cnn_features: list, embed_dim: int, num_classes: int) -> nn.ModuleList:
        """Build decoder with skip connections"""
        decoder_blocks = nn.ModuleList()
        
        # First, project transformer features back to CNN space
        self.transformer_proj = nn.Sequential(
            nn.Linear(embed_dim, cnn_features[-1]),
            nn.ReLU(inplace=True)
        )
        
        # Decoder blocks (reverse order of encoder)
        features = cnn_features[::-1]  # [512, 256, 128, 64]
        
        for i in range(len(features) - 1):
            in_channels = features[i] + features[i + 1]  # Skip connection
            out_channels = features[i + 1]
            
            decoder_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(features[i], features[i], 2, 2),  # Upsample
                    nn.Conv2d(in_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Final classification layer
        decoder_blocks.append(nn.Conv2d(features[-1], num_classes, 1))
        
        return decoder_blocks
        
    def _init_weights(self):
        """Initialize model weights"""
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Initialize linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """Forward pass through TransUNet"""
        B, C, H, W = x.shape
        
        # CNN encoder for multi-scale features
        cnn_features, skip_connections = self.cnn_encoder(x)
        
        # Transformer processing
        # Convert CNN features to patches
        transformer_input = self.patch_embed(cnn_features)
        
        # Add positional embedding
        transformer_input = transformer_input + self.pos_embed
        
        # Apply transformer
        transformer_output = self.transformer(transformer_input)
        
        # Reshape transformer output back to spatial format
        patch_h = patch_w = self.img_size // (4 * self.patch_size)
        transformer_spatial = transformer_output.transpose(1, 2).reshape(
            B, self.embed_dim, patch_h, patch_w
        )
        
        # Project back to CNN feature space
        transformer_spatial = transformer_spatial.permute(0, 2, 3, 1)  # [B, H, W, C]
        transformer_spatial = self.transformer_proj(transformer_spatial)
        transformer_spatial = transformer_spatial.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # Decoder with skip connections
        x = transformer_spatial
        skip_connections = skip_connections[::-1]  # Reverse for decoder
        
        for i, decoder_block in enumerate(self.decoder[:-1]):
            # Upsample
            if i == 0:
                x = decoder_block[0](x)  # ConvTranspose2d
            else:
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            
            # Skip connection
            if i < len(skip_connections):
                skip = skip_connections[i]
                if x.shape != skip.shape:
                    x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
                x = torch.cat([x, skip], dim=1)
            
            # Apply decoder block (except the first ConvTranspose2d which we already applied)
            for layer in decoder_block[1:] if i == 0 else decoder_block:
                x = layer(x)
        
        # Final classification
        output = self.decoder[-1](x)
        
        # Ensure output matches input size
        if output.shape[2:] != (H, W):
            output = F.interpolate(output, size=(H, W), mode='bilinear', align_corners=False)
        
        if return_features:
            return output, transformer_spatial
        
        return output
    
    def get_transformer_parameters(self):
        """Get transformer parameters for separate optimization"""
        params = []
        params.extend(list(self.patch_embed.parameters()))
        params.extend(list(self.transformer.parameters()))
        params.extend(list(self.transformer_proj.parameters()))
        params.append(self.pos_embed)
        return params
    
    def get_cnn_parameters(self):
        """Get CNN parameters for separate optimization"""
        params = []
        params.extend(list(self.cnn_encoder.parameters()))
        params.extend(list(self.decoder.parameters()))
        return params 