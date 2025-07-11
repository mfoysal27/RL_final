# unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """Double convolution block for UNet"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """UNet architecture for image segmentation"""
    def __init__(self, in_channels=1, num_classes=6, features=[64, 128, 256, 512], temperature=1.0):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.n_classes = num_classes  # Store number of classes as attribute
        self.temperature = temperature  # Temperature parameter for softmax
        
        # Downsampling part
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
            
        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        
        # Upsampling part
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature*2, feature))
            
        # Final convolution
        self.final_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)
        
    def forward(self, x):
        skip_connections = []
        
        # Downsampling and save skip connections
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # Reverse list
        
        # Upsampling and concatenate with skip connections
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # Upsampling
            skip_connection = skip_connections[idx//2]
            
            # Handle cases where dimensions don't match due to rounding
            if x.shape != skip_connection.shape:
                x = F.interpolate(
                    x, size=skip_connection.shape[2:], mode="bilinear", align_corners=True
                )
                
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)  # Double conv
        
        # Apply final convolution
        logits = self.final_conv(x)
        
        # Apply temperature scaling to logits (but don't apply softmax here)
        # This helps control the "confidence" of the model's predictions
        if self.temperature != 1.0:
            logits = logits / self.temperature
            
        return logits