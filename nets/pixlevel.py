# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

'''pixel-level module'''


class PixLevelModule(nn.Module):
    def __init__(self, in_channels):
        super(PixLevelModule, self).__init__()
        self.middle_layer_size_ratio = 2  # Multiplier for bottleneck intermediate layer size
        
        # Convolution for average pooling path
        self.conv_avg = nn.Conv2d(in_channels, out_channels=in_channels, kernel_size=1, bias=False)
        self.relu_avg = nn.ReLU(inplace=True)
        
        # Convolution for max pooling path
        self.conv_max = nn.Conv2d(in_channels, out_channels=in_channels, kernel_size=1, bias=False)
        self.relu_max = nn.ReLU(inplace=True)
        
        # Bottleneck MLP to process the combined features
        self.bottleneck = nn.Sequential(
            nn.Linear(3, 3 * self.middle_layer_size_ratio),  # Expand from 3 to 6 channels
            nn.ReLU(inplace=True),
            nn.Linear(3 * self.middle_layer_size_ratio, 1)  # Reduce back to 1 channel
        )
        
        # Sigmoid attention map generator
        self.conv_sig = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    '''forward'''

    def forward(self, x):
        # Process through average pooling path
        x_avg = self.conv_avg(x)  
        x_avg = self.relu_avg(x_avg) 
        x_avg = torch.mean(x_avg, dim=1)  # Average across channels [B, H, W]
        x_avg = x_avg.unsqueeze(dim=1)    # Add channel dimension back [B, 1, H, W]
        
        # Process through max pooling path
        x_max = self.conv_max(x)
        x_max = self.relu_max(x_max)
        x_max = torch.max(x_max, dim=1).values  # Max across channels [B, H, W]
        x_max = x_max.unsqueeze(dim=1)         # Add channel dimension back [B, 1, H, W]
        
        # Combine average and max features
        x_out = x_max + x_avg
        
        # Concatenate all three representations
        x_output = torch.cat((x_avg, x_max, x_out), dim=1)  # [B, 3, H, W]
        
        # Reshape for MLP processing: [B, H, W, 3]
        x_output = x_output.transpose(1, 3) 
        
        # Apply bottleneck MLP to learn channel attention weights
        x_output = self.bottleneck(x_output)  # [B, H, W, 1]
        
        # Reshape back: [B, 1, H, W]
        x_output = x_output.transpose(1, 3) 
        
        # Apply attention to original input
        y = x_output * x  # Element-wise multiplication for attention
        return y