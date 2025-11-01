# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Vit import VisionTransformer, Reconstruct
from .pixlevel import PixLevelModule


def get_activation(activation_type):
    """Get activation function from string name, defaults to ReLU if not found."""
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()


def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    """Create a sequence of n convolutional layers with batch norm and activation."""
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))
    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class ConvBatchNorm(nn.Module):
    """Basic building block: Conv2d -> BatchNorm2d -> Activation"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        # 3x3 convolution with padding to maintain spatial dimensions
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        # Batch normalization for stable training
        self.norm = nn.BatchNorm2d(out_channels)
        # Get activation function based on input string
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class DownBlock(nn.Module):
    """Downsampling block: MaxPool -> Multiple Conv layers"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        # Max pooling to reduce spatial dimensions by half
        self.maxpool = nn.MaxPool2d(2)
        # Convolution layers for feature extraction after pooling
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)  # Reduce spatial size
        return self.nConvs(out)


class Flatten(nn.Module):
    """Flatten tensor to (batch_size, -1)"""
    def forward(self, x):
        return x.view(x.size(0), -1)


class UpblockAttention(nn.Module):
    """Upsampling block with pixel-level attention mechanism."""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        # Upsample to double spatial dimensions
        self.up = nn.Upsample(scale_factor=2)
        # Pixel-level attention module to enhance skip connections
        self.pixModule = PixLevelModule(in_channels // 2)
        # Convolution layers after concatenation
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        up = self.up(x)  # Upsample the input
        # Apply pixel-level attention to skip connection
        skip_x_att = self.pixModule(skip_x)
        # Concatenate attention-weighted skip with upsampled input
        x = torch.cat([skip_x_att, up], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)


class LViT(nn.Module):
    def __init__(self, config, n_channels=3, n_classes=1, img_size=224, vis=False):
        super().__init__()
        self.vis = vis
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = config.base_channel
        
        # Initial convolution block
        self.inc = ConvBatchNorm(n_channels, in_channels)
        
        # Vision Transformers for different scales in encoder
        self.downVit = VisionTransformer(config, vis, img_size=224, channel_num=64, patch_size=16, embed_dim=64)
        self.downVit1 = VisionTransformer(config, vis, img_size=112, channel_num=128, patch_size=8, embed_dim=128)
        self.downVit2 = VisionTransformer(config, vis, img_size=56, channel_num=256, patch_size=4, embed_dim=256)
        self.downVit3 = VisionTransformer(config, vis, img_size=28, channel_num=512, patch_size=2, embed_dim=512)
        
        # Vision Transformers for different scales in decoder
        self.upVit = VisionTransformer(config, vis, img_size=224, channel_num=64, patch_size=16, embed_dim=64)
        self.upVit1 = VisionTransformer(config, vis, img_size=112, channel_num=128, patch_size=8, embed_dim=128)
        self.upVit2 = VisionTransformer(config, vis, img_size=56, channel_num=256, patch_size=4, embed_dim=256)
        self.upVit3 = VisionTransformer(config, vis, img_size=28, channel_num=512, patch_size=2, embed_dim=512)
        
        # Down blocks for U-Net encoder path
        self.down1 = DownBlock(in_channels, in_channels * 2, nb_Conv=2)
        self.down2 = DownBlock(in_channels * 2, in_channels * 4, nb_Conv=2)
        self.down3 = DownBlock(in_channels * 4, in_channels * 8, nb_Conv=2)
        self.down4 = DownBlock(in_channels * 8, in_channels * 8, nb_Conv=2)
        
        # Up blocks with attention for U-Net decoder path
        self.up4 = UpblockAttention(in_channels * 16, in_channels * 4, nb_Conv=2)
        self.up3 = UpblockAttention(in_channels * 8, in_channels * 2, nb_Conv=2)
        self.up2 = UpblockAttention(in_channels * 4, in_channels, nb_Conv=2)
        self.up1 = UpblockAttention(in_channels * 2, in_channels, nb_Conv=2)
        
        # Output convolution to get desired number of classes
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1), stride=(1, 1))
        
        # Activation functions for output
        self.last_activation = nn.Sigmoid()  # if using BCELoss
        self.multi_activation = nn.Softmax()
        
        # Reconstruction modules to restore spatial information from transformer features
        self.reconstruct1 = Reconstruct(in_channels=64, out_channels=64, kernel_size=1, scale_factor=(16, 16))
        self.reconstruct2 = Reconstruct(in_channels=128, out_channels=128, kernel_size=1, scale_factor=(8, 8))
        self.reconstruct3 = Reconstruct(in_channels=256, out_channels=256, kernel_size=1, scale_factor=(4, 4))
        self.reconstruct4 = Reconstruct(in_channels=512, out_channels=512, kernel_size=1, scale_factor=(2, 2))
        
        # Pixel-level attention modules for skip connections
        self.pix_module1 = PixLevelModule(64)
        self.pix_module2 = PixLevelModule(128)
        self.pix_module3 = PixLevelModule(256)
        self.pix_module4 = PixLevelModule(512)
        
        # Text processing modules for multimodal fusion
        self.text_module4 = nn.Conv1d(in_channels=768, out_channels=512, kernel_size=3, padding=1)
        self.text_module3 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.text_module2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.text_module1 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)

    def forward(self, x, text):
        x = x.float()  # x [4,3,224,224]
        x1 = self.inc(x)  # x1 [4, 64, 224, 224]
        
        # Process text features at different scales
        text4 = self.text_module4(text.transpose(1, 2)).transpose(1, 2)  # Downsample text to match scale 4
        text3 = self.text_module3(text4.transpose(1, 2)).transpose(1, 2)  # Downsample text to match scale 3
        text2 = self.text_module2(text3.transpose(1, 2)).transpose(1, 2)  # Downsample text to match scale 2
        text1 = self.text_module1(text2.transpose(1, 2)).transpose(1, 2)  # Downsample text to match scale 1
        
        # Encoder path with Vision Transformers
        y1 = self.downVit(x1, x1, text1)  # Process at 224x224 scale with text info
        x2 = self.down1(x1)  # Downsample spatially
        y2 = self.downVit1(x2, y1, text2)  # Process at 112x112 scale with text info
        x3 = self.down2(x2)  # Downsample spatially
        y3 = self.downVit2(x3, y2, text3)  # Process at 56x56 scale with text info
        x4 = self.down3(x3)  # Downsample spatially
        y4 = self.downVit3(x4, y3, text4)  # Process at 28x28 scale with text info
        x5 = self.down4(x4)  # Final downsample to bottleneck
        
        # Decoder path with Vision Transformers (reconstruction mode)
        y4 = self.upVit3(y4, y4, text4, True)  # Reconstruct at 28x28 scale
        y3 = self.upVit2(y3, y4, text3, True)  # Reconstruct at 56x56 scale
        y2 = self.upVit1(y2, y3, text2, True)  # Reconstruct at 112x112 scale
        y1 = self.upVit(y1, y2, text1, True)  # Reconstruct at 224x224 scale
        
        # Add reconstructed transformer features back to CNN features
        x1 = self.reconstruct1(y1) + x1  # Residual connection at scale 1
        x2 = self.reconstruct2(y2) + x2  # Residual connection at scale 2
        x3 = self.reconstruct3(y3) + x3  # Residual connection at scale 3
        x4 = self.reconstruct4(y4) + x4  # Residual connection at scale 4
        
        # U-Net decoder with attention
        x = self.up4(x5, x4)  # Upsample and combine with attention
        x = self.up3(x, x3)   # Upsample and combine with attention
        x = self.up2(x, x2)   # Upsample and combine with attention
        x = self.up1(x, x1)   # Upsample and combine with attention
        
        # Final output with appropriate activation
        if self.n_classes == 1:
            logits = self.last_activation(self.outc(x))  # Sigmoid for binary segmentation
        else:
            logits = self.outc(x)  # if not using BCEWithLogitsLoss or class>1
        return logits