# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
from timm.layers import DropPath
from torch.nn import Dropout, Conv2d
from torch.nn.modules.utils import _pair

class Reconstruct(nn.Module):
    """Upsample and reconstruct feature maps from patch embeddings."""
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Reconstruct, self).__init__()
        # Set padding based on kernel size
        if kernel_size == 3:
            padding = 1
        else:
            padding = 0
        # Convolution to reconstruct spatial information
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        # Batch normalization and activation
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        # Upsampling scale factor
        self.scale_factor = scale_factor

    def forward(self, x):
        if x is None:
            return None

        # Reshape from patch embeddings back to spatial format
        B, n_patch, hidden = x.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1)  # [B, hidden, n_patch]
        x = x.contiguous().view(B, hidden, h, w)  # [B, hidden, h, w]
        x = nn.Upsample(scale_factor=self.scale_factor)(x)  # Upsample spatial dimensions

        # Apply reconstruction layers
        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out


class Embeddings(nn.Module):
    # Construct the patch, position embeddings
    def __init__(self, config, patch_size, img_size, in_channels):
        super().__init__()
        # Convert to pairs for 2D processing
        img_size = _pair(img_size)
        patch_size = _pair(patch_size)
        # Calculate number of patches
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        
        # Patch embedding using convolution (extract patches and project to hidden dimension)
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=in_channels,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        # Learnable position embeddings for patch sequence
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, in_channels))
        # Dropout for regularization
        self.dropout = Dropout(0.1)

    def forward(self, x):
        if x is None:
            return None
        # Extract patches using convolution
        x = self.patch_embeddings(x)  # (B, hidden, n_patches^(1/2), n_patches^(1/2))
        # Flatten spatial dimensions
        x = x.flatten(2)  # (B, hidden, n_patches)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        # Add position embeddings and apply dropout
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class MLP(nn.Module):
    """Multi-Layer Perceptron with GELU activation."""
    def __init__(self, in_dim, hidden_dim=None, out_dim=None):
        super().__init__()
        # Linear layers with GELU activation
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act_layer = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        # Dropout for regularization
        self.dropout = Dropout(0.1)
        self._init_weights()

    def _init_weights(self):
        # Initialize weights with Xavier uniform and biases with normal distribution
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        # Forward pass through MLP
        x = self.fc1(x)  # [B, num_patches, hidden_dim]
        x = self.act_layer(x)
        x = self.dropout(x)
        x = self.fc2(x)  # [B, num_patches, out_dim]
        x = self.act_layer(x)  # Apply activation after second linear layer
        x = self.dropout(x)
        return x


class Attention(nn.Module):
    """Multi-Head Self-Attention mechanism."""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5  # Scale factor for attention scores

        # Linear layer to generate Q, K, V from input
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        # Output projection
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape  # Batch, Num_patches, Embedding_dim
        
        # Generate Q, K, V matrices
        qkv = self.qkv(x)  # [B, num_patches, 3*embed_dim]
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)  # [B, num_patches, 3, num_heads, per_HeadDim]
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, num_patches, per_HeadDim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, num_patches, per_HeadDim]

        # Calculate attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, num_patches, num_patches]
        attn = attn.softmax(dim=-1)  # Apply softmax to get attention weights
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v)  # [B, num_heads, num_patches, per_HeadDim]
        x = x.transpose(1, 2)  # [B, num_patches, num_heads, per_HeadDim]
        x = x.reshape(B, N, C)  # [B, num_patches, embed_dim]
        x = self.proj(x)  # [B, num_patches, embed_dim]
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """Transformer block: attention -> MLP with residual connections."""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        # Layer normalization before attention
        self.norm1 = norm_layer(dim)
        # Multi-head self-attention module
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # Stochastic depth for regularization
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # Layer normalization before MLP
        self.norm2 = norm_layer(dim)
        # MLP hidden dimension calculation
        self.mlp_hidden_dim = int(dim * mlp_ratio)
        # MLP module
        self.mlp = MLP(in_dim=dim, hidden_dim=self.mlp_hidden_dim, out_dim=dim)

    def forward(self, x):
        # Pre-norm transformer block with residual connections
        x = x + self.drop_path(self.attn(self.norm1(x)))  # Attention with residual
        x = x + self.drop_path(self.mlp(self.norm2(x)))   # MLP with residual
        return x


class ConvTransBN(nn.Module):  # (convolution => [BN] => ReLU)
    """1D Convolution block: Conv1d -> BatchNorm1d -> ReLU."""
    def __init__(self, in_channels, out_channels):
        super(ConvTransBN, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class VisionTransformer(nn.Module):  # Transformer-branch
    def __init__(self, config, vis, img_size, channel_num, patch_size, embed_dim, depth=1, num_heads=8,
                 mlp_ratio=4., qkv_bias=True, num_classes=1, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super(VisionTransformer, self).__init__()
        self.config = config
        self.vis = vis
        
        # Patch embedding layer
        self.embeddings = Embeddings(config=config, patch_size=patch_size, img_size=img_size, in_channels=channel_num)
        self.depth = depth
        self.dim = embed_dim
        
        # Normalization and activation layers
        norm_layer = nn.LayerNorm
        self.norm = norm_layer(embed_dim)
        act_layer = nn.GELU

        # Stochastic depth decay rates
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        
        # Transformer encoder blocks
        self.Encoder_blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                  attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(self.depth)])

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        # 1D convolution blocks for processing transformer features
        self.CTBN = ConvTransBN(in_channels=embed_dim, out_channels=embed_dim//2)  # For downsampling
        self.CTBN2 = ConvTransBN(in_channels=embed_dim*2, out_channels=embed_dim)  # For reconstruction
        self.CTBN3 = ConvTransBN(in_channels=10, out_channels=196)  # For text processing

    def forward(self, x, skip_x, text, reconstruct=False):
        if not reconstruct:
            # Initial patch embedding
            x = self.embeddings(x)
            # Add text information if this is the final transformer layer
            if self.dim == 64:
                x = x+self.CTBN3(text)  # [B, num_patches, embed_dim]
            # Process through transformer blocks
            x = self.Encoder_blocks(x)
        else:
            # Process through transformer blocks during reconstruction
            x = self.Encoder_blocks(x)
            
        # Return intermediate results for different transformer stages
        if (self.dim == 64 and not reconstruct) or (self.dim == 512 and reconstruct):
            return x
        elif not reconstruct:
            # Process for skip connection: downsample and concatenate with skip
            x = x.transpose(1, 2)  # [B, embed_dim, num_patches]
            x = self.CTBN(x)  # [B, embed_dim//2, num_patches] - downsample
            x = x.transpose(1, 2)  # [B, num_patches, embed_dim//2]
            y = torch.cat([x, skip_x], dim=2)  # [B, num_patches, embed_dim] - concatenate with skip connection
            return y
        elif reconstruct:
            # Process skip connection and add to transformer output during reconstruction
            skip_x = skip_x.transpose(1, 2)  # [B, embed_dim, num_patches]
            skip_x = self.CTBN2(skip_x)  # [B, embed_dim, num_patches] - process skip
            skip_x = skip_x.transpose(1, 2)  # [B, num_patches, embed_dim]
            y = x+skip_x  # Add processed skip to transformer output
            return y