# # -*- coding: utf-8 -*-

# import torch
# import torch.nn as nn
# import numpy as np
# from timm.layers import DropPath
# from torch.nn import Dropout, Conv2d
# from torch.nn.modules.utils import _pair

# class Reconstruct(nn.Module):
#     """Upsample and reconstruct feature maps from patch embeddings."""
#     def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
#         super(Reconstruct, self).__init__()
#         # Set padding based on kernel size
#         if kernel_size == 3:
#             padding = 1
#         else:
#             padding = 0
#         # Convolution to reconstruct spatial information
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
#         # Batch normalization and activation
#         self.norm = nn.BatchNorm2d(out_channels)
#         self.activation = nn.ReLU(inplace=True)
#         # Upsampling scale factor
#         self.scale_factor = scale_factor

#     def forward(self, x):
#         if x is None:
#             return None

#         # Reshape from patch embeddings back to spatial format
#         B, n_patch, hidden = x.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
#         h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
#         x = x.permute(0, 2, 1)  # [B, hidden, n_patch]
#         x = x.contiguous().view(B, hidden, h, w)  # [B, hidden, h, w]
#         x = nn.Upsample(scale_factor=self.scale_factor)(x)  # Upsample spatial dimensions

#         # Apply reconstruction layers
#         out = self.conv(x)
#         out = self.norm(out)
#         out = self.activation(out)
#         return out


# class Embeddings(nn.Module):
#     # Construct the patch, position embeddings
#     def __init__(self, config, patch_size, img_size, in_channels):
#         super().__init__()
#         # Convert to pairs for 2D processing
#         img_size = _pair(img_size)
#         patch_size = _pair(patch_size)
#         # Calculate number of patches
#         n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        
#         # Patch embedding using convolution (extract patches and project to hidden dimension)
#         self.patch_embeddings = Conv2d(in_channels=in_channels,
#                                        out_channels=in_channels,
#                                        kernel_size=patch_size,
#                                        stride=patch_size)
#         # Learnable position embeddings for patch sequence
#         self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, in_channels))
#         # Dropout for regularization
#         self.dropout = Dropout(0.1)

#     def forward(self, x):
#         if x is None:
#             return None
#         # Extract patches using convolution
#         x = self.patch_embeddings(x)  # (B, hidden, n_patches^(1/2), n_patches^(1/2))
#         # Flatten spatial dimensions
#         x = x.flatten(2)  # (B, hidden, n_patches)
#         x = x.transpose(-1, -2)  # (B, n_patches, hidden)
#         # Add position embeddings and apply dropout
#         embeddings = x + self.position_embeddings
#         embeddings = self.dropout(embeddings)
#         return embeddings


# class MLP(nn.Module):
#     """Multi-Layer Perceptron with GELU activation."""
#     def __init__(self, in_dim, hidden_dim=None, out_dim=None):
#         super().__init__()
#         # Linear layers with GELU activation
#         self.fc1 = nn.Linear(in_dim, hidden_dim)
#         self.act_layer = nn.GELU()
#         self.fc2 = nn.Linear(hidden_dim, out_dim)
#         # Dropout for regularization
#         self.dropout = Dropout(0.1)
#         self._init_weights()

#     def _init_weights(self):
#         # Initialize weights with Xavier uniform and biases with normal distribution
#         nn.init.xavier_uniform_(self.fc1.weight)
#         nn.init.xavier_uniform_(self.fc2.weight)
#         nn.init.normal_(self.fc1.bias, std=1e-6)
#         nn.init.normal_(self.fc2.bias, std=1e-6)

#     def forward(self, x):
#         # Forward pass through MLP
#         x = self.fc1(x)  # [B, num_patches, hidden_dim]
#         x = self.act_layer(x)
#         x = self.dropout(x)
#         x = self.fc2(x)  # [B, num_patches, out_dim]
#         x = self.act_layer(x)  # Apply activation after second linear layer
#         x = self.dropout(x)
#         return x


# class Attention(nn.Module):
#     """Multi-Head Self-Attention mechanism."""
#     def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = head_dim ** -0.5  # Scale factor for attention scores

#         # Linear layer to generate Q, K, V from input
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         # Output projection
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x):
#         B, N, C = x.shape  # Batch, Num_patches, Embedding_dim
        
#         # Generate Q, K, V matrices
#         qkv = self.qkv(x)  # [B, num_patches, 3*embed_dim]
#         qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)  # [B, num_patches, 3, num_heads, per_HeadDim]
#         qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, num_patches, per_HeadDim]
#         q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, num_patches, per_HeadDim]

#         # Calculate attention scores
#         attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, num_patches, num_patches]
#         attn = attn.softmax(dim=-1)  # Apply softmax to get attention weights
#         attn = self.attn_drop(attn)
        
#         # Apply attention to values
#         x = (attn @ v)  # [B, num_heads, num_patches, per_HeadDim]
#         x = x.transpose(1, 2)  # [B, num_patches, num_heads, per_HeadDim]
#         x = x.reshape(B, N, C)  # [B, num_patches, embed_dim]
#         x = self.proj(x)  # [B, num_patches, embed_dim]
#         x = self.proj_drop(x)
#         return x


# class Block(nn.Module):
#     """Transformer block: attention -> MLP with residual connections."""
#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
#                  drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
#         super().__init__()
#         # Layer normalization before attention
#         self.norm1 = norm_layer(dim)
#         # Multi-head self-attention module
#         self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
#         # Stochastic depth for regularization
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         # Layer normalization before MLP
#         self.norm2 = norm_layer(dim)
#         # MLP hidden dimension calculation
#         self.mlp_hidden_dim = int(dim * mlp_ratio)
#         # MLP module
#         self.mlp = MLP(in_dim=dim, hidden_dim=self.mlp_hidden_dim, out_dim=dim)

#     def forward(self, x):
#         # Pre-norm transformer block with residual connections
#         x = x + self.drop_path(self.attn(self.norm1(x)))  # Attention with residual
#         x = x + self.drop_path(self.mlp(self.norm2(x)))   # MLP with residual
#         return x


# class ConvTransBN(nn.Module):  # (convolution => [BN] => ReLU)
#     """1D Convolution block: Conv1d -> BatchNorm1d -> ReLU."""
#     def __init__(self, in_channels, out_channels):
#         super(ConvTransBN, self).__init__()
#         self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.norm = nn.BatchNorm1d(out_channels)
#         self.activation = nn.ReLU()

#     def forward(self, x):
#         out = self.conv(x)
#         out = self.norm(out)
#         return self.activation(out)

# class CrossAttention(nn.Module):
#     # Q = image tokens ; KV = text tokens
#     def __init__(self, dim, num_heads=8):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = head_dim ** -0.5

#         self.q = nn.Linear(dim, dim)
#         self.k = nn.Linear(dim, dim)
#         self.v = nn.Linear(dim, dim)
#         self.proj = nn.Linear(dim, dim)

#     def forward(self, x_img, x_txt):  # [B, N1, C], [B, N2, C]
#         B, N1, C = x_img.shape
#         _, N2, _ = x_txt.shape

#         q = self.q(x_img).reshape(B, N1, self.num_heads, C//self.num_heads).transpose(1,2)
#         k = self.k(x_txt).reshape(B, N2, self.num_heads, C//self.num_heads).transpose(1,2)
#         v = self.v(x_txt).reshape(B, N2, self.num_heads, C//self.num_heads).transpose(1,2)

#         att = (q @ k.transpose(-2,-1)) * self.scale
#         att = att.softmax(dim=-1)
#         out = att @ v
#         out = out.transpose(1,2).reshape(B,N1,C)
#         return self.proj(out)


# class VisionTransformer(nn.Module):  # Transformer-branch
#     def __init__(self, config, vis, img_size, channel_num, patch_size, embed_dim, depth=1, num_heads=8,
#                  mlp_ratio=4., qkv_bias=True, num_classes=1, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
#         super(VisionTransformer, self).__init__()
#         self.config = config
#         self.vis = vis
        
#         # Patch embedding layer
#         self.embeddings = Embeddings(config=config, patch_size=patch_size, img_size=img_size, in_channels=channel_num)
#         self.depth = depth
#         self.dim = embed_dim
        
#         # Normalization and activation layers
#         norm_layer = nn.LayerNorm
#         self.norm = norm_layer(embed_dim)
#         act_layer = nn.GELU

#         # Stochastic depth decay rates
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        
#         # Transformer encoder blocks
#         self.Encoder_blocks = nn.Sequential(*[
#             Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
#                   attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
#             for i in range(self.depth)])

#         # Classification head
#         self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
#         # 1D convolution blocks for processing transformer features
#         self.CTBN = ConvTransBN(in_channels=embed_dim, out_channels=embed_dim//2)  # For downsampling
#         self.CTBN2 = ConvTransBN(in_channels=embed_dim*2, out_channels=embed_dim)  # For reconstruction
#         # self.CTBN3 = ConvTransBN(in_channels=10, out_channels=196)  # For text processing
#         self.cross_att = CrossAttention(embed_dim, num_heads=num_heads)


#     def forward(self, x, skip_x, text, reconstruct=False):
#         if not reconstruct:
#             # Initial patch embedding
#             x = self.embeddings(x)
#             # Add text information if this is the final transformer layer
#             # if self.dim == 64:
#             #     x = x+self.CTBN3(text)  # [B, num_patches, embed_dim]
#             if self.dim == 64:
#                 x = x + self.cross_att(x, text)   # cross attention fuse

#             # Process through transformer blocks
#             x = self.Encoder_blocks(x)
#         else:
#             # Process through transformer blocks during reconstruction
#             x = self.Encoder_blocks(x)
            
#         # Return intermediate results for different transformer stages
#         if (self.dim == 64 and not reconstruct) or (self.dim == 512 and reconstruct):
#             return x
#         elif not reconstruct:
#             # Process for skip connection: downsample and concatenate with skip
#             x = x.transpose(1, 2)  # [B, embed_dim, num_patches]
#             x = self.CTBN(x)  # [B, embed_dim//2, num_patches] - downsample
#             x = x.transpose(1, 2)  # [B, num_patches, embed_dim//2]
#             y = torch.cat([x, skip_x], dim=2)  # [B, num_patches, embed_dim] - concatenate with skip connection
#             return y
#         elif reconstruct:
#             # Process skip connection and add to transformer output during reconstruction
#             skip_x = skip_x.transpose(1, 2)  # [B, embed_dim, num_patches]
#             skip_x = self.CTBN2(skip_x)  # [B, embed_dim, num_patches] - process skip
#             skip_x = skip_x.transpose(1, 2)  # [B, num_patches, embed_dim]
#             y = x+skip_x  # Add processed skip to transformer output
#             return y

# ViT.py
# Clean Vision Transformer (Option A). Exposes:
# - Embeddings: patch -> tokens
# - CrossAttention: image queries, text KV
# - Block: transformer encoder block
# - VisionTransformer: embed -> (optional cross-attn) -> encoder -> tokens
#
# Requirements: torch

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, Conv2d
from torch.nn.modules.utils import _pair


def _pair(x):
    return (x, x) if not isinstance(x, (tuple, list)) else x


class Reconstruct(nn.Module):
    """Optional helper: tokens [B, n_patches, C] -> spatial map [B, C_out, H, W].
    Keep here for convenience (you can also move to LViT)."""
    def __init__(self, in_channels, out_channels, kernel_size=1, scale_factor=1, mode='bilinear'):
        super().__init__()
        padding = 1 if kernel_size == 3 else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        # scale_factor can be int or (h_scale, w_scale) or output size tuple
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        # x: [B, n_patches, C]
        if x is None:
            return None
        B, n_patch, C = x.size()
        side = int(math.sqrt(n_patch))
        assert side * side == n_patch, f"n_patch ({n_patch}) not perfect square"
        x = x.permute(0, 2, 1).contiguous().view(B, C, side, side)
        if isinstance(self.scale_factor, (tuple, list)):
            size = (side * self.scale_factor[0], side * self.scale_factor[1])
            x = F.interpolate(x, size=size, mode=self.mode, align_corners=False if self.mode == 'bilinear' else None)
        else:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False if self.mode == 'bilinear' else None)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class Embeddings(nn.Module):
    """Patch embedding layer: conv projection -> flatten tokens -> add positional embeddings."""
    def __init__(self, in_channels, embed_dim, patch_size, img_size, dropout=0.1):
        super().__init__()
        img_size = _pair(img_size)
        patch_size = _pair(patch_size)
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, "img_size must be divisible by patch_size"
        self.patch_size = patch_size
        self.n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        # Project input channels -> embed_dim per patch
        self.proj = Conv2d(in_channels=in_channels, out_channels=embed_dim,
                           kernel_size=patch_size, stride=patch_size)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        self.dropout = Dropout(dropout)

    def forward(self, x):
        # x: [B, C_in, H, W]
        B = x.shape[0]
        x = self.proj(x)                        # [B, embed_dim, H', W']
        x = x.flatten(2).transpose(1, 2).contiguous()  # [B, n_patches, embed_dim]
        x = x + self.pos_emb
        return self.dropout(x)


class CrossAttention(nn.Module):
    """Cross-attention where queries come from image tokens and keys/values from text tokens."""
    def __init__(self, dim, num_heads=8, qkv_bias=True, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x_img, x_txt):
        # x_img: [B, N_img, C], x_txt: [B, N_txt, C]
        B, N1, C = x_img.shape
        _, N2, _ = x_txt.shape

        q = self.q(x_img).view(B, N1, self.num_heads, C // self.num_heads).permute(0,2,1,3)  # [B, heads, N1, hd]
        k = self.k(x_txt).view(B, N2, self.num_heads, C // self.num_heads).permute(0,2,1,3)  # [B, heads, N2, hd]
        v = self.v(x_txt).view(B, N2, self.num_heads, C // self.num_heads).permute(0,2,1,3)  # [B, heads, N2, hd]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, N1, N2]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v)  # [B, heads, N1, hd]
        out = out.permute(0,2,1,3).reshape(B, N1, C)  # [B, N1, C]
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class _MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

        # weight init
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """Transformer encoder block (pre-ln). Uses nn.MultiheadAttention for simplicity."""
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.0, attn_dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=attn_dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x):
        # x: [B, N, C]
        residual = x
        x = self.norm1(x)
        x_attn, _ = self.attn(x, x, x, need_weights=False)
        x = residual + x_attn

        residual = x
        x = self.norm2(x)
        x = residual + self.mlp(x)
        return x


class VisionTransformer(nn.Module):
    """Pure Vision Transformer (Option A).
    Usage:
        vit = VisionTransformer(img_size=224, patch_size=16, in_channels=3, embed_dim=64, depth=1, heads=8)
        tokens = vit(img, text_tokens=None)  # tokens: [B, n_patches, embed_dim]
    If you want text fusion, pass text_tokens of shape [B, n_text, embed_dim] (embedding/projection to embed_dim must be done outside).
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=64, depth=1, heads=8,
                 mlp_ratio=4.0, attn_dropout=0.0, dropout=0.0, text_fuse=False):
        super().__init__()
        self.embed = Embeddings(in_channels=in_channels, embed_dim=embed_dim,
                                patch_size=patch_size, img_size=img_size, dropout=dropout)
        self.n_patches = self.embed.n_patches
        self.blocks = nn.Sequential(*[Block(embed_dim, num_heads=heads, mlp_ratio=mlp_ratio,
                                            dropout=dropout, attn_dropout=attn_dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.text_fuse = text_fuse
        if self.text_fuse:
            self.cross = CrossAttention(embed_dim, num_heads=heads, dropout=attn_dropout)

    def _ensure_text_len(self, text_tokens, target_len):
        """Interpolate text token sequence along sequence length to match target_len.
        text_tokens: [B, L, C] -> returns [B, target_len, C]"""
        if text_tokens is None:
            return None
        B, L, C = text_tokens.shape
        if L == target_len:
            return text_tokens
        # interpolate in length dimension
        t = text_tokens.permute(0, 2, 1).contiguous()  # [B, C, L]
        t = F.interpolate(t, size=target_len, mode='linear', align_corners=False)
        t = t.permute(0, 2, 1).contiguous()  # [B, target_len, C]
        return t

    def forward(self, x, text_tokens=None):
        """
        x: spatial image tensor [B, in_channels, H, W]
        text_tokens: optional tensor [B, L, embed_dim] (if provided and text_fuse=True).
                     If L != n_patches it will be interpolated to n_patches.
        returns: tokens [B, n_patches, embed_dim]
        """
        tokens = self.embed(x)  # [B, n_patches, embed_dim]
        if self.text_fuse and text_tokens is not None:
            # ensure same sequence length and channel dims match
            text_tokens = self._ensure_text_len(text_tokens, tokens.shape[1])
            if text_tokens.size(-1) != tokens.size(-1):
                raise ValueError(f"text_tokens last dim {text_tokens.size(-1)} != embed_dim {tokens.size(-1)}")
            tokens = tokens + self.cross(tokens, text_tokens)

        tokens = self.blocks(tokens)
        tokens = self.norm(tokens)
        return tokens
