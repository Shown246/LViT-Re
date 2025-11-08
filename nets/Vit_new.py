# ViT.py
# Clean Vision Transformer (Option A) with optional timm pretrained loading helper.
# - Pure ViT: Embeddings -> (optional cross-attn) -> Transformer blocks -> tokens
# - Optional: try to use timm pretrained ViT forward_features or copy weights

import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, Conv2d
from torch.nn.modules.utils import _pair

# try to import timm; it's optional
try:
    import timm
    _HAS_TIMM = True
except Exception:
    _HAS_TIMM = False


def _pair(x):
    return (x, x) if not isinstance(x, (tuple, list)) else x


class Reconstruct(nn.Module):
    """Helper: tokens [B, n_patches, C] -> spatial map [B, C_out, H, W]."""
    def __init__(self, in_channels, out_channels, kernel_size=1, scale_factor=1, mode='bilinear'):
        super().__init__()
        padding = 1 if kernel_size == 3 else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        if x is None:
            return None
        B, n_patch, C = x.size()
        side = int(math.sqrt(n_patch))
        assert side * side == n_patch, f"n_patches ({n_patch}) is not a perfect square"
        x = x.permute(0, 2, 1).view(B, C, side, side)
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
    """Patch embedding: conv projection -> flatten tokens -> add positional embeddings."""
    def __init__(self, in_channels, embed_dim, patch_size, img_size, dropout=0.1):
        super().__init__()
        img_size = _pair(img_size)
        patch_size = _pair(patch_size)
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, "img_size must be divisible by patch_size"
        self.patch_size = patch_size
        self.n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.proj = Conv2d(in_channels=in_channels, out_channels=embed_dim,
                           kernel_size=patch_size, stride=patch_size)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        self.dropout = Dropout(dropout)

    def forward(self, x):
        # x: [B, C_in, H, W]
        B = x.shape[0]
        x = self.proj(x)  # [B, embed_dim, H', W']
        x = x.flatten(2).transpose(1, 2).contiguous()  # [B, n_patches, embed_dim]
        # x: [B, N, C], self.pos_emb: [1, 196, C] (old)
        if x.size(1) != self.pos_emb.size(1):
            # interpolate pos embedding to match current token count
            pos = self.pos_emb
            pos = pos.transpose(1,2)                          # [1, C, 196]
            H = int(x.size(1) ** 0.5)
            pos = pos.view(1, -1, int(self.pos_emb.size(1)**0.5), int(self.pos_emb.size(1)**0.5))
            pos = torch.nn.functional.interpolate(pos, size=(H,H), mode='bicubic', align_corners=False)
            pos = pos.view(1, x.size(2), x.size(1)).transpose(1,2)
        else:
            pos = self.pos_emb
        
        x = x + pos

        return self.dropout(x)


class CrossAttention(nn.Module):
    """Cross-attention: queries from image tokens, keys/values from text tokens."""
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
        q = self.q(x_img).view(B, N1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # [B, heads, N1, hd]
        k = self.k(x_txt).view(B, N2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)   # [B, heads, N2, hd]
        v = self.v(x_txt).view(B, N2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)   # [B, heads, N2, hd]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, N1, N2]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v)  # [B, heads, N1, hd]
        out = out.permute(0, 2, 1, 3).reshape(B, N1, C)  # [B, N1, C]
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
        # init
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
    """Transformer encoder block (pre-LN). MultiheadAttention used for clarity."""
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.0, attn_dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=attn_dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x):
        residual = x
        x_norm = self.norm1(x)
        x_attn, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = residual + x_attn
        residual = x
        x_norm = self.norm2(x)
        x = residual + self.mlp(x_norm)
        return x


class VisionTransformer(nn.Module):
    """
    Pure Vision Transformer (Option A), optional text fusion and optional timm pretrained helper.

    Args:
        img_size, patch_size, in_channels, embed_dim, depth, heads, mlp_ratio, attn_dropout, dropout
        text_fuse: whether to enable cross-attention fusion with text tokens passed to forward()
        pretrained_timm_name: optional timm model name to attempt to load/preload weights from (best-effort)
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=64, depth=1, heads=8,
                 mlp_ratio=4.0, attn_dropout=0.0, dropout=0.0, text_fuse=False, pretrained_timm_name=None, pretrained=False):
        super().__init__()
        self.embed = Embeddings(in_channels=in_channels, embed_dim=embed_dim, patch_size=patch_size, img_size=img_size, dropout=dropout)
        self.n_patches = self.embed.n_patches
        self.blocks = nn.Sequential(*[Block(embed_dim, num_heads=heads, mlp_ratio=mlp_ratio, dropout=dropout, attn_dropout=attn_dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.text_fuse = text_fuse
        if text_fuse:
            self.cross = CrossAttention(embed_dim, num_heads=heads, qkv_bias=True, dropout=attn_dropout)

        # optionally manage a timm model or copy weights from it
        self._timm_model = None
        if pretrained_timm_name is not None:
            self.load_pretrained_from_timm(pretrained_timm_name, pretrained=pretrained)

        # small init of position embeddings if not zero
        nn.init.trunc_normal_(self.embed.pos_emb, std=0.02)

    # ----------------------------
    # Helper: attempt to use timm pretrained model
    # ----------------------------
    def load_pretrained_from_timm(self, model_name, pretrained=True, use_forward_features=True):
        """
        Best-effort attempt to use a timm pretrained backbone.
        Behavior:
          - If timm not installed => raises informative ImportError.
          - Creates timm model = timm.create_model(model_name, pretrained=pretrained)
          - If use_forward_features and the timm model exposes `forward_features`, we will keep that model
            and during forward() call its forward_features to produce features (then adapt shapes).
          - Otherwise we attempt to copy parameters from timm model into this ViT where shapes match.

        WARNING: mapping is model-specific. This is a best-effort helper to accelerate experimentation.
        """
        if not _HAS_TIMM:
            raise ImportError("timm is required to load pretrained models. Install via `pip install timm`.")

        tm = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='')  # we want features
        # prefer using forward_features if available
        if use_forward_features and hasattr(tm, 'forward_features'):
            self._timm_model = tm
            # check expected shapes quickly (we cannot guarantee)
            warnings.warn(f"Using timm model {model_name} forward_features as feature extractor. "
                          "Make sure returned shape matches token expectations (n_patches, embed_dim).")
            return

        # otherwise attempt parameter copying where keys / shapes match
        tm_state = tm.state_dict()
        my_state = self.state_dict()
        copied = 0
        for k_tm, v_tm in tm_state.items():
            # try to find a matching parameter in our state dict by suffix (best-effort)
            # simple heuristic: match by name endings or exact name
            if k_tm in my_state and my_state[k_tm].shape == v_tm.shape:
                my_state[k_tm].copy_(v_tm)
                copied += 1
            else:
                # try suffix match
                for k_my in my_state:
                    if k_my.endswith(k_tm) and my_state[k_my].shape == v_tm.shape:
                        my_state[k_my].copy_(v_tm)
                        copied += 1
                        break
        self.load_state_dict(my_state)
        warnings.warn(f"Attempted copying pretrained weights from {model_name}. Parameters copied: {copied}/{len(my_state)}. "
                      "If this is small, mapping may not be correct for this backbone.")

    # ----------------------------
    # Helpers
    # ----------------------------
    def _ensure_text_len(self, text_tokens, target_len):
        """Interpolate text token sequence along length to match target_len."""
        if text_tokens is None:
            return None
        B, L, C = text_tokens.shape
        if L == target_len:
            return text_tokens
        t = text_tokens.permute(0, 2, 1).contiguous()  # [B, C, L]
        t = F.interpolate(t, size=target_len, mode='linear', align_corners=False)
        t = t.permute(0, 2, 1).contiguous()
        return t

    # ----------------------------
    # Forward
    # ----------------------------
    def forward(self, x, text_tokens=None):
        """
        x: [B, in_channels, H, W]
        text_tokens (optional): [B, L, embed_dim] -> will be interpolated if needed
        returns: tokens [B, n_patches, embed_dim]
        """
        # If we have a timm model with forward_features, try to use it and adapt output
        if self._timm_model is not None:
            # Use timm model's forward_features if available
            if hasattr(self._timm_model, 'forward_features'):
                feat = self._timm_model.forward_features(x)
                # timm feature types vary: could be [B, C, H', W'] or [B, N, C]
                if feat.ndim == 4:
                    B, C, Hf, Wf = feat.shape
                    n_patches = Hf * Wf
                    tokens = feat.flatten(2).transpose(1, 2).contiguous()  # [B, n_patches, C]
                elif feat.ndim == 2:
                    # sometimes forward_features returns pooled feature [B, C]
                    raise RuntimeError("timm forward_features returned pooled features, not tokens. Can't adapt automatically.")
                else:
                    tokens = feat
                # if text fusion requested here, ensure dims match
                if self.text_fuse and text_tokens is not None:
                    text_tokens = self._ensure_text_len(text_tokens, tokens.shape[1])
                    if text_tokens.size(-1) != tokens.size(-1):
                        raise ValueError(f"text_tokens dim {text_tokens.size(-1)} != feature dim {tokens.size(-1)}")
                    tokens = tokens + self.cross(tokens, text_tokens)
                # pass through our transformer blocks if embed dims match, else return tokens
                if tokens.size(-1) == self.norm.normalized_shape[0]:
                    tokens = self.blocks(tokens)
                    tokens = self.norm(tokens)
                return tokens

            # fallback: timm model present but no forward_features -> attempt copy (already attempted in loader)
            # continue to normal path

        # Normal internal path (our embed -> cross-attn -> blocks)
        tokens = self.embed(x)  # [B, n_patches, embed_dim]
        if self.text_fuse and text_tokens is not None:
            text_tokens = self._ensure_text_len(text_tokens, tokens.shape[1])
            if text_tokens.size(-1) != tokens.size(-1):
                raise ValueError(f"text_tokens last dim {text_tokens.size(-1)} != embed_dim {tokens.size(-1)}")
            tokens = tokens + self.cross(tokens, text_tokens)

        tokens = self.blocks(tokens)
        tokens = self.norm(tokens)
        return tokens
