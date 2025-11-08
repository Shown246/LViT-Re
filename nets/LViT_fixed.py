# LViT.py
# LViT model that uses the pure VisionTransformer from ViT.py (Option A).
# Expects ViT.py in the same folder, exposing VisionTransformer and Reconstruct.

import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
from .Vit import VisionTransformer, Reconstruct

# -------------------------
# Utility building blocks
# -------------------------
def get_activation(name='ReLU'):
    name = name.lower()
    if hasattr(nn, name):
        return getattr(nn, name)()
    return nn.ReLU()

class ConvBatchNorm(nn.Module):
    def __init__(self, in_ch, out_ch, activation='ReLU'):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = get_activation(activation)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, nb_conv=2, activation='ReLU'):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        layers = [ConvBatchNorm(in_ch, out_ch, activation)]
        for _ in range(nb_conv - 1):
            layers.append(ConvBatchNorm(out_ch, out_ch, activation))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(self.pool(x))

# Pixel-level attention (same approach you had)
class PixLevelModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_avg = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.conv_max = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        # tiny MLP: 3 -> 6 -> 1 per pixel
        self.mlp = nn.Sequential(
            nn.Linear(3, 3 * 2),
            nn.ReLU(inplace=True),
            nn.Linear(3 * 2, 1)
        )
    def forward(self, x):
        # x: B,C,H,W
        B, C, H, W = x.shape
        xa = self.relu(self.conv_avg(x))
        xm = self.relu(self.conv_max(x))
        xa_s = xa.mean(dim=1, keepdim=True)               # [B,1,H,W]
        xm_s = xm.max(dim=1).values.unsqueeze(1)          # [B,1,H,W]
        xsum = xa_s + xm_s                                # [B,1,H,W]
        stacked = torch.cat([xa_s, xm_s, xsum], dim=1)    # [B,3,H,W]
        px = stacked.permute(0,2,3,1).contiguous()        # [B,H,W,3]
        out = self.mlp(px)                                # [B,H,W,1]
        out = out.permute(0,3,1,2).contiguous()           # [B,1,H,W]
        return out * x

class UpblockAttention(nn.Module):
    """Upsample x and fuse with skip via pixel-attention, then conv stack."""
    def __init__(self, up_channels, skip_channels, out_channels, nb_conv=2, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.pix = PixLevelModule(skip_channels)
        layers = [ConvBatchNorm(up_channels + skip_channels, out_channels, activation)]
        for _ in range(nb_conv - 1):
            layers.append(ConvBatchNorm(out_channels, out_channels, activation))
        self.net = nn.Sequential(*layers)
    def forward(self, x, skip):
        x_up = self.up(x)
        skip_att = self.pix(skip)
        cat = torch.cat([x_up, skip_att], dim=1)
        return self.net(cat)

# -------------------------
# LViT - top-level model
# -------------------------
class LViTf(nn.Module):
    """
    LViT that uses separate VisionTransformer instances (pure ViT from ViT.py).
    - config should have attribute base_channel (e.g., SimpleNamespace(base_channel=64))
    - text input expected shape: [B, L, 768] (BERT). It's projected per-scale inside.
    """
    def __init__(self, config, n_channels=3, n_classes=1, img_size=224, vis=False):
        super().__init__()
        base = config.base_channel
        self.n_classes = n_classes
        self.inc = ConvBatchNorm(n_channels, base)

        # CNN encoder
        self.down1 = DownBlock(base, base*2, nb_conv=2)
        self.down2 = DownBlock(base*2, base*4, nb_conv=2)
        self.down3 = DownBlock(base*4, base*8, nb_conv=2)
        self.down4 = DownBlock(base*8, base*8, nb_conv=2)

        # Vision Transformers per scale (pure ViT)
        # ensure in_channels passed matches the CNN feature channels at that scale
        self.vit1 = VisionTransformer(img_size=224, patch_size=16, in_channels=base, embed_dim=base, depth=1, heads=8, text_fuse=True)
        self.vit2 = VisionTransformer(img_size=112, patch_size=8,  in_channels=base*2, embed_dim=base*2, depth=1, heads=8, text_fuse=True)
        self.vit3 = VisionTransformer(img_size=56,  patch_size=4,  in_channels=base*4, embed_dim=base*4, depth=1, heads=8, text_fuse=True)
        self.vit4 = VisionTransformer(img_size=28,  patch_size=2,  in_channels=base*8, embed_dim=base*8, depth=1, heads=8, text_fuse=True)

        # Reconstruct tokens -> spatial maps (use scale factor so patch grid -> original map size)
        # For patch_size p and image size S, patch grid side = S/p, to get back to S use scale_factor = p
        self.reconstruct1 = Reconstruct(in_channels=base, out_channels=base, kernel_size=1, scale_factor=16)
        self.reconstruct2 = Reconstruct(in_channels=base*2, out_channels=base*2, kernel_size=1, scale_factor=8)
        self.reconstruct3 = Reconstruct(in_channels=base*4, out_channels=base*4, kernel_size=1, scale_factor=4)
        self.reconstruct4 = Reconstruct(in_channels=base*8, out_channels=base*8, kernel_size=1, scale_factor=2)

        # CNN decoder (U-Net style with attention ups)
        self.up4 = UpblockAttention(up_channels=base*8, skip_channels=base*8, out_channels=base*4)
        self.up3 = UpblockAttention(up_channels=base*4, skip_channels=base*4, out_channels=base*2)
        self.up2 = UpblockAttention(up_channels=base*2, skip_channels=base*2, out_channels=base)
        self.up1 = UpblockAttention(up_channels=base,   skip_channels=base,   out_channels=base)

        self.outc = nn.Conv2d(base, n_classes, kernel_size=1)
        self.last_act = nn.Sigmoid() if n_classes == 1 else nn.Identity()

        # Text projection conv1d chain: 768 -> 512 -> 256 -> 128 -> 64 (map down progressively)
        # conv1d expects [B, C, L]
        self.text_proj4 = nn.Conv1d(in_channels=768, out_channels=base*8, kernel_size=1)  # for vit4 (embed_dim = base*8)
        self.text_proj3 = nn.Conv1d(in_channels=base*8, out_channels=base*4, kernel_size=1)
        self.text_proj2 = nn.Conv1d(in_channels=base*4, out_channels=base*2, kernel_size=1)
        self.text_proj1 = nn.Conv1d(in_channels=base*2, out_channels=base,   kernel_size=1)

    # small helper to convert conv1d output to [B, L, C]
    def _proj_text(self, text, conv1d):
        # text: [B, L, 768] -> conv1d wants [B, C, L]
        if text is None:
            return None
        t = text.permute(0, 2, 1).contiguous()   # [B, 768, L]
        t = conv1d(t)                            # [B, embedC, L]
        t = t.permute(0, 2, 1).contiguous()      # [B, L, embedC]
        return t

    def forward(self, x, text):
        """
        x: [B, 3, 224, 224]
        text: [B, L, 768] (optional)
        """
        assert x.dim() == 4 and x.size(1) == 3

        # CNN encoder
        x1 = self.inc(x)         # [B, base, 224,224]
        x2 = self.down1(x1)      # [B, base*2, 112,112]
        x3 = self.down2(x2)      # [B, base*4, 56,56]
        x4 = self.down3(x3)      # [B, base*8, 28,28]
        x5 = self.down4(x4)      # [B, base*8, 14,14]  bottleneck

        # Project text for each scale (output dims = embed_dim for each vit)
        if text is not None:
            t4 = self._proj_text(text, self.text_proj4)  # [B, L, base*8]
            t3 = self._proj_text(t4, self.text_proj3)    # [B, L', base*4] (L' still equals original L; VisionTransformer will interpolate)
            t2 = self._proj_text(t3, self.text_proj2)    # [B, L'', base*2]
            t1 = self._proj_text(t2, self.text_proj1)    # [B, L''', base]
        else:
            t1 = t2 = t3 = t4 = None

        # --- VisionTransformer encoder per scale ---
        # Each Vit will embed its input spatial map internally using its own Embeddings
        # and optionally perform cross-attention with its projected text tokens.
        y1_tokens = self.vit1(x1, text_tokens=t1)  # [B, n_patches1, base]
        y2_tokens = self.vit2(x2, text_tokens=t2)  # [B, n_patches2, base*2]
        y3_tokens = self.vit3(x3, text_tokens=t3)  # [B, n_patches3, base*4]
        y4_tokens = self.vit4(x4, text_tokens=t4)  # [B, n_patches4, base*8]

        # --- Reconstruct token sequences to spatial feature maps and fuse to CNN skips ---
        # Reconstruct returns spatial maps sized to match original feature maps
        rec1 = self.reconstruct1(y1_tokens)  # [B, base, 224,224]
        rec2 = self.reconstruct2(y2_tokens)  # [B, base*2, 112,112]
        rec3 = self.reconstruct3(y3_tokens)  # [B, base*4, 56,56]
        rec4 = self.reconstruct4(y4_tokens)  # [B, base*8, 28,28]

        # Add (residual) transformer info back into CNN feature maps
        x1 = x1 + rec1
        x2 = x2 + rec2
        x3 = x3 + rec3
        x4 = x4 + rec4

        # U-Net decoder with pixel-attention
        x = self.up4(x5, x4)   # [B, base*4, 28,28]
        x = self.up3(x, x3)    # [B, base*2, 56,56]
        x = self.up2(x, x2)    # [B, base,   112,112]
        x = self.up1(x, x1)    # [B, base,   224,224]

        logits = self.outc(x)
        if self.n_classes == 1:
            return self.last_act(logits)
        return logits

# -------------------------
# Quick smoke-test helper
# -------------------------
# if __name__ == "__main__":
#     # quick forward pass with dummy data to ensure shapes match
#     cfg = SimpleNamespace(base_channel=64)
#     model = LViT(cfg, n_channels=3, n_classes=1)
#     x = torch.randn(2, 3, 224, 224)
#     text = torch.randn(2, 10, 768)  # 10 tokens — will be interpolated by ViT to n_patches per scale
#     out = model(x, text)
#     print("Output shape:", out.shape)  # expected [2, 1, 224, 224]
