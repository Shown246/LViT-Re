# LViT.py
# LViT with pretrained ConvNeXt backbone (timm) + multi-scale ViT (from ViT.py)
# - deep supervision heads
# - combined loss (BCE + soft Dice)

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the VisionTransformer and Reconstruct from ViT.py
# Adjust import path if you use a package (e.g., from .ViT import VisionTransformer, Reconstruct)
from .Vit_new import VisionTransformer, Reconstruct

# timm is required for pretrained backbone
try:
    import timm
except Exception as e:
    timm = None

# ---------------------------
# Utilities
# ---------------------------
def get_backbone_features(backbone_name='convnext_small', pretrained=True, out_indices=(0,1,2,3)):
    """Create a timm backbone with features_only=True and return model and its feature_info."""
    if timm is None:
        raise ImportError("timm is required for pretrained backbones. Install with `pip install timm`.")
    model = timm.create_model(backbone_name, features_only=True, pretrained=pretrained, out_indices=out_indices)
    # model.feature_info.channels gives channel dims per stage
    feature_info = model.feature_info
    channels = [f['num_chs'] for f in feature_info.info]  # e.g., [96, 192, 384, 768]
    return model, channels


# ---------------------------
# Small CNN blocks for decoder
# ---------------------------
def conv_bn_relu(in_ch, out_ch, kernel=3, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )

class DecoderUpBlock(nn.Module):
    """Upsample + concat(skip) -> conv stack"""
    def __init__(self, in_ch, skip_ch, out_ch, nb_conv=2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        layers = [conv_bn_relu(in_ch + skip_ch, out_ch)]
        for _ in range(nb_conv - 1):
            layers.append(conv_bn_relu(out_ch, out_ch))
        self.net = nn.Sequential(*layers)
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.net(x)

# ---------------------------
# Deep supervision + final head
# ---------------------------
class DeepSupervisionHeads(nn.Module):
    """Simple 1x1 conv heads for deep supervision; returns upsampled maps to input size."""
    def __init__(self, in_channels_list, n_classes):
        """
        in_channels_list: list of channels from decoder levels where DS will be computed.
                          They should correspond to sizes (descend -> ascend).
        """
        super().__init__()
        self.heads = nn.ModuleList([nn.Conv2d(ch, n_classes, kernel_size=1) for ch in in_channels_list])
    def forward(self, feats, target_size):
        """
        feats: list of feature maps in same order as in_channels_list
        target_size: (H, W) upsample target
        returns list of upsampled logits (raw) matching target_size
        """
        outs = []
        for f, head in zip(feats, self.heads):
            o = head(f)
            o = F.interpolate(o, size=target_size, mode='bilinear', align_corners=False)
            outs.append(o)
        return outs

# ---------------------------
# LViT main model
# ---------------------------
class LViTN(nn.Module):
    def __init__(self, config, n_channels=3, n_classes=1, img_size=224,
                 backbone_name='convnext_small', backbone_pretrained=True, vit_depth=1, vit_heads=8,
                 text_fuse_scales=(True, True, True, True)):
        """
        config: SimpleNamespace or similar; not strictly used here except maybe for compatibility.
        n_channels: input image channels
        n_classes: segmentation classes (1 => binary)
        img_size: input spatial resolution (assumed square)
        backbone_name: timm ConvNeXt variant (e.g., 'convnext_tiny','convnext_small','convnext_base')
        text_fuse_scales: tuple of 4 booleans deciding if text fusion enabled per ViT scale (stage1..4)
        """
        super().__init__()
        if timm is None:
            raise ImportError("timm is required. Install with `pip install timm`.")

        # create pretrained backbone
        self.backbone, channels = get_backbone_features(backbone_name, pretrained=backbone_pretrained, out_indices=(0,1,2,3))
        # channels is list of in_channels for stage outputs (encoder features)
        # e.g., convnext_small -> [96, 192, 384, 768] (depends on model)
        if len(channels) < 4:
            raise RuntimeError("Backbone must expose 4 stages (found {}).".format(len(channels)))

        # expose channels
        ch1, ch2, ch3, ch4 = channels  # low -> high

        # Use ViT per scale with embed_dim = channel dim of that scale
        # Patch sizes chosen to map to feature map sizes (these depend on input img size)
        # For typical convnext features:
        # stage1 corresponds to spatial size ~img/4? We'll use common mapping:
        self.vit1 = VisionTransformer(img_size=img_size, patch_size=16, in_channels=ch1, embed_dim=ch1,
                                      depth=vit_depth, heads=vit_heads, text_fuse=text_fuse_scales[0])
        self.vit2 = VisionTransformer(img_size=img_size//2, patch_size=8, in_channels=ch2, embed_dim=ch2,
                                      depth=vit_depth, heads=vit_heads, text_fuse=text_fuse_scales[1])
        self.vit3 = VisionTransformer(img_size=img_size//4, patch_size=4, in_channels=ch3, embed_dim=ch3,
                                      depth=vit_depth, heads=vit_heads, text_fuse=text_fuse_scales[2])
        self.vit4 = VisionTransformer(img_size=img_size//8, patch_size=2, in_channels=ch4, embed_dim=ch4,
                                      depth=vit_depth, heads=vit_heads, text_fuse=text_fuse_scales[3])

        # Reconstruct modules: tokens -> spatial (project tokens back to CNN feature shape)
        # scale_factor should be patch_size used in ViT to go back to original spatial sizes:
        self.reconstruct1 = Reconstruct(in_channels=ch1, out_channels=ch1, kernel_size=1, scale_factor=16)
        self.reconstruct2 = Reconstruct(in_channels=ch2, out_channels=ch2, kernel_size=1, scale_factor=8)
        self.reconstruct3 = Reconstruct(in_channels=ch3, out_channels=ch3, kernel_size=1, scale_factor=4)
        self.reconstruct4 = Reconstruct(in_channels=ch4, out_channels=ch4, kernel_size=1, scale_factor=2)

        # Decoder: choose decoder channel widths (we'll reduce gradually)
        dec_ch4 = ch4 // 2
        dec_ch3 = ch3 // 2
        dec_ch2 = ch2 // 2
        dec_ch1 = ch1 // 2
        # Bottleneck conv to move backbone final feature -> decoder channels
        self.bottleneck = conv_bn_relu(ch4, dec_ch4)

        # Up blocks: in_ch = prev_out, skip_ch = backbone_x (after recon), out_ch = dec_chX
        self.up3 = DecoderUpBlock(in_ch=dec_ch4, skip_ch=ch3, out_ch=dec_ch3)  # from bottleneck + skip3
        self.up2 = DecoderUpBlock(in_ch=dec_ch3, skip_ch=ch2, out_ch=dec_ch2)
        self.up1 = DecoderUpBlock(in_ch=dec_ch2, skip_ch=ch1, out_ch=dec_ch1)

        # final conv to project decoder -> base channel then to n_classes
        self.final_conv = nn.Sequential(
            conv_bn_relu(dec_ch1, dec_ch1),
            nn.Conv2d(dec_ch1, n_classes, kernel_size=1)
        )

        # Deep supervision heads: take outputs from up3, up2, up1 (before final conv)
        # We'll upsample them to input image size in forward
        self.ds_heads = DeepSupervisionHeads(in_channels_list=[dec_ch3, dec_ch2, dec_ch1], n_classes=n_classes)

        # Text projection conv1d per scale (project BERT 768 -> embed_dim)
        self.text_proj4 = nn.Conv1d(in_channels=768, out_channels=ch4, kernel_size=1)
        self.text_proj3 = nn.Conv1d(in_channels=ch4, out_channels=ch3, kernel_size=1)
        self.text_proj2 = nn.Conv1d(in_channels=ch3, out_channels=ch2, kernel_size=1)
        self.text_proj1 = nn.Conv1d(in_channels=ch2, out_channels=ch1, kernel_size=1)

    def _proj_text(self, text, conv1d):
        """Project text embeddings: text [B,L,768] -> conv1d expects [B,C,L] -> returns [B,L,C_out]"""
        if text is None:
            return None
        t = text.permute(0, 2, 1).contiguous()  # [B, C=768, L]
        t = conv1d(t)                            # [B, C_out, L]
        t = t.permute(0, 2, 1).contiguous()      # [B, L, C_out]
        return t

    def forward(self, x, text=None):
        """
        x: [B, 3, H, W]  (assumes square H==W==img_size given at init, but interpolation is allowed)
        text: optional [B, L, 768] (e.g. BERT last hidden states). Will be projected and interpolated in ViT.
        returns: dict {'out': logits (B, n_classes, H, W), 'ds': [ds3, ds2, ds1]} (raw logits)
        """
        B, C, H, W = x.shape
        # backbone features: returns list of feature maps [stage1, stage2, stage3, stage4]
        feats = self.backbone(x)
        # ensure order is low->high (timm's features_only returns that)
        f1, f2, f3, f4 = feats  # f1: earliest (largest spatial), f4: deepest (smallest spatial)

        # Project text per-scale
        if text is not None:
            t4 = self._proj_text(text, self.text_proj4)  # [B, L, ch4]
            t3 = self._proj_text(t4, self.text_proj3)    # cascade projections
            t2 = self._proj_text(t3, self.text_proj2)
            t1 = self._proj_text(t2, self.text_proj1)
        else:
            t1 = t2 = t3 = t4 = None

        # Pass per-scale features through respective ViT (ViT expects spatial map for embedding)
        y1_tokens = self.vit1(f1, text_tokens=t1)  # tokens
        y2_tokens = self.vit2(f2, text_tokens=t2)
        y3_tokens = self.vit3(f3, text_tokens=t3)
        y4_tokens = self.vit4(f4, text_tokens=t4)

        # Reconstruct tokens -> spatial maps and residual-add into CNN features
        rec1 = self.reconstruct1(y1_tokens)  # [B, ch1, H, W]
        rec2 = self.reconstruct2(y2_tokens)  # [B, ch2, H/2, W/2]
        rec3 = self.reconstruct3(y3_tokens)  # [B, ch3, H/4, W/4]
        rec4 = self.reconstruct4(y4_tokens)  # [B, ch4, H/8, W/8]

        # Fuse transformer info back into CNN backbone maps
        rec1 = F.interpolate(rec1, size=f1.shape[-2:], mode='bilinear', align_corners=False)
        f1 = f1 + rec1
        rec2 = F.interpolate(rec2, size=f2.shape[-2:], mode='bilinear', align_corners=False)
        f2 = f2 + rec2
        rec3 = F.interpolate(rec3, size=f3.shape[-2:], mode='bilinear', align_corners=False)
        f3 = f3 + rec3
        # make sure rec4 matches f4 spatially
        rec4 = F.interpolate(rec4, size=f4.shape[-2:], mode='bilinear', align_corners=False)
        f4 = f4 + rec4

        # Decoder path
        b = self.bottleneck(f4)  # [B, dec_ch4, H/8, W/8]
        u3 = self.up3(b, f3)      # [B, dec_ch3, H/4, W/4]
        u2 = self.up2(u3, f2)     # [B, dec_ch2, H/2, W/2]
        u1 = self.up1(u2, f1)     # [B, dec_ch1, H, W]

        # final logits
        logits = self.final_conv(u1)  # [B, n_classes, H, W]

        # deep supervision heads: produce maps from u3, u2, u1 upsampled to input size
        ds_maps = self.ds_heads([u3, u2, u1], target_size=(H, W))  # list of logits

        return {'out': logits, 'ds': ds_maps}

# ---------------------------
# Combined loss utility (BCE + soft Dice)
# ---------------------------
class SoftDiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, logits, target):
        """
        logits: raw logits [B, C, H, W]
        target: binary mask [B, C, H, W] or [B, H, W] if C==1
        returns dice loss (1 - dice)
        """
        probs = torch.sigmoid(logits)
        if target.dim() == 3:
            target = target.unsqueeze(1)
        assert probs.shape == target.shape
        dims = (0, 2, 3)
        intersection = (probs * target).sum(dims)
        cardinality = (probs + target).sum(dims)
        dice_score = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        dice_loss = 1.0 - dice_score
        # average over batch & channels
        return dice_loss.mean()

class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss()
        self.bce_w = bce_weight
        self.dice_w = dice_weight
    def forward(self, logits, target):
        """
        logits: raw logits [B, C, H, W]
        target: binary mask [B, H, W] or [B, C, H, W]
        returns scalar loss
        """
        if target.dim() == 3:
            target = target.unsqueeze(1)
        # BCE expects probabilities over each pixel; BCEWithLogitsLoss handles logits
        bce_loss = self.bce(logits, target)
        dice_loss = self.dice(logits, target)
        return self.bce_w * bce_loss + self.dice_w * dice_loss

# # ---------------------------
# # Smoke test (quick)
# # ---------------------------
# if __name__ == "__main__":
#     from types import SimpleNamespace
#     cfg = SimpleNamespace(base_channel=64)
#     # instantiate model with convnext_small backbone
#     model = LViT(cfg, n_channels=3, n_classes=1, img_size=224, backbone_name='convnext_tiny', backbone_pretrained=True)
#     x = torch.randn(2, 3, 224, 224)
#     # dummy text: 10 tokens, BERT dim 768
#     text = torch.randn(2, 10, 768)
#     out = model(x, text)
#     print("out['out'] shape:", out['out'].shape)
#     print("deep supervision maps:", [d.shape for d in out['ds']])
#     # test loss
#     loss_fn = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
#     target = torch.randint(0, 2, (2, 224, 224)).float()
#     loss = loss_fn(out['out'], target)
#     print("loss:", loss.item())
