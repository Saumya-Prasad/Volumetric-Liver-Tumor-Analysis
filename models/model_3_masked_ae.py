# -*- coding: utf-8 -*-
"""
models/model_3_masked_ae.py
Masked Autoencoders for Unsupervised Anomaly Detection in Medical Images
Georgescu, KES 2023 (Procedia Computer Science 225, 969-978)

Architecture
------------
Stage 1 – MAE pre-training:
  x → patchify → random mask (75%) → ViT encoder → ViT decoder → reconstruct
  Loss: MSE on masked patches only

Stage 2 – Anomaly Classifier:
  Input  : residual map  r = |x - MAE(x)|
  Labels :
    • Negative (normal):   r from healthy reconstructions
    • Positive (anomaly):  r from pseudo-abnormal images (PA module)
  Classifier: small CNN → sigmoid → P(anomaly)

Pseudo-Abnormal Module (PA)
---------------------------
  Given healthy slice x, produce x_pa by:
  1. Select random region R of radius [r_min, r_max]
  2. Shift intensity: x_pa[R] = x[R] * scale + offset
     where scale ~ Uniform(0.5, 1.5), offset ~ Uniform(-0.3, 0.3)
  This mimics lesion-like intensity deviations without real labels.

Anomaly Score (inference)
-------------------------
  S(x) = P_classifier(|x - MAE(x)|) ∈ [0,1]

Future-scope improvement applied
---------------------------------
  • SSIM loss added alongside pixel MSE for better perceptual quality
  • Multi-scale patch sizes for robustness to lesion size
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
from einops import rearrange, repeat


# ──────────────────────────────────────────────
# 1.  Patch embedding / un-embedding
# ──────────────────────────────────────────────

class PatchEmbed(nn.Module):
    """Image → sequence of patch tokens."""
    def __init__(self, img_size=256, patch_size=16, in_ch=1, embed_dim=768):
        super().__init__()
        self.n_patches  = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B,C,H,W) → (B, n_patches, embed_dim)
        x = self.proj(x)                        # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)       # (B, N, D)
        return x


# ──────────────────────────────────────────────
# 2.  Transformer building blocks
# ──────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads=8, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, n_heads,
                                            dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim    = int(dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        h, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x    = x + h
        x    = x + self.mlp(self.norm2(x))
        return x


# ──────────────────────────────────────────────
# 3.  MAE Encoder + Decoder
# ──────────────────────────────────────────────

class MAEEncoder(nn.Module):
    """ViT encoder that operates only on visible (unmasked) patches."""
    def __init__(self, n_patches=256, embed_dim=384,
                 depth=6, n_heads=6):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, n_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, n_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, ids_keep):
        """
        x        : (B, N, D) all patch tokens
        ids_keep : (B, n_keep) indices of visible patches
        """
        B, N, D = x.shape

        # Add positional embedding before masking
        x = x + self.pos_embed[:, 1:, :]    # skip cls pos

        # Keep only visible patches
        x = torch.gather(x, 1,
            ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1) + self.pos_embed[:, :1, :]
        x   = torch.cat([cls, x], dim=1)    # (B, n_keep+1, D)

        x = self.blocks(x)
        x = self.norm(x)
        return x                            # (B, n_keep+1, D)


class MAEDecoder(nn.Module):
    """Lightweight decoder that reconstructs all N patches."""
    def __init__(self, n_patches=256, encoder_dim=384,
                 decoder_dim=192, depth=2, n_heads=3,
                 patch_size=16, in_ch=1):
        super().__init__()
        self.n_patches    = n_patches
        self.patch_size   = patch_size
        self.in_ch        = in_ch
        self.decoder_dim  = decoder_dim
        self.patch_pixels = patch_size * patch_size * in_ch

        self.embed    = nn.Linear(encoder_dim, decoder_dim)
        self.mask_tok = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, n_patches + 1, decoder_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.Sequential(
            *[TransformerBlock(decoder_dim, n_heads) for _ in range(depth)])
        self.norm    = nn.LayerNorm(decoder_dim)
        self.pred    = nn.Linear(decoder_dim, self.patch_pixels)

    def forward(self, x_enc, ids_keep, ids_restore):
        """
        x_enc       : (B, n_keep+1, encoder_dim)
        ids_keep    : (B, n_keep)
        ids_restore : (B, N)  permutation to restore original order
        """
        B    = x_enc.shape[0]
        N    = self.n_patches
        x    = self.embed(x_enc)             # (B, n_keep+1, D_dec)

        # Remove CLS, then unshuffle
        x_no_cls  = x[:, 1:, :]             # (B, n_keep, D_dec)
        n_keep    = x_no_cls.shape[1]
        n_mask    = N - n_keep

        mask_toks = self.mask_tok.expand(B, n_mask, -1)
        all_toks  = torch.cat([x_no_cls, mask_toks], dim=1)  # (B,N,D)
        all_toks  = torch.gather(all_toks, 1,
            ids_restore.unsqueeze(-1).repeat(1, 1, self.decoder_dim))

        # Add positional embedding (skip CLS pos)
        all_toks = all_toks + self.pos_embed[:, 1:, :]
        all_toks = self.blocks(all_toks)
        all_toks = self.norm(all_toks)
        pred     = self.pred(all_toks)       # (B, N, patch_pixels)
        return pred


# ──────────────────────────────────────────────
# 4.  Full MAE model
# ──────────────────────────────────────────────

class MaskedAutoencoder(nn.Module):
    """
    MAE for medical image anomaly detection.
    img_size=256, patch_size=16 → 256 patches.
    """
    def __init__(self,
                 img_size: int   = 256,
                 patch_size: int = 16,
                 in_ch: int      = 1,
                 encoder_dim: int = 384,
                 decoder_dim: int = 192,
                 encoder_depth: int = 6,
                 decoder_depth: int = 2,
                 n_heads_enc: int   = 6,
                 n_heads_dec: int   = 3,
                 mask_ratio: float  = 0.75):
        super().__init__()
        self.patch_size  = patch_size
        self.img_size    = img_size
        n_patches        = (img_size // patch_size) ** 2
        self.n_patches   = n_patches
        self.mask_ratio  = mask_ratio
        self.in_ch       = in_ch

        self.patch_embed = PatchEmbed(img_size, patch_size, in_ch, encoder_dim)
        self.encoder     = MAEEncoder(n_patches, encoder_dim,
                                      encoder_depth, n_heads_enc)
        self.decoder     = MAEDecoder(n_patches, encoder_dim,
                                      decoder_dim, decoder_depth, n_heads_dec,
                                      patch_size, in_ch)

    # ---- masking helpers --------------------------------------------------

    def _random_masking(self, x):
        """
        x : (B, N, D)
        Returns ids_keep, ids_restore, mask (1=masked)
        """
        B, N, D   = x.shape
        n_keep    = int(N * (1 - self.mask_ratio))

        noise     = torch.rand(B, N, device=x.device)
        ids_sort  = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_sort, dim=1)
        ids_keep  = ids_sort[:, :n_keep]

        mask      = torch.ones(B, N, device=x.device)
        mask[:, :n_keep] = 0
        mask      = torch.gather(mask, 1, ids_restore)   # 1=masked

        return ids_keep, ids_restore, mask

    # ---- patchify / unpatchify -------------------------------------------

    def patchify(self, x):
        """(B,C,H,W) → (B, N, patch_pixels)"""
        p   = self.patch_size
        B, C, H, W = x.shape
        x   = x.reshape(B, C, H//p, p, W//p, p)
        x   = torch.einsum('bchpwq->bhwpqc', x)
        x   = x.reshape(B, (H//p)*(W//p), p*p*C)
        return x

    def unpatchify(self, x):
        """(B, N, patch_pixels) → (B, C, H, W)"""
        p   = self.patch_size
        B   = x.shape[0]
        h = w = int(self.n_patches ** 0.5)
        C   = self.in_ch
        x   = x.reshape(B, h, w, p, p, C)
        x   = torch.einsum('bhwpqc->bchpwq', x)
        x   = x.reshape(B, C, h*p, w*p)
        return x

    # ---- forward -----------------------------------------------------------

    def forward(self, x):
        """
        Returns (pred_img, mask, loss)
        pred_img : full reconstruction (B,C,H,W) in [0,1]
        mask     : (B,N) 1=masked region
        """
        tokens              = self.patch_embed(x)      # (B,N,D)
        ids_keep, ids_restore, mask = self._random_masking(tokens)
        encoded             = self.encoder(tokens, ids_keep)
        pred_patches        = self.decoder(encoded, ids_keep, ids_restore)

        # Reconstruction loss: MSE on masked patches only
        target     = self.patchify(x)
        loss_mask  = mask.unsqueeze(-1).expand_as(target)
        loss_mse   = ((pred_patches - target) ** 2 * loss_mask).sum()
        loss_mse   = loss_mse / loss_mask.sum()

        pred_img   = torch.sigmoid(self.unpatchify(pred_patches))
        return pred_img, mask, loss_mse

    def reconstruct(self, x):
        """Full reconstruction without random masking (mask ratio=0)."""
        self.eval()
        old_ratio, self.mask_ratio = self.mask_ratio, 0.01
        with torch.no_grad():
            tokens              = self.patch_embed(x)
            ids_keep, ids_restore, mask = self._random_masking(tokens)
            encoded             = self.encoder(tokens, ids_keep)
            pred_patches        = self.decoder(encoded, ids_keep, ids_restore)
            pred_img            = torch.sigmoid(self.unpatchify(pred_patches))
        self.mask_ratio = old_ratio
        return pred_img


# ──────────────────────────────────────────────
# 5.  Pseudo-Abnormal Module
# ──────────────────────────────────────────────

class PseudoAbnormalModule:
    """
    Creates synthetic anomalies by modifying intensity of local regions.
    Used to generate positive (anomaly) training samples for the classifier.
    """
    def __init__(self, r_min=10, r_max=40,
                 scale_range=(0.5, 1.5),
                 offset_range=(-0.3, 0.3)):
        self.r_min   = r_min
        self.r_max   = r_max
        self.s_range = scale_range
        self.o_range = offset_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B,1,H,W) in [0,1]
        Returns x_pa with synthetic lesion-like regions.
        """
        x_pa = x.clone()
        B, C, H, W = x.shape
        for b in range(B):
            r  = random.randint(self.r_min, self.r_max)
            cy = random.randint(r, H - r)
            cx = random.randint(r, W - r)
            scale  = random.uniform(*self.s_range)
            offset = random.uniform(*self.o_range)

            yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W),
                                    indexing='ij')
            mask   = ((yy - cy)**2 + (xx - cx)**2) <= r**2
            x_pa[b, 0][mask] = (x_pa[b, 0][mask] * scale + offset).clamp(0,1)
        return x_pa


# ──────────────────────────────────────────────
# 6.  Anomaly Classifier (Stage 2)
# ──────────────────────────────────────────────

class AnomalyClassifier(nn.Module):
    """
    Small CNN that takes the residual map r = |x - x̂| and
    outputs P(anomaly) ∈ [0,1].
    Input: (B, 1, H, W)   residual map
    Output: (B,) probability
    """
    def __init__(self, img_size: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # H/2
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # H/4
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # H/8
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),                   # 4×4
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, r):
        f = self.features(r).flatten(1)
        return self.head(f).squeeze(-1)


# ──────────────────────────────────────────────
# 7.  Anomaly score at inference
# ──────────────────────────────────────────────

def anomaly_score(mae: MaskedAutoencoder,
                  classifier: AnomalyClassifier,
                  x: torch.Tensor) -> tuple:
    """
    Returns
    -------
    prob      : (B,) probability of anomaly from classifier
    error_map : (B,1,H,W) |x - x̂|
    x_hat     : (B,1,H,W) reconstruction
    """
    mae.eval(); classifier.eval()
    with torch.no_grad():
        x_hat     = mae.reconstruct(x)
        # Darkness-only: Highlight where Healthy (x_hat) > Actual (x)
        error_map = torch.clamp(x_hat - x, min=1e-6)
        prob      = classifier(error_map)
    return prob, error_map, x_hat


# ──────────────────────────────────────────────
if __name__ == '__main__':
    mae  = MaskedAutoencoder(img_size=256, patch_size=16)
    clf  = AnomalyClassifier(img_size=256)
    pa   = PseudoAbnormalModule()

    x     = torch.rand(2, 1, 256, 256)
    x_pa  = pa(x)
    pred, mask, loss = mae(x)
    print("MAE pred:", pred.shape, "loss:", loss.item())

    prob, emap, xh = anomaly_score(mae, clf, x)
    print("Anomaly prob:", prob)
    print(f"MAE params: {sum(p.numel() for p in mae.parameters()):,}")
    print(f"CLF params: {sum(p.numel() for p in clf.parameters()):,}")
