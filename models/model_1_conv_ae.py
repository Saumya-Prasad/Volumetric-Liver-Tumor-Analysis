# -*- coding: utf-8 -*-
"""
models/model_1_conv_ae.py
Vanilla Convolutional Autoencoder — baseline.

Architecture
------------
Encoder : Conv(1→16,s2) → Conv(16→32,s2) → Conv(32→64,s2) → Conv(64→128,s2)
Decoder : ConvTranspose mirror

Anomaly score
-------------
  S(x) = ||x - AE(x)||²   (per-pixel MSE)

Loss
----
  L = MSE(x, x̂)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────

def conv_block(in_ch, out_ch, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3,
                  stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )


def deconv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3,
                           stride=2, padding=1, output_padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────

class ConvAutoencoder(nn.Module):
    """
    Simple 4-level convolutional autoencoder.

    Input  : (B, 1, H, W)   H=W=256 by default
    Output : (B, 1, H, W)   values in [0, 1] (Sigmoid)
    """

    def __init__(self, in_channels: int = 1, latent_channels: int = 128):
        super().__init__()

        # Encoder: 256→128→64→32→16
        self.encoder = nn.Sequential(
            conv_block(in_channels,    32),   # 256→128
            conv_block(32,             64),   # 128→64
            conv_block(64,            128),   # 64→32
            conv_block(128, latent_channels), # 32→16
        )

        # Decoder: 16→32→64→128→256
        self.decoder = nn.Sequential(
            deconv_block(latent_channels, 128),  # 16→32
            deconv_block(128, 64),               # 32→64
            deconv_block(64,  32),               # 64→128
            nn.ConvTranspose2d(32, in_channels,
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 128→256
            nn.Sigmoid(),
        )

    def forward(self, x):
        z    = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def encode(self, x):
        return self.encoder(x)


# ──────────────────────────────────────────────
# Anomaly scoring
# ──────────────────────────────────────────────

def anomaly_score(model: ConvAutoencoder,
                  x: torch.Tensor) -> tuple:
    """
    Returns
    -------
    score : (B,) mean per-image anomaly score
    error_map : (B,1,H,W) pixel-wise absolute error
    x_hat : (B,1,H,W) reconstruction
    """
    model.eval()
    with torch.no_grad():
        x_hat     = model(x)
        # Standard absolute error captures both hyper-dense and hypo-dense tumors
        error_map = torch.abs(x - x_hat)
        score     = error_map.mean(dim=[1,2,3])  # scalar per image
    return score, error_map, x_hat


# ──────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────

if __name__ == '__main__':
    m = ConvAutoencoder()
    x = torch.randn(4, 1, 256, 256)
    out = m(x)
    print("Input :", x.shape)
    print("Output:", out.shape)
    s, emap, xh = anomaly_score(m, x)
    print("Score :", s)
    total = sum(p.numel() for p in m.parameters())
    print(f"Params: {total:,}")
