# -*- coding: utf-8 -*-
"""
models/model_2_ae_flow.py
AE-FLOW: Autoencoders with Normalizing Flows for Medical Image Anomaly Detection
ICLR 2023  –  Zhao, Ding, Zhang (Shanghai Jiao Tong University)

Architecture
------------
  x ──► Encoder f ──► z ──► NF Φ ──► z' ~ N(0,I)
                                 │
                                 ▼
  x'◄── Decoder g ◄──────────── z'

Loss
----
  L_recon = MSE(x, g(z'))          reconstruction loss
  L_flow  = -Σ log|det(∂Φ_k/∂z)| + ½||z'||²   NLL of standard Gaussian
  L_total = L_recon + λ * L_flow     λ=1 default

Anomaly Score (at inference)
----------------------------
  S(x) = α * ||x - x'||² + β * L_flow(z')
  α=β=1 default; both terms are normalised

References
----------
  Paper Fig. 2 + Sec. 2  (pretrained ConvNet encoder,
  RealNVP-style coupling layers, symmetric decoder).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ──────────────────────────────────────────────
# 1.  Encoder  (pretrained-style ConvNet)
# ──────────────────────────────────────────────

class EncoderBlock(nn.Module):
    """Single strided-conv block used by the encoder."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class AEFlowEncoder(nn.Module):
    """
    Downsamples 256×256 → 16×16 (4 strides of 2).
    Output feature z ∈ ℝ^{C × H/16 × W/16}.
    """
    def __init__(self, in_ch=1, base_ch=32):
        super().__init__()
        self.blocks = nn.Sequential(
            EncoderBlock(in_ch,       base_ch),      # 256→128
            EncoderBlock(base_ch,     base_ch * 2),  # 128→64
            EncoderBlock(base_ch * 2, base_ch * 4),  # 64→32
            EncoderBlock(base_ch * 4, base_ch * 8),  # 32→16
        )

    def forward(self, x):
        return self.blocks(x)   # (B, C, H/16, W/16)


# ──────────────────────────────────────────────
# 2.  Normalizing Flow  (RealNVP coupling layers)
# ──────────────────────────────────────────────

class CouplingLayer(nn.Module):
    """
    Affine coupling layer from RealNVP (Dinh et al., 2017).

    Splits z into (z₁, z₂).
    z₁' = z₁
    z₂' = z₂ * exp(s(z₁)) + t(z₁)

    log|det J| = Σ s(z₁)
    """
    def __init__(self, n_features: int, mask: torch.Tensor):
        super().__init__()
        self.register_buffer('mask', mask)

        # Scale & translate networks (1×1 conv in feature space)
        self.s_net = nn.Sequential(
            nn.Conv2d(n_features, n_features, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_features, n_features, 1),
            nn.Tanh(),          # bounded scale
        )
        self.t_net = nn.Sequential(
            nn.Conv2d(n_features, n_features, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_features, n_features, 1),
        )

    def forward(self, z, reverse=False):
        z_masked = z * self.mask
        s = self.s_net(z_masked) * (1 - self.mask)
        t = self.t_net(z_masked) * (1 - self.mask)

        if not reverse:
            z_out   = z_masked + (1 - self.mask) * (z * torch.exp(s) + t)
            log_det = (s).sum(dim=[1, 2, 3])          # Σ s_i
            return z_out, log_det
        else:
            z_out = z_masked + (1 - self.mask) * ((z - t) * torch.exp(-s))
            return z_out


class ActNorm(nn.Module):
    """Activation normalization (Glow, Kingma & Dhariwal 2018)."""
    def __init__(self, n_features: int):
        super().__init__()
        self.initialized = False
        self.log_scale = nn.Parameter(torch.zeros(1, n_features, 1, 1))
        self.bias      = nn.Parameter(torch.zeros(1, n_features, 1, 1))

    def forward(self, z, reverse=False):
        if not self.initialized:
            with torch.no_grad():
                mean = z.mean(dim=[0, 2, 3], keepdim=True)
                std  = z.std(dim=[0, 2, 3], keepdim=True).clamp(min=1e-6)
                self.bias.data      = -mean
                self.log_scale.data = -torch.log(std)
            self.initialized = True

        hw = z.shape[2] * z.shape[3]
        if not reverse:
            z_out   = (z + self.bias) * torch.exp(self.log_scale)
            log_det = self.log_scale.sum() * hw
            return z_out, log_det
        else:
            return z * torch.exp(-self.log_scale) - self.bias


class NormalizingFlow(nn.Module):
    """
    Stack of K coupling + actnorm blocks.
    Alternates channel-split masks (checkerboard).
    K=8 as in the paper.
    """
    def __init__(self, n_features: int, K: int = 8):
        super().__init__()
        self.flows = nn.ModuleList()

        # Alternating channel masks
        for k in range(K):
            mask = torch.zeros(1, n_features, 1, 1)
            # Split channels: first half / second half alternating
            if k % 2 == 0:
                mask[0, :n_features // 2, 0, 0] = 1.0
            else:
                mask[0, n_features // 2:, 0, 0] = 1.0
            self.flows.append(ActNorm(n_features))
            self.flows.append(CouplingLayer(n_features, mask))

    def forward(self, z):
        """z → z' (Gaussian) + log_det_sum"""
        log_det_total = torch.zeros(z.shape[0], device=z.device)
        for layer in self.flows:
            if isinstance(layer, ActNorm):
                z, ld = layer(z, reverse=False)
                log_det_total = log_det_total + ld
            elif isinstance(layer, CouplingLayer):
                z, ld = layer(z, reverse=False)
                log_det_total = log_det_total + ld
        return z, log_det_total

    def reverse(self, z_prime):
        """z' → z  (reconstruction path)"""
        for layer in reversed(self.flows):
            if isinstance(layer, ActNorm):
                z_prime = layer(z_prime, reverse=True)
            elif isinstance(layer, CouplingLayer):
                z_prime = layer(z_prime, reverse=True)
        return z_prime


# ──────────────────────────────────────────────
# 3.  Decoder (symmetric to encoder)
# ──────────────────────────────────────────────

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 3, stride=2,
                               padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class AEFlowDecoder(nn.Module):
    """16×16 → 256×256."""
    def __init__(self, out_ch=1, base_ch=32):
        super().__init__()
        self.blocks = nn.Sequential(
            DecoderBlock(base_ch * 8, base_ch * 4),  # 16→32
            DecoderBlock(base_ch * 4, base_ch * 2),  # 32→64
            DecoderBlock(base_ch * 2, base_ch),       # 64→128
            nn.ConvTranspose2d(base_ch, out_ch, 3,
                               stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.blocks(z)


# ──────────────────────────────────────────────
# 4.  Full AE-FLOW model
# ──────────────────────────────────────────────

class AEFlow(nn.Module):
    """
    Full AE-FLOW model.

    Forward pass
    ------------
    x → Encoder → z → NF → z' ~ N(0,I)
                        ↓
    x' ← Decoder ← z'
    Returns (x', z', log_det)
    """
    def __init__(self, in_ch: int = 1, base_ch: int = 32, K: int = 8):
        super().__init__()
        latent_ch    = base_ch * 8
        self.encoder = AEFlowEncoder(in_ch, base_ch)
        self.flow    = NormalizingFlow(latent_ch, K)
        self.decoder = AEFlowDecoder(in_ch, base_ch)

    def forward(self, x):
        z           = self.encoder(x)           # (B,C,H/16,W/16)
        z_prime, ld = self.flow(z)              # z' ~ N(0,I)
        x_hat       = self.decoder(z_prime)     # reconstruction
        return x_hat, z_prime, ld

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


# ──────────────────────────────────────────────
# 5.  Loss function
# ──────────────────────────────────────────────

class AEFlowLoss(nn.Module):
    """
    L_total = L_recon + λ * L_flow

    L_recon = MSE(x, x')
    L_flow  = mean[ ½||z'||² - log_det ]   (NLL of standard Gaussian)
    """
    def __init__(self, lam: float = 1.0):
        super().__init__()
        self.lam = lam

    def forward(self, x, x_hat, z_prime, log_det):
        # Reconstruction loss
        l_recon = F.mse_loss(x_hat, x)

        # Flow loss:  NLL = ½||z'||² - log|det J|
        hw       = z_prime.shape[2] * z_prime.shape[3]
        l_flow   = (0.5 * z_prime.pow(2).sum(dim=[1,2,3]) - log_det)
        l_flow   = l_flow.mean() / hw   # normalise by spatial size

        total = l_recon + self.lam * l_flow
        return total, l_recon, l_flow


# ──────────────────────────────────────────────
# 6.  Anomaly score
# ──────────────────────────────────────────────

def anomaly_score(model: AEFlow,
                  x: torch.Tensor,
                  alpha: float = 1.0,
                  beta: float  = 1.0) -> tuple:
    """
    S(x) = α * ||x - x'||² + β * (½||z'||² - log_det)

    Returns (score, recon_error_map, x_hat)
    """
    model.eval()
    with torch.no_grad():
        x_hat, z_prime, log_det = model(x)

        # Per-image reconstruction error
        recon_map  = (x - x_hat).pow(2)             # (B,1,H,W)
        recon_err  = recon_map.mean(dim=[1,2,3])    # (B,)

        # Per-image flow score
        hw         = z_prime.shape[2] * z_prime.shape[3]
        flow_score = (0.5 * z_prime.pow(2).sum(dim=[1,2,3])
                      - log_det) / hw               # (B,)

        score = alpha * recon_err + beta * flow_score

    return score, recon_map.sqrt(), x_hat


# ──────────────────────────────────────────────
if __name__ == '__main__':
    model = AEFlow(in_ch=1, base_ch=32, K=8)
    x     = torch.randn(2, 1, 256, 256)
    x_hat, z_prime, ld = model(x)
    print("x_hat  :", x_hat.shape)
    print("z'     :", z_prime.shape)
    print("log_det:", ld)
    loss_fn = AEFlowLoss(lam=1.0)
    total, lr, lf = loss_fn(x, x_hat, z_prime, ld)
    print(f"Loss total={total:.4f}  recon={lr:.4f}  flow={lf:.4f}")
    s, emap, xh = anomaly_score(model, x)
    print("Anomaly scores:", s)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {total_params:,}")
