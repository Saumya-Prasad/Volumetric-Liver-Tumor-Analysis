# -*- coding: utf-8 -*-
"""
models/model_6_ensemble_ae.py
Ensemble of Autoencoders for Anomaly Detection in Biomedical Data
(Narrative Review approach – combined architecture)

Ensemble Strategy
-----------------
Train N diverse autoencoders independently on the same healthy data.
At inference:

  S_i(x) = ||x - AE_i(x)||²     (individual scores)
  S_ens(x) = mean_i S_i(x)      (ensemble mean)
  U_ens(x) = std_i  S_i(x)      (epistemic uncertainty)

Diversity is achieved through:
  1. Different architectures (Conv-AE, VAE, Bottleneck widths)
  2. Different random initialisations
  3. Different dropout masks during training

Combined Future-Scope Enhancement
-----------------------------------
  • Disagreement-weighted ensemble:
      S(x) = Σ_i w_i * S_i(x)   where w_i ∝ 1/val_loss_i
  • Uncertainty thresholding:
      if U_ens(x) > τ → flag for human review
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ──────────────────────────────────────────────
# 1.  Member autoencoders (varied architectures)
# ──────────────────────────────────────────────

class _ConvAE(nn.Module):
    """Standard convolutional AE member."""
    def __init__(self, in_ch=1, latent_ch=64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch,     32, 4, 2, 1, bias=False), nn.BatchNorm2d(32),  nn.ReLU(True),
            nn.Conv2d(32,        64, 4, 2, 1, bias=False), nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.Conv2d(64, latent_ch, 4, 2, 1, bias=False), nn.BatchNorm2d(latent_ch), nn.ReLU(True),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(latent_ch, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64,        32, 4, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32,     in_ch, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.dec(self.enc(x))


class _WiderAE(nn.Module):
    """Wider bottleneck AE member."""
    def __init__(self, in_ch=1, latent_ch=128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch,  64, 3, 2, 1, bias=False), nn.BatchNorm2d(64),  nn.GELU(),
            nn.Conv2d(64,    128, 3, 2, 1, bias=False), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, latent_ch, 3, 2, 1, bias=False), nn.BatchNorm2d(latent_ch), nn.GELU(),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(latent_ch, 128, 3, 2, 1, 1, bias=False), nn.BatchNorm2d(128), nn.GELU(),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1, bias=False), nn.BatchNorm2d(64), nn.GELU(),
            nn.ConvTranspose2d(64, in_ch, 3, 2, 1, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.dec(self.enc(x))


class _VAEMember(nn.Module):
    """Variational AE member with reparameterisation."""
    def __init__(self, in_ch=1, latent_dim=256):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 32, 4, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32,    64, 4, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64,    64, 4, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64,    64, 4, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(4),   # fixed 4×4 output regardless of input size
        )
        self.enc_flat  = 64 * 4 * 4   # = 1024, fixed
        self.fc_mu     = nn.Linear(self.enc_flat, latent_dim)
        self.fc_logvar = nn.Linear(self.enc_flat, latent_dim)
        self.fc_dec    = nn.Linear(latent_dim, self.enc_flat)
        # Decoder: 4×4 → original size via AdaptiveUpsample at the end
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, in_ch, 4, 2, 1), nn.Sigmoid(),
        )

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h      = self.enc(x).flatten(1)
        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z      = self.reparameterise(mu, logvar)
        # Reshape to (B, 64, 4, 4) — enc_flat = 64*4*4 = 1024
        z_sp   = self.fc_dec(z).view(-1, 64, 4, 4)
        x_hat  = self.dec(z_sp)   # 4→8→16→32→64
        # Match input size
        if x_hat.shape[-1] != x.shape[-1]:
            x_hat = torch.nn.functional.interpolate(
                x_hat, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return x_hat, mu, logvar


class _DropoutAE(nn.Module):
    """MC-Dropout AE member for uncertainty estimation."""
    def __init__(self, in_ch=1, latent_ch=64, p_drop=0.3):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch,     32, 4, 2, 1), nn.ReLU(True), nn.Dropout2d(p_drop),
            nn.Conv2d(32,        64, 4, 2, 1), nn.ReLU(True), nn.Dropout2d(p_drop),
            nn.Conv2d(64, latent_ch, 4, 2, 1), nn.ReLU(True),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(latent_ch, 64, 4, 2, 1), nn.ReLU(True), nn.Dropout2d(p_drop),
            nn.ConvTranspose2d(64,        32, 4, 2, 1), nn.ReLU(True), nn.Dropout2d(p_drop),
            nn.ConvTranspose2d(32,     in_ch, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.dec(self.enc(x))


# ──────────────────────────────────────────────
# 2.  Ensemble model
# ──────────────────────────────────────────────

class EnsembleAE(nn.Module):
    """
    Ensemble of 4 diverse autoencoders.

    Call model(x) for training (returns individual losses).
    Call anomaly_score(model, x) for inference.
    """
    def __init__(self, in_ch: int = 1, img_size: int = 256):
        super().__init__()
        self.img_size = img_size
        # 4 members with different designs
        self.members = nn.ModuleList([
            _ConvAE(in_ch,   latent_ch=64),
            _WiderAE(in_ch,  latent_ch=128),
            _VAEMember(in_ch, latent_dim=512),
            _DropoutAE(in_ch, latent_ch=64, p_drop=0.2),
        ])
        self.member_names = ['ConvAE', 'WiderAE', 'VAE', 'DropoutAE']

    def forward(self, x):
        """
        Returns dict of {name: (x_hat, loss)} for each member.
        """
        out = {}
        for name, m in zip(self.member_names, self.members):
            if isinstance(m, _VAEMember):
                x_hat, mu, logvar = m(x)
                recon = F.mse_loss(x_hat, x)
                kld   = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
                loss  = recon + 0.001 * kld
            else:
                x_hat = m(x)
                loss  = F.mse_loss(x_hat, x)
            out[name] = (x_hat, loss)
        return out

    def reconstruct_all(self, x):
        """Returns list of x_hat tensors from all members."""
        outs = []
        for m in self.members:
            if isinstance(m, _VAEMember):
                x_hat, _, _ = m(x)
            else:
                x_hat = m(x)
            outs.append(x_hat)
        return outs


# ──────────────────────────────────────────────
# 3.  Disagreement-weighted anomaly scoring
# ──────────────────────────────────────────────

class EnsembleScorer:
    """
    Manages per-member validation weights and ensemble scoring.

    Usage
    -----
    scorer = EnsembleScorer(model)
    scorer.fit_weights(val_loader, device)   # once after training
    score, emap, uncertainty = scorer.score(x, device)
    """

    def __init__(self, model: EnsembleAE):
        self.model   = model
        self.weights = None   # will be set in fit_weights

    def fit_weights(self, val_loader, device='cpu'):
        """
        w_i ∝ 1 / val_loss_i   (better members get higher weight).
        Computes one epoch on val_loader.
        """
        self.model.eval()
        losses = {n: [] for n in self.model.member_names}
        with torch.no_grad():
            for x in val_loader:
                x    = x.to(device)
                outs = self.model(x)
                for name, (x_hat, loss) in outs.items():
                    losses[name].append(loss.item())

        mean_losses = {n: np.mean(v) for n, v in losses.items()}
        inv         = np.array([1.0 / (l + 1e-8)
                                 for l in mean_losses.values()])
        self.weights = inv / inv.sum()
        print("[Ensemble] Member weights:", dict(zip(self.model.member_names,
                                                      self.weights.round(3))))

    @torch.no_grad()
    def score(self, x: torch.Tensor) -> tuple:
        """
        S(x)  = Σ_i w_i * ||x - AE_i(x)||²    weighted ensemble score
        U(x)  = std_i ||x - AE_i(x)||²          epistemic uncertainty
        emap  = per-pixel mean error

        Returns (score, emap, uncertainty)
        """
        self.model.eval()
        if self.weights is None:
            w = np.ones(len(self.model.members)) / len(self.model.members)
        else:
            w = self.weights

        all_errors = []
        for m in self.model.members:
            m.eval()
            if isinstance(m, _VAEMember):
                x_hat, _, _ = m(x)
            else:
                x_hat = m(x)
            # Darkness-only error: reconstruction - input
            err_map = torch.clamp(x_hat - x, min=1e-6)
            err = (err_map ** 2).mean(dim=[1,2,3])  # (B,)
            all_errors.append(err)

        # Stack: (N_members, B)
        err_stack = torch.stack(all_errors, dim=0)

        weights_t = torch.tensor(w, dtype=x.dtype, device=x.device)
        score     = (weights_t[:, None] * err_stack).sum(0)   # (B,)
        uncertainty = err_stack.std(0)                          # (B,)

        # Mean pixel-error map
        all_maps = []
        for m in self.model.members:
            if isinstance(m, _VAEMember):
                xh, _, _ = m(x)
            else:
                xh = m(x)
            # Darkness-only: Highlight where Healthy (xh) > Actual (x)
            all_maps.append(torch.clamp(xh - x, min=1e-6))
        emap = torch.stack(all_maps).mean(0)                    # (B,1,H,W)

        return score, emap, uncertainty


# ──────────────────────────────────────────────
if __name__ == '__main__':
    model   = EnsembleAE(in_ch=1, img_size=256)
    x       = torch.rand(2, 1, 256, 256)
    results = model(x)
    for name, (xh, loss) in results.items():
        print(f"  {name}: x_hat={xh.shape}  loss={loss:.4f}")

    scorer  = EnsembleScorer(model)
    score, emap, unc = scorer.score(x)
    print("Score      :", score)
    print("Uncertainty:", unc)
    n = sum(p.numel() for p in model.parameters())
    print(f"Total params: {n:,}")
