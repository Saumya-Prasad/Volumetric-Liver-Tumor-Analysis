# -*- coding: utf-8 -*-
"""
models/model_4_ccb_aae.py
Unsupervised Deep Anomaly Detection for Medical Images Using an
Improved Adversarial Autoencoder (Zhang et al., J. Digital Imaging 2022)

Architecture
------------
Key innovation: Chain of Convolution Block (CCB) replaces the
conventional skip connections in adversarial autoencoders.

CCB structure:
  z_enc_i  ──► [Conv→BN→ReLU → Conv→BN] ──► non-linear residual ──► z̃_i
  z̃_i is used at the corresponding decoder level (no direct bypass).

This bridges the semantic gap between encoder and decoder features
while preventing information leakage that trivially solves reconstruction.

Full model:
  Encoder E : x → {z_1,...,z_L, z_latent}
  CCB       : z_i → z̃_i for each skip
  Decoder G : z_latent, {z̃_1,...,z̃_L} → x̂
  Discriminator D: judges x real vs x̂ fake (also in latent space)

Loss
----
  L_recon   = MSE(x, x̂)                    pixel-level recon
  L_adv_img = -log D_img(x̂)               GAN loss (image space)
  L_adv_lat = MSE(z_latent, N(0,1) sample) latent regularisation
  L_G = L_recon + λ1*L_adv_img + λ2*L_adv_lat
  L_D = BCE(D(x), 1) + BCE(D(x̂.detach()), 0)

Anomaly Score
-------------
  S(x) = α * ||x - x̂||² + β * ||z_latent - μ_train||²
  where μ_train is the mean latent computed on training data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# 1.  Chain of Convolution Block (CCB)
# ──────────────────────────────────────────────

class CCB(nn.Module):
    """
    Chain of Convolution Block.
    Transforms encoder feature z_i → z̃_i using two conv layers
    with a non-linear (tanh-gated) residual path.

    This bridges the semantic gap without direct feature copying.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        # Non-linear gate: tanh activation controls information flow
        self.gate  = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        h = F.relu(self.bn1(self.conv1(z)), inplace=True)
        h = self.bn2(self.conv2(h))
        g = self.gate(z)               # ∈ (-1, 1)
        # Non-linear residual: z + gate(z) * transformed_features
        return F.relu(z + g * h, inplace=True)


# ──────────────────────────────────────────────
# 2.  Encoder
# ──────────────────────────────────────────────

class CCBEncoder(nn.Module):
    """
    Produces hierarchical features at 4 scales + bottleneck latent.
    256→128→64→32→16 (latent)
    """
    def __init__(self, in_ch=1, base=32):
        super().__init__()
        def block(ic, oc):
            return nn.Sequential(
                nn.Conv2d(ic, oc, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(oc),
                nn.LeakyReLU(0.2, inplace=True),
            )
        self.e1 = block(in_ch,   base)      # 256→128   (base)
        self.e2 = block(base,    base*2)    # 128→64
        self.e3 = block(base*2,  base*4)    # 64→32
        self.e4 = block(base*4,  base*8)    # 32→16  ← bottleneck

    def forward(self, x):
        z1 = self.e1(x)      # (B, 32,  128,128)
        z2 = self.e2(z1)     # (B, 64,   64, 64)
        z3 = self.e3(z2)     # (B,128,   32, 32)
        z4 = self.e4(z3)     # (B,256,   16, 16)  ← latent
        return z1, z2, z3, z4


# ──────────────────────────────────────────────
# 3.  Decoder with CCB skip connections
# ──────────────────────────────────────────────

class CCBDecoder(nn.Module):
    """
    Mirrors the encoder; uses CCB-transformed skip features at each level.
    """
    def __init__(self, out_ch=1, base=32):
        super().__init__()
        def up_block(ic, oc):
            return nn.Sequential(
                nn.ConvTranspose2d(ic, oc, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(oc),
                nn.ReLU(inplace=True),
            )
        # Channels: at each level we concatenate CCB skip (doubles channels)
        self.d1 = up_block(base*8,    base*4)   # 16→32   (skip from e3)
        self.d2 = up_block(base*4*2,  base*2)   # 32→64   (skip from e2)
        self.d3 = up_block(base*2*2,  base)     # 64→128  (skip from e1)
        self.d4 = nn.Sequential(
            nn.ConvTranspose2d(base*2, out_ch, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )                                        # 128→256

        # CCB for each skip connection
        self.ccb3 = CCB(base*4)
        self.ccb2 = CCB(base*2)
        self.ccb1 = CCB(base)

    def forward(self, z1, z2, z3, z4):
        d = self.d1(z4)                                  # 16→32
        d = self.d2(torch.cat([d,  self.ccb3(z3)], 1))  # 32→64
        d = self.d3(torch.cat([d,  self.ccb2(z2)], 1))  # 64→128
        x_hat = self.d4(torch.cat([d, self.ccb1(z1)], 1))  # 128→256
        return x_hat


# ──────────────────────────────────────────────
# 4.  Discriminator
# ──────────────────────────────────────────────

class Discriminator(nn.Module):
    """
    PatchGAN discriminator (image-level realness scoring).
    Also used in latent space as a Gaussian prior regulariser.
    """
    def __init__(self, in_ch=1, base=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,  base,   4, stride=2, padding=1),   # 256→128
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base,   base*2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base*2, base*4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base*4, 1, 4, stride=2, padding=1),        # patch output
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)       # (B,1,H',W')


# ──────────────────────────────────────────────
# 5.  Full CCB-AAE
# ──────────────────────────────────────────────

class CCBAAE(nn.Module):
    """
    Chain-of-Convolution-Block Adversarial Autoencoder.
    Use `.generator_params()` and `.discriminator_params()` for
    separate optimisers.
    """
    def __init__(self, in_ch=1, base=32):
        super().__init__()
        self.encoder       = CCBEncoder(in_ch, base)
        self.decoder       = CCBDecoder(in_ch, base)
        self.discriminator = Discriminator(in_ch, base)

    def encode(self, x):
        return self.encoder(x)      # (z1,z2,z3,z4)

    def decode(self, z1, z2, z3, z4):
        return self.decoder(z1, z2, z3, z4)

    def forward(self, x):
        z1, z2, z3, z4 = self.encoder(x)
        x_hat           = self.decoder(z1, z2, z3, z4)
        return x_hat, z4            # z4 is the bottleneck latent

    def generator_params(self):
        return list(self.encoder.parameters()) + \
               list(self.decoder.parameters())

    def discriminator_params(self):
        return list(self.discriminator.parameters())


# ──────────────────────────────────────────────
# 6.  Loss functions
# ──────────────────────────────────────────────

class CCBAAELoss:
    """
    Generator:     L_G = L_recon + λ1*L_adv_img + λ2*L_adv_lat
    Discriminator: L_D = BCE real + BCE fake
    """
    def __init__(self, lam1=0.1, lam2=0.1):
        self.lam1 = lam1
        self.lam2 = lam2

    def generator_loss(self, x, x_hat, z_latent, disc_fake):
        """
        x, x_hat    : (B,1,H,W)
        z_latent    : (B,C,h,w)  encoder bottleneck
        disc_fake   : D(x_hat) patch output
        """
        # Reconstruction loss (dual: pixel + latent)
        l_recon = F.mse_loss(x_hat, x)

        # Adversarial image loss: fool discriminator
        l_adv   = F.binary_cross_entropy(disc_fake,
                                          torch.ones_like(disc_fake))

        # Latent regularisation: push z towards N(0,1)
        z_prior = torch.randn_like(z_latent)
        l_lat   = F.mse_loss(z_latent, z_prior)

        total = l_recon + self.lam1 * l_adv + self.lam2 * l_lat
        return total, l_recon, l_adv, l_lat

    def discriminator_loss(self, disc_real, disc_fake):
        l_real = F.binary_cross_entropy(disc_real, torch.ones_like(disc_real))
        l_fake = F.binary_cross_entropy(disc_fake.detach(),
                                         torch.zeros_like(disc_fake))
        return (l_real + l_fake) * 0.5


# ──────────────────────────────────────────────
# 7.  Anomaly scoring
# ──────────────────────────────────────────────

def compute_train_latent_mean(model: CCBAAE,
                               loader,
                               device: str = 'cpu') -> torch.Tensor:
    """Compute mean bottleneck latent over training set for anomaly scoring."""
    model.eval()
    latents = []
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            _, z4 = model(x)
            latents.append(z4.mean(dim=[2,3]))   # (B,C)
    return torch.cat(latents, 0).mean(0)         # (C,)


def anomaly_score(model: CCBAAE,
                  x: torch.Tensor,
                  z_mean: torch.Tensor,
                  alpha: float = 1.0,
                  beta: float  = 0.5) -> tuple:
    """
    S(x) = α * ||x - x̂||² + β * ||z - μ_train||²

    Returns (score, error_map, x_hat)
    """
    model.eval()
    with torch.no_grad():
        x_hat, z4 = model(x)
        error_map = (x - x_hat).pow(2)
        recon_err = error_map.mean(dim=[1,2,3])

        z_flat    = z4.mean(dim=[2,3])             # (B,C)
        lat_err   = ((z_flat - z_mean.to(x.device)).pow(2)).mean(-1)

        score = alpha * recon_err + beta * lat_err
    return score, error_map.sqrt(), x_hat


# ──────────────────────────────────────────────
if __name__ == '__main__':
    model = CCBAAE(in_ch=1, base=32)
    x     = torch.randn(2, 1, 256, 256)
    x_hat, z4 = model(x)
    print("x_hat :", x_hat.shape)
    print("z4    :", z4.shape)

    d_real = model.discriminator(x)
    d_fake = model.discriminator(x_hat)
    print("D_real:", d_real.shape)

    loss_fn = CCBAAELoss()
    lg, lr, la, ll = loss_fn.generator_loss(x, x_hat, z4, d_fake)
    ld  = loss_fn.discriminator_loss(d_real, d_fake)
    print(f"L_G={lg:.4f}  L_D={ld:.4f}")

    n_gen  = sum(p.numel() for p in model.generator_params())
    n_disc = sum(p.numel() for p in model.discriminator_params())
    print(f"Generator params: {n_gen:,}  Discriminator: {n_disc:,}")
