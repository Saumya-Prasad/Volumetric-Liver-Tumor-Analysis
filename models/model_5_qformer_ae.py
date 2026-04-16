# -*- coding: utf-8 -*-
"""
models/model_5_qformer_ae.py
Q-Former Autoencoder: A Modern Framework for Medical Anomaly Detection
Dalmonte, Bayar, Akbas, Georgescu – arXiv 2507.18481 (Jul 2025)

Architecture
------------
x ──► Frozen Foundation Encoder (DINOv2-small or simple CNN fallback)
      Produces multi-scale features E = {e1, e2, e3}
           │
           ▼ concat + project → E ∈ ℝ^{N×D_enc}
      Q-Former Bottleneck
        • M learnable query tokens Q ∈ ℝ^{M×D_q}
        • Self-Attention:   Q  ← SA(Q)
        • Cross-Attention:  Q  ← CA(Q, E)
        • MLP:              Q  ← MLP(Q)
        → output Z ∈ ℝ^{M×D_q}   (fixed-length bottleneck)
           │
           ▼ reshape / project → spatial latent
      CNN Decoder  →  x̂ ∈ [0,1]

Loss
----
  L_total = MSE(x, x̂) + λ_p * L_perceptual
  L_perceptual = 1 - cos_sim(F_mae(x), F_mae(x̂))
  where F_mae are features from a *frozen* MAE model (pretrained on ImageNet)

Future-scope improvement
------------------------
  • Cosine annealing of λ_p during training
  • Multi-scale query sets (Q_coarse, Q_fine) with separate cross-attentions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ──────────────────────────────────────────────
# 1.  Foundation encoder
#     Uses DINOv2-small if available, else a plain ResNet-like CNN
# ──────────────────────────────────────────────

class SimpleCNNFoundation(nn.Module):
    """
    Lightweight substitute for DINOv2 when torchvision hub is unavailable.
    Returns 3 intermediate feature maps (multi-scale).
    Input: (B,1,H,W) or (B,3,H,W)
    """
    def __init__(self, in_ch=1, base=32):
        super().__init__()
        def blk(ic,oc,s=2):
            return nn.Sequential(
                nn.Conv2d(ic,oc,3,stride=s,padding=1,bias=False),
                nn.BatchNorm2d(oc), nn.GELU())

        self.stage1 = nn.Sequential(blk(in_ch,base),   blk(base,  base,1))  # /2
        self.stage2 = nn.Sequential(blk(base,  base*2),blk(base*2,base*2,1))# /4
        self.stage3 = nn.Sequential(blk(base*2,base*4),blk(base*4,base*4,1))# /8
        self.stage4 = nn.Sequential(blk(base*4,base*8),blk(base*8,base*8,1))# /16
        self.out_dims = [base, base*2, base*4, base*8]  # per stage

    def forward(self, x):
        e1 = self.stage1(x)   # /2
        e2 = self.stage2(e1)  # /4
        e3 = self.stage3(e2)  # /8
        e4 = self.stage4(e3)  # /16
        return [e1, e2, e3, e4]   # list of feature maps


def build_foundation_encoder(in_ch=1, base=32, use_dinov2=False):
    """
    If use_dinov2=True and torch.hub is available, load frozen DINOv2.
    Otherwise fall back to SimpleCNNFoundation.
    """
    if use_dinov2:
        try:
            model = torch.hub.load('facebookresearch/dinov2',
                                   'dinov2_vits14', pretrained=True)
            for p in model.parameters():
                p.requires_grad = False
            print("[Q-FAE] Using frozen DINOv2-small encoder.")
            return model, None   # caller must handle DINOv2 specially
        except Exception as e:
            print(f"[Q-FAE] DINOv2 unavailable ({e}). Using CNN fallback.")

    enc = SimpleCNNFoundation(in_ch=in_ch, base=base)
    for p in enc.parameters():   # freeze foundation model
        p.requires_grad = False
    return enc, enc.out_dims


# ──────────────────────────────────────────────
# 2.  Multi-scale feature projector
# ──────────────────────────────────────────────

class MultiScaleProjector(nn.Module):
    """
    Projects and concatenates multi-scale encoder outputs into
    a single token sequence E ∈ ℝ^{N_total × D_enc}.
    Uses adaptive pooling so all maps become (8×8).
    """
    def __init__(self, in_dims: list, d_enc: int = 256):
        super().__init__()
        self.pool  = nn.AdaptiveAvgPool2d(8)      # all → 8×8 = 64 tokens
        self.projs = nn.ModuleList([
            nn.Conv2d(d, d_enc, 1) for d in in_dims
        ])

    def forward(self, features: list):
        """features: list of (B,C_i,H_i,W_i)"""
        tokens = []
        for feat, proj in zip(features, self.projs):
            p = self.pool(proj(feat))              # (B, D_enc, 8, 8)
            t = p.flatten(2).transpose(1, 2)       # (B, 64, D_enc)
            tokens.append(t)
        E = torch.cat(tokens, dim=1)               # (B, 64*n_scales, D_enc)
        return E


# ──────────────────────────────────────────────
# 3.  Q-Former bottleneck
# ──────────────────────────────────────────────

class QFormerLayer(nn.Module):
    """
    One Q-Former layer:
      SA  : Q  ← LayerNorm + SelfAttn(Q)  + Q
      CA  : Q  ← LayerNorm + CrossAttn(Q, E) + Q
      MLP : Q  ← LayerNorm + FFN(Q) + Q
    """
    def __init__(self, d_q: int, d_enc: int, n_heads: int = 8):
        super().__init__()
        self.norm_sa  = nn.LayerNorm(d_q)
        self.sa       = nn.MultiheadAttention(d_q, n_heads, batch_first=True)
        self.norm_ca  = nn.LayerNorm(d_q)
        self.norm_kv  = nn.LayerNorm(d_enc)
        # Project E to Q dimension for cross-attn K/V
        self.kv_proj  = nn.Linear(d_enc, d_q)
        self.ca       = nn.MultiheadAttention(d_q, n_heads, batch_first=True)
        self.norm_ffn = nn.LayerNorm(d_q)
        mlp_dim       = d_q * 4
        self.ffn      = nn.Sequential(
            nn.Linear(d_q, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, d_q))

    def forward(self, Q, E):
        # Self-attention
        h, _ = self.sa(self.norm_sa(Q), self.norm_sa(Q), self.norm_sa(Q))
        Q = Q + h

        # Cross-attention: Q queries E
        KV  = self.kv_proj(self.norm_kv(E))
        Qn  = self.norm_ca(Q)
        h, _= self.ca(Qn, KV, KV)
        Q   = Q + h

        # FFN
        Q = Q + self.ffn(self.norm_ffn(Q))
        return Q


class QFormer(nn.Module):
    """
    Q-Former bottleneck: M learnable queries aggregate E → Z.

    Parameters
    ----------
    M       : number of learnable query tokens
    d_q     : query dimension
    d_enc   : encoder feature dimension
    depth   : number of Q-Former layers
    """
    def __init__(self, M=64, d_q=256, d_enc=256, depth=4, n_heads=8):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, M, d_q) * 0.02)
        self.layers  = nn.ModuleList([
            QFormerLayer(d_q, d_enc, n_heads) for _ in range(depth)
        ])
        self.norm_out = nn.LayerNorm(d_q)

    def forward(self, E):
        """E: (B, N_e, D_enc)  →  Z: (B, M, D_q)"""
        B   = E.shape[0]
        Q   = self.queries.expand(B, -1, -1)
        for layer in self.layers:
            Q = layer(Q, E)
        return self.norm_out(Q)   # (B, M, D_q)


# ──────────────────────────────────────────────
# 4.  CNN Decoder
# ──────────────────────────────────────────────

class QFAEDecoder(nn.Module):
    """
    Projects Q-Former output Z back to spatial feature, then upsamples.
    Input  : (B, M, D_q)
    Output : (B, 1, H, W)  in [0,1]

    Assumes M=64 → reshape to (8,8) spatial start.
    """
    def __init__(self, M=64, d_q=256, out_ch=1, base=32):
        super().__init__()
        self.h0    = int(math.sqrt(M))     # 8
        self.proj  = nn.Linear(d_q, base*8)
        self.up = nn.Sequential(
            # 8→16
            nn.ConvTranspose2d(base*8, base*4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base*4), nn.ReLU(inplace=True),
            # 16→32
            nn.ConvTranspose2d(base*4, base*2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base*2), nn.ReLU(inplace=True),
            # 32→64
            nn.ConvTranspose2d(base*2, base,   4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base), nn.ReLU(inplace=True),
            # 64→128
            nn.ConvTranspose2d(base, base,     4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base), nn.ReLU(inplace=True),
            # 128→256
            nn.ConvTranspose2d(base, out_ch,   4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, Z, target_size=None):
        B, M, D = Z.shape
        x = self.proj(Z)                        # (B, M, base*8)
        h = self.h0
        x = x.transpose(1,2).reshape(B, -1, h, h)  # (B, C, 8, 8)
        out = self.up(x)                        # (B, 1, 256, 256)
        if target_size is not None and (out.shape[2] != target_size or out.shape[3] != target_size):
            out = torch.nn.functional.interpolate(out, size=(target_size, target_size), mode='bilinear', align_corners=False)
        return out


# ──────────────────────────────────────────────
# 5.  Perceptual Loss (frozen MAE features)
# ──────────────────────────────────────────────

class PerceptualLoss(nn.Module):
    """
    Cosine-similarity perceptual loss using features from a frozen
    shallow CNN (substitute for Masked AE features from paper).

    1 - cos_sim(F(x), F(x̂))
    """
    def __init__(self, in_ch=1):
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(8),
        )
        for p in self.feature_net.parameters():
            p.requires_grad = False  # frozen

    def forward(self, x, x_hat):
        fx    = self.feature_net(x).flatten(1)
        fx_h  = self.feature_net(x_hat).flatten(1)
        cos   = F.cosine_similarity(fx, fx_h, dim=-1)  # (B,)
        return (1 - cos).mean()


# ──────────────────────────────────────────────
# 6.  Full Q-Former Autoencoder
# ──────────────────────────────────────────────

class QFormerAE(nn.Module):
    """
    Q-Former Autoencoder for medical anomaly detection.

    Components
    ----------
    • Frozen foundation encoder (CNN or DINOv2)
    • Multi-scale projector → token sequence E
    • Q-Former bottleneck → Z
    • CNN decoder → x̂
    • Frozen perceptual loss network
    """
    def __init__(self,
                 in_ch: int  = 1,
                 base: int   = 32,
                 M: int      = 64,     # query tokens
                 d_q: int    = 256,    # query dim
                 d_enc: int  = 256,    # enc proj dim
                 qf_depth: int = 4,
                 n_heads: int  = 8,
                 lam_p: float  = 0.5):
        super().__init__()
        self.lam_p = lam_p

        # Foundation encoder (frozen)
        self.foundation, out_dims = build_foundation_encoder(in_ch, base)
        if out_dims is None:   # DINOv2 path (not yet implemented here)
            raise NotImplementedError("DINOv2 path requires special handling.")

        # Multi-scale projector
        self.projector = MultiScaleProjector(out_dims, d_enc)

        # Q-Former
        n_enc_tokens = 64 * len(out_dims)  # 64 tokens per scale
        self.qformer = QFormer(M, d_q, d_enc, qf_depth, n_heads)

        # Decoder
        self.decoder = QFAEDecoder(M, d_q, in_ch, base)

        # Perceptual loss (frozen)
        self.perceptual = PerceptualLoss(in_ch)

    def forward(self, x):
        # Step 1: frozen multi-scale features
        with torch.no_grad():
            feats = self.foundation(x)   # list of feature maps

        # Step 2: project to token sequence
        E = self.projector(feats)        # (B, N_e, D_enc)

        # Step 3: Q-Former bottleneck
        Z = self.qformer(E)              # (B, M, D_q)

        # Step 4: decode
        x_hat = self.decoder(Z, target_size=x.shape[2])  # match input H/W

        # Step 5: losses
        l_mse = F.mse_loss(x_hat, x)
        l_p   = self.perceptual(x, x_hat)
        loss  = l_mse + self.lam_p * l_p
        return x_hat, loss, l_mse, l_p

    def reconstruct(self, x):
        self.eval()
        with torch.no_grad():
            feats = self.foundation(x)
            E     = self.projector(feats)
            Z     = self.qformer(E)
            x_hat = self.decoder(Z)
        return x_hat


# ──────────────────────────────────────────────
# 7.  Anomaly score
# ──────────────────────────────────────────────

def anomaly_score(model: QFormerAE, x: torch.Tensor) -> tuple:
    """
    S(x) = MSE(x, x̂) + λ_p * (1 - cos_sim(F(x), F(x̂)))
    Returns (score, error_map, x_hat)
    """
    model.eval()
    with torch.no_grad():
        x_hat = model.reconstruct(x)
        # Bidirectional error captures both hyper-dense and hypo-dense tumors
        error_map = torch.abs(x - x_hat)
        l_mse  = error_map.pow(2).mean(dim=[1,2,3])
        l_p    = model.perceptual(x, x_hat)
        score  = l_mse + model.lam_p * l_p
    return score, error_map, x_hat


# ──────────────────────────────────────────────
if __name__ == '__main__':
    model = QFormerAE(in_ch=1, base=32, M=64, d_q=256, d_enc=256)
    x     = torch.rand(2, 1, 256, 256)
    x_hat, loss, lm, lp = model(x)
    print("x_hat:", x_hat.shape)
    print(f"loss={loss:.4f}  mse={lm:.4f}  perceptual={lp:.4f}")
    s, emap, xh = anomaly_score(model, x)
    print("Score:", s)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n:,}")
