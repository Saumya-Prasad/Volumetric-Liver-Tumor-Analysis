# -*- coding: utf-8 -*-
"""
train.py
Unified training script for all 6 anomaly-detection models.

Usage
-----
  # Original (full-slice) training — unchanged behaviour
  python train.py --model ae_flow --epochs 50 --batch 16

  # Liver-mask mode: AE sees full slice with non-liver pixels zeroed out.
  # Fastest to try; requires no changes to model architecture.
  python train.py --model ae_flow --epochs 50 --batch 16 --liver_only

  # Liver-crop mode: AE sees only the liver bounding-box crop resized to
  # img_size × img_size.  Best results; requires full retraining.
  python train.py --model ae_flow --epochs 50 --batch 16 --liver_crop

  # All models with liver-crop (recommended after reading the paper):
  python train.py --model conv_ae   --epochs 50  --batch 16 --liver_crop
  python train.py --model ae_flow   --epochs 50  --batch 16 --liver_crop
  python train.py --model masked_ae --epochs 100 --batch 8  --liver_crop
  python train.py --model ccb_aae   --epochs 60  --batch 16 --liver_crop
  python train.py --model qformer   --epochs 40  --batch 8  --liver_crop
  python train.py --model ensemble  --epochs 50  --batch 16 --liver_crop

Why liver-crop is important
----------------------------
Without liver isolation, the MSE loss that drives every AE in this
codebase is dominated by high-contrast non-liver structures (vertebrae,
ribs, air–tissue boundary).  The liver represents ≈10-15 % of the image
area and therefore ≈10-15 % of the gradient signal.  A tumour inside the
liver — which appears as a small, poorly-reconstructed dark region —
contributes <2 % of the gradient and the model never internalises what
"normal liver texture" looks like.

With --liver_crop every training image is exclusively liver parenchyma.
The entire bottleneck is used for liver-specific features.  At inference
time, any deviation from expected texture (dark necrotic core, cystic
region, heterogeneous enhancement) produces a reconstruction error that
dominates the anomaly score because there is nothing else to compare it
against.

Checkpoint naming convention
------------------------------
  liver_crop  → checkpoints/<model>_liver_crop_best.pt
  liver_only  → checkpoints/<model>_liver_only_best.pt
  default     → checkpoints/<model>_best.pt

This lets you keep multiple trained variants side-by-side.

All models are trained on HEALTHY CT slices only (CHAOS dataset).
"""

import os
import argparse
import math
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import kagglehub

# ── local imports ────────────────────────────
from dataset import get_dataloaders
from models.model_1_conv_ae   import ConvAutoencoder
from models.model_2_ae_flow   import AEFlow, AEFlowLoss
from models.model_3_masked_ae import (MaskedAutoencoder, AnomalyClassifier,
                                       PseudoAbnormalModule)
from models.model_4_ccb_aae   import CCBAAE, CCBAAELoss
from models.model_5_qformer_ae import QFormerAE
from models.model_6_ensemble_ae import EnsembleAE

# ─────────────────────────────────────────────
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[train.py] Using device: {DEVICE}")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def save_ckpt(model, path, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    obj = {'model': model.state_dict()}
    if extra:
        obj.update(extra)
    torch.save(obj, path)
    print(f"  Saved checkpoint -> {path}")


def load_ckpt(model, path):
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt['model'])
    return ckpt


def cosine_lr(optimizer, epoch, epochs, eta_min=1e-6):
    """Inline cosine annealing update."""
    for pg in optimizer.param_groups:
        pg['lr'] = eta_min + 0.5 * (pg['initial_lr'] - eta_min) * (
            1 + math.cos(math.pi * epoch / epochs))


def ckpt_name(base: str, save_dir: str, liver_crop: bool,
              liver_only: bool) -> str:
    """
    Build a checkpoint filename that encodes the liver mode so that
    liver-crop checkpoints do not silently overwrite full-slice ones.
    """
    if liver_crop:
        suffix = '_liver_crop'
    elif liver_only:
        suffix = '_liver_only'
    else:
        suffix = ''
    return os.path.join(save_dir, f"{base}{suffix}_best.pt")


# ─────────────────────────────────────────────
# 1.  Conv-AE trainer
# ─────────────────────────────────────────────

def train_conv_ae(train_loader, val_loader, epochs, lr, save_dir,
                  liver_crop=False, liver_only=False):
    model = ConvAutoencoder().to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=lr)
    for pg in opt.param_groups:
        pg['initial_lr'] = lr
    best_val = float('inf')
    log      = []

    ckpt_path = ckpt_name('conv_ae', save_dir, liver_crop, liver_only)

    for ep in range(1, epochs + 1):
        cosine_lr(opt, ep, epochs)
        model.train()
        t_loss = 0.0
        for x in train_loader:
            x = x.to(DEVICE)
            opt.zero_grad()
            x_hat = model(x)
            loss  = nn.functional.mse_loss(x_hat, x)
            loss.backward()
            opt.step()
            t_loss += loss.item()

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(DEVICE)
                v_loss += nn.functional.mse_loss(model(x), x).item()

        t_loss /= len(train_loader)
        v_loss /= len(val_loader)
        log.append({'epoch': ep, 'train': t_loss, 'val': v_loss})
        print(f"[ConvAE] ep {ep:3d}/{epochs}  train={t_loss:.4f}  val={v_loss:.4f}")

        if v_loss < best_val:
            best_val = v_loss
            save_ckpt(model, ckpt_path)

    log_base = ckpt_path.replace('_best.pt', '_log.json')
    json.dump(log, open(log_base, 'w'))
    return model


# ─────────────────────────────────────────────
# 2.  AE-FLOW trainer
# ─────────────────────────────────────────────

def train_ae_flow(train_loader, val_loader, epochs, lr, save_dir,
                  liver_crop=False, liver_only=False):
    model   = AEFlow(in_ch=1, base_ch=32, K=8).to(DEVICE)
    loss_fn = AEFlowLoss(lam=1.0)
    opt     = optim.Adam(model.parameters(), lr=lr)
    for pg in opt.param_groups:
        pg['initial_lr'] = lr
    best_val  = float('inf')
    log       = []
    ckpt_path = ckpt_name('ae_flow', save_dir, liver_crop, liver_only)

    for ep in range(1, epochs + 1):
        cosine_lr(opt, ep, epochs)
        model.train()
        t_loss = 0.0
        for x in train_loader:
            x = x.to(DEVICE)
            opt.zero_grad()
            x_hat, z_prime, ld = model(x)
            loss, _, _ = loss_fn(x, x_hat, z_prime, ld)
            loss.backward()
            opt.step()
            t_loss += loss.item()

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(DEVICE)
                xh, zp, ld = model(x)
                v_loss += loss_fn(x, xh, zp, ld)[0].item()

        t_loss /= len(train_loader)
        v_loss /= len(val_loader)
        log.append({'epoch': ep, 'train': t_loss, 'val': v_loss})
        print(f"[AEFLOW] ep {ep:3d}/{epochs}  train={t_loss:.4f}  val={v_loss:.4f}")

        if v_loss < best_val:
            best_val = v_loss
            save_ckpt(model, ckpt_path)

    json.dump(log, open(ckpt_path.replace('_best.pt', '_log.json'), 'w'))
    return model


# ─────────────────────────────────────────────
# 3.  Masked AE trainer (2-stage)
# ─────────────────────────────────────────────

def train_masked_ae(train_loader, val_loader, epochs, lr, save_dir,
                    liver_crop=False, liver_only=False):
    # Stage 1: MAE pre-training
    mae = MaskedAutoencoder(img_size=256, patch_size=16).to(DEVICE)
    opt = optim.AdamW(mae.parameters(), lr=lr, weight_decay=0.05)
    for pg in opt.param_groups:
        pg['initial_lr'] = lr
    best_val  = float('inf')
    log       = []
    ckpt_path = ckpt_name('masked_ae_stage1', save_dir, liver_crop, liver_only)

    print("=== Stage 1: MAE pre-training ===")
    for ep in range(1, epochs + 1):
        cosine_lr(opt, ep, epochs)
        mae.train()
        t_loss = 0.0
        for x in train_loader:
            x = x.to(DEVICE)
            opt.zero_grad()
            _, _, loss = mae(x)
            loss.backward()
            opt.step()
            t_loss += loss.item()

        mae.eval()
        v_loss = 0.0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(DEVICE)
                _, _, loss = mae(x)
                v_loss += loss.item()

        t_loss /= len(train_loader)
        v_loss /= len(val_loader)
        log.append({'stage': 1, 'epoch': ep, 'train': t_loss, 'val': v_loss})
        print(f"[MAE-s1] ep {ep:3d}/{epochs}  train={t_loss:.5f}  val={v_loss:.5f}")

        if v_loss < best_val:
            best_val = v_loss
            save_ckpt(mae, ckpt_path)

    # Stage 2: Train anomaly classifier
    print("=== Stage 2: Anomaly classifier ===")
    clf_path = ckpt_name('anomaly_classifier', save_dir, liver_crop, liver_only)
    clf      = AnomalyClassifier(img_size=256).to(DEVICE)
    pa_mod   = PseudoAbnormalModule()
    opt2     = optim.Adam(clf.parameters(), lr=lr)
    mae.eval()
    cls_log  = []

    for ep in range(1, 20 + 1):
        clf.train()
        t_acc = 0.0
        n     = 0
        for x in train_loader:
            x    = x.to(DEVICE)
            x_pa = pa_mod(x).to(DEVICE)
            opt2.zero_grad()

            with torch.no_grad():
                xh_normal = mae.reconstruct(x)
                xh_pa     = mae.reconstruct(x_pa)

            r_neg = (x    - xh_normal).abs()
            r_pos = (x_pa - xh_pa).abs()
            r     = torch.cat([r_neg, r_pos])
            y     = torch.cat([torch.zeros(x.shape[0]),
                                torch.ones(x_pa.shape[0])]).to(DEVICE)

            pred = clf(r)
            loss = nn.functional.binary_cross_entropy(pred, y)
            loss.backward()
            opt2.step()

            t_acc += ((pred > 0.5).float() == y).float().mean().item()
            n     += 1

        print(f"[MAE-s2] ep {ep:3d}/20  acc={t_acc/n:.3f}")
        cls_log.append({'epoch': ep, 'acc': t_acc / n})

    save_ckpt(clf, clf_path)
    log.extend(cls_log)
    json.dump(log, open(ckpt_path.replace('_best.pt', '_log.json'), 'w'))
    return mae, clf


# ─────────────────────────────────────────────
# 4.  CCB-AAE trainer
# ─────────────────────────────────────────────

def train_ccb_aae(train_loader, val_loader, epochs, lr, save_dir,
                  liver_crop=False, liver_only=False):
    model   = CCBAAE(in_ch=1, base=32).to(DEVICE)
    loss_fn = CCBAAELoss(lam1=0.1, lam2=0.1)
    opt_G   = optim.Adam(model.generator_params(),     lr=lr, betas=(0.5, 0.999))
    opt_D   = optim.Adam(model.discriminator_params(), lr=lr, betas=(0.5, 0.999))
    best_val  = float('inf')
    log       = []
    ckpt_path = ckpt_name('ccb_aae', save_dir, liver_crop, liver_only)

    for ep in range(1, epochs + 1):
        model.train()
        t_lg = t_ld = 0.0
        for x in train_loader:
            x = x.to(DEVICE)

            opt_D.zero_grad()
            x_hat, _  = model(x)
            d_real     = model.discriminator(x)
            d_fake     = model.discriminator(x_hat.detach())
            ld         = loss_fn.discriminator_loss(d_real, d_fake)
            ld.backward()
            opt_D.step()

            opt_G.zero_grad()
            x_hat, z4  = model(x)
            d_fake2    = model.discriminator(x_hat)
            lg, _, _, _ = loss_fn.generator_loss(x, x_hat, z4, d_fake2)
            lg.backward()
            opt_G.step()

            t_lg += lg.item()
            t_ld += ld.item()

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(DEVICE)
                x_hat, _ = model(x)
                v_loss += nn.functional.mse_loss(x_hat, x).item()

        t_lg /= len(train_loader)
        t_ld /= len(train_loader)
        v_loss /= len(val_loader)
        log.append({'epoch': ep, 'g_loss': t_lg, 'd_loss': t_ld, 'val': v_loss})
        print(f"[CCBAAE] ep {ep:3d}/{epochs}  G={t_lg:.4f}  D={t_ld:.4f}  val={v_loss:.4f}")

        if v_loss < best_val:
            best_val = v_loss
            save_ckpt(model, ckpt_path)

    json.dump(log, open(ckpt_path.replace('_best.pt', '_log.json'), 'w'))
    return model


# ─────────────────────────────────────────────
# 5.  Q-Former AE trainer
# ─────────────────────────────────────────────

def train_qformer(train_loader, val_loader, epochs, lr, save_dir,
                  liver_crop=False, liver_only=False):
    model = QFormerAE(in_ch=1, base=32, M=64, d_q=256).to(DEVICE)
    opt   = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01)
    for pg in opt.param_groups:
        pg['initial_lr'] = lr
    best_val  = float('inf')
    log       = []
    ckpt_path = ckpt_name('qformer_ae', save_dir, liver_crop, liver_only)

    for ep in range(1, epochs + 1):
        cosine_lr(opt, ep, epochs)
        model.train()
        t_loss = 0.0
        for x in train_loader:
            x = x.to(DEVICE)
            opt.zero_grad()
            _, loss, _, _ = model(x)
            loss.backward()
            opt.step()
            t_loss += loss.item()

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(DEVICE)
                _, loss, _, _ = model(x)
                v_loss += loss.item()

        t_loss /= len(train_loader)
        v_loss /= len(val_loader)
        log.append({'epoch': ep, 'train': t_loss, 'val': v_loss})
        print(f"[QFmrAE] ep {ep:3d}/{epochs}  train={t_loss:.4f}  val={v_loss:.4f}")

        if v_loss < best_val:
            best_val = v_loss
            save_ckpt(model, ckpt_path)

    json.dump(log, open(ckpt_path.replace('_best.pt', '_log.json'), 'w'))
    return model


# ─────────────────────────────────────────────
# 6.  Ensemble AE trainer
# ─────────────────────────────────────────────

def train_ensemble(train_loader, val_loader, epochs, lr, save_dir,
                   liver_crop=False, liver_only=False):
    model = EnsembleAE(in_ch=1, img_size=256).to(DEVICE)
    opts  = [optim.Adam(m.parameters(), lr=lr) for m in model.members]
    best_val  = {n: float('inf') for n in model.member_names}
    log       = []
    ckpt_path = ckpt_name('ensemble_ae', save_dir, liver_crop, liver_only)

    for ep in range(1, epochs + 1):
        model.train()
        t_losses = {n: 0.0 for n in model.member_names}
        for x in train_loader:
            x    = x.to(DEVICE)
            outs = model(x)
            for i, (name, (xh, loss)) in enumerate(outs.items()):
                opts[i].zero_grad()
                loss.backward()
                opts[i].step()
                t_losses[name] += loss.item()

        model.eval()
        v_losses = {n: 0.0 for n in model.member_names}
        with torch.no_grad():
            for x in val_loader:
                x    = x.to(DEVICE)
                outs = model(x)
                for name, (xh, loss) in outs.items():
                    v_losses[name] += loss.item()

        for n in model.member_names:
            t_losses[n] /= len(train_loader)
            v_losses[n] /= len(val_loader)
            if v_losses[n] < best_val[n]:
                best_val[n] = v_losses[n]
                save_ckpt(model, ckpt_path)

        info = '  '.join(f"{n}={v_losses[n]:.4f}" for n in model.member_names)
        print(f"[Ensemb] ep {ep:3d}/{epochs}  {info}")
        log.append({'epoch': ep,
                    **{f'val_{n}': v_losses[n] for n in model.member_names}})

    json.dump(log, open(ckpt_path.replace('_best.pt', '_log.json'), 'w'))
    return model


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

TRAINERS = {
    'conv_ae'  : train_conv_ae,
    'ae_flow'  : train_ae_flow,
    'masked_ae': train_masked_ae,
    'ccb_aae'  : train_ccb_aae,
    'qformer'  : train_qformer,
    'ensemble' : train_ensemble,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',    type=str, default='conv_ae',
                        choices=list(TRAINERS.keys()))
    parser.add_argument('--epochs',   type=int,   default=50)
    parser.add_argument('--batch',    type=int,   default=16)
    parser.add_argument('--lr',       type=float, default=1e-3)
    parser.add_argument('--img_size', type=int,   default=256)
    parser.add_argument('--save_dir', type=str,   default='./checkpoints')
    parser.add_argument('--workers',  type=int,   default=4)
    parser.add_argument('--data_path', type=str,  default=None,
                        help='Override dataset path (skip kagglehub download)')

    # ── Liver-focus flags ──────────────────────────────────────────────
    liver_group = parser.add_mutually_exclusive_group()
    liver_group.add_argument(
        '--liver_only', action='store_true',
        help='Zero non-liver pixels before feeding slices to the AE. '
             'Slices without a GT liver mask are excluded from training. '
             'Quick option — no architecture change required.')
    liver_group.add_argument(
        '--liver_crop', action='store_true',
        help='[RECOMMENDED] Crop each slice to its liver bounding box and '
             'resize to img_size. The AE learns only liver texture. '
             'Requires full retraining but produces the best anomaly scores.')

    args = parser.parse_args()

    # Print liver mode clearly so logs are unambiguous
    if args.liver_crop:
        print("[train.py] Mode: LIVER CROP — training exclusively on liver ROI")
    elif args.liver_only:
        print("[train.py] Mode: LIVER ONLY — non-liver pixels zeroed out")
    else:
        print("[train.py] Mode: FULL SLICE (standard)")

    # ── Dataset ────────────────────────────────────────────────────────
    if args.data_path:
        root = args.data_path
    else:
        print("Downloading CHAOS dataset from Kaggle…")
        root = kagglehub.dataset_download(
            "omarxadel/chaos-combined-ct-mr-healthy-abdominal-organ")
    print(f"Dataset path: {root}")

    train_loader, val_loader, _ = get_dataloaders(
        root,
        target_size=args.img_size,
        batch_size=args.batch,
        num_workers=args.workers,
        liver_only=args.liver_only,
        liver_crop=args.liver_crop,
    )

    # ── Train ──────────────────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    trainer = TRAINERS[args.model]

    t0 = time.time()
    trainer(train_loader, val_loader, args.epochs, args.lr, args.save_dir,
            liver_crop=args.liver_crop, liver_only=args.liver_only)

    elapsed = time.time() - t0
    print(f"\n[DONE] Training complete in {elapsed/60:.1f} min -> {args.save_dir}/")


if __name__ == '__main__':
    main()
