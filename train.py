# -*- coding: utf-8 -*-
"""
train.py
Unified training script for all 6 anomaly-detection models.

Usage
-----
  python train.py --model ae_flow   --epochs 50 --batch 16
  python train.py --model conv_ae   --epochs 50
  python train.py --model masked_ae --epochs 100
  python train.py --model ccb_aae   --epochs 60
  python train.py --model qformer   --epochs 40
  python train.py --model ensemble  --epochs 50

All models are trained on HEALTHY CT slices only (CHAOS dataset).
Checkpoints and training logs are saved to ./checkpoints/.
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
from models.model_1_conv_ae  import ConvAutoencoder
from models.model_2_ae_flow  import AEFlow, AEFlowLoss
from models.model_3_masked_ae import MaskedAutoencoder, AnomalyClassifier, PseudoAbnormalModule
from models.model_4_ccb_aae  import CCBAAE, CCBAAELoss
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
    print(f"  Saved checkpoint → {path}")


def load_ckpt(model, path):
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt['model'])
    return ckpt


def cosine_lr(optimizer, epoch, epochs, eta_min=1e-6):
    """Inline cosine annealing update."""
    for pg in optimizer.param_groups:
        pg['lr'] = eta_min + 0.5 * (pg['initial_lr'] - eta_min) * (
            1 + math.cos(math.pi * epoch / epochs))


# ─────────────────────────────────────────────
# 1.  Conv-AE trainer
# ─────────────────────────────────────────────

def train_conv_ae(train_loader, val_loader, epochs, lr, save_dir):
    model = ConvAutoencoder().to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=lr)
    for pg in opt.param_groups:
        pg['initial_lr'] = lr
    best_val = float('inf')
    log = []

    for ep in range(1, epochs + 1):
        cosine_lr(opt, ep, epochs)
        model.train()
        t_loss = 0.0
        for x in train_loader:
            x = x.to(DEVICE)
            opt.zero_grad()
            x_hat = model(x)
            loss  = nn.functional.mse_loss(x_hat, x)
            loss.backward(); opt.step()
            t_loss += loss.item()

        # Validation
        model.eval(); v_loss = 0.0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(DEVICE)
                v_loss += nn.functional.mse_loss(model(x), x).item()

        t_loss /= len(train_loader); v_loss /= len(val_loader)
        log.append({'epoch': ep, 'train': t_loss, 'val': v_loss})
        print(f"[ConvAE] ep {ep:3d}/{epochs}  train={t_loss:.4f}  val={v_loss:.4f}")

        if v_loss < best_val:
            best_val = v_loss
            save_ckpt(model, f"{save_dir}/conv_ae_best.pt")

    json.dump(log, open(f"{save_dir}/conv_ae_log.json", 'w'))
    return model


# ─────────────────────────────────────────────
# 2.  AE-FLOW trainer
# ─────────────────────────────────────────────

def train_ae_flow(train_loader, val_loader, epochs, lr, save_dir):
    model   = AEFlow(in_ch=1, base_ch=32, K=8).to(DEVICE)
    loss_fn = AEFlowLoss(lam=1.0)
    opt     = optim.Adam(model.parameters(), lr=lr)
    for pg in opt.param_groups:
        pg['initial_lr'] = lr
    best_val = float('inf'); log = []

    for ep in range(1, epochs + 1):
        cosine_lr(opt, ep, epochs)
        model.train(); t_loss = 0.0
        for x in train_loader:
            x = x.to(DEVICE)
            opt.zero_grad()
            x_hat, z_prime, ld = model(x)
            loss, lr_, lf = loss_fn(x, x_hat, z_prime, ld)
            loss.backward(); opt.step()
            t_loss += loss.item()

        model.eval(); v_loss = 0.0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(DEVICE)
                xh, zp, ld = model(x)
                v_loss += loss_fn(x, xh, zp, ld)[0].item()

        t_loss /= len(train_loader); v_loss /= len(val_loader)
        log.append({'epoch': ep, 'train': t_loss, 'val': v_loss})
        print(f"[AEFLOW] ep {ep:3d}/{epochs}  train={t_loss:.4f}  val={v_loss:.4f}")

        if v_loss < best_val:
            best_val = v_loss
            save_ckpt(model, f"{save_dir}/ae_flow_best.pt")

    json.dump(log, open(f"{save_dir}/ae_flow_log.json", 'w'))
    return model


# ─────────────────────────────────────────────
# 3.  Masked AE trainer (2-stage)
# ─────────────────────────────────────────────

def train_masked_ae(train_loader, val_loader, epochs, lr, save_dir):
    # Stage 1: MAE pre-training
    mae = MaskedAutoencoder(img_size=256, patch_size=16).to(DEVICE)
    opt = optim.AdamW(mae.parameters(), lr=lr, weight_decay=0.05)
    for pg in opt.param_groups:
        pg['initial_lr'] = lr
    best_val = float('inf'); log = []

    print("=== Stage 1: MAE pre-training ===")
    for ep in range(1, epochs + 1):
        cosine_lr(opt, ep, epochs)
        mae.train(); t_loss = 0.0
        for x in train_loader:
            x = x.to(DEVICE)
            opt.zero_grad()
            _, _, loss = mae(x)
            loss.backward(); opt.step()
            t_loss += loss.item()

        mae.eval(); v_loss = 0.0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(DEVICE)
                _, _, loss = mae(x)
                v_loss += loss.item()

        t_loss /= len(train_loader); v_loss /= len(val_loader)
        log.append({'stage': 1, 'epoch': ep, 'train': t_loss, 'val': v_loss})
        print(f"[MAE-s1] ep {ep:3d}/{epochs}  train={t_loss:.5f}  val={v_loss:.5f}")
        if v_loss < best_val:
            best_val = v_loss
            save_ckpt(mae, f"{save_dir}/masked_ae_stage1_best.pt")

    # Stage 2: Train anomaly classifier
    print("=== Stage 2: Anomaly classifier ===")
    clf    = AnomalyClassifier(img_size=256).to(DEVICE)
    pa_mod = PseudoAbnormalModule()
    opt2   = optim.Adam(clf.parameters(), lr=lr)
    mae.eval()
    cls_log = []

    for ep in range(1, 20 + 1):
        clf.train(); t_acc = 0.0; n = 0
        for x in train_loader:
            x    = x.to(DEVICE)
            x_pa = pa_mod(x).to(DEVICE)
            opt2.zero_grad()

            with torch.no_grad():
                xh_normal = mae.reconstruct(x)
                xh_pa     = mae.reconstruct(x_pa)

            r_neg = (x    - xh_normal).abs()   # normal → low residual
            r_pos = (x_pa - xh_pa).abs()        # pseudo-abnormal → high residual

            r   = torch.cat([r_neg, r_pos])
            y   = torch.cat([torch.zeros(x.shape[0]),
                              torch.ones(x_pa.shape[0])]).to(DEVICE)

            pred = clf(r)
            loss = nn.functional.binary_cross_entropy(pred, y)
            loss.backward(); opt2.step()

            t_acc += ((pred > 0.5).float() == y).float().mean().item()
            n += 1

        print(f"[MAE-s2] ep {ep:3d}/20  acc={t_acc/n:.3f}")
        cls_log.append({'epoch': ep, 'acc': t_acc / n})

    save_ckpt(clf, f"{save_dir}/anomaly_classifier_best.pt")
    log.extend(cls_log)
    json.dump(log, open(f"{save_dir}/masked_ae_log.json", 'w'))
    return mae, clf


# ─────────────────────────────────────────────
# 4.  CCB-AAE trainer
# ─────────────────────────────────────────────

def train_ccb_aae(train_loader, val_loader, epochs, lr, save_dir):
    model   = CCBAAE(in_ch=1, base=32).to(DEVICE)
    loss_fn = CCBAAELoss(lam1=0.1, lam2=0.1)
    opt_G   = optim.Adam(model.generator_params(),     lr=lr, betas=(0.5, 0.999))
    opt_D   = optim.Adam(model.discriminator_params(), lr=lr, betas=(0.5, 0.999))
    best_val = float('inf'); log = []

    for ep in range(1, epochs + 1):
        model.train(); t_lg = t_ld = 0.0
        for x in train_loader:
            x = x.to(DEVICE)

            # ── Discriminator step ──────────────────
            opt_D.zero_grad()
            x_hat, _  = model(x)
            d_real     = model.discriminator(x)
            d_fake     = model.discriminator(x_hat.detach())
            ld         = loss_fn.discriminator_loss(d_real, d_fake)
            ld.backward(); opt_D.step()

            # ── Generator step ──────────────────────
            opt_G.zero_grad()
            x_hat, z4  = model(x)
            d_fake2    = model.discriminator(x_hat)
            lg, lr_, la, ll = loss_fn.generator_loss(x, x_hat, z4, d_fake2)
            lg.backward(); opt_G.step()

            t_lg += lg.item(); t_ld += ld.item()

        model.eval(); v_loss = 0.0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(DEVICE)
                x_hat, _ = model(x)
                v_loss += nn.functional.mse_loss(x_hat, x).item()

        t_lg /= len(train_loader); t_ld /= len(train_loader)
        v_loss /= len(val_loader)
        log.append({'epoch': ep, 'g_loss': t_lg, 'd_loss': t_ld, 'val': v_loss})
        print(f"[CCBAAE] ep {ep:3d}/{epochs}  G={t_lg:.4f}  D={t_ld:.4f}  val={v_loss:.4f}")

        if v_loss < best_val:
            best_val = v_loss
            save_ckpt(model, f"{save_dir}/ccb_aae_best.pt")

    json.dump(log, open(f"{save_dir}/ccb_aae_log.json", 'w'))
    return model


# ─────────────────────────────────────────────
# 5.  Q-Former AE trainer
# ─────────────────────────────────────────────

def train_qformer(train_loader, val_loader, epochs, lr, save_dir):
    model = QFormerAE(in_ch=1, base=32, M=64, d_q=256).to(DEVICE)
    # Only train non-frozen parameters
    opt   = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01)
    for pg in opt.param_groups:
        pg['initial_lr'] = lr
    best_val = float('inf'); log = []

    for ep in range(1, epochs + 1):
        cosine_lr(opt, ep, epochs)
        model.train(); t_loss = 0.0
        for x in train_loader:
            x = x.to(DEVICE)
            opt.zero_grad()
            _, loss, _, _ = model(x)
            loss.backward(); opt.step()
            t_loss += loss.item()

        model.eval(); v_loss = 0.0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(DEVICE)
                _, loss, _, _ = model(x)
                v_loss += loss.item()

        t_loss /= len(train_loader); v_loss /= len(val_loader)
        log.append({'epoch': ep, 'train': t_loss, 'val': v_loss})
        print(f"[QFmrAE] ep {ep:3d}/{epochs}  train={t_loss:.4f}  val={v_loss:.4f}")

        if v_loss < best_val:
            best_val = v_loss
            save_ckpt(model, f"{save_dir}/qformer_ae_best.pt")

    json.dump(log, open(f"{save_dir}/qformer_ae_log.json", 'w'))
    return model


# ─────────────────────────────────────────────
# 6.  Ensemble AE trainer
# ─────────────────────────────────────────────

def train_ensemble(train_loader, val_loader, epochs, lr, save_dir):
    model = EnsembleAE(in_ch=1, img_size=256).to(DEVICE)
    # One optimiser per member
    opts  = [optim.Adam(m.parameters(), lr=lr)
             for m in model.members]
    best_val = {n: float('inf') for n in model.member_names}
    log = []

    for ep in range(1, epochs + 1):
        model.train(); t_losses = {n: 0.0 for n in model.member_names}
        for x in train_loader:
            x    = x.to(DEVICE)
            outs = model(x)
            for i, (name, (xh, loss)) in enumerate(outs.items()):
                opts[i].zero_grad()
                loss.backward()
                opts[i].step()
                t_losses[name] += loss.item()

        # Validation
        model.eval(); v_losses = {n: 0.0 for n in model.member_names}
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
                save_ckpt(model, f"{save_dir}/ensemble_ae_best.pt")

        info = '  '.join(f"{n}={v_losses[n]:.4f}" for n in model.member_names)
        print(f"[Ensemb] ep {ep:3d}/{epochs}  {info}")
        log.append({'epoch': ep, **{f'val_{n}': v_losses[n]
                                     for n in model.member_names}})

    json.dump(log, open(f"{save_dir}/ensemble_ae_log.json", 'w'))
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
    parser.add_argument('--model',      type=str, default='conv_ae',
                        choices=list(TRAINERS.keys()))
    parser.add_argument('--epochs',     type=int, default=50)
    parser.add_argument('--batch',      type=int, default=16)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--img_size',   type=int, default=256)
    parser.add_argument('--save_dir',   type=str, default='./checkpoints')
    parser.add_argument('--workers',    type=int, default=4)
    parser.add_argument('--data_path',  type=str, default=None,
                        help='Override dataset path (skip kagglehub download)')
    args = parser.parse_args()

    # ── Dataset ──────────────────────────────────────
    if args.data_path:
        root = args.data_path
    else:
        print("Downloading CHAOS dataset from Kaggle…")
        root = kagglehub.dataset_download(
            "omarxadel/chaos-combined-ct-mr-healthy-abdominal-organ")
    print(f"Dataset path: {root}")

    train_loader, val_loader, test_loader = get_dataloaders(
        root, target_size=args.img_size,
        batch_size=args.batch, num_workers=args.workers)

    # ── Train ────────────────────────────────────────
    trainer = TRAINERS[args.model]
    os.makedirs(args.save_dir, exist_ok=True)

    t0 = time.time()
    if args.model == 'masked_ae':
        trainer(train_loader, val_loader, args.epochs, args.lr, args.save_dir)
    else:
        trainer(train_loader, val_loader, args.epochs, args.lr, args.save_dir)

    elapsed = time.time() - t0
    print(f"\n✓ Training complete in {elapsed/60:.1f} min → {args.save_dir}/")


if __name__ == '__main__':
    main()
