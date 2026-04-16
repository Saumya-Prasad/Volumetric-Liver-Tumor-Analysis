# -*- coding: utf-8 -*-
"""
inference.py
Run anomaly detection on a single DICOM file or a folder of DICOM files.

NEW: Liver-focused inference modes
-----------------------------------
  --liver_mask
    Apply a binary liver mask *after* reconstruction.  The error map is
    zeroed outside the liver so the anomaly score is computed exclusively
    over liver parenchyma.  Works with **existing checkpoints** — no
    retraining required.  Recommended as an immediate fix.

  --liver_crop
    Crop the input slice to the liver bounding box before running the AE,
    then map the error map back into the full-image canvas for display.
    Matches the ``--liver_crop`` training mode in train.py.  Requires a
    model that was retrained with ``--liver_crop``.

Why this matters
----------------
Without liver isolation, the global MSE between input and reconstruction
is dominated by high-contrast structures (ribs, vertebrae, air–tissue
boundaries).  A tumour inside the liver — which appears as a dark, poorly
reconstructed region — contributes <2 % of the total error and is buried
in the per-image score.

With liver masking the score becomes::

    score = mean(|x - x̂|²  over  liver pixels only)

A dark tumour that the AE cannot reconstruct faithfully now causes a
large spike in that restricted mean, raising it above threshold.

Usage
-----
  # Single DICOM — quick fix with existing checkpoint
  python inference.py --dicom path/to/slice.dcm --model ae_flow \\
        --ckpt checkpoints/ae_flow_best.pt --liver_mask

  # With retrained liver-crop model
  python inference.py --dicom path/to/slice.dcm --model ae_flow \\
        --ckpt checkpoints/ae_flow_liver_crop_best.pt --liver_crop

  # Folder (returns per-slice scores + GIF)
  python inference.py --dicom path/to/patient_folder/ --model ccb_aae \\
        --ckpt checkpoints/ccb_aae_best.pt --liver_mask --save_dir results/

  # Batch comparison across all trained models
  python inference.py --dicom path/to/slice.dcm --compare \\
        --ckpt_dir checkpoints/ --liver_mask

Output
------
  results/original.png
  results/preprocessed.png
  results/liver_mask.png         ← NEW: binary liver ROI
  results/reconstruction.png
  results/error_map.png          ← masked to liver region
  results/overlay.png
  results/result.json     ← { "score": float, "label": "normal"|"tumor" }
"""

from __future__ import annotations

import os
import argparse
import json
import glob
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import pydicom

# ── local imports 
from dataset import preprocess_dicom
from liver_segmenter import get_liver_mask, crop_to_liver, uncrop_error_map

from models.model_1_conv_ae   import ConvAutoencoder
from models.model_1_conv_ae   import anomaly_score as score_conv
from models.model_2_ae_flow   import AEFlow
from models.model_2_ae_flow   import anomaly_score as score_aeflow
from models.model_3_masked_ae import MaskedAutoencoder, AnomalyClassifier
from models.model_3_masked_ae import anomaly_score as score_masked
from models.model_4_ccb_aae   import CCBAAE
from models.model_4_ccb_aae   import anomaly_score as score_ccbaae
from models.model_5_qformer_ae import QFormerAE
from models.model_5_qformer_ae import anomaly_score as score_qformer
from models.model_6_ensemble_ae import EnsembleAE, EnsembleScorer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# Model registry


def load_model(model_name: str, ckpt_path: str):
    """Instantiate + load weights."""
    name = model_name.lower()

    if name == 'conv_ae':
        m = ConvAutoencoder()
    elif name == 'ae_flow':
        m = AEFlow(in_ch=1, base_ch=32, K=8)
    elif name == 'masked_ae':
        mae = MaskedAutoencoder(img_size=256, patch_size=16)
        clf_name = "anomaly_classifier_liver_crop_best.pt" if "liver_crop" in ckpt_path else "anomaly_classifier_best.pt"
        clf_path = os.path.join(os.path.dirname(ckpt_path), clf_name)
        clf = AnomalyClassifier(img_size=256)
        if os.path.exists(clf_path):
            ckpt = torch.load(clf_path, map_location=DEVICE)
            clf.load_state_dict(ckpt['model'])
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        mae.load_state_dict(ckpt['model'])
        return mae.to(DEVICE), clf.to(DEVICE)
    elif name == 'ccb_aae':
        m = CCBAAE(in_ch=1, base=32)
    elif name == 'qformer':
        m = QFormerAE(in_ch=1, base=32, M=64, d_q=256)
    elif name == 'ensemble':
        m = EnsembleAE(in_ch=1, img_size=256)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    m.load_state_dict(ckpt['model'])
    return m.to(DEVICE)


def run_inference(model, model_name: str, x: torch.Tensor,
                  z_mean=None) -> tuple:
    """
    Returns (score, error_map_np, x_hat_np)
    score       : float  (higher = more anomalous; NOT yet liver-masked)
    error_map   : numpy (H, W) float32
    x_hat       : numpy (H, W) float32
    """
    name = model_name.lower()

    if name == 'conv_ae':
        s, emap, xh = score_conv(model, x)
    elif name == 'ae_flow':
        s, emap, xh = score_aeflow(model, x)
    elif name == 'masked_ae':
        mae, clf = model
        s, emap, xh = score_masked(mae, clf, x)
    elif name == 'ccb_aae':
        if z_mean is None:
            z_mean = torch.zeros(256).to(DEVICE)
        s, emap, xh = score_ccbaae(model, x, z_mean)
    elif name == 'qformer':
        s, emap, xh = score_qformer(model, x)
    elif name == 'ensemble':
        scorer = EnsembleScorer(model)
        s, emap, _ = scorer.score(x)
        # Average all members for a smooth, high-quality reconstruction
        xh_tensors = model.reconstruct_all(x)
        xh = torch.stack(xh_tensors).mean(0)
    else:
        raise ValueError(name)

    # Normalise output shapes
    emap_np = emap.squeeze().cpu().detach().numpy()
    xh_np   = xh.squeeze().cpu().detach().numpy()

    score_val = float(s[0].item() if isinstance(s, torch.Tensor) else s)
    return score_val, emap_np, xh_np


# Liver-aware score computation

def apply_liver_mask_to_score(error_map_np: np.ndarray,
                               liver_mask:  np.ndarray) -> tuple[float, np.ndarray]:
    """
    Re-score the error map using only liver pixels.

    Parameters
    ----------
    error_map_np : (H, W) float32 — raw per-pixel reconstruction error
    liver_mask   : (H, W) float32 binary {0, 1}

    Returns
    -------
    liver_score    : mean squared error over liver pixels only
    masked_emap    : error map with non-liver pixels zeroed out
                     (used for visualisation — keeps spatial structure)

    Design note
    -----------
    We compute the *mean* over liver pixels (not the sum) so the score is
    invariant to liver size across patients.  A uniformly distributed error
    gives the same score regardless of how large the liver is; a localised
    tumour region gives a score proportional to the fraction of the liver
    it occupies and the severity of the reconstruction failure.
    """
    # Ensure mask matches error map shape
    if liver_mask.shape != error_map_np.shape:
        pil  = Image.fromarray((liver_mask * 255).astype(np.uint8))
        pil  = pil.resize(error_map_np.shape[::-1], Image.NEAREST)
        liver_mask = (np.array(pil) > 0).astype(np.float32)

    masked_emap = error_map_np * liver_mask

    liver_pixels = masked_emap[liver_mask > 0]
    liver_score  = float(liver_pixels.mean()) if len(liver_pixels) > 0 \
                   else float(error_map_np.mean())

    return liver_score, masked_emap


# Threshold calibration (set per model on val data)

# Thresholds tuned on global (full-image) error:
THRESHOLDS_GLOBAL = {
    'conv_ae'  : 0.015,
    'ae_flow'  : 0.020,
    'masked_ae': 0.40,
    'ccb_aae'  : 0.018,
    'qformer'  : 0.012,
    'ensemble' : 0.015,
}

# Thresholds tuned on liver-only error (lower absolute values because we
# average over fewer, more homogeneous pixels).  Re-calibrate these after
# running the model on a validation set with known labels.
THRESHOLDS_LIVER = {
    'conv_ae'  : 0.008,
    'ae_flow'  : 0.010,
    'masked_ae': 0.35,
    'ccb_aae'  : 0.009,
    'qformer'  : 0.007,
    'ensemble' : 0.008,
}


def classify(score: float, model_name: str, liver_mode: bool = False) -> str:
    table = THRESHOLDS_LIVER if liver_mode else THRESHOLDS_GLOBAL
    thr   = table.get(model_name.lower(), 0.01 if liver_mode else 0.02)
    return 'tumor' if score > thr else 'normal'


# Visualisation helpers

def save_results(original_np:     np.ndarray,
                 preprocessed_np: np.ndarray,
                 x_hat_np:        np.ndarray,
                 error_map_np:    np.ndarray,
                 score:           float,
                 label:           str,
                 save_dir:        str,
                 model_name:      str,
                 liver_mask_np:   np.ndarray | None = None,
                 threshold:       float | None      = None):
    """
    Saves a multi-panel figure + individual images + JSON result.

    When *liver_mask_np* is provided the figure gains a 'Liver Mask' panel
    and the overlay highlights only regions inside the liver, making it
    immediately clear whether the detected anomaly is anatomically
    plausible.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Post-process error map → binary anomaly mask
    if threshold is None:
        # Use 95th percentile of the liver region if mask available,
        # otherwise global 95th percentile
        if liver_mask_np is not None and liver_mask_np.sum() > 0:
            liver_errors = error_map_np[liver_mask_np > 0]
            threshold    = float(np.percentile(liver_errors, 95))
        else:
            threshold = float(np.percentile(error_map_np, 95))

    binary_mask = (error_map_np > threshold).astype(np.uint8)

    # ── Save individual images ────────────────
    def _save(arr, name):
        img = Image.fromarray((arr * 255).astype(np.uint8))
        img.save(os.path.join(save_dir, name))

    _save(original_np,     'original.png')
    _save(preprocessed_np, 'preprocessed.png')
    _save(x_hat_np,        'reconstruction.png')

    # Save liver mask if present
    if liver_mask_np is not None:
        _save(liver_mask_np, 'liver_mask.png')

    # Heatmap
    heatmap = cm.hot(error_map_np / (error_map_np.max() + 1e-8))[:, :, :3]
    _save(heatmap, 'error_map.png')

    # Overlay — red tint on detected anomaly region within liver
    overlay_base = np.stack([preprocessed_np] * 3, axis=-1)
    overlay      = overlay_base.copy()
    overlay[binary_mask == 1, 0] = 1.0
    overlay[binary_mask == 1, 1] = 0.0
    overlay[binary_mask == 1, 2] = 0.0
    blended = overlay_base.copy()
    blended[binary_mask == 1] = (0.4 * overlay_base[binary_mask == 1] +
                                  0.6 * overlay[binary_mask == 1])
    _save(blended, 'overlay.png')

    # ── Multi-panel summary figure ────────────
    has_liver_panel = liver_mask_np is not None
    n_panels = 6 if has_liver_panel else 5
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))

    liver_info = '  |  LIVER-FOCUSED' if has_liver_panel else ''
    fig.suptitle(
        f"Model: {model_name}   Score: {score:.4f}   "
        f"Label: {label.upper()}{liver_info}",
        fontsize=13, fontweight='bold',
        color='red' if label == 'tumor' else 'green')

    panels = [
        (original_np,     'Original DICOM',       'gray'),
        (preprocessed_np, 'Preprocessed (HU)',    'gray'),
        (x_hat_np,        'AE Reconstruction',    'gray'),
        (error_map_np,    'Error Map (liver only)', 'hot'),
        (blended,         'Detected Region',       None),
    ]
    if has_liver_panel:
        # Insert liver mask panel after preprocessed
        panels.insert(2, (liver_mask_np, 'Liver Mask', 'gray'))

    for ax, (img, title, cmap) in zip(axes, panels):
        kw = dict(vmin=0, vmax=1) if cmap else {}
        ax.imshow(img, cmap=cmap, **kw)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'summary.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ── JSON result ──────────────────────────
    result = {
        'model'       : model_name,
        'score'       : round(score, 6),
        'label'       : label,
        'threshold'   : THRESHOLDS_LIVER.get(model_name.lower(), 0.01)
                        if has_liver_panel
                        else THRESHOLDS_GLOBAL.get(model_name.lower(), 0.02),
        'liver_focused': has_liver_panel,
    }
    with open(os.path.join(save_dir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=2)

    print(f"  -> Results saved to {save_dir}/")
    return result


def compare_models(x_tensor:        torch.Tensor,
                   preprocessed_np:  np.ndarray,
                   original_np:      np.ndarray,
                   ckpt_dir:         str,
                   save_dir:         str,
                   liver_mask_np:    np.ndarray | None = None):
    """Run all available models and save a comparison JSON + vote."""
    results  = {}
    model_map = {
        'conv_ae'  : 'conv_ae_best.pt',
        'ae_flow'  : 'ae_flow_best.pt',
        'ccb_aae'  : 'ccb_aae_best.pt',
        'qformer'  : 'qformer_ae_best.pt',
        'ensemble' : 'ensemble_ae_best.pt',
    }
    liver_mode = liver_mask_np is not None

    for mname, ckpt_file in model_map.items():
        ckpt_path = os.path.join(ckpt_dir, ckpt_file)
        if not os.path.exists(ckpt_path):
            print(f"  [compare] Skipping {mname} (no checkpoint)")
            continue

        model = load_model(mname, ckpt_path)
        score, emap_np, xh_np = run_inference(model, mname, x_tensor)

        if liver_mode:
            score, emap_np = apply_liver_mask_to_score(emap_np, liver_mask_np)

        label = classify(score, mname, liver_mode)
        results[mname] = {'score': score, 'label': label}
        print(f"  {mname:12s}  score={score:.5f}  label={label}"
              f"{'  [liver]' if liver_mode else ''}")

    # Majority vote
    votes = sum(1 for r in results.values() if r['label'] == 'tumor')
    final = 'tumor' if votes > len(results) // 2 else 'normal'
    results['__final_vote__'] = {'label': final, 'votes': votes}

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'comparison.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Final vote: {final.upper()} ({votes}/{len(results)-1} models)")
    return results


# Main

def main():
    parser = argparse.ArgumentParser(description='Liver Anomaly Inference')
    parser.add_argument('--dicom',      required=True,
                        help='DICOM file or folder')
    parser.add_argument('--model',      default='ae_flow',
                        choices=['conv_ae', 'ae_flow', 'masked_ae',
                                 'ccb_aae', 'qformer', 'ensemble'])
    parser.add_argument('--ckpt',       default=None,
                        help='Checkpoint path')
    parser.add_argument('--ckpt_dir',   default='./checkpoints',
                        help='Checkpoint folder for --compare mode')
    parser.add_argument('--compare',    action='store_true',
                        help='Run all models and compare')
    parser.add_argument('--save_dir',   default='./results')
    parser.add_argument('--img_size',   type=int, default=256)

    # ── Liver-focus flags ─────────────────────────────────────────────
    liver_group = parser.add_mutually_exclusive_group()
    liver_group.add_argument(
        '--liver_mask', action='store_true',
        help='[RECOMMENDED] Zero the error map outside the liver ROI and '
             'compute the anomaly score over liver pixels only. '
             'Works with any existing checkpoint — no retraining needed.')
    liver_group.add_argument(
        '--liver_crop', action='store_true',
        help='Crop input to liver bounding box before the AE '
             '(requires a model retrained with --liver_crop in train.py).')

    args = parser.parse_args()

    # ── Choose DICOM path ─────────────────────────────────────────────
    dcm_path = args.dicom
    if os.path.isdir(dcm_path):
        files    = sorted(glob.glob(
            os.path.join(dcm_path, '**', '*.dcm'), recursive=True))
        dcm_path = files[len(files) // 2]      # middle slice
        print(f"Using slice: {dcm_path}")

    # ── Raw pixel display (unchanged HU) ─────────────────────────────
    dicom      = pydicom.dcmread(dcm_path)
    raw_pixels = dicom.pixel_array.astype(np.float32)
    raw_norm   = ((raw_pixels - raw_pixels.min()) /
                  (raw_pixels.max() - raw_pixels.min() + 1e-8))
    raw_pil    = np.array(
        Image.fromarray((raw_norm * 255).astype(np.uint8)).resize(
            (args.img_size, args.img_size))) / 255.0

    # ── Preprocessed (HU windowed) ────────────────────────────────────
    preprocessed = preprocess_dicom(dcm_path, target_size=args.img_size)

    # ── Liver mask computation ─────────────────────────────────────────
    liver_mask_np = None
    if args.liver_mask or args.liver_crop:
        print("Computing liver mask…")
        liver_mask_np = get_liver_mask(dcm_path, target_size=args.img_size)
        n_liver_px    = int(liver_mask_np.sum())
        n_total_px    = args.img_size * args.img_size
        print(f"  Liver pixels: {n_liver_px} / {n_total_px} "
              f"({100 * n_liver_px / n_total_px:.1f} % of slice)")

    # ── Build model input tensor ──────────────────────────────────────
    if args.liver_crop:
        # Crop the image to liver and resize — must match training mode
        cropped_img, cropped_mask, bbox = crop_to_liver(
            preprocessed, liver_mask_np, args.img_size)
        x        = torch.from_numpy(cropped_img).unsqueeze(0).unsqueeze(0).float()
        x        = x.to(DEVICE)
        inp_display = cropped_img   # what we show in the "preprocessed" panel
    else:
        x           = torch.from_numpy(preprocessed).unsqueeze(0).unsqueeze(0).float()
        x           = x.to(DEVICE)
        inp_display = preprocessed

    # ── Compare mode ──────────────────────────────────────────────────
    if args.compare:
        compare_models(x, inp_display, raw_pil,
                       args.ckpt_dir, args.save_dir, liver_mask_np)
        return

    # ── Single-model inference ────────────────────────────────────────
    if args.ckpt is None:
        ckpt_names = {
            'conv_ae'  : 'conv_ae_best.pt',
            'ae_flow'  : 'ae_flow_best.pt',
            'masked_ae': 'masked_ae_stage1_best.pt',
            'ccb_aae'  : 'ccb_aae_best.pt',
            'qformer'  : 'qformer_ae_best.pt',
            'ensemble' : 'ensemble_ae_best.pt',
        }
        args.ckpt = os.path.join(args.ckpt_dir, ckpt_names[args.model])

    if not os.path.exists(args.ckpt):
        print(f"No checkpoint at {args.ckpt}. Running with untrained model (demo).")
        model_map_init = {
            'conv_ae'  : ConvAutoencoder(),
            'ae_flow'  : AEFlow(),
            'masked_ae': (MaskedAutoencoder(), AnomalyClassifier()),
            'ccb_aae'  : CCBAAE(),
            'qformer'  : QFormerAE(),
            'ensemble' : EnsembleAE(),
        }
        model = model_map_init[args.model]
        if isinstance(model, tuple):
            model = tuple(m.to(DEVICE) for m in model)
        else:
            model = model.to(DEVICE)
    else:
        model = load_model(args.model, args.ckpt)

    score, emap_np, xh_np = run_inference(model, args.model, x)

    # ── Apply liver mask to error map ─────────────────────────────────
    if args.liver_crop:
        # Map the cropped error map back into a full-size canvas so the
        # summary figure aligns with the original anatomy display.
        emap_full = uncrop_error_map(emap_np, bbox, full_size=args.img_size)
        xh_full   = uncrop_error_map(xh_np,   bbox, full_size=args.img_size)

        # Score over the liver crop (already liver-only by construction)
        liver_score = float(emap_np.mean())
        emap_display = emap_full
        xh_display   = xh_full

    elif args.liver_mask:
        # Zero non-liver pixels in the error map and recompute score
        liver_score, emap_display = apply_liver_mask_to_score(
            emap_np, liver_mask_np)
        xh_display = xh_np

    else:
        liver_score  = score
        emap_display = emap_np
        xh_display   = xh_np

    liver_mode = args.liver_mask or args.liver_crop
    label = classify(liver_score, args.model, liver_mode)

    mode_str = (' [liver-crop]' if args.liver_crop else
                ' [liver-mask]' if args.liver_mask else '')
    print(f"Score{mode_str}: {liver_score:.5f} -> {label.upper()}")

    save_results(raw_pil, inp_display, xh_display, emap_display,
                 liver_score, label, args.save_dir, args.model,
                 liver_mask_np=liver_mask_np if liver_mode else None)


if __name__ == '__main__':
    main()
