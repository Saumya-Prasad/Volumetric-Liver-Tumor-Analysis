# -*- coding: utf-8 -*-
"""
inference.py
Run anomaly detection on a single DICOM file or a folder of DICOM files.

Usage
-----
  # Single DICOM
  python inference.py --dicom path/to/slice.dcm --model ae_flow \
        --ckpt checkpoints/ae_flow_best.pt

  # Folder (returns per-slice scores + GIF)
  python inference.py --dicom path/to/patient_folder/ --model ccb_aae \
        --ckpt checkpoints/ccb_aae_best.pt --save_dir results/

  # Batch comparison across all trained models
  python inference.py --dicom path/to/slice.dcm --compare \
        --ckpt_dir checkpoints/

Output
------
  results/original.png
  results/preprocessed.png
  results/reconstruction.png
  results/error_map.png
  results/overlay.png
  results/result.json     ← { "score": float, "label": "normal"|"tumor" }
"""

import os
import argparse
import json
import glob
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import pydicom
import imageio

# ── local imports 
from dataset import preprocess_dicom
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
        clf_path = ckpt_path.replace('stage1', 'classifier').replace(
            'masked_ae_best', 'anomaly_classifier_best')
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
    Returns (score, error_map, x_hat, label)
    score     : float, higher = more anomalous
    error_map : (1,1,H,W) numpy
    x_hat     : (1,1,H,W) numpy
    label     : 'normal' | 'tumor'
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
        xh = model.reconstruct_all(x)[0]
    else:
        raise ValueError(name)

    score_val = float(s[0].item() if isinstance(s, torch.Tensor) else s)
    return score_val, emap, xh


# Threshold calibration (set per model on val data)

THRESHOLDS = {
    'conv_ae'  : 0.015,
    'ae_flow'  : 0.020,
    'masked_ae': 0.40,    # probability threshold
    'ccb_aae'  : 0.018,
    'qformer'  : 0.012,
    'ensemble' : 0.015,
}


def classify(score: float, model_name: str) -> str:
    thr = THRESHOLDS.get(model_name.lower(), 0.02)
    return 'tumor' if score > thr else 'normal'


# Visualisation helpers

def save_results(original_np: np.ndarray,
                 preprocessed_np: np.ndarray,
                 x_hat_np: np.ndarray,
                 error_map_np: np.ndarray,
                 score: float,
                 label: str,
                 save_dir: str,
                 model_name: str,
                 threshold: float = None):
    """
    Saves a 4-panel figure + individual images + JSON result.

    Parameters (all 2-D arrays in [0,1])
    """
    os.makedirs(save_dir, exist_ok=True)

    # Post-process error map → binary mask
    if threshold is None:
        threshold = float(np.percentile(error_map_np, 95))
    binary_mask = (error_map_np > threshold).astype(np.uint8)

    # ── Save individual images
    def _save(arr, name):
        img = Image.fromarray((arr * 255).astype(np.uint8))
        img.save(os.path.join(save_dir, name))

    _save(original_np,    'original.png')
    _save(preprocessed_np,'preprocessed.png')
    _save(x_hat_np,       'reconstruction.png')

    # Heatmap
    heatmap = cm.hot(error_map_np / (error_map_np.max() + 1e-8))[:, :, :3]
    _save(heatmap,        'error_map.png')

    # Overlay
    overlay_base = np.stack([preprocessed_np]*3, axis=-1)
    overlay      = overlay_base.copy()
    overlay[binary_mask == 1, 0] = 1.0   # red channel
    overlay[binary_mask == 1, 1] = 0.0
    overlay[binary_mask == 1, 2] = 0.0
    alpha_blend  = 0.5
    blended = (1 - alpha_blend) * overlay_base + alpha_blend * overlay
    blended[binary_mask == 1] = overlay[binary_mask == 1]
    _save(blended, 'overlay.png')

    # ── 4-panel summary figure ───────────────
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle(f"Model: {model_name}   Score: {score:.4f}   Label: {label.upper()}",
                 fontsize=14, fontweight='bold',
                 color='red' if label == 'tumor' else 'green')

    panels = [
        (original_np,     'Original DICOM',       'gray'),
        (preprocessed_np, 'Preprocessed (HU)',    'gray'),
        (x_hat_np,        'AE Reconstruction',    'gray'),
        (error_map_np,    'Error Map',             'hot'),
        (blended,         'Detected Region',       None),
    ]
    for ax, (img, title, cmap) in zip(axes, panels):
        if cmap:
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        else:
            ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'summary.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ── JSON result ──────────────────────────
    result = {
        'model'     : model_name,
        'score'     : round(score, 6),
        'label'     : label,
        'threshold' : THRESHOLDS.get(model_name.lower(), 0.02),
    }
    with open(os.path.join(save_dir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  → Results saved to {save_dir}/")
    return result


def compare_models(x_tensor: torch.Tensor,
                   preprocessed_np: np.ndarray,
                   original_np: np.ndarray,
                   ckpt_dir: str,
                   save_dir: str):
    """Run all available models and save a comparison figure."""
    results = {}
    model_map = {
        'conv_ae'  : 'conv_ae_best.pt',
        'ae_flow'  : 'ae_flow_best.pt',
        'ccb_aae'  : 'ccb_aae_best.pt',
        'qformer'  : 'qformer_ae_best.pt',
        'ensemble' : 'ensemble_ae_best.pt',
    }

    for mname, ckpt_file in model_map.items():
        ckpt_path = os.path.join(ckpt_dir, ckpt_file)
        if not os.path.exists(ckpt_path):
            print(f"  [compare] Skipping {mname} (no checkpoint)")
            continue
        model = load_model(mname, ckpt_path)
        score, emap, xh = run_inference(model, mname, x_tensor)
        label = classify(score, mname)
        results[mname] = {'score': score, 'label': label}
        print(f"  {mname:12s}  score={score:.5f}  label={label}")

    # Vote ensemble
    votes   = sum(1 for r in results.values() if r['label'] == 'tumor')
    final   = 'tumor' if votes > len(results) // 2 else 'normal'
    results['__final_vote__'] = {'label': final, 'votes': votes}

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'comparison.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Final vote: {final.upper()} ({votes}/{len(results)-1} models)")
    return results


# Main

def main():
    parser = argparse.ArgumentParser(description='Liver Anomaly Inference')
    parser.add_argument('--dicom',    required=True, help='DICOM file or folder')
    parser.add_argument('--model',    default='ae_flow',
                        choices=['conv_ae','ae_flow','masked_ae',
                                 'ccb_aae','qformer','ensemble'])
    parser.add_argument('--ckpt',     default=None, help='Checkpoint path')
    parser.add_argument('--ckpt_dir', default='./checkpoints',
                        help='Checkpoint folder for --compare mode')
    parser.add_argument('--compare',  action='store_true',
                        help='Run all models and compare')
    parser.add_argument('--save_dir', default='./results')
    parser.add_argument('--img_size', type=int, default=256)
    args = parser.parse_args()

    # ── Load DICOM ────────────────────────────
    dcm_path = args.dicom
    if os.path.isdir(dcm_path):
        # Use middle slice
        files   = sorted(glob.glob(os.path.join(dcm_path, '**', '*.dcm'), recursive=True))
        dcm_path = files[len(files) // 2]
        print(f"Using slice: {dcm_path}")

    # Raw pixel (for display)
    dicom       = pydicom.dcmread(dcm_path)
    raw_pixels  = dicom.pixel_array.astype(np.float32)
    raw_norm    = (raw_pixels - raw_pixels.min()) / ((raw_pixels.max() - raw_pixels.min()) + 1e-8)
    raw_pil     = np.array(Image.fromarray(
        (raw_norm * 255).astype(np.uint8)).resize(
        (args.img_size, args.img_size))) / 255.0

    # Preprocessed
    preprocessed = preprocess_dicom(dcm_path, target_size=args.img_size)

    # Tensor
    x = torch.from_numpy(preprocessed).unsqueeze(0).unsqueeze(0).float()
    x = x.to(DEVICE)

    if args.compare:
        compare_models(x, preprocessed, raw_pil, args.ckpt_dir, args.save_dir)
        return

    # Single model
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
        model_map = {
            'conv_ae'  : ConvAutoencoder(),
            'ae_flow'  : AEFlow(),
            'masked_ae': (MaskedAutoencoder(), AnomalyClassifier()),
            'ccb_aae'  : CCBAAE(),
            'qformer'  : QFormerAE(),
            'ensemble' : EnsembleAE(),
        }
        model = model_map[args.model]
        if isinstance(model, tuple):
            model = tuple(m.to(DEVICE) for m in model)
        else:
            model = model.to(DEVICE)
    else:
        model = load_model(args.model, args.ckpt)

    score, emap, xh = run_inference(model, args.model, x)
    label           = classify(score, args.model)
    print(f"Score: {score:.5f}  →  {label.upper()}")

    # Squeeze to numpy
    emap_np = emap.squeeze().cpu().detach().numpy()
    xh_np   = xh.squeeze().cpu().detach().numpy()

    save_results(raw_pil, preprocessed, xh_np, emap_np,
                 score, label, args.save_dir, args.model)


if __name__ == '__main__':
    main()
