# -*- coding: utf-8 -*-
"""
debug_inference.py
==================
Step-by-step diagnostic for the liver anomaly pipeline.
Run this FIRST to find out exactly which stage is failing.

Usage
-----
  python debug_inference.py --dicom path/to/slice.dcm

Outputs (all saved to ./debug_output/)
---------------------------------------
  01_raw_dicom.png          -- raw pixel values before any processing
  02_hu_image.png           -- after HU windowing [-100, 200]
  03_liver_mask.png         -- what liver_segmenter actually returned
  04_masked_image.png       -- image * mask (what the AE actually sees)
  05_mask_coverage.txt      -- % of image that is "liver"
  06_error_map_raw.png      -- raw error map from AE (before any masking)
  07_error_map_masked.png   -- error map * liver mask
  08_error_histogram.png    -- distribution of errors (liver vs background)
  09_threshold_analysis.txt -- where the 95th/98th pct thresholds land

Read 05 and 08 first. If mask_coverage > 90% the mask has failed.
If 08 shows two clearly separated distributions you have a chance.
"""

from __future__ import annotations
import os
import argparse
import numpy as np
import pydicom
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

from dataset import preprocess_dicom
from liver_segmenter import get_liver_mask, hu_liver_mask, load_chaos_mask

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT    = './debug_output'
os.makedirs(OUT, exist_ok=True)


def save_img(arr: np.ndarray, name: str, cmap='gray'):
    arr = np.clip(arr, 0, 1)
    if cmap == 'hot':
        rgb = cm.hot(arr)[:, :, :3]
        Image.fromarray((rgb * 255).astype(np.uint8)).save(os.path.join(OUT, name))
    else:
        Image.fromarray((arr * 255).astype(np.uint8)).save(os.path.join(OUT, name))
    print(f"  Saved: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dicom',    required=True)
    parser.add_argument('--ckpt',     default='./checkpoints/ae_flow_best.pt')
    parser.add_argument('--model',    default='ae_flow',
                        choices=['conv_ae','ae_flow','ccb_aae','qformer','ensemble'])
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--seg_ckpt', default='./checkpoints/liver_seg_best.pt')
    args = parser.parse_args()

    dcm_path = args.dicom
    SZ       = args.img_size

    print("\n" + "="*60)
    print("STEP 1: Raw DICOM")
    print("="*60)
    dcm     = pydicom.dcmread(dcm_path)
    raw     = dcm.pixel_array.astype(np.float32)
    slope   = float(getattr(dcm, 'RescaleSlope',     1))
    inter   = float(getattr(dcm, 'RescaleIntercept', 0))
    hu      = raw * slope + inter
    print(f"  PixelArray shape : {raw.shape}")
    print(f"  HU range         : [{hu.min():.0f}, {hu.max():.0f}]")
    print(f"  HU mean          : {hu.mean():.1f}")

    raw_norm = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
    raw_pil  = np.array(
        Image.fromarray((raw_norm*255).astype(np.uint8)).resize((SZ,SZ))) / 255.0
    save_img(raw_pil, '01_raw_dicom.png')

    print("\n" + "="*60)
    print("STEP 2: HU Windowed Image [-100, 200 HU]")
    print("="*60)
    preprocessed = preprocess_dicom(dcm_path, target_size=SZ)
    print(f"  Preprocessed range : [{preprocessed.min():.3f}, {preprocessed.max():.3f}]")
    print(f"  Preprocessed mean  : {preprocessed.mean():.3f}")
    # How many pixels are soft tissue (liver-range)?
    windowed_hu = np.clip(hu, -100, 200)
    n_soft = ((hu >= 20) & (hu <= 180)).sum()
    print(f"  Liver-HU pixels (20-180) : {n_soft} ({100*n_soft/hu.size:.1f}%)")
    save_img(preprocessed, '02_hu_image.png')

    print("\n" + "="*60)
    print("STEP 3: Liver Mask Diagnosis")
    print("="*60)

    # Check each tier independently
    print("\n  --- Tier 1: MONAI checkpoint ---")
    if os.path.exists(args.seg_ckpt):
        print(f"  MONAI ckpt EXISTS: {args.seg_ckpt}")
        try:
            from liver_seg_model import LiverSegmenter
            seg  = LiverSegmenter(args.seg_ckpt, SZ)
            monai_mask = seg.predict(dcm_path)
            n = int(monai_mask.sum())
            print(f"  MONAI mask coverage: {n}/{SZ*SZ} ({100*n/(SZ*SZ):.1f}%)")
            if n/(SZ*SZ) > 0.90:
                print("  !! WARNING: MONAI mask coverage >90% -- model may be untrained/wrong")
            save_img(monai_mask, '03a_mask_monai.png')
        except Exception as e:
            print(f"  MONAI FAILED: {e}")
    else:
        print(f"  MONAI ckpt NOT FOUND: {args.seg_ckpt}")
        print("  -> Will fall through to GT / HU fallback")

    print("\n  --- Tier 2: CHAOS GT PNG ---")
    gt_mask = load_chaos_mask(dcm_path, SZ)
    if gt_mask is not None:
        n = int(gt_mask.sum())
        print(f"  GT mask FOUND. Coverage: {n}/{SZ*SZ} ({100*n/(SZ*SZ):.1f}%)")
        save_img(gt_mask, '03b_mask_chaos_gt.png')
    else:
        print(f"  GT mask NOT FOUND for: {dcm_path}")
        print("  -> Check that CHAOS Ground/ folder exists next to DICOM_anon/")

    print("\n  --- Tier 3: HU-threshold fallback ---")
    hu_mask = hu_liver_mask(dcm_path, SZ)
    n_hu    = int(hu_mask.sum())
    print(f"  HU mask coverage: {n_hu}/{SZ*SZ} ({100*n_hu/(SZ*SZ):.1f}%)")
    if n_hu/(SZ*SZ) > 0.90:
        print("  !! WARNING: HU mask is nearly full image!")
        print("  !! This means hu_liver_mask() fell back to np.ones()")
        print("  !! Likely cause: no soft tissue found in HU range [20, 180]")
        print("  !! Check if RescaleSlope/Intercept were applied correctly")
    elif n_hu/(SZ*SZ) < 0.03:
        print("  !! WARNING: HU mask is nearly empty (<3%)")
        print("  !! Liver may not be visible in this slice")
    else:
        print(f"  HU mask looks plausible (3-60% is typical for liver slices)")
    save_img(hu_mask, '03c_mask_hu_threshold.png')

    print("\n  --- Final mask used by get_liver_mask() ---")
    final_mask = get_liver_mask(dcm_path, SZ)
    n_final    = int(final_mask.sum())
    pct        = 100 * n_final / (SZ * SZ)
    print(f"  Final mask coverage: {n_final}/{SZ*SZ} ({pct:.1f}%)")
    if pct > 85:
        print("  !! DIAGNOSIS: Liver mask has FAILED. The entire image is being")
        print("  !! treated as liver. This is why anomalies appear everywhere and")
        print("  !! the actual tumor is missed (threshold set by global artifacts).")
    elif pct < 3:
        print("  !! DIAGNOSIS: Liver mask is near-empty. No liver found.")
        print("  !! The slice may not contain the liver (too high/low).")
    else:
        print(f"  DIAGNOSIS: Mask looks reasonable ({pct:.1f}% coverage).")
    save_img(final_mask, '03_liver_mask_FINAL.png')

    # Save coverage report
    with open(os.path.join(OUT, '05_mask_coverage.txt'), 'w') as f:
        f.write(f"HU range of raw slice: [{hu.min():.0f}, {hu.max():.0f}]\n")
        f.write(f"Liver-HU pixels (20-180): {n_soft} ({100*n_soft/hu.size:.1f}%)\n")
        f.write(f"HU mask coverage: {n_hu}/{SZ*SZ} ({100*n_hu/(SZ*SZ):.1f}%)\n")
        f.write(f"GT mask found: {gt_mask is not None}\n")
        f.write(f"Final mask coverage: {n_final}/{SZ*SZ} ({pct:.1f}%)\n")
        f.write("\nDiagnosis:\n")
        if pct > 85:
            f.write("FAIL: Mask is effectively the full image. No liver isolation.\n")
            f.write("The anomaly threshold will be set by border/edge artifacts,\n")
            f.write("not by liver content. Real liver anomalies will be missed.\n")
        elif pct < 3:
            f.write("FAIL: Mask is near-empty. Liver not present in this slice.\n")
        else:
            f.write("OK: Mask coverage looks plausible.\n")
    print(f"  Report saved: 05_mask_coverage.txt")

    print("\n" + "="*60)
    print("STEP 4: Masked image (what the AE actually sees)")
    print("="*60)
    masked_img = preprocessed * final_mask
    save_img(masked_img, '04_masked_image.png')
    print(f"  Non-zero pixels in masked image: {(masked_img > 0).sum()}")

    print("\n" + "="*60)
    print("STEP 5: AE Reconstruction + Error Map")
    print("="*60)

    if not os.path.exists(args.ckpt):
        print(f"  !! Checkpoint not found: {args.ckpt}")
        print("  !! Skipping AE step. Train a model first.")
        return

    # Load model
    x = torch.from_numpy(preprocessed).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

    try:
        if args.model == 'conv_ae':
            from models.model_1_conv_ae import ConvAutoencoder, anomaly_score as ascore
            model = ConvAutoencoder()
            model.load_state_dict(torch.load(args.ckpt, map_location=DEVICE)['model'])
            model = model.to(DEVICE).eval()
            _, emap, xh = ascore(model, x)
        elif args.model == 'ae_flow':
            from models.model_2_ae_flow import AEFlow, anomaly_score as ascore
            model = AEFlow(in_ch=1, base_ch=32, K=8)
            model.load_state_dict(torch.load(args.ckpt, map_location=DEVICE)['model'])
            model = model.to(DEVICE).eval()
            _, emap, xh = ascore(model, x)
        elif args.model == 'ccb_aae':
            from models.model_4_ccb_aae import CCBAAE, anomaly_score as ascore
            model = CCBAAE(in_ch=1, base=32)
            model.load_state_dict(torch.load(args.ckpt, map_location=DEVICE)['model'])
            model = model.to(DEVICE).eval()
            z_mean = torch.zeros(256).to(DEVICE)
            _, emap, xh = ascore(model, x, z_mean)
        else:
            print(f"  Use --model conv_ae, ae_flow, or ccb_aae for this debug script.")
            return

        emap_np = emap.squeeze().cpu().detach().numpy()
        xh_np   = xh.squeeze().cpu().detach().numpy()

        save_img(xh_np, '06a_reconstruction.png')
        save_img(emap_np / (emap_np.max() + 1e-8), '06_error_map_raw.png', cmap='hot')
        save_img((emap_np * final_mask) / (emap_np.max() + 1e-8),
                 '07_error_map_masked.png', cmap='hot')

        print(f"  Error map range: [{emap_np.min():.5f}, {emap_np.max():.5f}]")
        print(f"  Error map mean : {emap_np.mean():.5f}")

        # Histogram
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        liver_errors = emap_np[final_mask > 0] if pct < 85 else None
        bg_errors    = emap_np[final_mask < 0.5]

        axes[0].hist(emap_np.ravel(), bins=100, color='gray', alpha=0.7,
                     label='All pixels', log=True)
        if liver_errors is not None:
            axes[0].hist(liver_errors, bins=100, color='red', alpha=0.7,
                         label='Liver pixels', log=True)
        axes[0].axvline(np.quantile(emap_np.ravel(), 0.95),
                        color='blue', ls='--', label='95th pct (global)')
        if liver_errors is not None:
            axes[0].axvline(np.quantile(liver_errors, 0.95),
                            color='red', ls='--', label='95th pct (liver)')
        axes[0].set_title('Error distribution')
        axes[0].legend()

        # Border vs interior
        border = 10
        border_mask = np.zeros_like(emap_np)
        border_mask[:border, :]   = 1
        border_mask[-border:, :]  = 1
        border_mask[:, :border]   = 1
        border_mask[:, -border:]  = 1
        border_errors   = emap_np[border_mask > 0]
        interior_errors = emap_np[border_mask < 0.5]
        axes[1].hist(border_errors,   bins=100, color='orange', alpha=0.7,
                     label=f'Border pixels (10px ring)', log=True)
        axes[1].hist(interior_errors, bins=100, color='blue', alpha=0.7,
                     label='Interior pixels', log=True)
        axes[1].set_title('Border vs interior errors')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(OUT, '08_error_histogram.png'), dpi=120)
        plt.close()
        print("  Saved: 08_error_histogram.png")

        # Threshold analysis
        q95_global = np.quantile(emap_np.ravel(), 0.95)
        q98_global = np.quantile(emap_np.ravel(), 0.98)
        q95_border = np.quantile(border_errors,   0.95)
        q95_inter  = np.quantile(interior_errors, 0.95)
        q95_liver  = np.quantile(liver_errors, 0.95) if liver_errors is not None else None

        with open(os.path.join(OUT, '09_threshold_analysis.txt'), 'w') as f:
            f.write(f"95th pct (all pixels)    : {q95_global:.6f}\n")
            f.write(f"98th pct (all pixels)    : {q98_global:.6f}\n")
            f.write(f"95th pct (border 10px)   : {q95_border:.6f}\n")
            f.write(f"95th pct (interior)      : {q95_inter:.6f}\n")
            if q95_liver:
                f.write(f"95th pct (liver only)    : {q95_liver:.6f}\n")
            f.write(f"\nBorder 95th vs interior 95th ratio: "
                    f"{q95_border/(q95_inter+1e-10):.2f}x\n")
            if q95_border > q95_inter * 2:
                f.write("\nDIAGNOSIS: Border artifacts are >2x interior errors.\n")
                f.write("The 98th pct threshold is being set by BORDER pixels,\n")
                f.write("not by pathological liver content.\n")
                f.write("Fix: Exclude border pixels from threshold computation.\n")

        print("\n  Threshold analysis:")
        print(f"    95th pct (global) : {q95_global:.6f}")
        print(f"    95th pct (border) : {q95_border:.6f}")
        print(f"    95th pct (interior): {q95_inter:.6f}")
        if q95_border > q95_inter * 2:
            print("  !! DIAGNOSIS: Border artifacts are dominating the threshold!")
            print("  !! This causes the tumor to fall below threshold -> missed.")
        print(f"\n  Full report saved: 09_threshold_analysis.txt")

    except Exception as e:
        print(f"  AE step failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
    print(f"Debug output saved to: {OUT}/")
    print("Check files in this order: 05_mask_coverage.txt -> 03_liver_mask_FINAL.png")
    print("-> 09_threshold_analysis.txt -> 08_error_histogram.png")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
