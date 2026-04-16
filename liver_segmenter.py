# -*- coding: utf-8 -*-
"""
liver_segmenter.py  (v2)
========================
Liver ROI extraction with a three-tier priority chain:

  Priority 1 — Trained MONAI UNet (best accuracy)
    Loaded automatically if  checkpoints/liver_seg_best.pt  exists and
    MONAI is installed.  Produces smooth, closed liver contours that track
    the true capsule even in contrast-enhanced scans.

  Priority 2 — CHAOS ground-truth PNG mask (exact, zero cost)
    Used during CHAOS dataset training when the slice has a pre-existing
    label file at  CT/Ground/IMG-xxxx.png.

  Priority 3 — HU-threshold + morphology fallback (no labels required)
    Works on any abdominal CT.  Less accurate near the liver dome/tip and
    on contrast-enhanced studies, but robust enough for the crop pipeline.

Public API
----------
  get_liver_mask(dicom_path, target_size, prefer_gt, seg_ckpt) -> np.ndarray
  has_liver_mask(dicom_path)                                    -> bool
  crop_to_liver(image, mask, target_size, margin)               -> (img, mask, bbox)
  uncrop_error_map(error_crop, bbox, full_size)                 -> np.ndarray
"""

from __future__ import annotations

import os
import numpy as np
import pydicom
from PIL import Image
from scipy import ndimage

# Lazy-load the MONAI segmenter to avoid crashing when MONAI is absent
_segmenter_cache: dict = {}   # ckpt_path -> LiverSegmenter instance


def _get_segmenter(ckpt_path: str, target_size: int):
    """Cache the LiverSegmenter so it is loaded only once per session."""
    key = (ckpt_path, target_size)
    if key not in _segmenter_cache:
        try:
            from liver_seg_model import LiverSegmenter
            _segmenter_cache[key] = LiverSegmenter(ckpt_path, target_size)
        except (ImportError, RuntimeError, FileNotFoundError) as e:
            _segmenter_cache[key] = None
            print(f"[liver_segmenter] MONAI segmenter unavailable: {e}")
    return _segmenter_cache[key]


# -------------------------------------------------------------------------
# Priority 2 -- CHAOS ground-truth PNG
# -------------------------------------------------------------------------

def _chaos_mask_path(dicom_path: str) -> str | None:
    for dicom_sub in ('DICOM_anon', 'DICOM'):
        if dicom_sub in dicom_path:
            candidate = dicom_path.replace(dicom_sub, 'Ground')
            base, _   = os.path.splitext(candidate)
            for ext in ('.png', '.PNG'):
                p = base + ext
                if os.path.exists(p):
                    return p
    return None


def load_chaos_mask(dicom_path: str, target_size: int = 256) -> np.ndarray | None:
    path = _chaos_mask_path(dicom_path)
    if path is None:
        return None
    mask_pil = Image.open(path).convert('L')
    if np.array(mask_pil).max() == 0:
        return None
    mask_pil = mask_pil.resize((target_size, target_size), Image.NEAREST)
    return (np.array(mask_pil) > 0).astype(np.float32)


def has_liver_mask(dicom_path: str) -> bool:
    """Fast check: non-empty CHAOS GT mask exists on disk (no DICOM load)."""
    path = _chaos_mask_path(dicom_path)
    if path is None:
        return False
    try:
        return bool(np.array(Image.open(path).convert('L')).max() > 0)
    except Exception:
        return False


# -------------------------------------------------------------------------
# Priority 3 -- HU-threshold fallback
# -------------------------------------------------------------------------

def hu_liver_mask(dicom_path: str,
                  target_size: int  = 256,
                  hu_low:  float    = 20.0,
                  hu_high: float    = 180.0) -> np.ndarray:
    dcm       = pydicom.dcmread(dicom_path)
    pix       = dcm.pixel_array.astype(np.float32)
    slope     = float(getattr(dcm, 'RescaleSlope',     1))
    intercept = float(getattr(dcm, 'RescaleIntercept', 0))
    hu        = pix * slope + intercept

    # --- Soft-tissue threshold + Cleaning --------------------------------
    soft   = ((hu >= hu_low) & (hu <= hu_high)).astype(np.uint8)
    soft   = ndimage.binary_fill_holes(soft).astype(np.uint8)
    
    # Aggressively erode to sever muscle/fat connections
    eroded = ndimage.binary_erosion(soft, iterations=10)

    # --- Anatomical Filtering: Favor Image-Left (Patient-Right) -----------
    labeled, n = ndimage.label(eroded)
    if n == 0:
        # If erosion killed everything, fallback to soft mask without erosion
        labeled, n = ndimage.label(soft)
        if n == 0:
            return np.ones((target_size, target_size), dtype=np.float32)

    if n == 1:
        liver = (labeled == 1).astype(np.float32)
    else:
        # Check top 3 components to find the one that looks most like a liver (on the left)
        component_info = []
        for i in range(1, n + 1):
            mask_i  = (labeled == i)
            size_i  = np.sum(mask_i)
            com_i   = ndimage.center_of_mass(mask_i) # (y, x)
            component_info.append({'id': i, 'size': size_i, 'x_center': com_i[1]})
            
        # Filter for components that aren't purely on the right side of the image (Stomach/Spleen area)
        # In a 256px image, x < 160 is a safe threshold for liver bulk.
        liver_candidates = [c for c in component_info if c['x_center'] < 160]
        
        if not liver_candidates:
            # Fallback to absolute largest if no candidate fits spatial rule
            largest_id = int(np.argmax([c['size'] for c in component_info])) + 1
            liver = (labeled == largest_id).astype(np.float32)
        else:
            # Pick largest among valid spatial candidates
            best_candidate = max(liver_candidates, key=lambda x: x['size'])
            liver = (labeled == best_candidate['id']).astype(np.float32)

    # Restore volume + generous boundary margin
    liver = ndimage.binary_dilation(liver, iterations=12).astype(np.float32)

    pil = Image.fromarray((liver * 255).astype(np.uint8))
    pil = pil.resize((target_size, target_size), Image.NEAREST)
    return (np.array(pil) > 0).astype(np.float32)


# -------------------------------------------------------------------------
# Public mask entry point  (three-tier priority chain)
# -------------------------------------------------------------------------

_DEFAULT_SEG_CKPT = './checkpoints/liver_seg_best.pt'


def get_liver_mask(dicom_path:  str,
                   target_size: int  = 256,
                   prefer_gt:   bool = True,
                   seg_ckpt:    str  = _DEFAULT_SEG_CKPT) -> np.ndarray:
    """
    Tier 1: MONAI UNet  -- if seg_ckpt exists and MONAI is installed.
    Tier 2: CHAOS GT    -- if prefer_gt=True and a PNG mask exists.
    Tier 3: HU heuristic -- always available as final fallback.
    """
    # Tier 1
    if os.path.exists(seg_ckpt):
        seg = _get_segmenter(seg_ckpt, target_size)
        if seg is not None:
            try:
                return seg.predict(dicom_path)
            except Exception as e:
                print(f"[liver_segmenter] MONAI predict failed ({e}); "
                      "falling back to GT / HU mask.")

    # Tier 2
    if prefer_gt:
        gt = load_chaos_mask(dicom_path, target_size)
        if gt is not None:
            return gt

    # Tier 3
    return hu_liver_mask(dicom_path, target_size)


# -------------------------------------------------------------------------
# Crop / uncrop utilities
# -------------------------------------------------------------------------

def get_liver_bbox(mask: np.ndarray,
                   margin: int = 20) -> tuple[int, int, int, int]:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        H, W = mask.shape
        return 0, H, 0, W
    r0, r1 = int(np.where(rows)[0][0]),  int(np.where(rows)[0][-1])
    c0, c1 = int(np.where(cols)[0][0]),  int(np.where(cols)[0][-1])
    H, W   = mask.shape
    return (max(0, r0 - margin), min(H, r1 + margin + 1),
            max(0, c0 - margin), min(W, c1 + margin + 1))


def crop_to_liver(image:       np.ndarray,
                  mask:        np.ndarray,
                  target_size: int = 128,
                  margin:      int = 20
                  ) -> tuple[np.ndarray, np.ndarray, tuple]:
    """
    Crop image + mask to the liver bounding box and resize to target_size.

    Default target_size is 128, NOT 64.
    64px discards the subtle HU gradient transitions between normal
    parenchyma and hypo-dense lesions. 128px retains them while still
    excluding all non-liver anatomy.
    """
    y0, y1, x0, x1 = get_liver_bbox(mask, margin)
    crop_img  = image[y0:y1, x0:x1]
    crop_mask = mask [y0:y1, x0:x1]

    pil_img  = Image.fromarray((crop_img  * 255).astype(np.uint8))
    pil_mask = Image.fromarray((crop_mask * 255).astype(np.uint8))
    pil_img  = pil_img.resize( (target_size, target_size), Image.BILINEAR)
    pil_mask = pil_mask.resize((target_size, target_size), Image.NEAREST)

    out_img  = np.array(pil_img ).astype(np.float32) / 255.0
    out_mask = (np.array(pil_mask) > 0).astype(np.float32)
    return out_img, out_mask, (y0, y1, x0, x1)


def uncrop_error_map(error_crop: np.ndarray,
                     bbox:       tuple,
                     full_size:  int = 256) -> np.ndarray:
    """Place a cropped error map back into a full-size canvas (zeros outside liver)."""
    y0, y1, x0, x1 = bbox
    canvas          = np.zeros((full_size, full_size), dtype=np.float32)
    crop_h, crop_w  = y1 - y0, x1 - x0
    pil = Image.fromarray(error_crop.astype(np.float32))
    pil = pil.resize((crop_w, crop_h), Image.BILINEAR)
    canvas[y0:y1, x0:x1] = np.array(pil)
    return canvas
