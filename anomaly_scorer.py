"""
anomaly_scorer.py  (v2)
========================
Root cause of the missed detection:

  Failure 1 -- Liver mask silently returns np.ones()
    When no MONAI ckpt and no CHAOS GT PNGs exist, hu_liver_mask() can
    return np.ones() (full-image fallback).  With coverage=100%, the mask
    does nothing.

  Failure 2 -- Border artifacts dominate the threshold
    Conv AEs with zero-padding produce high error at image borders
    (decoder sees zeros across padding edges).  With a full-image mask
    these border pixels set the 98th-pct threshold, which is 3-10x
    higher than interior liver errors.  Tumor falls below -> missed.

  Failure 3 -- Wrong statistic (mean) dilutes concentrated tumor signal

Fixes
-----
  1. border_margin: exclude outer N pixels from ALL scoring (default 16)
  2. validate_liver_mask(): detect mask failure (coverage > 85%) and
     fall back to interior-only mask automatically, with a warning
  3. Score is clamped to >= 0 (untrained models can return negative values)
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def make_interior_mask(size: int, border_margin: int = 16) -> np.ndarray:
    """Float32 (size,size) mask: 1.0 interior, 0.0 outer border ring."""
    mask = np.ones((size, size), dtype=np.float32)
    m = border_margin
    mask[:m, :]  = 0.0
    mask[-m:, :] = 0.0
    mask[:, :m]  = 0.0
    mask[:, -m:] = 0.0
    return mask


def validate_liver_mask(liver_mask:    np.ndarray | None,
                         img_size:      int   = 256,
                         border_margin: int   = 16,
                         max_coverage:  float = 0.85,
                         min_coverage:  float = 0.02
                         ) -> tuple[np.ndarray, str]:
    """
    Validate and fix the liver mask.

    Returns (mask_to_use, status) where status is one of:
      'ok'          -- coverage 2-85%, border pixels additionally excluded
      'mask_failed' -- coverage >85%, silently returned np.ones(); use interior-only
      'no_liver'    -- coverage <2%, liver not in this slice; use interior-only
      'no_mask'     -- liver_mask is None; use interior-only
    """
    interior = make_interior_mask(img_size, border_margin)

    if liver_mask is None:
        return interior, 'no_mask'

    coverage = float(liver_mask.mean())

    if coverage > max_coverage:
        print(f"  [AnomalyScorer] WARNING: liver mask coverage={coverage:.1%} > "
              f"{max_coverage:.0%} -- mask FAILED (returned np.ones). "
              "Using interior-only mask.")
        return interior, 'mask_failed'

    if coverage < min_coverage:
        return interior, 'no_liver'

    # Good mask: additionally zero border pixels
    combined = liver_mask * interior
    if combined.sum() < 10:
        return liver_mask, 'ok'
    return combined.astype(np.float32), 'ok'


# Scoring primitives

def quantile_score(error_map:     np.ndarray,
                   liver_mask:    np.ndarray | None = None,
                   q:             float             = 0.95,
                   border_margin: int               = 16) -> float:
    h, w   = error_map.shape
    size   = min(h, w)
    mask, status = validate_liver_mask(liver_mask, size, border_margin)
    pixels = error_map[mask > 0]
    return float(max(np.quantile(pixels, q), 0.0)) if len(pixels) > 0 else 0.0


def threshold_error_map(error_map:     np.ndarray,
                         liver_mask:    np.ndarray | None = None,
                         q:             float             = 0.98,
                         border_margin: int               = 16
                         ) -> tuple[float, np.ndarray]:
    h, w   = error_map.shape
    size   = min(h, w)
    mask, _ = validate_liver_mask(liver_mask, size, border_margin)
    pixels  = error_map[mask > 0]
    thr     = float(np.quantile(pixels, q)) if len(pixels) > 0 else 0.0
    binary  = ((error_map > thr) & (mask > 0)).astype(np.uint8)
    return thr, binary


def postprocess_mask(binary_mask:   np.ndarray,
                     liver_mask:    np.ndarray | None = None,
                     min_area_frac: float             = 0.001,
                     morphology:    bool              = True,
                     border_margin: int               = 16) -> np.ndarray:
    h, w    = binary_mask.shape
    size    = min(h, w)
    mask, _ = validate_liver_mask(liver_mask, size, border_margin)
    binary_mask = (binary_mask * (mask > 0)).astype(np.uint8)

    labeled, n = ndimage.label(binary_mask)
    if n == 0:
        return binary_mask

    liver_area = max(1, int(mask.sum()))
    min_area   = max(4, int(liver_area * min_area_frac))

    clean = np.zeros_like(binary_mask)
    for comp_id in range(1, n + 1):
        comp = (labeled == comp_id)
        if comp.sum() >= min_area:
            clean[comp] = 1

    if morphology and clean.sum() > 0:
        struct = ndimage.generate_binary_structure(2, 1)
        clean  = ndimage.binary_closing(clean, structure=struct,
                                        iterations=2).astype(np.uint8)
    return clean


# High-level scorer
class AnomalyScorer:
    """
    Full pipeline: error_map + liver_mask
      -> validate mask (auto-detect failures, exclude border)
      -> q_score quantile  (scalar score for classification)
      -> q_threshold       (binary mask)
      -> connected-component denoising

    Parameters
    ----------
    q_score        : quantile for scalar score  (default 0.95)
    q_threshold    : quantile for binary mask   (default 0.98)
    min_area_frac  : min CC area fraction of liver area  (default 0.001)
    border_margin  : pixels excluded at image border  (default 16)
    use_morphology : binary closing on final mask
    """

    def __init__(self,
                 q_score:        float = 0.95,
                 q_threshold:    float = 0.98,
                 min_area_frac:  float = 0.001,
                 border_margin:  int   = 16,
                 use_morphology: bool  = True):
        self.q_score       = q_score
        self.q_threshold   = q_threshold
        self.min_area_frac = min_area_frac
        self.border_margin = border_margin
        self.morphology    = use_morphology

    def score(self, error_map: np.ndarray,
              liver_mask: np.ndarray | None = None
              ) -> tuple[float, float, np.ndarray]:
        """Returns (score, threshold, clean_binary_mask)."""
        bm = self.border_margin
        s  = quantile_score(error_map, liver_mask, self.q_score, bm)
        t, raw_mask = threshold_error_map(error_map, liver_mask,
                                          self.q_threshold, bm)
        clean = postprocess_mask(raw_mask, liver_mask,
                                 self.min_area_frac, self.morphology, bm)
        return s, t, clean


# Thresholds (recalibrate on val set after retraining with liver_crop)

THRESHOLDS_Q95 = {
    'conv_ae'  : 0.025,
    'ae_flow'  : 0.032,
    'masked_ae': 0.45,
    'ccb_aae'  : 0.028,
    'qformer'  : 0.020,
    'ensemble' : 0.022,
}


def classify_quantile(score: float, model_name: str) -> str:
    thr = THRESHOLDS_Q95.get(model_name.lower(), 0.03)
    return 'tumor' if score > thr else 'normal'
