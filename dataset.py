# -*- coding: utf-8 -*-
"""
dataset.py
CHAOS CT Dataset Loader with HU preprocessing + optional liver ROI focus.

Key addition vs. the original file
------------------------------------
Two new parameters on ``CHAOSDataset`` control how slices are presented to
the model:

  liver_only (bool, default False)
    Zero-out every pixel outside the liver mask.  The AE still receives a
    256×256 image, but background, ribs, and spine are silenced.  Works
    with *any* existing checkpoint — no retraining required.  Good for a
    quick first fix.

  liver_crop (bool, default False)
    Stronger option: crop the image to the liver bounding box and resize
    to target_size.  The AE now only ever sees liver parenchyma at full
    resolution.  Requires retraining, but the reconstruction error is
    entirely liver-specific, so tumour-induced deviations dominate the
    anomaly score.

  --liver_crop implies liver_only (cropped image already contains only
    liver pixels; there is nothing else to mask).

Slice filtering (when liver_only or liver_crop is active)
----------------------------------------------------------
Slices that do *not* intersect the liver are useless for training an
organ-specific AE and actively harmful (they teach the model to reconstruct
non-liver tissue as if it were normal).  During ``__init__``, we filter the
file list to slices that have a non-empty CHAOS ground-truth mask.  For
slices without GT masks (non-CHAOS scans, inference-time data), the full
image is returned and ``liver_segmenter.hu_liver_mask`` is used as fallback
in ``__getitem__``.
"""

import os
import glob
import random
import numpy as np
import pydicom
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from liver_segmenter import get_liver_mask, crop_to_liver, has_liver_mask


# ──────────────────────────────────────────────
# 1.  DICOM  →  Normalised numpy [0,1]
# ──────────────────────────────────────────────

def preprocess_dicom(dcm_path: str,
                     window_min: int = -100,
                     window_max: int = 200,
                     target_size: int = 256) -> np.ndarray:
    """
    Load one DICOM slice and return a float32 numpy array in [0,1].

    Steps
    -----
    1. Read pixel array
    2. Apply RescaleSlope / RescaleIntercept  →  Hounsfield Units (HU)
    3. Soft-tissue windowing  (-100 HU … 200 HU)
    4. Normalise to [0, 1]
    5. Resize to target_size × target_size
    """
    dicom = pydicom.dcmread(dcm_path)
    img   = dicom.pixel_array.astype(np.float32)

    # HU conversion
    slope     = float(getattr(dicom, 'RescaleSlope',     1))
    intercept = float(getattr(dicom, 'RescaleIntercept', 0))
    hu_image  = img * slope + intercept

    # Soft-tissue windowing
    windowed = np.clip(hu_image, window_min, window_max)

    # Normalise
    normalised = (windowed - window_min) / float(window_max - window_min)

    # Resize
    pil = Image.fromarray((normalised * 255).astype(np.uint8))
    pil = pil.resize((target_size, target_size), Image.BILINEAR)
    out = np.array(pil).astype(np.float32) / 255.0
    return out   # shape: (H, W)  values: [0,1]


def load_volume(patient_dir: str, **kw) -> list:
    """Load and sort all DICOM slices in a patient folder."""
    files  = sorted(glob.glob(os.path.join(patient_dir, '*.dcm')))
    slices = [pydicom.dcmread(f) for f in files]
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except AttributeError:
        slices.sort(key=lambda x: int(x.InstanceNumber))
    imgs = [preprocess_dicom(f, **kw) for f in files]
    return imgs


# ──────────────────────────────────────────────
# 2.  PyTorch Dataset
# ──────────────────────────────────────────────

class CHAOSDataset(Dataset):
    """
    Iterates over individual DICOM slices from the CHAOS CT dataset.

    Parameters
    ----------
    root_path   : path returned by kagglehub.dataset_download(...)
    split       : 'train' | 'val' | 'test'
    target_size : spatial resolution fed to the model
    augment     : apply random flips / rotations during training
    liver_only  : zero-mask pixels outside the liver ROI before returning
                  the tensor.  Requires only a binary mask — no retraining.
    liver_crop  : crop the image to the liver bounding box and resize to
                  target_size.  Forces the AE to learn only liver texture.
                  Implies liver_only=True.  Requires retraining.

    Notes on liver_crop
    -------------------
    When liver_crop=True, ``__init__`` filters out slices with no non-empty
    CHAOS GT mask (i.e. slices where the liver is not visible).  This keeps
    training clean: the AE never sees cross-sections through the pelvis or
    lungs.  The filtering uses ``has_liver_mask()`` which checks the PNG
    file system without loading DICOM data, so the overhead is minimal.
    """

    def __init__(self,
                 root_path:   str,
                 split:       str  = 'train',
                 target_size: int  = 256,
                 augment:     bool = True,
                 liver_only:  bool = False,
                 liver_crop:  bool = False):

        self.target_size = target_size
        self.augment     = augment and (split == 'train')
        self.liver_only  = liver_only or liver_crop   # crop implies masking
        self.liver_crop  = liver_crop

        # ── Gather all .dcm paths recursively ────────────────────────
        all_dcm = sorted(glob.glob(
            os.path.join(root_path, '**', '*.dcm'), recursive=True))

        if len(all_dcm) == 0:
            raise FileNotFoundError(
                f"No .dcm files found under {root_path}. "
                "Check your kagglehub download path.")

        # ── Deterministic 80 / 10 / 10 split ─────────────────────────
        random.seed(42)
        shuffled = random.sample(all_dcm, len(all_dcm))
        n        = len(shuffled)
        splits   = {
            'train': shuffled[:int(0.8 * n)],
            'val'  : shuffled[int(0.8 * n):int(0.9 * n)],
            'test' : shuffled[int(0.9 * n):],
        }
        self.files = splits[split]

        # ── Liver-mode: filter to slices that contain the liver ───────
        if self.liver_only:
            pre_filter = len(self.files)
            # has_liver_mask() is fast (checks PNG on disk, no DICOM load).
            # Keep slices that have a GT mask  AND  that mask is non-empty.
            # Slices without any GT mask (e.g. non-CHAOS data) are kept and
            # fall back to the HU-based segmenter at __getitem__ time.
            has_gt  = [f for f in self.files if has_liver_mask(f)]
            no_gt   = [f for f in self.files if not has_liver_mask(f)]

            if len(has_gt) > 0:
                # Prefer GT-confirmed liver slices for training quality
                self.files = has_gt
                print(f"[CHAOSDataset] liver filter: kept {len(has_gt)}"
                      f" / {pre_filter} slices with GT mask  "
                      f"(dropped {len(no_gt)} slices without liver).")
            else:
                # No GT masks found (non-CHAOS dataset) — keep all slices,
                # HU fallback will handle masking.
                print("[CHAOSDataset] No GT masks found; "
                      "HU-based liver mask will be used for all slices.")

        print(f"[CHAOSDataset] split={split}  slices={len(self.files)}"
              f"  liver_only={self.liver_only}  liver_crop={self.liver_crop}")

        # ── Augmentation transforms ───────────────────────────────────
        self.tf_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
        ])

    # ──────────────────────────────────────────
    def __len__(self):
        return len(self.files)

    # ──────────────────────────────────────────
    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.files[idx]
        img  = preprocess_dicom(path, target_size=self.target_size)
        # img: (H, W) float32 in [0, 1]

        # ── Liver masking / crop ──────────────────────────────────────
        if self.liver_only:
            mask = get_liver_mask(path, target_size=self.target_size)
            # mask: (H, W) float32 binary {0.0, 1.0}

            if self.liver_crop:
                # Crop to liver bounding box then resize back to target_size.
                # After this, img is 100 % liver — no background to confuse
                # the autoencoder.
                img, mask, _ = crop_to_liver(img, mask, self.target_size)
            else:
                # Zero-out pixels outside the liver.
                # Simple, works with existing checkpoints, no retraining.
                img = img * mask

        # ── Tensor conversion  (H, W) → (1, H, W) ───────────────────
        tensor = torch.from_numpy(img).unsqueeze(0)    # float32, [0,1]

        # ── Augmentation (training only) ──────────────────────────────
        if self.augment:
            pil    = transforms.ToPILImage()(tensor)
            pil    = self.tf_aug(pil)
            tensor = transforms.ToTensor()(pil)

        return tensor   # float32, shape (1, H, W), values [0,1]


# ──────────────────────────────────────────────
# 3.  Convenience factory
# ──────────────────────────────────────────────

def get_dataloaders(root_path:   str,
                    target_size: int  = 256,
                    batch_size:  int  = 16,
                    num_workers: int  = 4,
                    liver_only:  bool = False,
                    liver_crop:  bool = False):
    """
    Return (train_loader, val_loader, test_loader).

    Parameters
    ----------
    liver_only : zero-mask pixels outside the liver in every slice
    liver_crop : crop each slice to its liver bounding box  (stronger;
                 also triggers slice filtering)
    """
    train_ds = CHAOSDataset(root_path, 'train', target_size,
                            augment=True,
                            liver_only=liver_only, liver_crop=liver_crop)
    val_ds   = CHAOSDataset(root_path, 'val',   target_size,
                            augment=False,
                            liver_only=liver_only, liver_crop=liver_crop)
    test_ds  = CHAOSDataset(root_path, 'test',  target_size,
                            augment=False,
                            liver_only=liver_only, liver_crop=liver_crop)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=1,
                              shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


# ──────────────────────────────────────────────
# Quick smoke-test
# ──────────────────────────────────────────────
if __name__ == '__main__':
    import kagglehub
    path = kagglehub.dataset_download(
        "omarxadel/chaos-combined-ct-mr-healthy-abdominal-organ")
    print("Dataset path:", path)

    # --- Standard mode --------------------------------------------------
    tl, vl, testl = get_dataloaders(path, target_size=256, batch_size=8)
    batch = next(iter(tl))
    print("Standard  — Batch shape:", batch.shape,
          " Min/Max:", batch.min().item(), batch.max().item())

    # --- Liver-only mode (mask) -----------------------------------------
    tl2, _, _ = get_dataloaders(path, target_size=256, batch_size=8,
                                liver_only=True)
    batch2 = next(iter(tl2))
    print("liver_only — Batch shape:", batch2.shape,
          " Non-zero px:", (batch2 > 0).float().mean().item())

    # --- Liver-crop mode ------------------------------------------------
    tl3, _, _ = get_dataloaders(path, target_size=256, batch_size=8,
                                liver_crop=True)
    batch3 = next(iter(tl3))
    print("liver_crop — Batch shape:", batch3.shape,
          " Min/Max:", batch3.min().item(), batch3.max().item())
