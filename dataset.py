import os
import glob
import random
import numpy as np
import pydicom
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# 1.  DICOM  →  Normalised numpy [0,1]

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


# 2.  PyTorch Dataset

class CHAOSDataset(Dataset):
    """
    Iterates over individual DICOM slices from the CHAOS CT dataset.

    Parameters
    ----------
    root_path   : path returned by kagglehub.dataset_download(...)
    split       : 'train' | 'val' | 'test'
    target_size : spatial resolution fed to the model
    augment     : apply random flips / rotations during training
    """

    def __init__(self,
                 root_path: str,
                 split: str      = 'train',
                 target_size: int = 256,
                 augment: bool   = True):
        self.target_size = target_size
        self.augment     = augment and (split == 'train')

        # Gather all .dcm paths recursively
        all_dcm = sorted(glob.glob(
            os.path.join(root_path, '**', '*.dcm'), recursive=True))

        if len(all_dcm) == 0:
            raise FileNotFoundError(
                f"No .dcm files found under {root_path}. "
                "Check your kagglehub download path.")

        # Deterministic train / val / test split (80 / 10 / 10)
        random.seed(42)
        shuffled = random.sample(all_dcm, len(all_dcm))
        n        = len(shuffled)
        splits   = {'train': shuffled[:int(0.8*n)],
                    'val'  : shuffled[int(0.8*n):int(0.9*n)],
                    'test' : shuffled[int(0.9*n):]}
        self.files = splits[split]
        print(f"[CHAOSDataset] split={split}  slices={len(self.files)}")

        # Augmentation transforms
        self.tf_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img  = preprocess_dicom(path, target_size=self.target_size)

        # (H, W) → Tensor (1, H, W)
        tensor = torch.from_numpy(img).unsqueeze(0)

        if self.augment:
            pil    = transforms.ToPILImage()(tensor)
            pil    = self.tf_aug(pil)
            tensor = transforms.ToTensor()(pil)

        return tensor   # float32, shape (1, H, W), values [0,1]


# 3.  Convenience factory

def get_dataloaders(root_path: str,
                    target_size: int = 256,
                    batch_size: int  = 16,
                    num_workers: int = 4):
    """Return (train_loader, val_loader, test_loader)."""
    train_ds = CHAOSDataset(root_path, 'train', target_size, augment=True)
    val_ds   = CHAOSDataset(root_path, 'val',   target_size, augment=False)
    test_ds  = CHAOSDataset(root_path, 'test',  target_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=1,
                              shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader

# Quick smoke-test
if __name__ == '__main__':
    import kagglehub
    path = kagglehub.dataset_download(
        "omarxadel/chaos-combined-ct-mr-healthy-abdominal-organ")
    print("Dataset path:", path)

    tl, vl, testl = get_dataloaders(path, target_size=256, batch_size=8)
    batch = next(iter(tl))
    print("Batch shape:", batch.shape)   # (8, 1, 256, 256)
    print("Min / Max :", batch.min().item(), batch.max().item())
