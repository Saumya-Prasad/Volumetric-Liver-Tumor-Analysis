import torch
import numpy as np
import pydicom
from PIL import Image
from models.model_6_ensemble_ae import EnsembleAE
from liver_segmenter import get_liver_mask, crop_to_liver, uncrop_error_map

def debug():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Init Ensemble
    print("Loading checkpoints/ensemble_ae_liver_crop_best.pt...")
    scorer = EnsembleAE().to(device)
    scorer.load_state_dict(torch.load("checkpoints/ensemble_ae_liver_crop_best.pt", map_location=device))
    scorer.eval()

    # Load a sample DICOM
    dcm_path = "dataset/i0041,0000b.dcm"
    print(f"Processing {dcm_path}...")
    ds = pydicom.dcmread(dcm_path)
    img = ds.pixel_array.astype(np.float32)
    
    # Mask
    mask = get_liver_mask(dcm_path)
    
    # Crop
    try:
        c_img, c_mask, bbox = crop_to_liver(img, mask, target_size=256)
    except ValueError as e:
        print(f"Crop Error: {e}")
        return

    # Prepare for model
    x = torch.from_numpy(c_img).unsqueeze(0).unsqueeze(0).to(device)
    
    # Scale x to 0-1 if not already (check liver_segmenter logic)
    # The liver_segmenter already scales it.
    
    # Inference
    score, score_map, x_hat = scorer.score(x)
    
    emap = score_map[0,0].cpu().numpy()
    
    # Masked EMAP
    masked_emap = emap * c_mask
    
    print(f"Score: {score.item():.5f}")
    print(f"EMAP - Max: {emap.max():.5f}, Min: {emap.min():.5f}, Mean: {emap.mean():.5f}")
    print(f"Masked EMAP - Max: {masked_emap.max():.5f}, Mean: {masked_emap.mean():.5f}")
    
    thresh = 0.012
    vcount = np.sum(masked_emap > thresh)
    print(f"Voxels > {thresh}: {vcount}")
    
    # Compare with non-masked
    vcount_raw = np.sum(emap > thresh)
    print(f"Voxels (No Mask) > {thresh}: {vcount_raw}")

if __name__ == "__main__":
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    debug()
