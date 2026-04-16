# -*- coding: utf-8 -*-
"""
volumetric_analysis.py
======================
3D Reconstruction engine for liver tumor anomaly detection.
Usage:
  python volumetric_analysis.py --dicom path/to/patient/folder/ --ckpt checkpoints/ensemble_ae_liver_crop_best.pt
"""

import os
import argparse
import numpy as np
import torch
import pydicom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from glob import glob
from tqdm import tqdm

from dataset import preprocess_dicom
from liver_segmenter import get_liver_mask, crop_to_liver
from models.model_6_ensemble_ae import EnsembleAE, EnsembleScorer
from PIL import Image

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

from skimage import measure

def load_volume_masks(dicom_dir, scorer, target_size=128):
    """
    Process all DICOMs in directory and stack anomaly masks into a 3D volume.
    target_size is lower than training to ensure 3D rendering is possible.
    """
    files = sorted(glob(os.path.join(dicom_dir, "*.dcm")))
    if not files:
        # Try recursive if flat fails
        files = sorted(glob(os.path.join(dicom_dir, "**", "*.dcm"), recursive=True))
    
    print(f"  Found {len(files)} slices. Processing Volume...")
    
    # 3D Array: Depth x H x W
    volume = np.zeros((len(files), target_size, target_size), dtype=np.bool_)
    
    for i, f in enumerate(tqdm(files)):
        try:
            # 1. Preprocess and Segment at model resolution (256)
            img  = preprocess_dicom(f, target_size=256)
            mask = get_liver_mask(f, target_size=256)
            
            # 2. MATCH TRAINING: Crop to Liver ROI
            # Fixed: unpack 3 values
            cropped_img, _, _ = crop_to_liver(img, mask, target_size=256)
            
            # 3. Ensemble Inference
            x = torch.from_numpy(cropped_img).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
            with torch.no_grad():
                score, emap, _ = scorer.score(x)
            
            emap_np = emap.squeeze().cpu().detach().numpy()
            
            # 4. Extract Tumor Voxels (Threshold: 0.012 based on final ensemble metrics)
            anomaly_mask = emap_np > 0.012
            
            # 5. Downsample for 3D render display stability
            if target_size != 256:
                pil = Image.fromarray((anomaly_mask * 255).astype(np.uint8))
                pil = pil.resize((target_size, target_size), Image.NEAREST)
                anomaly_mask = np.array(pil) > 0

            volume[i] = anomaly_mask
            
        except Exception as e:
            # print(f"Error on slice {i}: {e}")
            continue
            
    return volume

def export_to_obj(volume, output_path):
    """
    Saves the 3D volume as an .obj mesh using Marching Cubes.
    This file can be loaded directly in Android Studio / Filament.
    """
    print(f"  Generating 3D Mesh (OBJ)...")
    if volume.sum() == 0:
        print("  Warning: No anomaly voxels found to export.")
        return

    # Use Marching Cubes to find a mesh surface
    # spacing is [z, y, x]
    verts, faces, normals, values = measure.marching_cubes(volume.astype(float), level=0.5)

    with open(output_path, 'w') as f:
        f.write("# Tumor Reconstruction OBJ\n")
        for v in verts:
            f.write(f"v {v[2]} {v[1]} {v[0]}\n") # Flip to X, Y, Z for standard viewers
        for face in faces:
            # OBJ is 1-indexed
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    
    print(f"  3D OBJ mesh saved: {output_path}")

def render_3d_tumor(volume, output_path):
    """
    Renders the 3D boolean volume using Matplotlib voxels for a quick preview.
    """
    if volume.sum() == 0:
        print("  Warning: Empty volume, skipping plot.")
        return

    print(f"  Rendering 3D Volume Preview (Size: {volume.shape})...")
    
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Matplotlib voxels expects H, W, D
    # Our volume is D, H, W (Z, Y, X)
    voxels = volume.transpose(1, 2, 0) # Y, X, Z
    
    # Set color (Solid Red for Tumor)
    colors = np.empty(voxels.shape + (4,), dtype=np.float32)
    colors[:] = [1, 0, 0, 0.8] # Red with some alpha
    
    ax.voxels(voxels, facecolors=colors, edgecolor='k', linewidth=0.1)
    
    # Medical Axes Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z (Slices)')
    ax.set_title('3D Volumetric Anomaly Reconstruction')
    
    ax.set_box_aspect([1, 1, 0.5]) 
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  3D Preview saved: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dicom', required=True, help='Path to DICOM directory')
    parser.add_argument('--ckpt', default='./checkpoints/ensemble_ae_liver_crop_best.pt')
    parser.add_argument('--out_img', default='results/3d_preview.png')
    parser.add_argument('--out_obj', default='results/tumor_volume.obj')
    parser.add_argument('--res', type=int, default=128, help='Resolution for 3D plot')
    args = parser.parse_args()
    
    # 1. Load Scorer
    print(f"Loading Ensemble Scorer from {args.ckpt}...")
    model = EnsembleAE(in_ch=1, img_size=256)
    ckpt_data = torch.load(args.ckpt, map_location=DEVICE)
    model.load_state_dict(ckpt_data['model'] if 'model' in ckpt_data else ckpt_data)
    model = model.to(DEVICE).eval()
    scorer = EnsembleScorer(model)
    
    # 2. Extract Volume
    volume = load_volume_masks(args.dicom, scorer, target_size=args.res)
    
    # 3. Export & Render
    os.makedirs('results', exist_ok=True)
    export_to_obj(volume, args.out_obj)
    render_3d_tumor(volume, args.out_img)
    
    # Summary
    n_vox = volume.sum()
    print("\n" + "="*40)
    print("3D RECONSTRUCTION COMPLETE")
    print("="*40)
    print(f"Total Tumor Voxels: {n_vox}")
    print(f"Volume coverage   : {100 * n_vox / volume.size:.4f} %")
    print(f"OBJ File (Android): {args.out_obj}")
    print(f"Preview Image     : {args.out_img}")
    print("="*40 + "\n")


if __name__ == "__main__":
    main()
