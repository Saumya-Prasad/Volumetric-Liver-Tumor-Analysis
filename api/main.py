# -*- coding: utf-8 -*-
"""
api/main.py
FastAPI REST backend for the Liver Anomaly Detection Android app.

Endpoints
---------
  POST /predict          Upload one DICOM file → JSON result + base64 images
  POST /predict/batch    Upload multiple DICOM files → per-slice results
  GET  /models           List available models
  GET  /health           Health check

Run
---
  pip install fastapi uvicorn python-multipart
  uvicorn api.main:app --host 0.0.0.0 --port 8000

Android calls
  POST http://<server-ip>:8000/predict
  Content-Type: multipart/form-data
  Body: file=<.dcm binary>, model_name=ae_flow
"""

import os
import sys
import io
import json
import base64
import tempfile
import traceback
import zipfile
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import pydicom

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ── Make parent importable ────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset import preprocess_dicom
from inference import load_model, run_inference, classify, THRESHOLDS_GLOBAL, THRESHOLDS_LIVER, save_results
from liver_segmenter import get_liver_mask, crop_to_liver, uncrop_error_map

LIVER_CROP_MODE = True # Set to True once your models finish training with --liver_crop parameter!

# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────

app = FastAPI(
    title="Liver Anomaly Detection API",
    description="REST API for unsupervised CT liver anomaly detection using 6 research-paper models.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT_DIR   = os.environ.get('CKPT_DIR', './checkpoints')
IMG_SIZE   = 256

# ── Model cache (loaded once) ─────────────────
_model_cache: dict = {}

AVAILABLE_MODELS = {
    'conv_ae'  : 'conv_ae_best.pt',
    'ae_flow'  : 'ae_flow_best.pt',
    'masked_ae': 'masked_ae_stage1_best.pt',
    'ccb_aae'  : 'ccb_aae_best.pt',
    'qformer'  : 'qformer_ae_best.pt',
    'ensemble' : 'ensemble_ae_best.pt',
    'attention_ae': 'attention_ae_liver_crop_best.pt',
}

MODEL_DESCRIPTIONS = {
    'conv_ae'  : 'Vanilla Convolutional Autoencoder (baseline)',
    'ae_flow'  : 'AE-FLOW: Autoencoder + Normalizing Flows (ICLR 2023)',
    'masked_ae': 'Masked Autoencoder + Pseudo-Abnormal Classifier (KES 2023)',
    'ccb_aae'  : 'Improved Adversarial AE with CCB (J. Digital Imaging 2022)',
    'qformer'  : 'Q-Former Autoencoder with DINOv2 (arXiv 2025)',
    'ensemble' : 'Ensemble of 4 diverse autoencoders',
    'attention_ae': 'Spatial Attention AE with Multi-Domain Loss',
}


def get_model(model_name: str):
    if model_name not in _model_cache:
        base_name = AVAILABLE_MODELS[model_name]
        if LIVER_CROP_MODE:
            crop_name = base_name.replace('_best.pt', '_liver_crop_best.pt')
            crop_path = os.path.join(CKPT_DIR, crop_name)
            ckpt_path = crop_path if os.path.exists(crop_path) else os.path.join(CKPT_DIR, base_name)
        else:
            ckpt_path = os.path.join(CKPT_DIR, base_name)
            
        if not os.path.exists(ckpt_path):
            # Return untrained model for demo
            from models.model_1_conv_ae   import ConvAutoencoder
            from models.model_2_ae_flow   import AEFlow
            from models.model_3_masked_ae import MaskedAutoencoder, AnomalyClassifier
            from models.model_4_ccb_aae   import CCBAAE
            from models.model_5_qformer_ae import QFormerAE
            from models.model_6_ensemble_ae import EnsembleAE
            demo = {
                'conv_ae'  : ConvAutoencoder(),
                'ae_flow'  : AEFlow(),
                'masked_ae': (MaskedAutoencoder(), AnomalyClassifier()),
                'ccb_aae'  : CCBAAE(),
                'qformer'  : QFormerAE(),
                'ensemble' : EnsembleAE(),
            }[model_name]
            if isinstance(demo, tuple):
                _model_cache[model_name] = tuple(m.to(DEVICE).eval() for m in demo)
            else:
                _model_cache[model_name] = demo.to(DEVICE).eval()
        else:
            _model_cache[model_name] = load_model(model_name, ckpt_path)
    return _model_cache[model_name]


# ─────────────────────────────────────────────
# Image encoding helpers
# ─────────────────────────────────────────────

def _np_to_b64(arr: np.ndarray, fmt='PNG') -> str:
    """Convert float32 numpy [0,1] to base64 PNG string."""
    arr_uint8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    if arr_uint8.ndim == 3 and arr_uint8.shape[2] == 3:
        pil = Image.fromarray(arr_uint8, 'RGB')
    else:
        # For grayscale (L), we apply a small contrast normalization if pixel range is tight
        pil = Image.fromarray(arr_uint8, 'L')
        # If it's a reconstruction, optimize the brightness to match clinical expectations
        if arr.max() < 0.6:  # If image is too dim
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(pil)
            pil = enhancer.enhance(1.4) # 40% boost for better medical visibility
    buf = io.BytesIO()
    pil.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def _make_gif_b64(images: list, duration=150) -> str:
    """Combines a list of base64 PNGs into a single base64 GIF."""
    if not images: return ""
    import io
    from PIL import Image
    try:
        pil_images = [Image.open(io.BytesIO(base64.b64decode(b))) for b in images]
        buf = io.BytesIO()
        pil_images[0].save(buf, format='GIF', save_all=True, append_images=pil_images[1:], loop=0, duration=duration)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        print("GIF Compilation Error:", e)
        return images[0]

def _get_slice_location(dcm_bytes: bytes) -> float:
    """Tries to extract spatial Z-axis order from raw DICOM bytes to prevent flickering GIFs."""
    try:
        d = pydicom.dcmread(io.BytesIO(dcm_bytes), stop_before_pixels=True)
        return float(d.SliceLocation) if hasattr(d, 'SliceLocation') else float(d.InstanceNumber)
    except:
        return 0.0

def _make_overlay(preprocessed: np.ndarray,
                  error_map: np.ndarray,
                  model_name: str) -> np.ndarray:
    """Red overlay on detected regions."""
    # Use the calibrated threshold for this specific model
    # SPECIAL CASE: masked_ae threshold (0.35) is a probability, not a pixel intensity.
    # We use a standard intensity floor (0.05) for its visual overlay.
    if model_name == 'masked_ae':
        fixed_threshold = 0.05
    else:
        fixed_threshold = float(THRESHOLDS_LIVER.get(model_name, 0.02))
    
    # Only calculate stats on non-zero liver pixels to avoid background dilution
    active_errors = error_map[error_map > 1e-6]
    if len(active_errors) > 0:
        # Strictly use the calibrated fixed threshold for this model
        # This allows users to see the actual sensitivity differences between models
        threshold = fixed_threshold
    else:
        threshold = fixed_threshold
        
    # Spatial Consistency Filter: Real tumors are 'blobs', noise is 'pepper dots'.
    # We apply a 3x3 median filter to eliminate isolated noisy pixels.
    from scipy.ndimage import median_filter
    error_map_smoothed = median_filter(error_map, size=3)
    
    # Black Exclusion Filter: Ignore air/gas (HU < -40)
    # White Exclusion Filter: Ignore bone/spine (HU > 120)
    # real tumors are hypo-dense soft-tissue, not cavities or bones.
    is_tumor_tissue = (preprocessed > 0.3) & (preprocessed < 0.65) 
    binary_mask  = (error_map_smoothed > threshold) & is_tumor_tissue
    rgb          = np.stack([preprocessed]*3, axis=-1)
    overlay      = rgb.copy()
    overlay[binary_mask, 0] = 1.0
    overlay[binary_mask, 1] = 0.0
    overlay[binary_mask, 2] = 0.0
    blended      = 0.5 * rgb + 0.5 * overlay
    blended[binary_mask] = overlay[binary_mask]
    return blended


def _make_heatmap(error_map: np.ndarray) -> np.ndarray:
    normed = error_map / (error_map.max() + 1e-8)
    return cm.hot(normed)[:, :, :3].astype(np.float32)


# ─────────────────────────────────────────────
# Shared DICOM processing
# ─────────────────────────────────────────────

def process_dicom_bytes(dicom_bytes: bytes, model_name: str, img_size: int, include_images: bool = True):
    """Core logic: bytes → prediction result dict."""
    # Save to temp file (pydicom needs a path)
    with tempfile.NamedTemporaryFile(suffix='.dcm', delete=False) as tmp:
        tmp.write(dicom_bytes)
        tmp_path = tmp.name

    try:
        # Raw pixels setup for display
        dcm       = pydicom.dcmread(tmp_path)
        raw       = dcm.pixel_array.astype(np.float32)
        raw_norm  = (raw - raw.min()) / ((raw.max() - raw.min()) + 1e-8)
        raw_pil   = np.array(Image.fromarray(
            (raw_norm * 255).astype(np.uint8)).resize((img_size, img_size))) / 255.0

        # Extract Preprocessed HU slice
        prep = preprocess_dicom(tmp_path, target_size=img_size)

        # Extract Robust Geometry Mask via the newly created External Framework!
        mask = get_liver_mask(tmp_path, target_size=img_size, prefer_gt=False)
        
        # Model Injection Configuration Safeties
        if LIVER_CROP_MODE:
            model_input, _, bbox = crop_to_liver(prep, mask, target_size=img_size)
        else:
            model_input = prep * mask

        # Tensor Build
        x = torch.from_numpy(model_input).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

        # Run Neural Model
        model = get_model(model_name)
        score, emap_np, xh_np = run_inference(model, model_name, x)

        # Rescale Error Map outwards if physically cropped, else zero-shift.
        if LIVER_CROP_MODE:
            emap_np = uncrop_error_map(emap_np, bbox, full_size=img_size)
            xh_np   = uncrop_error_map(xh_np,   bbox, full_size=img_size)
            
        # Anatomical Blending: Since the reconstruction is liver-only, we blend it 
        # back into the preprocessed background (prep) to provide anatomical context 
        # and prevent 'jumping' GIF artifacts.
        complete_recon = prep.copy()
        complete_recon[mask > 0.5] = xh_np[mask > 0.5]
            
        # CRITICAL: Always zero-out pixels outside the anatomical mask to prevent 
        # background noise (lungs/air/spine) from being detected as anomalies.
        emap_np = emap_np * mask
        
        # Diagnostic print to help verify mask quality in the terminal
        coverage = float(mask.mean() * 100)
        print(f"  [API] Liver Coverage: {coverage:.1f}% | Anomaly Thresholding Focused")
            
        label = classify(score, model_name, liver_mode=True)

        res = {
            'score'          : float(score),
            'label'          : str(label),
            'model'          : str(model_name),
            'threshold'      : float(THRESHOLDS_LIVER.get(model_name, 0.02))
        }

        if include_images:
            overlay = _make_overlay(prep, emap_np, model_name)
            heatmap = _make_heatmap(emap_np)
            res['images'] = {
                'original'     : _np_to_b64(raw_pil),
                'preprocessed' : _np_to_b64(prep),
                'reconstruction': _np_to_b64(complete_recon),
                'error_map'    : _np_to_b64(heatmap),
                'overlay'      : _np_to_b64(overlay),
            }
            
        return res
    finally:
        os.unlink(tmp_path)


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get('/health')
async def health():
    return {'status': 'ok', 'device': DEVICE,
            'checkpoint_dir': CKPT_DIR}


@app.get('/models')
async def list_models():
    """List all available models + whether their checkpoint exists."""
    result = []
    for name, ckpt_file in AVAILABLE_MODELS.items():
        ckpt_path = os.path.join(CKPT_DIR, ckpt_file)
        result.append({
            'name'       : name,
            'description': MODEL_DESCRIPTIONS[name],
            'trained'    : os.path.exists(ckpt_path),
            'threshold'  : THRESHOLDS_LIVER.get(name, 0.02),
        })
    return {'models': result}


@app.post('/predict')
async def predict(
    file: UploadFile = File(..., description='DICOM (.dcm) file or ZIP volume'),
    model_name: str  = Form('ae_flow', description='Model to use for inference'),
    img_size: int    = Form(256),
):
    """
    Upload a DICOM slice OR a ZIP volume and get anomaly detection results.
    Returns JSON with score, label, and base64 images (PNGs or Animated GIFs depending on input).
    """
    if model_name != "ALL MODELS" and model_name not in AVAILABLE_MODELS:
        raise HTTPException(400, f"Unknown model '{model_name}'.")

    try:
        file_bytes = await file.read()
        if file.filename.endswith('.zip'):
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
                dcm_files = [n for n in z.namelist() if not n.endswith('/') and '__macosx' not in n.lower() and '.ds_store' not in n.lower()]
                if not dcm_files: raise HTTPException(400, "No valid files.")
                
                slices_data = []
                for n in dcm_files:
                    b = z.read(n)
                    slices_data.append((_get_slice_location(b), b))
                slices_data.sort(key=lambda x: x[0])
                
                # Sub-sampling logic: 50 frames
                num_slices = len(slices_data)
                target_frames = 50
                if num_slices > target_frames:
                    indices = [int(i * (num_slices - 1) / (target_frames - 1)) for i in range(target_frames)]
                    slices_data = [slices_data[idx] for idx in indices]
                slices_data = slices_data[:target_frames]

                # --- Handle Comparison Mode (ALL MODELS) -----------------
                if model_name == "ALL MODELS":
                    comparison_results = {m: {'score': -float('inf'), 'label': 'normal', 'images': None} for m in AVAILABLE_MODELS}
                    # We only generate GIFs for the Ensemble model to keep payload small
                    accumulated_ensemble = {
                        'original': [], 'overlay': [], 'reconstruction': [],
                        'preprocessed': [], 'error_map': []
                    }
                    
                    for _, b in slices_data:
                        # 1. Run Ensemble for the full visual diagnostic
                        r_ensemble = process_dicom_bytes(b, 'ensemble', img_size, include_images=True)
                        for k in accumulated_ensemble:
                            accumulated_ensemble[k].append(r_ensemble['images'][k])
                        
                        # 2. Run all others for the numerical labels/scores ONLY (speed + efficiency)
                        for m in AVAILABLE_MODELS:
                            if m == 'ensemble':
                                r = r_ensemble
                            else:
                                r = process_dicom_bytes(b, m, img_size, include_images=False)
                            
                            if r['score'] > comparison_results[m]['score']:
                                comparison_results[m]['score'] = r['score']
                                comparison_results[m]['label'] = r['label']

                    # Post-process: Add Ensemble GIFs and Skeleton images for others
                    for m in AVAILABLE_MODELS:
                        if m == 'ensemble':
                            comparison_results[m]['images'] = {
                                'original': _make_gif_b64(accumulated_ensemble['original']),
                                'overlay' : _make_gif_b64(accumulated_ensemble['overlay']),
                                'reconstruction': _make_gif_b64(accumulated_ensemble['reconstruction']),
                                'preprocessed': _make_gif_b64(accumulated_ensemble['preprocessed']),
                                'error_map': _make_gif_b64(accumulated_ensemble['error_map'])
                            }
                        else:
                            # Skeletal structure for Android compatibility without the bloat
                            comparison_results[m]['images'] = {
                                'original': "", 'overlay': "", 'error_map': "", 'preprocessed': "", 'reconstruction': ""
                            }

                    labels = [v['label'] for v in comparison_results.values()]
                    majority_vote = 'tumor' if labels.count('tumor') > len(labels) // 2 else 'normal'
                    
                    return JSONResponse(content={
                        'majority_vote': majority_vote,
                        'model_results': comparison_results
                    })

                # --- Handle Single Model Mode -----------------------------
                accumulated = {'original': [], 'overlay': [], 'reconstruction': [], 
                               'preprocessed': [], 'error_map': []} 
                max_score, final_label = -float('inf'), 'normal'
                for _, b in slices_data:
                    r = process_dicom_bytes(b, model_name, img_size, include_images=True)
                    if r['score'] > max_score:
                        max_score = r['score']
                        final_label = r['label']
                    for k in accumulated:
                        accumulated[k].append(r['images'][k])

                return JSONResponse(content={
                    'score': float(max_score),
                    'label': str(final_label),
                    'model': str(model_name),
                    'threshold': float(THRESHOLDS_LIVER.get(model_name, 0.02)),
                    'images': {
                        'original': _make_gif_b64(accumulated['original']),
                        'overlay': _make_gif_b64(accumulated['overlay']),
                        'reconstruction': _make_gif_b64(accumulated['reconstruction']),
                        'preprocessed': _make_gif_b64(accumulated['preprocessed']),
                        'error_map': _make_gif_b64(accumulated['error_map'])
                    }
                })
        else:
            if model_name == "ALL MODELS":
                results = {}
                # Images only for Ensemble
                for m in AVAILABLE_MODELS:
                    include_imgs = True # Include images for all models in comparison mode
                    results[m] = process_dicom_bytes(file_bytes, m, img_size, include_images=include_imgs)
                
                labels = [v['label'] for v in results.values()]
                vote   = 'tumor' if labels.count('tumor') > len(labels) // 2 else 'normal'
                
                return JSONResponse(content={
                    'majority_vote': vote,
                    'model_results': results
                })
            else:
                result = process_dicom_bytes(file_bytes, model_name, img_size)
                return JSONResponse(content=result)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Inference failed: {str(e)}")


@app.post('/predict/batch')
async def predict_batch(
    files: List[UploadFile] = File(...),
    model_name: str          = Form('ae_flow'),
    img_size: int            = Form(256),
):
    """
    Upload multiple DICOM slices (e.g., a full series).
    Returns per-slice scores + a volume-level classification.
    """
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(400, f"Unknown model '{model_name}'.")

    slice_results = []
    for f in files:
        try:
            data   = await f.read()
            result = process_dicom_bytes(data, model_name, img_size, include_images=False)
            result['filename'] = f.filename
            slice_results.append(result)
        except Exception as e:
            slice_results.append({'filename': f.filename, 'error': str(e)})

    # Volume-level decision: any slice score above threshold → tumor
    tumor_slices = [r for r in slice_results
                    if isinstance(r.get('label'), str) and r['label'] == 'tumor']
    volume_label = 'tumor' if len(tumor_slices) > 0 else 'normal'
    max_score    = max((r.get('score', 0.0) for r in slice_results
                        if 'score' in r), default=0.0)

    return JSONResponse(content={
        'volume_label' : volume_label,
        'max_score'    : round(max_score, 6),
        'tumor_slices' : len(tumor_slices),
        'total_slices' : len(slice_results),
        'slices'       : slice_results,
    })


@app.post('/predict/zip')
async def predict_zip(
    file: UploadFile = File(..., description='ZIP file containing DICOM (.dcm) slices'),
    model_name: str  = Form('ae_flow'),
    img_size: int    = Form(256),
):
    """
    Upload a ZIP archive containing a volume of DICOM slices.
    Returns per-slice scores + a volume-level classification.
    """
    if not file.filename.endswith('.zip'):
        raise HTTPException(400, "File must be a .zip archive")
        
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(400, f"Unknown model '{model_name}'.")

    slice_results = []
    try:
        zip_bytes = await file.read()
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            # Drop the strict .dcm extension check to support raw file uploads without extensions
            dcm_files = [n for n in z.namelist() if not n.endswith('/') and '__macosx' not in n.lower() and '.ds_store' not in n.lower()]
            # Safety limit: don't process more than 50 slices in one go to prevent server timeout
            dcm_files = dcm_files[:50]
            
            for dcm_name in dcm_files:
                try:
                    data = z.read(dcm_name)
                    result = process_dicom_bytes(data, model_name, img_size, include_images=False)
                    result['filename'] = dcm_name
                    slice_results.append(result)
                except Exception as e:
                    slice_results.append({'filename': dcm_name, 'error': str(e)})
                    
    except zipfile.BadZipFile:
        raise HTTPException(400, "Invalid or corrupted ZIP file.")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"ZIP processing failed: {str(e)}")

    tumor_slices = [r for r in slice_results if isinstance(r.get('label'), str) and r['label'] == 'tumor']
    volume_label = 'tumor' if len(tumor_slices) > 0 else 'normal'
    max_score    = max((r.get('score', 0.0) for r in slice_results if 'score' in r), default=0.0)

    return JSONResponse(content={
        'volume_label' : volume_label,
        'max_score'    : round(max_score, 6),
        'tumor_slices' : len(tumor_slices),
        'total_slices' : len(slice_results),
        'slices'       : slice_results,
    })


@app.post('/predict/compare')
async def predict_compare(
    file: UploadFile = File(...),
    img_size: int    = Form(256),
):
    """
    Run ALL trained models on a single slice OR a ZIP volume and return comparison results.
    Useful for clinical review / model disagreement analysis.
    """
    results = {m: {'score': -float('inf'), 'label': 'normal'} for m in AVAILABLE_MODELS}

    try:
        file_bytes = await file.read()
        
        if file.filename.endswith('.zip'):
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
                dcm_files = [n for n in z.namelist() if not n.endswith('/') and '__macosx' not in n.lower() and '.ds_store' not in n.lower()]
                if not dcm_files: raise HTTPException(400, "No valid files.")
                
                slices_data = []
                for n in dcm_files:
                    b = z.read(n)
                    slices_data.append((_get_slice_location(b), b))
                slices_data.sort(key=lambda x: x[0])

                accumulated_all = {m: {'original': [], 'preprocessed': [], 'reconstruction': [], 'error_map': [], 'overlay': []} for m in AVAILABLE_MODELS}

                for _, b in slices_data:
                    for model_name in AVAILABLE_MODELS:
                        try:
                            if 'error' in results[model_name]: continue
                            r = process_dicom_bytes(b, model_name, img_size)
                            if r['score'] > results[model_name]['score']:
                                results[model_name]['score'] = r['score']
                                results[model_name]['label'] = r['label']
                            for k in accumulated_all[model_name]:
                                accumulated_all[model_name][k].append(r['images'][k])
                        except Exception as e:
                            results[model_name] = {'error': str(e)}

                for model_name in AVAILABLE_MODELS:
                    if 'error' not in results[model_name]:
                        results[model_name]['images'] = {k: _make_gif_b64(v) for k, v in accumulated_all[model_name].items()}

        else:
            for model_name in AVAILABLE_MODELS:
                try:
                    r = process_dicom_bytes(file_bytes, model_name, img_size)
                    results[model_name] = {
                        'score' : r['score'],
                        'label' : r['label'],
                        'images': r['images']
                    }
                except Exception as e:
                    results[model_name] = {'error': str(e)}
                    
    except zipfile.BadZipFile:
        raise HTTPException(400, "Invalid archive.")
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Processing failed: {str(e)}")

    labels = [v['label'] for v in results.values() if 'label' in v]
    vote   = 'tumor' if labels.count('tumor') > len(labels) // 2 else 'normal'

    return JSONResponse(content={
        'majority_vote': vote,
        'model_results': results,
    })


# ─────────────────────────────────────────────
if __name__ == '__main__':
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
