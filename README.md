# Liver Anomaly Detection — Research-to-Code

Unsupervised CT liver anomaly detection implementing **6 architectures** from
peer-reviewed research papers, trained on the CHAOS abdominal CT dataset.
Includes a FastAPI backend and Android (Kotlin) integration.

---

## Quick Start

```bash
pip install -r requirements.txt

# Download CHAOS dataset & train one model
python train.py --model ae_flow --epochs 50 --batch 16

# Predict on a DICOM file
python inference.py --dicom path/to/slice.dcm --model ae_flow

# Start REST API
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

---

## Project Structure

```
liver_anomaly/
├── dataset.py              CHAOS CT dataset with DICOM HU preprocessing
├── train.py                Unified training script (all 6 models)
├── inference.py            Single/batch DICOM inference + visualisation
├── requirements.txt
├── models/
│   ├── model_1_conv_ae.py     Baseline Convolutional Autoencoder
│   ├── model_2_ae_flow.py     AE-FLOW (ICLR 2023)
│   ├── model_3_masked_ae.py   Masked AE + Pseudo-Abnormal Module (KES 2023)
│   ├── model_4_ccb_aae.py     CCB Adversarial AE (J. Digital Imaging 2022)
│   ├── model_5_qformer_ae.py  Q-Former Autoencoder (arXiv 2025)
│   └── model_6_ensemble_ae.py Ensemble of 4 Autoencoders
├── api/
│   └── main.py             FastAPI REST backend
└── android/
    └── INTEGRATION_GUIDE.md  Android Studio / Kotlin integration guide
```

---

## Model Architectures

### 1. Vanilla Conv-AE (`model_1_conv_ae.py`)
Baseline 4-level strided-conv encoder + transposed-conv decoder.
- **Loss:** `MSE(x, x̂)`
- **Anomaly score:** `‖x - x̂‖²`

### 2. AE-FLOW (`model_2_ae_flow.py`)
*Zhao, Ding, Zhang — ICLR 2023*

Encoder → **RealNVP Normalizing Flow** bottleneck → Decoder.
The flow maps latent features to a standard Gaussian.

- **Loss:** `L_recon + λ · L_flow`
  - `L_recon = MSE(x, x̂)`
  - `L_flow = ½‖z'‖² − Σlog|det(∂Φₖ/∂z)|`
- **Anomaly score:** `α · ‖x−x̂‖² + β · (½‖z'‖² − log_det)`

### 3. Masked AE (`model_3_masked_ae.py`)
*Georgescu — KES 2023, Procedia Computer Science*

ViT-based Masked Autoencoder (75% masking ratio) trained to reconstruct
masked patches. Anomaly classifier is trained on residual maps using the
**Pseudo-Abnormal Module** to synthesise positive (anomaly) examples.

- **Loss (stage 1):** `MSE on masked patches only`
- **Loss (stage 2):** `BCE(classifier(|x−x̂|), label)`
- **Pseudo-Abnormal Module:** random elliptical region with intensity shift
  `x_pa[R] = x[R] * U(0.5,1.5) + U(-0.3,0.3)`
- **Anomaly score:** `P_classifier(|x − x̂|) ∈ [0,1]`

### 4. CCB-AAE (`model_4_ccb_aae.py`)
*Zhang et al. — Journal of Digital Imaging 2022*

Adversarial AE where skip connections are replaced by **Chain of
Convolution Blocks (CCB)**. CCBs bridge the semantic gap via
non-linear (tanh-gated) residual transforms.

- **Generator loss:**
  `L_G = MSE(x,x̂) + λ₁·BCE_adv(D(x̂),1) + λ₂·MSE(z, N(0,I))`
- **Discriminator loss:**
  `L_D = ½[BCE(D(x),1) + BCE(D(x̂),0)]`
- **Anomaly score:** `α·‖x−x̂‖² + β·‖z−μ_train‖²`

### 5. Q-Former AE (`model_5_qformer_ae.py`)
*Dalmonte, Bayar, Akbas, Georgescu — arXiv 2507.18481, Jul 2025*

Frozen foundation model encoder (DINOv2 / CNN fallback) → multi-scale
feature projector → **Q-Former bottleneck** with M learnable query tokens
(Self-Attn + Cross-Attn + FFN) → CNN decoder. Perceptual loss uses frozen
Masked AE features.

- **Loss:** `MSE(x, x̂) + λ_p · (1 − cos_sim(F_mae(x), F_mae(x̂)))`
- **Anomaly score:** same as loss (pixel + perceptual)

### 6. Ensemble AE (`model_6_ensemble_ae.py`)
*Narrative Review approach — combined from multiple papers*

4 diverse members (ConvAE, WiderAE, VAE, MC-Dropout AE) trained
independently. Weights assigned inversely proportional to validation loss.

- **Anomaly score:** `S(x) = Σᵢ wᵢ · ‖x − AEᵢ(x)‖²`
- **Uncertainty:** `U(x) = std_i ‖x − AEᵢ(x)‖²` → flag for human review

---

## Dataset — CHAOS CT

```python
import kagglehub
path = kagglehub.dataset_download(
    "omarxadel/chaos-combined-ct-mr-healthy-abdominal-organ")
```

**DICOM Preprocessing pipeline** (matches `dataset.py`):
1. Load DICOM pixel array
2. HU conversion: `hu = pixel × RescaleSlope + RescaleIntercept`
3. Soft-tissue windowing: `clip(hu, -100, 200)`
4. Normalise to `[0, 1]`
5. Resize to 256 × 256

All models train only on **healthy** slices — anomalies are detected
as high reconstruction error at test time.

---

## Training

```bash
# Choose any model:
python train.py --model conv_ae   --epochs 50  --batch 16
python train.py --model ae_flow   --epochs 50  --batch 16
python train.py --model masked_ae --epochs 100 --batch 8
python train.py --model ccb_aae   --epochs 60  --batch 16
python train.py --model qformer   --epochs 40  --batch 8
python train.py --model ensemble  --epochs 50  --batch 16

# All options:
python train.py --help
```

Checkpoints are saved to `./checkpoints/<model>_best.pt`.

---

## Inference

```bash
# Single DICOM slice
python inference.py --dicom patient/slice_042.dcm --model ae_flow

# Compare all trained models
python inference.py --dicom slice.dcm --compare --ckpt_dir checkpoints/

# Outputs saved to ./results/:
#   original.png, preprocessed.png, reconstruction.png,
#   error_map.png, overlay.png, summary.png, result.json
```

`result.json` example:
```json
{
  "model": "ae_flow",
  "score": 0.023451,
  "label": "tumor",
  "threshold": 0.02
}
```

---

## REST API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health + device |
| `/models` | GET | List models + trained status |
| `/predict` | POST | Single DICOM → score + 5 images (base64) |
| `/predict/batch` | POST | Multiple DICOMs → volume-level result |
| `/predict/compare` | POST | All models → majority vote |

All image responses are base64-encoded PNG strings, ready for Android `BitmapFactory.decodeByteArray`.

---

## Android Integration

See `android/INTEGRATION_GUIDE.md` for the complete Kotlin implementation including:
- Retrofit API client
- ViewModel with coroutine upload
- 5-panel image display (Original / Preprocessed / Reconstruction / Error Map / Overlay)
- Model picker spinner
- DICOM file picker

---

## Future Scope (implemented from papers)

| Improvement | Source | Location |
|-------------|--------|----------|
| Cosine LR annealing | Standard | `train.py` |
| Perceptual loss (MAE features) | Q-Former paper | `model_5_qformer_ae.py` |
| Pseudo-Abnormal Module | Masked AE paper | `model_3_masked_ae.py` |
| Latent space regularisation | CCB-AAE paper | `model_4_ccb_aae.py` |
| Disagreement-weighted ensemble | Ensemble review | `model_6_ensemble_ae.py` |
| Epistemic uncertainty flag | Ensemble review | `model_6_ensemble_ae.py` |
| Multi-scale Q-Former queries | Q-Former future scope | `model_5_qformer_ae.py` |

---

## Requirements

```
torch>=2.0.0 · torchvision · einops · pydicom
scikit-image · numpy · Pillow · matplotlib
fastapi · uvicorn · python-multipart · kagglehub
```
