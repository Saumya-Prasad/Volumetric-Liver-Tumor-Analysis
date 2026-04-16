import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

from models.model import build_model
from loss import CombinedLoss
from dataset import get_dataloaders


SAVE_DIR = "outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

EPOCHS = 15
BATCH_SIZE = 1
LR = 1e-4
VIZ_INTERVAL = 5
MODEL_PATH = "model.pth"


def save_reconstruction_viz(epoch, inputs, recons, save_dir):
    inputs_np = inputs[0, 0].cpu().numpy()   # (H,W)
    recons_np = recons[0, 0].cpu().detach().numpy()

    inp_slice = inputs_np
    rec_slice = recons_np
    anom_map = np.abs(inp_slice - rec_slice)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(inp_slice, cmap='gray')
    axes[0].set_title(f"Input (epoch {epoch})")
    axes[0].axis('off')

    axes[1].imshow(rec_slice, cmap='gray')
    axes[1].set_title("Reconstruction")
    axes[1].axis('off')

    im = axes[2].imshow(anom_map, cmap='hot')
    axes[2].set_title("Anomaly Map |x - x̂|")
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"recon_epoch_{epoch:03d}.png"), dpi=150)
    plt.close()


def save_loss_curves(history, save_dir):
    epochs = range(1, len(history['train_total']) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, history['train_total'], label='Train Loss')
    ax.plot(epochs, history['val_total'], label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training vs Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=150)
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    components = ['l1', 'ssim', 'freq', 'edge']
    titles = ['L1 Loss', 'SSIM Loss', 'Frequency Loss', 'Edge Loss']
    for ax, comp, title in zip(axes.flat, components, titles):
        ax.plot(epochs, history[f'train_{comp}'], label='Train')
        ax.plot(epochs, history[f'val_{comp}'], label='Val')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.suptitle('Individual Loss Components', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_components.png"), dpi=150)
    plt.close()


def save_anomaly_histogram(normal_scores, tumor_scores=None, save_dir="."):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(normal_scores, bins=30, alpha=0.7, label='Normal', color='steelblue')
    if tumor_scores is not None and len(tumor_scores) > 0:
        ax.hist(tumor_scores, bins=30, alpha=0.7, label='Tumor', color='tomato')
    ax.axvline(x=0.02, color='black', linestyle='--', label='Threshold (0.02)')
    ax.set_xlabel('Anomaly Score')
    ax.set_ylabel('Count')
    ax.set_title('Anomaly Score Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "anomaly_histogram.png"), dpi=150)
    plt.close()


def save_roc_curve(labels, scores, save_dir="."):
    from sklearn.metrics import roc_curve, auc
    if len(set(labels)) < 2:
        print("ROC: only one class present, skipping.")
        return 0.0
    fpr, tpr, _ = roc_curve(labels, scores)
    auroc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f'AUROC = {auroc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title('ROC Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "roc_curve.png"), dpi=150)
    plt.close()
    return auroc


def save_threshold_viz(inputs, recons, save_dir, epoch):
    inputs_np = inputs[0, 0].cpu().numpy()
    recons_np = recons[0, 0].cpu().detach().numpy()

    inp_slice = inputs_np
    anom_map = np.abs(inp_slice - recons_np)
    threshold = anom_map.mean() + 2 * anom_map.std()
    binary_mask = (anom_map > threshold).astype(np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(inp_slice, cmap='gray')
    axes[0].set_title("Input")
    axes[0].axis('off')

    axes[1].imshow(anom_map, cmap='hot')
    axes[1].set_title("Anomaly Map")
    axes[1].axis('off')

    axes[2].imshow(binary_mask, cmap='Reds')
    axes[2].set_title(f"Thresholded Mask (>{threshold:.4f})")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"threshold_viz_epoch_{epoch:03d}.png"), dpi=150)
    plt.close()


def run_epoch(model, loader, criterion, optimizer, device, training=True):
    model.train(training)
    total_metrics = defaultdict(float)
    n = 0

    with torch.set_grad_enabled(training):
        for batch in loader:
            x = batch.to(device)
            recon = model(x)
            loss, metrics = criterion(recon, x)

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            for k, v in metrics.items():
                total_metrics[k] += v
            n += 1

    return {k: v / max(n, 1) for k, v in total_metrics.items()}, \
           (x, recon) if n > 0 else (None, None)


def train(dataset_root):
    device = torch.device("cuda")
    print(f"Using device: {device}")

    train_loader, val_loader, _ = get_dataloaders(dataset_root, batch_size=BATCH_SIZE, liver_crop=True)

    if len(train_loader.dataset) == 0:
        print("No training data found. Check dataset path.")
        return

    model = build_model(device)
    criterion = CombinedLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = defaultdict(list)
    normal_scores = []

    for epoch in range(1, EPOCHS + 1):
        train_metrics, (tx, tr) = run_epoch(model, train_loader, criterion, optimizer, device, training=True)
        val_metrics, (vx, vr) = run_epoch(model, val_loader, criterion, optimizer, device, training=False)
        scheduler.step()

        for k in ['total', 'l1', 'ssim', 'freq', 'edge']:
            history[f'train_{k}'].append(train_metrics.get(k, 0))
            history[f'val_{k}'].append(val_metrics.get(k, 0))

        print(f"Epoch [{epoch:3d}/{EPOCHS}] "
              f"Train: {train_metrics['total']:.4f} "
              f"Val: {val_metrics['total']:.4f} "
              f"(L1:{train_metrics['l1']:.4f} SSIM:{train_metrics['ssim']:.4f} "
              f"Freq:{train_metrics['freq']:.4f} Edge:{train_metrics['edge']:.4f})")

        if epoch % VIZ_INTERVAL == 0 or epoch == 1:
            if vx is not None:
                save_reconstruction_viz(epoch, vx, vr, SAVE_DIR)
                save_threshold_viz(vx, vr, SAVE_DIR, epoch)
            save_loss_curves(history, SAVE_DIR)

        # Collect anomaly scores from val set
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                x = batch.to(device)
                recon = model(x)
                score = torch.mean(torch.abs(x - recon)).item()
                normal_scores.append(score)

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    save_loss_curves(history, SAVE_DIR)
    save_anomaly_histogram(normal_scores, save_dir=SAVE_DIR)

    # Attempt ROC if we have both classes (will only work if tumor data exists)
    if len(normal_scores) > 1:
        labels = [0] * len(normal_scores)
        try:
            save_roc_curve(labels, normal_scores, SAVE_DIR)
        except Exception as e:
            print(f"ROC skipped (need both classes): {e}")

    print("Training complete. Plots saved in:", SAVE_DIR)


if __name__ == "__main__":
    import sys
    import kagglehub

    dataset_root = None
    if len(sys.argv) > 1:
        dataset_root = sys.argv[1]
    else:
        print("Downloading CHAOS dataset via kagglehub...")
        dataset_root = kagglehub.dataset_download(
            "omarxadel/chaos-combined-ct-mr-healthy-abdominal-organ"
        )
        print(f"Dataset path: {dataset_root}")

    train(dataset_root)
