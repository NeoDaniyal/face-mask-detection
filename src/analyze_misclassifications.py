import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import (
    IDX_TO_CLASS,
    MODEL_DIR,
    OUTPUT_DIR,
    PLOTS_DIR,
)
from dataloader import create_dataloaders
from model import MaskCNNBaseline


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Reverses ImageNet normalization for plotting (CHW -> HWC, range [0, 1])."""
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * std + mean
    return np.clip(img, 0, 1)


def plot_misclassified_grid(
    samples: list[dict], title: str, save_path: Path
) -> None:
    """Plots a 3x4 grid (up to 12 images) of misclassified samples."""
    if not samples:
        print(f"No samples found for: {title}")
        return

    num_samples = min(12, len(samples))
    cols = 4
    rows = 3

    fig, axes = plt.subplots(rows, cols, figsize=(14, 10))
    axes = axes.flatten()

    for i in range(num_samples):
        item = samples[i]
        img = denormalize_image(item["image"])
        true_cls = IDX_TO_CLASS[item["true_label"]]
        pred_cls = IDX_TO_CLASS[item["pred_label"]]
        conf = item["confidence"] * 100

        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(
            f"True: {true_cls}\nPred: {pred_cls} ({conf:.1f}%)",
            fontsize=9,
            fontweight="bold",
            color="red",
        )

    # Turn off unused subplots
    for j in range(num_samples, len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"{title} (Showing {num_samples}/{len(samples)})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path.resolve()}")


def analyze_misclassifications() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Output setup
    save_dir = PLOTS_DIR / "misclassifications"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load DataLoader & Model
    _, _, test_loader = create_dataloaders()
    model = MaskCNNBaseline(dropout_rate=0.5).to(device)
    checkpoint_path = MODEL_DIR / "cnn_baseline_best.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("=" * 60)
    print("MISCLASSIFICATION ERROR ANALYSIS")
    print("=" * 60)

    false_negatives = []  # True: with_mask (0), Pred: without_mask (1)
    false_positives = []  # True: without_mask (1), Pred: with_mask (0)

    # 3. Inference Pass & Tracking
    with torch.no_grad():
        for images, labels in test_loader:
            images_gpu = images.to(device)
            logits = model(images_gpu)
            probs = torch.sigmoid(logits).squeeze(1).cpu()

            for i in range(len(labels)):
                true_label = int(labels[i].item())
                prob_without_mask = float(probs[i].item())

                # Binary thresholding at 0.5 probability
                pred_label = 1 if prob_without_mask >= 0.5 else 0

                if pred_label != true_label:
                    # Calculate confidence in the (wrong) prediction
                    confidence = prob_without_mask if pred_label == 1 else (1.0 - prob_without_mask)

                    sample_data = {
                        "image": images[i],
                        "true_label": true_label,
                        "pred_label": pred_label,
                        "confidence": confidence,
                    }

                    if true_label == 0 and pred_label == 1:
                        false_negatives.append(sample_data)
                    elif true_label == 1 and pred_label == 0:
                        false_positives.append(sample_data)

    # Sort errors by highest confidence first (worst mistakes)
    false_negatives.sort(key=lambda x: x["confidence"], reverse=True)
    false_positives.sort(key=lambda x: x["confidence"], reverse=True)

    print(f"Total Test Samples   : 1088")
    print(f"Total Errors Found   : {len(false_negatives) + len(false_positives)}")
    print(f"False Negatives (FN) : {len(false_negatives)} (Actual: with_mask → Pred: without_mask)")
    print(f"False Positives (FP) : {len(false_positives)} (Actual: without_mask → Pred: with_mask)")
    print("-" * 60)

    # 4. Generate Visualizations
    plot_misclassified_grid(
        false_negatives,
        title="False Negatives: Masked Faces Predicted as No-Mask",
        save_path=save_dir / "false_negatives.png",
    )

    plot_misclassified_grid(
        false_positives,
        title="False Positives: Unmasked Faces Predicted as Mask",
        save_path=save_dir / "false_positives.png",
    )

    print("=" * 60 + "\n")


if __name__ == "__main__":
    analyze_misclassifications()