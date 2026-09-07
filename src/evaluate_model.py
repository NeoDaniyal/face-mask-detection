import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from config import (
    IDX_TO_CLASS,
    MODEL_DIR,
    OUTPUT_DIR,
    PLOTS_DIR,
)
from dataloader import create_dataloaders
from model import MaskCNNBaseline
from model_transfer import MaskResNet18


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
    """Plots a 3x4 grid (up to 12 images) from the prediction table."""
    if not samples:
        print(f"No samples found for: {title}")
        return

    num_samples = min(12, len(samples))
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    axes = axes.flatten()

    for i in range(num_samples):
        item = samples[i]
        img = denormalize_image(item["tensor"])
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

    for j in range(num_samples, len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"{title} (Showing {num_samples}/{len(samples)})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path.resolve()}")


def run_evaluation(model_type: str = "resnet18", checkpoint_name: str = "resnet18_best.pth") -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Output setup
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    misclass_dir = PLOTS_DIR / f"misclassifications_{model_type}"
    misclass_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Data & Model Architecture
    _, _, test_loader = create_dataloaders()

    if model_type == "resnet18":
        model = MaskResNet18(dropout_rate=0.3).to(device)
    else:
        model = MaskCNNBaseline(dropout_rate=0.5).to(device)

    # Check MODEL_DIR first, fall back to current directory
    checkpoint_path = MODEL_DIR / checkpoint_name
    if not checkpoint_path.exists():
        checkpoint_path = Path(checkpoint_name)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract state dict safely
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    print("=" * 60)
    print(f"TEST EVALUATION & ERROR ANALYSIS: {model_type.upper()}")
    print("=" * 60)
    print(f"Device                   : {device}")
    print(f"Checkpoint Loaded        : {checkpoint_path.name}")
    print("-" * 60)

    # 2. Inference Pass
    records = []
    sample_index = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images_gpu = images.to(device)
            logits = model(images_gpu)
            probs_without_mask = torch.sigmoid(logits).squeeze(1).cpu()

            for i in range(len(labels)):
                true_lbl = int(labels[i].item())
                prob_no_mask = float(probs_without_mask[i].item())

                pred_lbl = 1 if prob_no_mask >= 0.5 else 0
                is_correct = (pred_lbl == true_lbl)
                confidence = prob_no_mask if pred_lbl == 1 else (1.0 - prob_no_mask)

                records.append({
                    "sample_id": sample_index,
                    "true_label": true_lbl,
                    "pred_label": pred_lbl,
                    "prob_without_mask": prob_no_mask,
                    "confidence": confidence,
                    "is_correct": is_correct,
                    "tensor": images[i],
                })
                sample_index += 1

    df_preds = pd.DataFrame(records)

    # Save Prediction CSV
    csv_path = OUTPUT_DIR / f"test_predictions_{model_type}.csv"
    df_preds.drop(columns=["tensor"]).to_csv(csv_path, index=False)
    print(f"Saved Prediction CSV to: {csv_path.resolve()}")

    # 3. Calculate Core Metrics
    all_targets = df_preds["true_label"].values
    all_predictions = df_preds["pred_label"].values

    acc = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, pos_label=1, zero_division=0)
    recall = recall_score(all_targets, all_predictions, pos_label=1, zero_division=0)
    f1 = f1_score(all_targets, all_predictions, pos_label=1, zero_division=0)
    cm = confusion_matrix(all_targets, all_predictions)

    print(f"\nTest Samples Count       : {len(df_preds)}")
    print(f"Total Correct            : {df_preds['is_correct'].sum()}")
    print(f"Total Incorrect          : {(~df_preds['is_correct']).sum()}")
    print(f"Test Accuracy            : {acc * 100:.2f}%")
    print(f"Precision (without_mask) : {precision * 100:.2f}%")
    print(f"Recall (without_mask)    : {recall * 100:.2f}%")
    print(f"F1-Score (without_mask)  : {f1 * 100:.2f}%")

    print("\n" + "-" * 60)
    print("CONFUSION MATRIX")
    print("-" * 60)
    print(f"{'':<18} Predicted Mask    Predicted No Mask")
    print(f"Actual Mask        {cm[0, 0]:<18} {cm[0, 1]:<18} (Total: {cm[0].sum()})")
    print(f"Actual No Mask     {cm[1, 0]:<18} {cm[1, 1]:<18} (Total: {cm[1].sum()})")

    fn_samples = df_preds[(df_preds["true_label"] == 0) & (df_preds["pred_label"] == 1)].sort_values(by="confidence", ascending=False).to_dict("records")
    fp_samples = df_preds[(df_preds["true_label"] == 1) & (df_preds["pred_label"] == 0)].sort_values(by="confidence", ascending=False).to_dict("records")

    print("\n" + "-" * 60)
    print(f"False Negatives (FN) [with_mask → without_mask] : {len(fn_samples)}")
    print(f"False Positives (FP) [without_mask → with_mask] : {len(fp_samples)}")
    print("-" * 60)

    # 4. Generate & Save Visual Outputs
    class_labels = [IDX_TO_CLASS[0], IDX_TO_CLASS[1]]
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels, cbar=False)
    plt.title(f"Confusion Matrix - {model_type.upper()}", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=10)
    plt.ylabel("True Label", fontsize=10)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"confusion_matrix_{model_type}.png", dpi=300)
    plt.close()

    plot_misclassified_grid(
        fn_samples,
        title=f"False Negatives ({model_type.upper()}): Masked predicted as No-Mask",
        save_path=misclass_dir / "false_negatives.png",
    )
    plot_misclassified_grid(
        fp_samples,
        title=f"False Positives ({model_type.upper()}): Unmasked predicted as Mask",
        save_path=misclass_dir / "false_positives.png",
    )

    metrics_data = {
        "model_type": model_type,
        "test_samples": int(len(df_preds)),
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
        "false_negatives": len(fn_samples),
        "false_positives": len(fp_samples),
    }
    with open(OUTPUT_DIR / f"{model_type}_test_metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=4)

    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="resnet18", choices=["baseline", "resnet18"])
    parser.add_argument("--checkpoint", type=str, default="resnet18_best.pth")
    args = parser.parse_args()

    run_evaluation(model_type=args.model_type, checkpoint_name=args.checkpoint)