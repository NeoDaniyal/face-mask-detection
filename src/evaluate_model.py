import json
from pathlib import Path
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from config import (
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    MODEL_DIR,
    OUTPUT_DIR,
    PLOTS_DIR,
)
from dataloader import create_dataloaders
from model import MaskCNNBaseline

def evaluate_test_set() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _,_, test_loader = create_dataloaders()
    model = MaskCNNBaseline(dropout_rate=0.5).to(device)
    checkpoint_path = MODEL_DIR / "cnn_baseline_best.pth"
    if not checkpoint_path.exists():
        print(f"[⚠️ WARNING] Checkpoint not found at {checkpoint_path}. Please train the model first.")
        return
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("=" * 60)
    print("BASELINE CNN TEST EVALUATION")
    print("=" * 60)
    print(f"Device           : {device}")
    print(f"Checkpoint Path  : {checkpoint_path}")
    print(f"Best Checkpoint Epoch : {checkpoint.get('epoch', 'N/A')}")
    print("-"*60)

    all_targets = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits).sequeeeze(1).cpu().numpy()
            pred = (logits >= 0.0).sequeeeze(1).cpu().numpy()

            all_targets.extend(labels.numpy())
            all_predictions.extend(pred)
            all_probabilities.extend(probs)
    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)

    acc = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, pos_label=1, zero_division=0)
    recall = recall_score(all_targets, all_predictions, pos_label=1, zero_division=0)
    f1 = f1_score(all_targets, all_predictions, pos_label=1, zero_division=0)
    cm = confusion_matrix(all_targets, all_predictions)
    print(f"Test Samples Count       : {len(all_targets)}")
    print(f"Test Accuracy            : {acc * 100:.2f}%")
    print(f"Precision (without_mask) : {precision * 100:.2f}%")
    print(f"Recall (without_mask)    : {recall * 100:.2f}%")
    print(f"F1-Score (without_mask)  : {f1 * 100:.2f}%")

    print("\n" + "-" * 60)
    print("CONFUSION MATRIX")
    print("-" * 60)
    print(f"{'':<18} Predicted Mask    Predicted No Mask")
    print(
        f"Actual Mask        {cm[0, 0]:<18} {cm[0, 1]:<18} (Total: {cm[0].sum()})"
    )
    print(
        f"Actual No Mask     {cm[1, 0]:<18} {cm[1, 1]:<18} (Total: {cm[1].sum()})"
    )

    class_labels = [IDX_TO_CLASS[0], IDX_TO_CLASS[1]]
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels, cbar=False)
    plt.title("Confusion Matrix - CNN Baseline", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=10)
    plt.ylabel("True Label", fontsize=10)
    plt.tight_layout()

    cm_plot_path = PLOTS_DIR/ "confusion_matrix_baseline.png"
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"\nSaved Confusion Matrix Plot to: {cm_plot_path.resove()}")

    metrics_data = {
        "test_samples": int(len(all_targets)),
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
    }

    metrics_json_path = OUTPUT_DIR / "baseline_test_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"Saved Metrics Report to: {metrics_json_path.resolve()}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    evaluate_test_set()

