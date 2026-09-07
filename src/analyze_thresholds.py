import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score
from config import OUTPUT_DIR, PLOTS_DIR


def optimize_classification_threshold() -> None:
    csv_path = OUTPUT_DIR / "test_predictions_resnet18.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Predictions CSV missing at {csv_path}")

    df = pd.read_csv(csv_path)
    y_true = df["true_label"].values
    probs_no_mask = df["prob_without_mask"].values

    thresholds = np.linspace(0.01, 0.99, 100)
    f1_scores = []

    for t in thresholds:
        preds = (probs_no_mask >= t).astype(int)
        f1_scores.append(f1_score(y_true, preds, pos_label=1))

    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    print("=" * 60)
    print("CLASSIFICATION THRESHOLD ANALYSIS")
    print("=" * 60)
    print(f"Default Threshold (0.50) F1-Score : {f1_score(y_true, (probs_no_mask >= 0.5).astype(int)):.4f}")
    print(f"Optimal Threshold ({best_thresh:.2f}) F1-Score : {best_f1:.4f}")
    print("=" * 60)

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, f1_scores, color="#2ca02c", linewidth=2, label="F1-Score Curve")
    plt.axvline(0.5, color="gray", linestyle="--", label="Default Threshold (0.5)")
    plt.axvline(best_thresh, color="red", linestyle=":", label=f"Optimal Threshold ({best_thresh:.2f})")
    plt.xlabel("Decision Threshold (Probability of 'without_mask')", fontsize=10)
    plt.ylabel("F1-Score", fontsize=10)
    plt.title("ResNet-18 Decision Threshold vs F1-Score", fontsize=12, fontweight="bold")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    save_path = PLOTS_DIR / "resnet18_threshold_analysis.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved Threshold Plot to: {save_path.resolve()}\n")


if __name__ == "__main__":
    optimize_classification_threshold()