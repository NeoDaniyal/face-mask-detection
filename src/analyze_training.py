import json
import matplotlib.pyplot as plt
from config import OUTPUT_DIR, PLOTS_DIR

def plot_training_history()->None:
    history_path = OUTPUT_DIR / "history.json"
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    if not history_path.exists():
        raise FileNotFoundError(f"History file not found at {history_path.resolve()}. "
            "Please run training again with updated train.py.")
    with open(history_path, "r") as f:
        history = json.load(f)

    epochs = history["epochs"]
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    train_acc = [acc * 100 for acc in history["train_acc"]]
    val_acc = [acc * 100 for acc in history["val_acc"]]

    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_loss, 'o-', color="#1f77b4", linewidth=2, label="Train Loss")
    plt.plot(epochs, val_loss, 'o-', color="#ff7f0e", linewidth=2, label="Val Loss")
    plt.title("Training vs Validation Loss (CNN Baseline)", fontsize=12, fontweight="bold")
    plt.xlabel("Epochs", fontsize=10)
    plt.ylabel("BCE Loss (%)", fontsize=10)
    plt.xticks(epochs)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()

    loss_plot_path = PLOTS_DIR/"training_validation_loss.png"
    plt.savefig(loss_plot_path, dpi=300)
    plt.close()
    print(f"Saved Loss Curve To: {loss_plot_path.resolve()}")

    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_acc, 'o-', color="#2ca02c", linewidth=2, label="Train Accuracy")
    plt.plot(epochs, val_acc, 'o-', color="#d62728", linewidth=2, label="Val Accuracy")
    plt.title("Training vs Validation Accuracy (CNN Baseline)", fontsize=12, fontweight="bold")
    plt.xlabel("Epochs", fontsize=10)
    plt.ylabel("Accuracy (%)", fontsize=10)
    plt.xticks(epochs)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()

    acc_plot_path = PLOTS_DIR/"training_validation_accuracy.png"
    plt.savefig(acc_plot_path, dpi=300)
    plt.close()
    print(f"Saved Accuracy Curve To: {acc_plot_path.resolve()}")

if __name__ == "__main__":
    plot_training_history()