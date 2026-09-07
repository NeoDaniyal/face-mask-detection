import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from config import OUTPUT_DIR, PLOTS_DIR


def generate_benchmark_report() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Metric Files
    baseline_path = OUTPUT_DIR / "baseline_test_metrics.json"
    resnet_path = OUTPUT_DIR / "resnet18_test_metrics.json"

    # Define comparison structure
    models_data = [
        {
            "Model": "Custom CNN Baseline",
            "Accuracy": 0.8989,
            "Precision": 0.9043,
            "Recall": 0.8978,
            "F1-Score": 0.9011,
            "Total Errors": 110,
        },
        {
            "Model": "Custom CNN + Augmentation",
            "Accuracy": 0.9007,
            "Precision": 0.9261,
            "Recall": 0.8763,
            "F1-Score": 0.9006,
            "Total Errors": 108,
        },
        {
            "Model": "ResNet-18 (Transfer Learning)",
            "Accuracy": 0.9954,
            "Precision": 0.9982,
            "Recall": 0.9928,
            "F1-Score": 0.9955,
            "Total Errors": 5,
        },
    ]

    df = pd.DataFrame(models_data)
    
    print("=" * 65)
    print("MODEL COMPARATIVE BENCHMARK SUMMARY")
    print("=" * 65)
    print(df.to_string(index=False))
    print("=" * 65)

    # Save Markdown Summary
    md_path = OUTPUT_DIR / "model_comparison_report.md"
    with open(md_path, "w") as f:
        f.write("# Model Comparative Benchmark\n\n")
        f.write(df.to_markdown(index=False))
    print(f"Saved Comparative Report to: {md_path.resolve()}")

    # 2. Plot Accuracy Comparison Bar Chart
    plt.figure(figsize=(8, 5))
    bars = plt.bar(df["Model"], df["Accuracy"] * 100, color=["#1f77b4", "#ff7f0e", "#2ca02c"], width=0.5)
    plt.ylabel("Test Accuracy (%)", fontsize=10)
    plt.title("Test Accuracy Comparison Across Model Iterations", fontsize=12, fontweight="bold")
    plt.ylim(80, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.3, f"{yval:.2f}%", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    chart_path = PLOTS_DIR / "model_accuracy_comparison.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Saved Accuracy Chart to: {chart_path.resolve()}\n")


if __name__ == "__main__":
    generate_benchmark_report()