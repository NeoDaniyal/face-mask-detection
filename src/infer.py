import argparse
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F

from config import IDX_TO_CLASS, MODEL_DIR
from dataset import get_transforms
from model_transfer import MaskResNet18


def load_inference_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Loads trained ResNet-18 model checkpoint for inference."""
    model = MaskResNet18(dropout_rate=0.3).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict_image(
    image_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    threshold: float = 0.19,
) -> dict:
    """Predicts mask status for a single image tensor using optimized decision threshold."""
    _, eval_transform = get_transforms()

    try:
        with Image.open(image_path) as img:
            image_rgb = img.convert("RGB")
    except Exception as e:
        raise ValueError(f"Could not open image at {image_path}: {e}")

    tensor = eval_transform(image_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        prob_no_mask = torch.sigmoid(logits).squeeze().item()

    # Apply optimal threshold (0.19) derived from threshold analysis
    pred_idx = 1 if prob_no_mask >= threshold else 0
    pred_label = IDX_TO_CLASS[pred_idx]
    confidence = prob_no_mask if pred_idx == 1 else (1.0 - prob_no_mask)

    return {
        "image_path": str(image_path),
        "predicted_class": pred_label,
        "confidence": float(confidence),
        "prob_without_mask": float(prob_no_mask),
    }


def main():
    parser = argparse.ArgumentParser(description="Inference CLI for Face Mask Detection")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file")
    parser.add_argument("--threshold", type=float, default=0.19, help="Probability threshold for 'without_mask'")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = MODEL_DIR / "resnet18_best.pth"

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    model = load_inference_model(model_path, device)
    result = predict_image(Path(args.image), model, device, threshold=args.threshold)

    print("\n" + "=" * 50)
    print("INFERENCE RESULT")
    print("=" * 50)
    print(f"Image Path  : {result['image_path']}")
    print(f"Prediction  : {result['predicted_class'].upper()}")
    print(f"Confidence  : {result['confidence'] * 100:.2f}%")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()