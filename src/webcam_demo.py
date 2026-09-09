import cv2
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from config import IDX_TO_CLASS, MODEL_DIR
from dataset import get_transforms
from model_transfer import MaskResNet18


def run_face_mask_detector(threshold: float = 0.19) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = MODEL_DIR / "resnet18_best.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    # 1. Load Model & Transforms
    model = MaskResNet18(dropout_rate=0.3).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _, eval_transform = get_transforms()

    # 2. Load Haar Cascade Face Detector
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # 3. Initialize Video Stream (0 = default webcam)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    print("=" * 60)
    print("Starting Real-Time Face Mask Detection (Press 'q' to exit)")
    print("=" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        for (x, y, w, h) in faces:
            # Crop face region
            face_roi = frame[y : y + h, x : x + w]
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(face_rgb)

            # Preprocess tensor
            tensor = eval_transform(pil_img).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(tensor)
                prob_no_mask = torch.sigmoid(logits).squeeze().item()

            # Classification decision using threshold 0.19
            is_no_mask = prob_no_mask >= threshold
            label = "No Mask" if is_no_mask else "Mask"
            conf = prob_no_mask if is_no_mask else (1.0 - prob_no_mask)

            # Assign color scheme: Green for Mask, Red for No Mask
            color = (0, 0, 255) if is_no_mask else (0, 255, 0)

            # Draw bounding box & text
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                f"{label} ({conf * 100:.1f}%)",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        cv2.imshow("Face Mask Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_face_mask_detector(threshold=0.19)