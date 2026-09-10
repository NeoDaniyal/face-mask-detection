import sys
import time
from pathlib import Path

# Add project root to sys.path for clean imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
if str(project_root / "src") not in sys.path:
    sys.path.append(str(project_root / "src"))

import cv2
import numpy as np
import torch
from PIL import Image

from config import MODEL_DIR
from dataset import get_transforms
from model_transfer import MaskResNet18


def draw_hud_header(frame: np.ndarray, fps: float) -> None:
    """Draws a sleek top status bar showing system state and live FPS."""
    h, w, _ = frame.shape
    
    # Semi-transparent top banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 45), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Header indicators
    cv2.circle(frame, (20, 22), 6, (0, 255, 0), -1)
    cv2.putText(
        frame,
        "AI FACE MASK MONITORING SYSTEM",
        (35, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(
        frame,
        fps_text,
        (w - 110, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 215, 255),
        1,
        cv2.LINE_AA,
    )


def draw_styled_bounding_box(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    label: str,
    confidence: float,
    color: tuple,
) -> None:
    """Draws corner reticles, semi-transparent label tag, and confidence bar."""
    length = int(min(w, h) * 0.2)
    thickness = 2

    # Corner reticles for high-tech look
    cv2.line(frame, (x, y), (x + length, y), color, thickness + 1)
    cv2.line(frame, (x, y), (x, y + length), color, thickness + 1)
    
    cv2.line(frame, (x + w, y), (x + w - length, y), color, thickness + 1)
    cv2.line(frame, (x + w, y), (x + w, y + length), color, thickness + 1)
    
    cv2.line(frame, (x, y + h), (x + length, y + h), color, thickness + 1)
    cv2.line(frame, (x, y + h), (x, y + h - length), color, thickness + 1)
    
    cv2.line(frame, (x + w, y + h), (x + w - length, y + h), color, thickness + 1)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - length), color, thickness + 1)

    # Bounding box frame
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)

    # Text Tag background
    tag_text = f"{label} {confidence * 100:.1f}%"
    (text_w, text_h), baseline = cv2.getTextSize(
        tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )
    
    tag_y1 = max(0, y - text_h - 14)
    tag_y2 = y

    # Semi-transparent background panel for text
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, tag_y1), (x + text_w + 12, tag_y2), color, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Text label
    cv2.putText(
        frame,
        tag_text,
        (x + 6, y - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # Progress bar below box
    bar_y = y + h + 6
    bar_width = int(w * confidence)
    cv2.rectangle(frame, (x, bar_y), (x + w, bar_y + 4), (50, 50, 50), -1)
    cv2.rectangle(frame, (x, bar_y), (x + bar_width, bar_y + 4), color, -1)


def run_desktop_webcam(threshold: float = 0.19) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = MODEL_DIR / "resnet18_best.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    # Initialize ResNet-18 model
    model = MaskResNet18(dropout_rate=0.3).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _, eval_transform = get_transforms()

    # Haar Cascade detector
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access video capture device.")
        return

    prev_time = time.time()
    fps = 0.0

    print("=" * 60)
    print("  LIVE INFERENCE ENGINE ACTIVATED")
    print("  Press 'q' or 'ESC' to terminate session.")
    print("=" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate FPS
        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / (curr_time - prev_time + 1e-6))
        prev_time = curr_time

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )

        # 1. PRIMARY LOOP: Detect faces using Haar Cascade
        for x, y, w, h in faces:
            face_roi = frame[y : y + h, x : x + w]
            if face_roi.size == 0:
                continue

            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            pil_face = Image.fromarray(face_rgb)

            tensor = eval_transform(pil_face).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(tensor)
                prob_no_mask = torch.sigmoid(logits).squeeze().item()

            is_no_mask = prob_no_mask >= threshold
            label = "NO MASK" if is_no_mask else "MASK"
            confidence = prob_no_mask if is_no_mask else (1.0 - prob_no_mask)
            color = (40, 40, 235) if is_no_mask else (50, 205, 50)

            draw_styled_bounding_box(
                frame, x, y, w, h, label, confidence, color
            )

        # 2. FALLBACK BLOCK: Place immediately after the face loop
        if len(faces) == 0:
            h, w, _ = frame.shape
            cx, cy = w // 2, h // 2
            cw, ch = int(w * 0.5), int(h * 0.5)
            x1, y1 = max(0, cx - cw // 2), max(0, cy - ch // 2)

            face_roi = frame[y1 : y1 + ch, x1 : x1 + cw]
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            pil_face = Image.fromarray(face_rgb)

            tensor = eval_transform(pil_face).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(tensor)
                prob_no_mask = torch.sigmoid(logits).squeeze().item()

            is_no_mask = prob_no_mask >= threshold
            label = "NO MASK" if is_no_mask else "MASK"
            confidence = prob_no_mask if is_no_mask else (1.0 - prob_no_mask)
            color = (40, 40, 235) if is_no_mask else (50, 205, 50)

            draw_styled_bounding_box(
                frame, x1, y1, cw, ch, label, confidence, color
            )

        # Draw HUD header bar
        draw_hud_header(frame, fps)

        cv2.imshow("Face Mask Detection - ResNet-18 HUD", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_desktop_webcam(threshold=0.19)