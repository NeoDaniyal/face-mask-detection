import base64
from pathlib import Path
import cv2
from IPython.display import display, Javascript
from google.colab.output import eval_js
import numpy as np
from PIL import Image
import torch

from config import MODEL_DIR
from dataset import get_transforms
from model_transfer import MaskResNet18


def js_to_image(js_reply: str) -> np.ndarray:
    """Converts JavaScript base64 image string to OpenCV BGR image matrix."""
    image_bytes = base64.b64decode(js_reply.split(",")[1])
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(image_array, cv2.IMREAD_COLOR)


def take_photo_and_predict(threshold: float = 0.19) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = MODEL_DIR / "resnet18_best.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    # 1. Load Model
    model = MaskResNet18(dropout_rate=0.3).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _, eval_transform = get_transforms()
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # 2. JavaScript webcam trigger
    js = Javascript("""
        async function takePhoto() {
          const div = document.createElement('div');
          const video = document.createElement('video');
          video.style.display = 'block';
          const stream = await navigator.mediaDevices.getUserMedia({video: true});
          document.body.appendChild(div);
          div.appendChild(video);
          video.srcObject = stream;
          await video.play();
          
          await new Promise((resolve) => setTimeout(resolve, 1000));
          
          const canvas = document.createElement('canvas');
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          canvas.getContext('2d').drawImage(video, 0, 0);
          stream.getTracks().forEach(track => track.stop());
          div.remove();
          return canvas.toDataURL('image/jpeg');
        }
    """)
    display(js)
    data = eval_js("takePhoto()")

    # 3. Process captured frame
    frame = js_to_image(data)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    for x, y, w, h in faces:
        face_roi = frame[y : y + h, x : x + w]
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(face_rgb)

        tensor = eval_transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            prob_no_mask = torch.sigmoid(logits).squeeze().item()

        is_no_mask = prob_no_mask >= threshold
        label = "No Mask" if is_no_mask else "Mask"
        conf = prob_no_mask if is_no_mask else (1.0 - prob_no_mask)
        color = (0, 0, 255) if is_no_mask else (0, 255, 0)

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

    # Output annotated frame to Colab
    save_path = Path("outputs/plots/webcam_prediction.jpg")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), frame)

    display(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    print(f"Prediction frame saved to: {save_path.resolve()}")


if __name__ == "__main__":
    take_photo_and_predict(threshold=0.19)