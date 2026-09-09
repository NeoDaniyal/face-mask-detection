import base64
import io
import sys
from pathlib import Path

# Fix relative import paths for Colab
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
if str(project_root / "src") not in sys.path:
    sys.path.append(str(project_root / "src"))

from google.colab.output import eval_js
from IPython.display import Image as IPImage, Javascript, display
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from config import MODEL_DIR
from dataset import get_transforms
from model_transfer import MaskResNet18


def take_photo_and_predict(threshold: float = 0.19) -> None:
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

    # 2. JavaScript Webcam Bridge
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
          
          await new Promise((resolve) => setTimeout(resolve, 800));
          
          const canvas = document.createElement('canvas');
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          canvas.getContext('2d').drawImage(video, 0, 0);
          
          stream.getTracks().forEach(track => track.stop());
          div.remove();
          return canvas.toDataURL('image/jpeg', 0.9);
        }
    """)
    display(js)
    data = eval_js("takePhoto()")

    # 3. Process Image directly using PIL
    binary_data = base64.b64decode(data.split(",")[1])
    pil_img = Image.open(io.BytesIO(binary_data)).convert("RGB")

    # 4. Model Prediction
    tensor = eval_transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        prob_no_mask = torch.sigmoid(logits).squeeze().item()

    is_no_mask = prob_no_mask >= threshold
    label = "No Mask" if is_no_mask else "Mask"
    conf = prob_no_mask if is_no_mask else (1.0 - prob_no_mask)
    color = "red" if is_no_mask else "green"

    # 5. Draw Visual Overlay (PIL Draw)
    draw = ImageDraw.Draw(pil_img)
    w, h = pil_img.size
    
    # Draw frame boundary
    draw.rectangle([10, 10, w - 10, h - 10], outline=color, width=4)
    text = f"{label} ({conf * 100:.1f}%)"
    draw.text((20, 20), text, fill=color)

    # Save output frame
    save_path = project_root / "outputs/plots/webcam_prediction.jpg"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    pil_img.save(save_path)

    print("=" * 50)
    print("WEBCAM PREDICTION RESULT")
    print("=" * 50)
    print(f"Prediction : {label.upper()}")
    print(f"Confidence : {conf * 100:.2f}%")
    print(f"Saved to   : {save_path.resolve()}")
    print("=" * 50)

    display(IPImage(filename=str(save_path)))


if __name__ == "__main__":
    take_photo_and_predict(threshold=0.19)