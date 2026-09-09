import base64
import io
import sys
from pathlib import Path

# Add project root and src directory to Python path
project_root = Path("/content/face-mask-detection")
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
if str(project_root / "src") not in sys.path:
    sys.path.append(str(project_root / "src"))

import numpy as np
import torch
from PIL import Image
from google.colab.output import eval_js
from IPython.display import Image as IPImage
from IPython.display import Javascript, display

# Import project modules
from config import IDX_TO_CLASS, MODEL_DIR
from dataset import get_transforms
from model_transfer import MaskResNet18


def run_colab_inference(threshold: float = 0.19):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = MODEL_DIR / "resnet18_best.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint missing at {checkpoint_path}")

    # Load ResNet-18 model
    model = MaskResNet18(dropout_rate=0.3).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _, eval_transform = get_transforms()

    # JS Webcam Capture Script
    js_code = Javascript(
        """
        async function captureFrame() {
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
    """
    )

    display(js_code)
    img_data = eval_js("captureFrame()")

    # Convert base64 string directly to PIL Image
    binary_data = base64.b64decode(img_data.split(",")[1])
    pil_image = Image.open(io.BytesIO(binary_data)).convert("RGB")

    # Run inference on entire captured frame tensor
    tensor = eval_transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        prob_no_mask = torch.sigmoid(logits).squeeze().item()

    is_no_mask = prob_no_mask >= threshold
    label = "without_mask" if is_no_mask else "with_mask"
    confidence = prob_no_mask if is_no_mask else (1.0 - prob_no_mask)

    print("=" * 50)
    print("WEBCAM PREDICTION RESULT")
    print("=" * 50)
    print(f"Predicted Class : {label.upper()}")
    print(f"Confidence      : {confidence * 100:.2f}%")
    print(f"Raw Prob (NoMask): {prob_no_mask:.4f}")
    print("=" * 50)

    # Save output snapshot
    out_path = project_root / "outputs/plots/webcam_snapshot.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil_image.save(out_path)
    display(IPImage(filename=str(out_path)))


run_colab_inference(threshold=0.19)