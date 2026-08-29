import random
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

from config import RAW_DATA_DIR

def visualize_random_samples(data_dir: Path, sample_per_class: int = 4, seed: int = None)->None:
   """Randomly Selectes and displays images from 'with_mask' and 'without_mask' classes."""
   if seed is not None:
       random.seed(seed)

   classes = ["with_mask", "without_mask"]
   valid_extension = {".jpg", ".jpeg", ".png", ".bmp"}

   fig, axes = plt.subplots(len(classes), sample_per_class, figsize=(16, 7))
   fig.suptitle("Face Mask Dataset - Visual Inspection", fontsize=16, fontweight="bold")

   for row_idx, class_name in enumerate(classes):
       class_folder = data_dir / class_name
       if not class_folder.exists():
           print(f"[❌ERROR] Directory Not Found: {class_folder}")
           continue
       image_paths = [
                   p
                   for p in class_folder.iterdir()
                   if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
               ]
       if len(image_paths) < sample_per_class:
           print(f"[⚠️ WARNING] Not enough images in {class_name} to the sample {sample_per_class}")
           selected_path = image_paths
       else:
           selected_path = random.sample(image_paths, sample_per_class)
       for col_idx, img_path in enumerate(selected_path):
           ax = axes[row_idx, col_idx]
           try:
                # comment: Convert into RGB for matplotlib rendering
                with Image.open(img_path) as img:
                    img_rgb = img.convert("RGB")
                    w, h = img.size

                    ax.imshow(img_rgb)
                    ax.set_title(f"{class_name}\n {w}x{h}", fontsize=10, color="navy" if class_name == "with_mask" else "darkred")
           except Exception as e:
                ax.set_title(f"Error loading\n{img_path.name}", fontsize=8)
           ax.axis("off")
   plt.tight_layout()
   plt.show()

if __name__ == "__main__":
    visualize_random_samples(RAW_DATA_DIR, sample_per_class=4)