from pathlib import Path
from typing import Callable, Tuple, Optional

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from config import CLASS_TO_IDX, IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE

class MaskDataset(Dataset):
    """Custom Pytorch Dataset for loading face mask images."""

    def __init__(self, split_dir: Path, transform: Optional[Callable] = None)->None:
        """Args:
        split_dir (Path): Path to a split folder (e.g., data/processed/split/train)
        transform (Callable, Optional):Torchvision transforms to apply to the images.
        """
        self.split_dir = split_dir
        self.transform = transform
        self.valid_extension = {".jpg", ".jpeg", ".png", "bmp"}

        self.image_paths = []
        self.labels = []

        #Find the image and assign labels using central configuration mapping
        for class_name, class_idx in CLASS_TO_IDX.items():
            class_folder = self.split_dir/class_name
            if not class_folder.exists():
                continue
            for p in class_folder.iterdir():
                if p.is_file() and p.suffix.lower() in self.valid_extension:
                    self.image_paths.append(p)
                    self.labels.append(class_idx)
    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx:int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]

            # Convert to RGB to handle grayscale/RGBA images consistently
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        return image, label

def get_transform()-> Tuple[transforms.Compose, transforms.Compose]:
    """Reuturn Training and validation/testing transform pipelines.
        Uses Resize(256) + CenterCrop(224) to preserve aspected ratio without facial distortion.
    """
    train_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    return train_transform, eval_transform