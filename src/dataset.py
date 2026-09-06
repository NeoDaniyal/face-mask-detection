from pathlib import Path
from typing import Callable, Optional, Tuple

from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from config import CLASS_TO_IDX, IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Returns training (augmented) and evaluation (deterministic) pipelines."""
    train_transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    return train_transform, eval_transform


class MaskDataset(Dataset):
    """Custom PyTorch Dataset for loading face mask images."""

    def __init__(
        self, split_dir: Path, transform: Optional[Callable] = None
    ) -> None:
        self.split_dir = split_dir
        self.transform = transform
        self.valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

        self.image_paths = []
        self.labels = []

        for class_name, class_idx in CLASS_TO_IDX.items():
            class_folder = self.split_dir / class_name
            if not class_folder.exists():
                continue

            for p in class_folder.iterdir():
                if p.is_file() and p.suffix.lower() in self.valid_extensions:
                    self.image_paths.append(p)
                    self.labels.append(class_idx)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            with Image.open(img_path) as img:
                if img.mode in ("P", "PA"):
                    img = img.convert("RGBA")
                image = img.convert("RGB")
        except Exception as e:
            image = Image.new("RGB", IMAGE_SIZE, (0, 0, 0))

        if self.transform is not None:
            image = self.transform(image)

        return image, label