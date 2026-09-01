from typing import Tuple
from torch.utils.data import DataLoader

from config import BATCH_SIZE, SPLIT_DATA_DIR
from dataset import MaskDataset, get_transform

def create_dataloaders(split_base_dir=SPLIT_DATA_DIR, batch_size=BATCH_SIZE)-> Tuple[DataLoader,DataLoader, DataLoader]:
    """Instantiate dataset and return Dataloader for train, val and test splits."""
    train_transform, eval_transform = get_transform()

    train_dataset = MaskDataset(split_dir=split_base_dir/"train", transform=train_transform)
    val_dataset = MaskDataset(split_dir=split_base_dir/"val", transform=eval_transform)
    test_dataset = MaskDataset(split_dir=split_base_dir/"test", transform=eval_transform)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader