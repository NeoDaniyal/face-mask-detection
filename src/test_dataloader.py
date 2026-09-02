from config import BATCH_SIZE, IDX_TO_CLASS
from src.dataloader import create_dataloaders


def test_pipeline() -> None:
    train_loader, val_loader, test_loader = create_dataloaders()

    train_dataset = train_loader.dataset
    images, labels = next(iter(train_loader))

    print("=" * 60)
    print("DATALOADER TEST REPORT")
    print("=" * 60)

    print("\n1. Dataset Sizes")
    print("-" * 60)
    print(f"Train Dataset Size : {len(train_dataset)}")
    print(f"Val Dataset Size   : {len(val_loader.dataset)}")
    print(f"Test Dataset Size  : {len(test_loader.dataset)}")

    print("\n2. Batch Verification")
    print("-" * 60)
    print(f"Configured Batch   : {BATCH_SIZE}")
    print(f"Image Batch Shape  : {images.shape}")
    print(f"Label Batch Shape  : {labels.shape}")

    print("\n3. Data Types & Pixel Statistics")
    print("-" * 60)
    print(f"Image Tensor Dtype : {images.dtype}")
    print(f"Label Tensor Dtype : {labels.dtype}")
    print(f"Min Pixel Value    : {images.min().item():.4f}")
    print(f"Max Pixel Value    : {images.max().item():.4f}")
    print(f"Mean Pixel Value   : {images.mean().item():.4f}")

    print("\n4. Sample Labels (First 8 in Batch)")
    print("-" * 60)
    sample_labels = labels[:8].tolist()
    sample_names = [IDX_TO_CLASS[idx] for idx in sample_labels]

    for idx, (lbl, name) in enumerate(zip(sample_labels, sample_names)):
        print(f" - Sample {idx + 1}: Label {lbl} -> {name}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    test_pipeline()