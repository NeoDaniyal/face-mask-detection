import torch
from dataloader import create_dataloaders
from model import MaskCNNBaseline

def test_model_forward_pass()-> None:
    train_loader, _, _ = create_dataloaders()
    images, labels = next(iter(train_loader))

    model = MaskCNNBaseline(dropout_rate=0.5)
    model.eval()

    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grid
    )

    with torch.no_grid():
        outputs = model(images)

    print("="*60)
    print("MODEL TEST REPORT")
    print("="*60)

    print("\n1. Tensor Dimensions:")
    print("-"*60)
    print(f"Input Image Batch Shape: {images.shape}")
    print(f"Output Logit Batch Shape: {outputs.shape}")

    print(f"\n2. Model Complexity:")
    print("-"*60)
    print(f"Trainable Parameters: {trainable_params:,}")

    print("\n3. Sample Output Logits(5 samples):")
    print("-"*60)
    for idx, logits in enumerate(outputs[:5]):
        print(f"Sample {idx+1}: Raw Logits: {logits.items():.4f}")

    print("\n4. Forward Pass varification:")
    print("-"*60)
    if outputs.shape == (images.shape[0], 1):
        print("Model Forward Pass       : SUCCESS ✅")
    else:
        print("Model Forward Pass       : FAILED ❌")

    print("="*60 + "\n")

if __name__ == "__main__":
    test_model_forward_pass()