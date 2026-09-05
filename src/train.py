import os
import torch
import torch.nn as nn
import time
from pathlib import Path
from torch.optim import Adam
from config import BATCH_SIZE, MODEL_DIR, RANDOM_SEED
from dataloader import create_dataloaders
from model import MaskCNNBaseline

def calculate_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate binary classification accuracy.
    
    Applies sigmoid thresholding at 0.0 logits (0.5 probability)"""
    preds = (logits>0.0).float()
    correct = (preds == targets).float().sum()
    return correct / targets.size(0)

def train_one_epoch(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
)-> tuple[float, float]:
    """Run a single training epoch."""
    model.train()
    running_loss = 0.0
    running_corrects = 0.0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1).float()  # Ensure labels are of shape (batch_size, 1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size

        pred = (outputs > 0.0).float()
        running_corrects += (pred == labels).sum().item()
        total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    return epoch_loss, epoch_acc

def evaluate(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        device: torch.device
        )-> tuple[float, float]:
    """Running validation evaluation."""
    running_loss = 0.0
    running_corrects = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1).float()

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size

            pred = (outputs >= 0.0).float()
            running_corrects += (pred == labels).sum().item()
            total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    return epoch_loss, epoch_acc

def run_training(epoch: int=5, lr: float= 0.001) -> None:
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("="*60)
    print("STARTING TRAINING PIPELINE")
    print("="*60)
    print(f"Device: {device}")
    print(f"Epochs: {epoch}")
    print(f"Learning Rate: {lr}")
    print(f"Batch Size: {BATCH_SIZE}")
    print("-"*60)

    train_loader, val_loader, _ = create_dataloaders()

    model = MaskCNNBaseline(dropout_rate=0.5).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_model_path = MODEL_DIR / "cnn_baseline_best.pth"

    start_time = time.time()

    for epoch in range(1, epoch + 1):
        epoch_start_time = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        elapsed_time = time.time() - epoch_start_time

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc
            },
            best_model_path
            )
            saved_str = "[★ SAVED BEST]"
        else:
            saved_str = ""
        print(
            f"Epoch [{epoch:02d}/{epoch:02d}] ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc * 100:.2f}% {saved_str}"
        )

    total_time = time.time() - start_time
    print("-" * 60)
    print(f"Training Complete in {total_time / 60:.2f} minutes.")
    print(f"Best Model Checkpoint: {best_model_path.resolve()}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_training(epochs=5, lr=0.001)

