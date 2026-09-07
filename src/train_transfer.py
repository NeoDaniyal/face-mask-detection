import json
import time
import torch
import torch.nn as nn
from torch.optim import Adam

from config import MODEL_DIR, OUTPUT_DIR, RANDOM_SEED
from dataloader import create_dataloaders
from model_transfer import MaskResNet18

def run_transfer_training(epoch: int=5, lr: float=1e-4)->None:
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("="*60)
    print("STARTING RESNET-18 TRANSFER LEARNING PIPELINE")
    print("="*60)

    train_loader, val_loader, _ = create_dataloaders()
    model = MaskResNet18(freeze_backbone=False, dropout_rate=0.3).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    best_model_path = MODEL_DIR/ "resnet18_best.pth"
    history_path = OUTPUT_DIR/ "history_resnet18.json"


    history = {
        "epochs": list(range(1, epoch+1)),
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
    start_time = time.time()

    for epoch in range(1, epoch+1):
        epoch_start = time.time()

        model.train()
        train_loss, train_correct, train_total = 0.0,0,0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).unsequeeze(1).float()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            train_loss += loss.item() * batch_size
            preds = (outputs >= 0.0).float()
            train_correct += (preds == labels).sum().item()
            train_total += batch_size

        train_loss /= train_total
        train_acc = train_correct / train_total

        model.eval()
        val_loss, val_correct, val_total = 0.0,0,0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).unsequeeze(1).float()
                outputs = model(images)
                loss = criterion(outputs, labels)

                batch_size = images.size(0)
                val_loss += loss.item() * batch_size
                preds = (outputs >= 0.0).float()
                val_correct += (preds == labels).sum().item()
                val_total += batch_size
        val_loss /= train_total
        val_acc = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - epoch_start

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_selection_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc
            },
            best_model_path,
            )
            saved_str = "[★ SAVED BEST]"
        else:
            saved_str = ""

        print(
        f"Epoch [{epoch:02d}/{epoch:02d}] ({elapsed:.1f}s) | "
        f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc * 100:.2f}% | "
        f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc * 100:.2f}% {saved_str}"
        )
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)

    total_time = time.time() - start_time
    print("-" * 60)
    print(f"Transfer Learning Complete in {total_time / 60:.2f} minutes.")
    print(f"Best Model Saved to: {best_model_path.resolve()}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_transfer_training(epoch=5, lr=1e-4)