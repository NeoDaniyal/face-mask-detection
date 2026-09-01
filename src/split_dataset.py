from collections import Counter
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split
from config import RANDOM_SEED, PROCCESSED_DATA_DIR

def create_dataset_split(
        processed_dir: Path, seed: int= RANDOM_SEED
)->None:
    """Performs a 70/15/15 stratified train/val/test split on deduplicated images."""
    dedup_dir = processed_dir/"deduplicated"
    split_dir = processed_dir/"split"

    if not dedup_dir.exists():
        print(f"[❌ERROR] Deduplicate directory not found at: {dedup_dir}")
        return
    classes = ["with_mask", "without_mask"]
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    filepaths = []
    labels = []
    for class_name in classes:
            class_folder = dedup_dir / class_name
            if not class_folder.exists():
                print(f"[❌ ERROR] Source Directory not found: {class_folder}")
                continue
            for p in class_folder.iterdir():
                if p.is_file() and p.suffix.lower() in valid_extensions:
                     filepaths.append(p)
                     labels.append(class_name)
    total_images = len(filepaths)
    if total_images == 0:
         print("[❌ERROR] No image found to split.")
         return
    #1. Temporarily split data: 70%(Train), 30%(Temp->(Test + Val))
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
         filepaths,
         labels,
         test_size=0.30,
         random_state=seed,
         stratify=labels
    )
    #2. Split 30%(Temp) data into 15%(Val) & 15%(Test)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
         temp_paths,
         temp_labels,
         test_size=0.50,
         random_state=seed,
         stratify=temp_labels
    )
    #Helper mapping for desk copying
    splits = {
         "trian": (train_paths, train_labels),
         "val": (val_paths, val_labels),
         "test": (test_paths, test_labels),
    }
    #Copy every file to its respective destination
    for split_name, (path, lbls) in splits.items():
         for src_path, label in zip(path, lbls):
            dest_dir = split_dir/ split_name/ label
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dest_dir/src_path.name)
    print("="*60)
    print("DATASET SPLIT REPORT")
    print("="*60)

    print("\n1. Overall Distribution")
    print("-"*60)
    print(f"Total Images: {total_images}")
    print(f"Train :{len(train_paths):<6} ({len(train_paths)/total_images*100:.2f}%)")
    print(f"Validation :{len(val_paths):<6} ({len(val_paths)/total_images*100:.2f}%)")
    print(f"Test :{len(test_paths):<6} ({len(test_paths)/total_images*100:.2f}%)")

    print("\n2. Class Distribution")
    print("-"*60)

    for split_name, (path, lbls) in splits.items():
         print(f"\n{split_name.upper()}")
         counts = Counter(lbls)
         split_total = len(lbls)
         for class_name in classes:
              c_count = counts[class_name]
              ratio = (c_count/split_total*100) if split_total>0 else 0
              print(f"-{class_name:<15}: {c_count:<6} ({ratio:.2f}%)")
    print("\n3.Random Seed")
    print("-"*60)
    print(f"Seed: {seed}")
    print("="*60 + "\n")

if __name__ == "__main__":
     create_dataset_split(PROCCESSED_DATA_DIR, seed=RANDOM_SEED)
