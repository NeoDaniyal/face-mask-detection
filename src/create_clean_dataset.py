import hashlib
from pathlib import Path
import shutil

from config import PROCCESSED_DATA_DIR, RAW_DATA_DIR

def create_deduplicated_dataset(raw_dir: Path, output_base_dir: Path) -> None:
    """"Copies from raw processed/deduplicate images based on MD5 content hashes."""

    classes = ["with_mask", "without_mask"]
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    target_dir = output_base_dir / "deduplicated"

    raw_counts = {}
    clean_counts = {}
    duplicates_removed = {}

    seen_hashes = set()

    print("="*60)
    print("CLEAN DATASET CREATION REPORT!")
    print("="*60)

    for class_name in classes:
        src_folder = raw_dir / class_name
        dest_folder = target_dir / class_name
        if not src_folder.exists():
            print(f"[❌ ERROR] Source Directory not found: {src_folder}")
            continue

        dest_folder.mkdir(parents=True, exist_ok=True)
        image_paths = [
                    p
                    for p in src_folder.iterdir()
                    if p.is_file() and p.suffix.lower() in valid_extensions
                ]
        raw_counts[class_name] = len(image_paths)
        copied_count = 0
        duplicates_count = 0

        for img_path in image_paths:
            try:
                # comment: Calculate byte level MD5 content hash
                with open(img_path, "rb") as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()

                if file_hash not in seen_hashes:
                    seen_hashes.add(file_hash)
                    dest_path = dest_folder / img_path.name
                    shutil.copy2(img_path, dest_path)
                    copied_count+=1
                else:
                    duplicates_count+=1
            except Exception as e:
                print(f"[⚠️ WARNING] Failed to process {img_path.name}: {e}")
            # end try
        clean_counts[class_name] = copied_count
        duplicates_removed[class_name] = duplicates_count
    total_raw = sum(raw_counts.values())
    total_duplicates = sum(duplicates_removed.values())
    total_clean = sum(clean_counts.values())

    print("\n1. Original Dataset (raw)")
    print("------------------------------------------------------------")
    for class_name, count in raw_counts.items():
        print(f" - {class_name:<15}: {count}")
    print(f" - Total Raw Images : {total_raw}\n")

    print("2. Duplicates Removed")
    print("------------------------------------------------------------")
    for class_name, count in duplicates_removed.items():
        print(f" - {class_name:<15}: {count}")
    print(f" - Total Duplicates : {total_duplicates}\n")

    print("3. Clean Dataset (deduplicated)")
    print("------------------------------------------------------------")
    for class_name, count in clean_counts.items():
        ratio = (count / total_clean * 100) if total_clean > 0 else 0
        print(f" - {class_name:<15}: {count:<6} ({ratio:.2f}%)")
    print(f" - Total Clean      : {total_clean}\n")

    print("4. Output Location")
    print("------------------------------------------------------------")
    print(f" - Directory        : {target_dir}")
    print("============================================================\n")


if __name__ == "__main__":
    create_deduplicated_dataset(RAW_DATA_DIR, PROCCESSED_DATA_DIR)