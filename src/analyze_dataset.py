import hashlib
from collections import Counter
from pathlib import Path
from PIL import Image

# Import root directory setup
from config import RAW_DATA_DIR


def analyze_dataset_quality(
    data_dir: Path, min_dim: int = 50, wide_ratio: float = 2.0, tall_ratio: float = 0.5
) -> None:
    """Analyzes dataset quality focusing on low-res images, extreme aspect ratios, and duplicates."""
    classes = ["with_mask", "without_mask"]
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    small_images = Counter()
    extreme_aspect_ratios = {"very_wide": 0, "very_tall": 0}

    # MD5 hash mapping: md5_hash -> list of file paths
    file_hashes = {}
    duplicate_count = 0

    print("=" * 60)
    print("DATASET QUALITY ANALYSIS REPORT")
    print("=" * 60)

    for class_name in classes:
        class_folder = data_dir / class_name
        if not class_folder.exists():
            print(f"[❌ ERROR] Directory not found: {class_folder}")
            continue

        image_paths = [
            p
            for p in class_folder.iterdir()
            if p.is_file() and p.suffix.lower() in valid_extensions
        ]

        for img_path in image_paths:
            try:
                # 1. Exact Duplicate Detection via MD5 Hash
                with open(img_path, "rb") as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()

                if file_hash in file_hashes:
                    duplicate_count += 1
                    file_hashes[file_hash].append(img_path)
                else:
                    file_hashes[file_hash] = [img_path]

                # 2. Image Resolution & Aspect Ratio Checks
                with Image.open(img_path) as img:
                    w, h = img.size

                    # Check for small images
                    if w < min_dim or h < min_dim:
                        small_images[class_name] += 1

                    # Check for extreme aspect ratios
                    aspect_ratio = w / h
                    if aspect_ratio >= wide_ratio:
                        extreme_aspect_ratios["very_wide"] += 1
                    elif aspect_ratio <= tall_ratio:
                        extreme_aspect_ratios["very_tall"] += 1

            except Exception as e:
                print(f"[⚠️ WARNING] Could not analyze {img_path.name}: {e}")

    # Output Statistics
    print(f"\n1. Small Images (Resolution < {min_dim}x{min_dim} px)")
    print("------------------------------------------------------------")
    total_small = sum(small_images.values())
    for class_name in classes:
        print(f" - {class_name:<15}: {small_images[class_name]}")
    print(f" - Total Small Images: {total_small}\n")

    print(
        f"2. Extreme Aspect Ratios (Wide >= {wide_ratio}, Tall <= {tall_ratio})"
    )
    print("------------------------------------------------------------")
    print(f" - Very Wide Images  : {extreme_aspect_ratios['very_wide']}")
    print(f" - Very Tall Images  : {extreme_aspect_ratios['very_tall']}\n")

    print("3. Duplicate Images (Exact File Matches)")
    print("------------------------------------------------------------")
    print(f" - Exact Duplicates  : {duplicate_count}")
    print("============================================================\n")


if __name__ == "__main__":
    analyze_dataset_quality(RAW_DATA_DIR)