from collections import defaultdict
import hashlib
from pathlib import Path

# Import root directory setup
from config import RAW_DATA_DIR


def analyze_duplicates_detailed(data_dir: Path) -> None:
    """Investigates exact duplicates in the dataset and checks for cross-class leakage."""
    classes = ["with_mask", "without_mask"]
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    # Map: md5_hash -> list of (class_name, file_path)
    hash_map = defaultdict(list)

    print("=" * 60)
    print("DUPLICATE INVESTIGATION REPORT")
    print("=" * 60)

    # 1. Collect MD5 Hashes and File References
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
                with open(img_path, "rb") as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                hash_map[file_hash].append((class_name, img_path))
            except Exception as e:
                print(f"[⚠️ WARNING] Could not read {img_path.name}: {e}")

    # 2. Categorize Duplicate Groups
    duplicate_groups = {h: items for h, items in hash_map.items() if len(items) > 1}
    total_duplicate_files = sum(len(items) - 1 for items in duplicate_groups.values())

    same_class_with_mask = 0
    same_class_without_mask = 0
    cross_class_groups = []

    for file_hash, items in duplicate_groups.items():
        classes_in_group = {item[0] for item in items}

        if len(classes_in_group) > 1:
            # Hash exists in BOTH with_mask and without_mask
            cross_class_groups.append(items)
        else:
            # All duplicates in group share the SAME class
            group_class = list(classes_in_group)[0]
            redundant_count = len(items) - 1
            if group_class == "with_mask":
                same_class_with_mask += redundant_count
            else:
                same_class_without_mask += redundant_count

    # 3. Print Results Summary
    print("\n1. Duplicate Overview")
    print("------------------------------------------------------------")
    print(f" Total Duplicate Groups : {len(duplicate_groups)}")
    print(f" Total Duplicate Files  : {total_duplicate_files}\n")

    print("2. Same-Class Duplicates")
    print("------------------------------------------------------------")
    print(f" - with_mask            : {same_class_with_mask} extra files")
    print(f" - without_mask         : {same_class_without_mask} extra files\n")

    print("3. Cross-Class Duplicates (🚨 Conflict Check)")
    print("------------------------------------------------------------")
    print(f" - Cross-Class Groups   : {len(cross_class_groups)}\n")

    if cross_class_groups:
        print("EXAMPLES OF CROSS-CLASS CONFLICTS:")
        print("------------------------------------------------------------")
        for idx, group in enumerate(cross_class_groups[:5], 1):
            print(f"Group {idx}:")
            for class_name, path in group:
                print(f"  [{class_name:<12}] {path.relative_to(data_dir.parent)}")
            print()
    else:
        print("✅ No cross-class duplicates found! All duplicates are strictly intra-class.")

    print("============================================================\n")


if __name__ == "__main__":
    analyze_duplicates_detailed(RAW_DATA_DIR)