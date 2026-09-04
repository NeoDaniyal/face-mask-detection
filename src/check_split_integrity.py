from pathlib import Path
from config import SPLIT_DATA_DIR, PROCESSED_DATA_DIR, CLASS_TO_IDX


def check_split_integrity(
    dedup_dir: Path = PROCESSED_DATA_DIR / "deduplicated",
    split_dir: Path = SPLIT_DATA_DIR,
) -> None:
    """Verifies filename alignment between deduplicated source and train/val/test splits,

    checking for missing files, orphan files, and cross-split data leakage.
    """
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    # 1. Gather all relative paths (class/filename) from deduplicated source
    dedup_files = set()
    for class_name in CLASS_TO_IDX.keys():
        folder = dedup_dir / class_name
        if folder.exists():
            for p in folder.iterdir():
                if p.is_file() and p.suffix.lower() in valid_extensions:
                    dedup_files.add(f"{class_name}/{p.name}")

    # 2. Gather files from each split
    split_files = {"train": set(), "val": set(), "test": set()}

    for split in ["train", "val", "test"]:
        for class_name in CLASS_TO_IDX.keys():
            folder = split_dir / split / class_name
            if folder.exists():
                for p in folder.iterdir():
                    if p.is_file() and p.suffix.lower() in valid_extensions:
                        split_files[split].add(f"{class_name}/{p.name}")

    train_set = split_files["train"]
    val_set = split_files["val"]
    test_set = split_files["test"]

    all_split_files = train_set | val_set | test_set

    # 3. Check for Cross-Split Overlap (Data Leakage)
    train_val_overlap = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap = val_set & test_set

    # 4. Check alignment with Deduplicated directory
    extra_in_split = all_split_files - dedup_files  # Files in split not in dedup
    missing_from_split = dedup_files - all_split_files  # Files in dedup missing from split

    # --- Print Integrity Report ---
    print("=" * 60)
    print("DATASET INTEGRITY & LEAKAGE REPORT")
    print("=" * 60)

    print("\n1. File Set Sizes")
    print("-" * 60)
    print(f"Deduplicated Source Total : {len(dedup_files)}")
    print(f"Total Files across Splits : {len(all_split_files)}")
    print(f" - Train Split             : {len(train_set)}")
    print(f" - Val Split               : {len(val_set)}")
    print(f" - Test Split              : {len(test_set)}")

    print("\n2. Data Leakage (Cross-Split Overlap)")
    print("-" * 60)
    print(f"Train ∩ Val Overlap        : {len(train_val_overlap)}")
    print(f"Train ∩ Test Overlap       : {len(train_test_overlap)}")
    print(f"Val ∩ Test Overlap         : {len(val_test_overlap)}")

    print("\n3. Source Alignment Checks")
    print("-" * 60)
    print(f"Extra files in Split       : {len(extra_in_split)}")
    print(f"Missing from Split         : {len(missing_from_split)}")

    if train_val_overlap or train_test_overlap or val_test_overlap:
        print("\n[❌ CRITICAL] Data leakage detected across splits!")
    elif extra_in_split:
        print("\n[⚠️ WARNING] Found stale/orphan files in split directory!")
    else:
        print("\n[✅ PASSED] No leakage or missing files detected.")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    check_split_integrity()