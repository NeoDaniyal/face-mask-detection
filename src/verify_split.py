from collections import Counter
from pathlib import Path

from config import CLASS_TO_IDX, SPLIT_DATA_DIR


def verify_split_directory(split_base_dir: Path = SPLIT_DATA_DIR) -> None:
    """Independently counts and reports file occurrences across dataset splits."""

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    splits = ["train", "val", "test"]

    print("=" * 60)
    print("INDEPENDENT SPLIT DIRECTORY VERIFICATION")
    print("=" * 60)

    grand_total = 0

    for split in splits:
        split_dir = split_base_dir / split
        print(f"\n{split.upper()}")
        print("-" * 60)

        if not split_dir.exists():
            print(f"[❌ ERROR] Directory not found: {split_dir}")
            continue

        split_total = 0
        counts = Counter()

        for class_name in CLASS_TO_IDX.keys():
            class_folder = split_dir / class_name
            if not class_folder.exists():
                counts[class_name] = 0
                continue

            # Count all matching valid images
            files = [
                p
                for p in class_folder.iterdir()
                if p.is_file() and p.suffix.lower() in valid_extensions
            ]
            count = len(files)
            counts[class_name] = count
            split_total += count

            print(f" - {class_name:<15}: {count}")

        print(f" Total {split.title():<12}: {split_total}")
        grand_total += split_total

    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)
    print(f" Disk Total Found  : {grand_total}")
    print(" Expected Total    : 7247")
    print(f" Discrepancy       : {grand_total - 7247:+d} images")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    verify_split_directory()