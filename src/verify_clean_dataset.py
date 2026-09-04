from pathlib import Path
from config import CLASS_TO_IDX, PROCESSED_DATA_DIR


def verify_clean_dataset(
    dedup_dir: Path = PROCESSED_DATA_DIR / "deduplicated",
) -> None:
    """Independently verifies file counts and inspects non-image files in the clean dataset."""

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    expected_counts = {"with_mask": 3529, "without_mask": 3718}

    print("=" * 60)
    print("DEDUPLICATED DATASET VERIFICATION")
    print("=" * 60)

    if not dedup_dir.exists():
        print(f"[❌ ERROR] Deduplicated directory not found: {dedup_dir}")
        return

    actual_totals = {}
    non_image_files = []

    for class_name, expected in expected_counts.items():
        class_folder = dedup_dir / class_name
        print(f"\nChecking class: {class_name}")
        print("-" * 60)

        if not class_folder.exists():
            print(f"[❌ ERROR] Class folder missing: {class_folder}")
            continue

        valid_images = []
        class_non_images = []

        for p in class_folder.iterdir():
            if p.is_file():
                if p.suffix.lower() in valid_extensions:
                    valid_images.append(p.name)
                else:
                    class_non_images.append(p.name)

        count = len(valid_images)
        actual_totals[class_name] = count
        diff = count - expected

        print(f" Expected Count : {expected}")
        print(f" Actual Count   : {count}")
        print(f" Difference     : {diff:+d}")

        if class_non_images:
            print(f" ⚠️ Non-image files found ({len(class_non_images)}):")
            for fname in class_non_images:
                print(f"   - {fname}")
            non_image_files.extend(
                [f"{class_name}/{fn}" for fn in class_non_images]
            )

    grand_actual = sum(actual_totals.values())
    grand_expected = sum(expected_counts.values())
    grand_diff = grand_actual - grand_expected

    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)
    print(f" Total Expected : {grand_expected}")
    print(f" Total Actual   : {grand_actual}")
    print(f" Difference     : {grand_diff:+d}")
    print(f" Non-Image Files: {len(non_image_files)}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    verify_clean_dataset()