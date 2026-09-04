import hashlib
from pathlib import Path
from config import CLASS_TO_IDX, PROCESSED_DATA_DIR, RAW_DATA_DIR


def get_image_hashes(folder_path: Path) -> dict[str, str]:
    """Computes MD5 hashes for all valid images in a folder.

    Returns dict mapping hash -> filename.
    """
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    hash_to_file = {}

    if not folder_path.exists():
        return hash_to_file

    for p in folder_path.iterdir():
        if p.is_file() and p.suffix.lower() in valid_extensions:
            try:
                with open(p, "rb") as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                hash_to_file[file_hash] = p.name
            except Exception as e:
                print(f"[⚠️ WARNING] Could not hash {p.name}: {e}")

    return hash_to_file


def check_deduplicated_integrity(
    raw_dir: Path = RAW_DATA_DIR,
    dedup_dir: Path = PROCESSED_DATA_DIR / "deduplicated",
) -> None:
    """Compares raw and deduplicated directories using MD5 hashes to identify unverified files."""

    print("=" * 60)
    print("HASH-BASED DEDUPLICATED DATASET INTEGRITY CHECK")
    print("=" * 60)

    for class_name in CLASS_TO_IDX.keys():
        raw_folder = raw_dir / class_name
        dedup_folder = dedup_dir / class_name

        print(f"\nAnalyzing Class: {class_name}")
        print("-" * 60)

        # 1. Gather Hashes
        raw_hash_map = get_image_hashes(raw_folder)
        dedup_hash_map = get_image_hashes(dedup_folder)

        raw_unique_count = len(raw_hash_map)
        dedup_count = len(dedup_hash_map)

        # 2. Hash Set Comparison
        raw_hashes = set(raw_hash_map.keys())
        dedup_hashes = set(dedup_hash_map.keys())

        unverified_hashes = (
            dedup_hashes - raw_hashes
        )  # Hashes in dedup NOT found in raw
        missing_hashes = raw_hashes - dedup_hashes  # Hashes in raw NOT in dedup

        print(f" Raw Directory Total Files      : {len(list(raw_folder.iterdir())) if raw_folder.exists() else 0}")
        print(f" Raw Unique MD5 Hashes          : {raw_unique_count}")
        print(f" Deduplicated Directory Total   : {dedup_count}")
        print(f" Deduplicated Hash Match Raw    : {len(dedup_hashes & raw_hashes)}")
        print(f" Unverified Files (Not in Raw) : {len(unverified_hashes)}")
        print(f" Missing Unique Hashes from Raw : {len(missing_hashes)}")

        if unverified_hashes:
            print(f"\n ⚠️ Unverified Files in 'deduplicated/{class_name}':")
            for h in list(unverified_hashes)[:20]:  # Print up to 20
                print(f"   - Filename: {dedup_hash_map[h]} (Hash: {h[:8]}...)")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    check_deduplicated_integrity()