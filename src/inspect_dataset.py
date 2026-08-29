from collections import Counter
from pathlib import Path
from PIL import Image
from config import RAW_DATA_DIR

def inspect_dataset(data_dir: Path) -> None:
    """Inspect raw Dataset files, verify formats, dimensions and readability."""
    classes = ["with_mask", "without_mask"]

    total_images = 0
    class_count = {}
    format_counts = Counter()
    mode_counts = Counter()
    corrupted_images = []

    min_width, min_height = float("inf"), float("inf")
    max_width, max_height = 0, 0
    total_width, total_height = 0, 0

    print("="*60)
    print("DATASET INSPECTION REPORT")
    print("="*60)

    for class_name in classes:
        class_folder = data_dir/ class_name
        if not class_folder.exists():
            print(f"[❌ERROR] Directory Not Found: {class_folder}")
            continue
        #Collect images files
        image_paths = [
            p
            for p in class_folder.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]

        count = len(image_paths)
        class_count[class_name] = count
        total_images += count

        for img_path in image_paths:
            try:
                # Check image validity & reopen for metadata: 
                with Image.open(img_path) as img:
                    img.verify()

                with Image.open(img_path) as img:
                    format_counts[img.format] +=1
                    mode_counts[img.mode] +=1

                    w, h = img.size
                    min_width, min_height = min(min_width, w), min(min_height, h) 
                    max_width, max_height = max(max_width, w), max(max_height, h)
                    total_width += w
                    total_height += h
            except Exception as e:
                corrupted_images.append((str(img_path), str(e)))

        #Summary Result
    print("1. Class Distribution Summary")
    print("-------------------------------------")
    for class_name, count in class_count.items():
        ratio = (count/total_images*100) if total_images>0 else 0
        print(f" - {class_name:<15}: {count:<6} ({ratio:.2f}%)")
    print(f" -Total Images: {total_images}\n")

    print("2. File Format and Color Modes")
    print(f" -Formats : {dict(format_counts)}")
    print(f" -Color Modes : {dict(mode_counts)}")

    print("3. Images Dimensions (Width x Height)")
    print("--------------------------------------")
    if total_images - len(corrupted_images) > 0:
        valid_count = total_images - len(corrupted_images)
        avg_w = total_width / valid_count
        avg_h = total_height / valid_count
        print(f" -Min Resolution: {min_width} x {min_height}")
        print(f" -Max Resolution: {max_width} x {max_height}")
        print(f" -Avg Resolution: {avg_w:.1f} x {avg_h:.1f}\n")
    else:
        print(" -No valid images proccessed.\n")

    print("4. Unreadable / Corrupted Images")
    print("-----------------------------------")
    print(f" -Corrupted Count: {len(corrupted_images)}")
    if corrupted_images:
        print(" -Corrupted Files:")
        for path, err in corrupted_images:
                print(f"  *{path} -> Error: {err}")
    print("\n==================================================================")

if __name__ == "__main__":
    inspect_dataset(RAW_DATA_DIR)
