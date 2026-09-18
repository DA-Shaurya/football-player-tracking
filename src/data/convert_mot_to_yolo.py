from pathlib import Path
import shutil


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SEQ_DIR = Path(
    "data/raw/SoccerNet/tracking-2023/train/train/SNMOT-060"
)

OUTPUT_DIR = Path(
    "data/processed/yolo_test"
)

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080


# --------------------------------------------------
# Create YOLO directory structure
# --------------------------------------------------

images_dir = OUTPUT_DIR / "images"
labels_dir = OUTPUT_DIR / "labels"

images_dir.mkdir(parents=True, exist_ok=True)
labels_dir.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Read ground-truth annotations
# --------------------------------------------------

annotations = {}

gt_path = SEQ_DIR / "gt" / "gt.txt"

with open(gt_path, "r") as f:
    for line in f:
        parts = line.strip().split(",")

        if len(parts) < 6:
            continue

        frame_id = int(parts[0])

        x = float(parts[2])
        y = float(parts[3])
        w = float(parts[4])
        h = float(parts[5])

        # YOLO uses center coordinates
        center_x = x + w / 2
        center_y = y + h / 2

        # Normalize
        center_x /= IMAGE_WIDTH
        center_y /= IMAGE_HEIGHT
        w /= IMAGE_WIDTH
        h /= IMAGE_HEIGHT

        # One class: player
        yolo_line = (
            f"0 {center_x:.6f} {center_y:.6f} "
            f"{w:.6f} {h:.6f}"
        )

        annotations.setdefault(frame_id, []).append(yolo_line)


# --------------------------------------------------
# Convert frames
# --------------------------------------------------

image_dir = SEQ_DIR / "img1"

converted = 0

for frame_id, labels in annotations.items():

    image_name = f"{frame_id:06d}.jpg"
    source_image = image_dir / image_name

    if not source_image.exists():
        continue

    # Copy image
    destination_image = images_dir / image_name
    shutil.copy2(source_image, destination_image)

    # Write YOLO labels
    label_path = labels_dir / f"{frame_id:06d}.txt"

    with open(label_path, "w") as f:
        f.write("\n".join(labels))

    converted += 1


print(f"Converted frames: {converted}")
print(f"Images: {images_dir}")
print(f"Labels: {labels_dir}")