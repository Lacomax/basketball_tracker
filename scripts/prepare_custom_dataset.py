#!/usr/bin/env python3
"""
Prepare custom basketball dataset combining manual annotations with filtered existing data.

This script:
1. Converts manual annotations from annotations.json to YOLO format
2. Filters existing dataset to keep only small balls (similar to your video)
3. Combines both datasets for optimal training
"""

import json
import os
import sys
import cv2
import shutil
from pathlib import Path
from collections import Counter

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config_loader import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)
config = get_config()

logger.info("=" * 70)
logger.info("CUSTOM BASKETBALL DATASET PREPARATION")
logger.info("=" * 70)

# Configuration
OUTPUT_DATASET = "data/basketball_custom"
MANUAL_ANNOTATIONS = f"{config.get_output_dir()}/annotations.json"
VIDEO_FILE = "data/input_video.mp4"
EXISTING_DATASET = "data/basketball_combined"

# Filtering criteria
MIN_BALL_SIZE = 20  # pixels (diameter)
MAX_BALL_SIZE = 80  # pixels (diameter)

# Create dataset structure
dataset_dir = Path(OUTPUT_DATASET)
for subdir in ['train/images', 'train/labels', 'valid/images', 'valid/labels']:
    (dataset_dir / subdir).mkdir(parents=True, exist_ok=True)

logger.info("Step 1: Converting manual annotations to YOLO format")
logger.info("-" * 70)

# Load manual annotations
if not os.path.exists(MANUAL_ANNOTATIONS):
    logger.error(f"Manual annotations not found: {MANUAL_ANNOTATIONS}")
    sys.exit(1)

with open(MANUAL_ANNOTATIONS, 'r') as f:
    annotations = json.load(f)

# Load video to get frame dimensions
if not os.path.exists(VIDEO_FILE):
    logger.error(f"Video not found: {VIDEO_FILE}")
    sys.exit(1)

cap = cv2.VideoCapture(VIDEO_FILE)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

logger.info(f"Video: {VIDEO_FILE}")
logger.info(f"  Resolution: {width}x{height}")
logger.info(f"  Total frames: {total_frames}")

# Convert annotations to YOLO format
manual_count = 0
frames_with_ball = {k: v for k, v in annotations.items() if v}

logger.info(f"Manual annotations: {len(frames_with_ball)} frames with ball")

for frame_id, ball_data in frames_with_ball.items():
    frame_num = int(frame_id)

    # Read frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()

    if not ret:
        logger.warning(f"Could not read frame {frame_num}")
        continue

    # Get ball info
    center = ball_data.get('center', [0, 0])
    radius = ball_data.get('radius', 15)

    # Convert to YOLO format (normalized)
    cx, cy = center
    diameter = radius * 2

    # Normalize to 0-1
    x_center_norm = cx / width
    y_center_norm = cy / height
    w_norm = diameter / width
    h_norm = diameter / height

    # Save image
    img_filename = f"manual_frame_{frame_num:06d}.jpg"
    img_path = dataset_dir / "train" / "images" / img_filename
    cv2.imwrite(str(img_path), frame)

    # Save label (class 0 = basketball)
    label_filename = f"manual_frame_{frame_num:06d}.txt"
    label_path = dataset_dir / "train" / "labels" / label_filename

    with open(label_path, 'w') as f:
        f.write(f"0 {x_center_norm:.6f} {y_center_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")

    manual_count += 1

cap.release()

logger.info(f"[+] Converted {manual_count} manual annotations to YOLO format")
logger.info("")

# Step 2: Filter existing dataset
logger.info("Step 2: Filtering existing dataset (small balls only)")
logger.info("-" * 70)

existing_train_labels = Path(EXISTING_DATASET) / "train" / "labels"
existing_train_images = Path(EXISTING_DATASET) / "train" / "images"

if not existing_train_labels.exists():
    logger.warning(f"Existing dataset not found: {EXISTING_DATASET}")
    logger.info("Will train only with manual annotations")
else:
    filtered_count = 0
    skipped_count = 0

    for label_file in existing_train_labels.glob("*.txt"):
        # Read label
        with open(label_file, 'r') as f:
            lines = f.readlines()

        # Filter lines (keep only small basketballs)
        filtered_lines = []

        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                w_norm = float(parts[3])
                h_norm = float(parts[4])

                # Convert to pixels (assuming 640x640 training size)
                avg_size = (w_norm + h_norm) / 2 * 640

                # Keep only if it's class 0 (basketball) and small size
                if class_id == 0 and MIN_BALL_SIZE <= avg_size <= MAX_BALL_SIZE:
                    filtered_lines.append(line)

        # If we have valid annotations, copy image and label
        if filtered_lines:
            # Copy image
            img_filename = label_file.stem + label_file.suffix.replace('.txt', '.jpg')
            src_img = existing_train_images / img_filename

            if src_img.exists():
                dst_img = dataset_dir / "train" / "images" / f"filtered_{img_filename}"
                shutil.copy(src_img, dst_img)

                # Write filtered labels
                dst_label = dataset_dir / "train" / "labels" / f"filtered_{label_file.name}"
                with open(dst_label, 'w') as f:
                    f.writelines(filtered_lines)

                filtered_count += len(filtered_lines)
            else:
                skipped_count += 1

    logger.info(f"[+] Filtered {filtered_count} small ball annotations from existing dataset")
    if skipped_count:
        logger.info(f"[!] Skipped {skipped_count} files (missing images)")

logger.info("")

# Step 3: Create validation set (20% of manual annotations)
logger.info("Step 3: Creating validation set")
logger.info("-" * 70)

train_images = list((dataset_dir / "train" / "images").glob("*.jpg"))
num_val = max(1, len(frames_with_ball) // 5)  # 20% for validation

# Move 20% of manual annotations to validation
manual_images = [img for img in train_images if img.name.startswith("manual_")]
val_images = manual_images[:num_val]

for img_path in val_images:
    # Move image
    dst_img = dataset_dir / "valid" / "images" / img_path.name
    shutil.move(str(img_path), str(dst_img))

    # Move label
    label_name = img_path.stem + ".txt"
    src_label = dataset_dir / "train" / "labels" / label_name
    dst_label = dataset_dir / "valid" / "labels" / label_name

    if src_label.exists():
        shutil.move(str(src_label), str(dst_label))

logger.info(f"[+] Created validation set: {len(val_images)} images")
logger.info("")

# Step 4: Create data.yaml
logger.info("Step 4: Creating data.yaml configuration")
logger.info("-" * 70)

data_yaml = dataset_dir / "data.yaml"
yaml_content = f"""# Custom basketball dataset with manual annotations
path: {dataset_dir.absolute()}
train: train/images
val: valid/images

names:
  0: basketball

nc: 1
"""

with open(data_yaml, 'w') as f:
    f.write(yaml_content)

logger.info(f"[+] Created {data_yaml}")
logger.info("")

# Summary
logger.info("=" * 70)
logger.info("DATASET READY!")
logger.info("=" * 70)

# Count final images
train_imgs = list((dataset_dir / "train" / "images").glob("*.jpg"))
val_imgs = list((dataset_dir / "valid" / "images").glob("*.jpg"))
train_labels = list((dataset_dir / "train" / "labels").glob("*.txt"))
val_labels = list((dataset_dir / "valid" / "labels").glob("*.txt"))

logger.info(f"Training set: {len(train_imgs)} images, {len(train_labels)} labels")
logger.info(f"Validation set: {len(val_imgs)} images, {len(val_labels)} labels")
logger.info(f"Total: {len(train_imgs) + len(val_imgs)} images")
logger.info("")

logger.info("Dataset composition:")
logger.info(f"  - Manual annotations: {manual_count}")
logger.info(f"  - Filtered existing: {len(train_imgs) - manual_count + len(val_imgs)}")
logger.info("")

logger.info("Next step:")
logger.info("  Run training script:")
logger.info(f"  python scripts/train_custom_model.py")
logger.info("")
logger.info("=" * 70)
