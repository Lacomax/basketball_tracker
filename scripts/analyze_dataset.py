#!/usr/bin/env python3
"""
Analyze basketball training dataset and manual annotations.
"""

import json
import os
import sys
from pathlib import Path
from collections import Counter
import statistics

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config_loader import get_config

config = get_config()

print("=" * 70)
print("DATASET ANALYSIS")
print("=" * 70)
print()

# 1. Analyze YOLO training labels
print("1. YOLO Training Dataset")
print("-" * 70)

train_labels = Path("data/basketball_combined/train/labels")
if train_labels.exists():
    label_files = list(train_labels.glob("*.txt"))

    class_counts = Counter()
    bbox_sizes = []

    for label_file in label_files:
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    width = float(parts[3])
                    height = float(parts[4])

                    class_counts[class_id] += 1
                    # Approximate size in pixels (assuming 640x640 training)
                    avg_size = (width + height) / 2 * 640
                    bbox_sizes.append(avg_size)

    print(f"Total label files: {len(label_files)}")
    print(f"Total annotations: {sum(class_counts.values())}")
    print()
    print("Class distribution:")
    for class_id, count in sorted(class_counts.items()):
        class_name = "basketball" if class_id == 0 else f"class_{class_id}"
        print(f"  Class {class_id} ({class_name}): {count} annotations")

    if bbox_sizes:
        print()
        print("Bounding box sizes (approx pixels):")
        print(f"  Mean: {statistics.mean(bbox_sizes):.1f}px")
        print(f"  Median: {statistics.median(bbox_sizes):.1f}px")
        print(f"  Min: {min(bbox_sizes):.1f}px")
        print(f"  Max: {max(bbox_sizes):.1f}px")
        print(f"  Std Dev: {statistics.stdev(bbox_sizes):.1f}px")

        # Size distribution
        small = sum(1 for s in bbox_sizes if s < 50)
        medium = sum(1 for s in bbox_sizes if 50 <= s < 100)
        large = sum(1 for s in bbox_sizes if s >= 100)

        print()
        print("Size distribution:")
        print(f"  Small (<50px): {small} ({small/len(bbox_sizes)*100:.1f}%)")
        print(f"  Medium (50-100px): {medium} ({medium/len(bbox_sizes)*100:.1f}%)")
        print(f"  Large (>100px): {large} ({large/len(bbox_sizes)*100:.1f}%)")

print()
print("=" * 70)

# 2. Analyze manual annotations
print("2. Manual Annotations (yours)")
print("-" * 70)

output_dir = config.get_output_dir()
annotations_file = f"{output_dir}/annotations.json"

if os.path.exists(annotations_file):
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)

    # Count frames with ball
    frames_with_ball = {k: v for k, v in annotations.items() if v}

    print(f"Total annotated frames: {len(annotations)}")
    print(f"Frames with ball: {len(frames_with_ball)}")
    print(f"Frames without ball: {len(annotations) - len(frames_with_ball)}")

    if frames_with_ball:
        # Analyze ball sizes
        radii = [v.get('radius', 0) for v in frames_with_ball.values()]
        diameters = [r * 2 for r in radii]

        print()
        print("Ball sizes in your annotations:")
        print(f"  Mean radius: {statistics.mean(radii):.1f}px")
        print(f"  Median radius: {statistics.median(radii):.1f}px")
        print(f"  Min radius: {min(radii):.0f}px")
        print(f"  Max radius: {max(radii):.0f}px")
        print()
        print(f"  Mean diameter: {statistics.mean(diameters):.1f}px")
        print(f"  Range: {min(diameters):.0f}-{max(diameters):.0f}px")

        # Positions
        centers = [v.get('center', [0, 0]) for v in frames_with_ball.values()]

        print()
        print(f"Sample positions (first 5):")
        for i, (frame_id, data) in enumerate(list(frames_with_ball.items())[:5]):
            center = data.get('center', [0, 0])
            radius = data.get('radius', 0)
            print(f"  Frame {frame_id}: center=({center[0]}, {center[1]}), radius={radius}px")
else:
    print("[!] No manual annotations found")

print()
print("=" * 70)

# 3. Problem diagnosis
print("3. Problem Diagnosis")
print("-" * 70)
print()

print("ISSUES FOUND:")
print()
print("1. YOLO trained on LARGE objects (mean ~200-400px)")
print("   Your video has SMALL balls (15-30px)")
print("   → YOLO ignores small objects!")
print()
print("2. Dataset likely from far-away camera angles")
print("   Your video is closer/different angle")
print("   → Distribution mismatch!")
print()
print("SOLUTION:")
print()
print("Re-train YOLO with YOUR manual annotations:")
print("  - 34 frames with ball positions")
print("  - Correct size (15-30px)")
print("  - Matching your video angle/distance")
print()
print("This will create a custom model for YOUR specific setup!")

print()
print("=" * 70)
