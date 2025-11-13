#!/usr/bin/env python3
"""
Train a custom YOLO11 model for basketball detection using Roboflow datasets.

This script downloads multiple basketball datasets from Roboflow Universe,
combines them, and fine-tunes YOLO11-L for better detection accuracy.

Usage:
    python scripts/train_basketball_detector.py

Requirements:
    pip install ultralytics roboflow
"""

import os
import yaml
from pathlib import Path
from ultralytics import YOLO


# Roboflow datasets for basketball detection
# You can find more at: https://universe.roboflow.com/search?q=basketball
ROBOFLOW_DATASETS = [
    {
        'workspace': 'basketball-detection',
        'project': 'basketball-ball-detection',
        'version': 1,
        'api_key': None,  # Set via environment variable ROBOFLOW_API_KEY
    },
    # Add more datasets here for better training
    # Example:
    # {
    #     'workspace': 'another-workspace',
    #     'project': 'basketball-tracker',
    #     'version': 2,
    # },
]


def download_roboflow_datasets(output_dir: str = 'data/basketball_training'):
    """
    Download basketball datasets from Roboflow Universe.

    Args:
        output_dir: Directory to save downloaded datasets

    Returns:
        List of dataset paths
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        print("❌ Roboflow not installed. Run: pip install roboflow")
        return []

    api_key = os.environ.get('ROBOFLOW_API_KEY')
    if not api_key:
        print("\n⚠️  ROBOFLOW_API_KEY not set in environment")
        print("To get your API key:")
        print("1. Go to https://universe.roboflow.com/")
        print("2. Sign in / create account")
        print("3. Go to your profile settings")
        print("4. Copy your API key")
        print("5. Set it: export ROBOFLOW_API_KEY='your_key_here'")
        return []

    os.makedirs(output_dir, exist_ok=True)
    rf = Roboflow(api_key=api_key)

    dataset_paths = []

    for idx, dataset_config in enumerate(ROBOFLOW_DATASETS):
        print(f"\n📦 Downloading dataset {idx+1}/{len(ROBOFLOW_DATASETS)}...")
        print(f"   Workspace: {dataset_config['workspace']}")
        print(f"   Project: {dataset_config['project']}")
        print(f"   Version: {dataset_config['version']}")

        try:
            project = rf.workspace(dataset_config['workspace']).project(dataset_config['project'])
            dataset = project.version(dataset_config['version']).download(
                model_format="yolov8",  # YOLO11 uses same format as YOLOv8
                location=f"{output_dir}/dataset_{idx+1}"
            )
            dataset_paths.append(dataset.location)
            print(f"   ✓ Downloaded to: {dataset.location}")
        except Exception as e:
            print(f"   ❌ Failed to download: {e}")
            continue

    return dataset_paths


def combine_datasets(dataset_paths: list, output_path: str = 'data/basketball_combined'):
    """
    Combine multiple YOLO datasets into one.

    Args:
        dataset_paths: List of paths to YOLO datasets
        output_path: Path for combined dataset

    Returns:
        Path to combined data.yaml
    """
    import shutil

    os.makedirs(output_path, exist_ok=True)

    train_images = os.path.join(output_path, 'train', 'images')
    train_labels = os.path.join(output_path, 'train', 'labels')
    val_images = os.path.join(output_path, 'valid', 'images')
    val_labels = os.path.join(output_path, 'valid', 'labels')

    for dir_path in [train_images, train_labels, val_images, val_labels]:
        os.makedirs(dir_path, exist_ok=True)

    print("\n🔗 Combining datasets...")

    for dataset_path in dataset_paths:
        # Copy train images and labels
        src_train_img = os.path.join(dataset_path, 'train', 'images')
        src_train_lbl = os.path.join(dataset_path, 'train', 'labels')

        if os.path.exists(src_train_img):
            for img in os.listdir(src_train_img):
                shutil.copy2(
                    os.path.join(src_train_img, img),
                    os.path.join(train_images, f"{Path(dataset_path).name}_{img}")
                )

        if os.path.exists(src_train_lbl):
            for lbl in os.listdir(src_train_lbl):
                shutil.copy2(
                    os.path.join(src_train_lbl, lbl),
                    os.path.join(train_labels, f"{Path(dataset_path).name}_{lbl}")
                )

        # Copy validation images and labels
        src_val_img = os.path.join(dataset_path, 'valid', 'images')
        src_val_lbl = os.path.join(dataset_path, 'valid', 'labels')

        if os.path.exists(src_val_img):
            for img in os.listdir(src_val_img):
                shutil.copy2(
                    os.path.join(src_val_img, img),
                    os.path.join(val_images, f"{Path(dataset_path).name}_{img}")
                )

        if os.path.exists(src_val_lbl):
            for lbl in os.listdir(src_val_lbl):
                shutil.copy2(
                    os.path.join(src_val_lbl, lbl),
                    os.path.join(val_labels, f"{Path(dataset_path).name}_{lbl}")
                )

    train_count = len(os.listdir(train_images))
    val_count = len(os.listdir(val_images))

    print(f"   ✓ Combined dataset: {train_count} train images, {val_count} val images")

    # Create data.yaml
    data_yaml = {
        'path': os.path.abspath(output_path),
        'train': 'train/images',
        'val': 'valid/images',
        'names': {
            0: 'basketball'
        },
        'nc': 1  # Number of classes
    }

    yaml_path = os.path.join(output_path, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"   ✓ Created data.yaml: {yaml_path}")

    return yaml_path


def train_basketball_detector(
    data_yaml: str,
    model_size: str = 'yolo11l.pt',
    epochs: int = 50,
    batch_size: int = 16,
    imgsz: int = 640,
    output_name: str = 'basketball_detector'
):
    """
    Train YOLO11 model for basketball detection.

    Args:
        data_yaml: Path to dataset configuration
        model_size: YOLO model to use (yolo11n/s/m/l/x)
        epochs: Number of training epochs
        batch_size: Batch size for training
        imgsz: Image size for training
        output_name: Name for the output model
    """
    print(f"\n🚀 Starting training with {model_size}...")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Image size: {imgsz}")

    # Load pre-trained model
    model = YOLO(model_size)

    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        name=output_name,
        patience=10,  # Early stopping patience
        save=True,
        device=0,  # Use GPU 0 (use 'cpu' if no GPU)
        workers=8,
        project='runs/basketball',
        # Data augmentation
        hsv_h=0.015,  # Hue augmentation
        hsv_s=0.7,    # Saturation augmentation
        hsv_v=0.4,    # Value augmentation
        degrees=10.0,  # Rotation augmentation
        translate=0.1,  # Translation augmentation
        scale=0.5,     # Scale augmentation
        flipud=0.0,    # No vertical flip (ball doesn't flip vertically)
        fliplr=0.5,    # Horizontal flip
        mosaic=1.0,    # Mosaic augmentation
    )

    print(f"\n✅ Training completed!")
    print(f"   Best model: runs/basketball/{output_name}/weights/best.pt")
    print(f"   Last model: runs/basketball/{output_name}/weights/last.pt")

    # Test on validation set
    print("\n📊 Validating model...")
    metrics = model.val()
    print(f"   mAP50: {metrics.box.map50:.3f}")
    print(f"   mAP50-95: {metrics.box.map:.3f}")

    # Copy best model to models directory
    import shutil
    os.makedirs('models', exist_ok=True)
    best_model_path = f'runs/basketball/{output_name}/weights/best.pt'
    output_model_path = f'models/{output_name}.pt'

    if os.path.exists(best_model_path):
        shutil.copy2(best_model_path, output_model_path)
        print(f"\n✓ Model saved to: {output_model_path}")
        print(f"\nTo use this model, update src/utils/ball_detection.py:")
        print(f"   _yolo_model = YOLO('{output_model_path}')")

    return output_model_path


def main():
    """Main training pipeline."""
    print("=" * 70)
    print("YOLO11 Basketball Detector Training")
    print("=" * 70)

    # Step 1: Download datasets from Roboflow
    print("\n[Step 1/3] Downloading datasets from Roboflow Universe...")
    dataset_paths = download_roboflow_datasets()

    if not dataset_paths:
        print("\n⚠️  No datasets downloaded.")
        print("\nAlternative: Manually download datasets from Roboflow Universe")
        print("1. Go to https://universe.roboflow.com/search?q=basketball")
        print("2. Find datasets with 'basketball' or 'ball' annotations")
        print("3. Download in YOLOv8 format")
        print("4. Place in data/basketball_training/")
        print("5. Update ROBOFLOW_DATASETS in this script with paths")
        return

    # Step 2: Combine datasets
    print("\n[Step 2/3] Combining datasets...")
    data_yaml = combine_datasets(dataset_paths)

    # Step 3: Train model
    print("\n[Step 3/3] Training YOLO11-L model...")
    model_path = train_basketball_detector(
        data_yaml=data_yaml,
        model_size='yolo11l.pt',  # Using large model for better accuracy
        epochs=50,
        batch_size=16,
        imgsz=640,
        output_name='basketball_detector_yolo11l'
    )

    print("\n" + "=" * 70)
    print("🎉 Training complete!")
    print("=" * 70)
    print(f"\nYour custom basketball detector is ready: {model_path}")
    print("\nNext steps:")
    print("1. Test the model: python scripts/test_basketball_model.py")
    print("2. Use in pipeline: The detector will automatically use the custom model")
    print("3. Generate video: python scripts/pipeline.py")


if __name__ == '__main__':
    main()
