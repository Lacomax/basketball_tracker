#!/usr/bin/env python3
"""
Simple YOLO11 basketball detector training script.
Works with manually downloaded datasets (no Roboflow API needed).

Usage:
1. Download basketball datasets from Roboflow Universe in YOLOv8 format
2. Extract to data/basketball_training/
3. Run: python scripts/train_basketball_detector_simple.py

Dataset structure expected:
data/basketball_training/
├── dataset_1/
│   ├── data.yaml
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   └── valid/
│       ├── images/
│       └── labels/
├── dataset_2/  (optional, for better results)
└── ...
"""

import os
import yaml
import shutil
from pathlib import Path
from ultralytics import YOLO


def find_datasets(base_dir='data/basketball_training'):
    """Find all YOLO datasets in the training directory."""
    datasets = []

    if not os.path.exists(base_dir):
        print(f"   ⚠️  Directorio no existe: {base_dir}")
        return datasets

    items = os.listdir(base_dir)
    if not items:
        print(f"   ⚠️  Directorio vacío: {base_dir}")
        return datasets

    print(f"   Buscando datasets en: {os.path.abspath(base_dir)}")
    print(f"   Encontrados {len(items)} item(s) en el directorio\n")

    for item in items:
        dataset_path = os.path.join(base_dir, item)

        if os.path.isdir(dataset_path):
            print(f"   Verificando: {item}/")

            # Check if it has data.yaml or is a valid YOLO dataset
            data_yaml = os.path.join(dataset_path, 'data.yaml')
            train_images_dir = os.path.join(dataset_path, 'train', 'images')
            train_dir = os.path.join(dataset_path, 'train')

            # Roboflow sometimes creates nested structure, check for that too
            # e.g., dataset_name/dataset_name/train/
            nested_check = os.path.join(dataset_path, item, 'train', 'images')

            has_data_yaml = os.path.exists(data_yaml)
            has_train_images = os.path.exists(train_images_dir)
            has_train_dir = os.path.exists(train_dir)
            has_nested = os.path.exists(nested_check)

            print(f"      - data.yaml: {'✓' if has_data_yaml else '✗'}")
            print(f"      - train/: {'✓' if has_train_dir else '✗'}")
            print(f"      - train/images/: {'✓' if has_train_images else '✗'}")

            # Use nested path if found
            if has_nested and not has_train_images:
                print(f"      - Estructura anidada detectada")
                dataset_path = os.path.join(dataset_path, item)
                train_images_dir = nested_check
                has_train_images = True

            if has_data_yaml or has_train_images:
                datasets.append(dataset_path)
                print(f"      ✅ Dataset válido encontrado\n")
            else:
                print(f"      ❌ No es un dataset YOLO válido")
                print(f"         (Falta data.yaml o train/images/)\n")

                # Show what's actually in this directory
                try:
                    contents = os.listdir(dataset_path)[:5]  # First 5 items
                    print(f"      Contenido: {', '.join(contents)}")
                    if len(os.listdir(dataset_path)) > 5:
                        print(f"      ... y {len(os.listdir(dataset_path)) - 5} más")
                except:
                    pass
                print()
        else:
            print(f"   Ignorando archivo: {item}\n")

    return datasets


def combine_datasets(dataset_paths, output_path='data/basketball_combined'):
    """Combine multiple YOLO datasets into one."""
    if not dataset_paths:
        print("❌ No datasets found to combine")
        return None

    os.makedirs(output_path, exist_ok=True)

    train_images = os.path.join(output_path, 'train', 'images')
    train_labels = os.path.join(output_path, 'train', 'labels')
    val_images = os.path.join(output_path, 'valid', 'images')
    val_labels = os.path.join(output_path, 'valid', 'labels')

    for dir_path in [train_images, train_labels, val_images, val_labels]:
        os.makedirs(dir_path, exist_ok=True)

    print(f"\n🔗 Combining {len(dataset_paths)} dataset(s)...")

    for idx, dataset_path in enumerate(dataset_paths):
        dataset_name = Path(dataset_path).name
        print(f"   Processing: {dataset_name}")

        # Copy train images and labels
        src_train_img = os.path.join(dataset_path, 'train', 'images')
        src_train_lbl = os.path.join(dataset_path, 'train', 'labels')

        if os.path.exists(src_train_img):
            for img in os.listdir(src_train_img):
                shutil.copy2(
                    os.path.join(src_train_img, img),
                    os.path.join(train_images, f"{dataset_name}_{img}")
                )

        if os.path.exists(src_train_lbl):
            for lbl in os.listdir(src_train_lbl):
                shutil.copy2(
                    os.path.join(src_train_lbl, lbl),
                    os.path.join(train_labels, f"{dataset_name}_{lbl}")
                )

        # Copy validation images and labels
        src_val_img = os.path.join(dataset_path, 'valid', 'images')
        src_val_lbl = os.path.join(dataset_path, 'valid', 'labels')

        if os.path.exists(src_val_img):
            for img in os.listdir(src_val_img):
                shutil.copy2(
                    os.path.join(src_val_img, img),
                    os.path.join(val_images, f"{dataset_name}_{img}")
                )

        if os.path.exists(src_val_lbl):
            for lbl in os.listdir(src_val_lbl):
                shutil.copy2(
                    os.path.join(src_val_lbl, lbl),
                    os.path.join(val_labels, f"{dataset_name}_{lbl}")
                )

    train_count = len(os.listdir(train_images))
    val_count = len(os.listdir(val_images))

    print(f"   ✓ Combined: {train_count} train images, {val_count} validation images")

    # Create data.yaml
    data_yaml = {
        'path': os.path.abspath(output_path),
        'train': 'train/images',
        'val': 'valid/images',
        'names': {0: 'basketball'},
        'nc': 1
    }

    yaml_path = os.path.join(output_path, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"   ✓ Created: {yaml_path}")

    return yaml_path


def train_model(data_yaml, epochs=50, batch_size=16):
    """Train YOLO11-L model."""
    print(f"\n🚀 Training YOLO11-L model...")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")

    model = YOLO('yolo11l.pt')

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=640,
        name='basketball_detector_yolo11l',
        patience=10,
        save=True,
        device=0,  # GPU 0, use 'cpu' if no GPU
        workers=8,
        project='runs/basketball',
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
    )

    print(f"\n✅ Training complete!")

    # Validate
    print("\n📊 Validating...")
    metrics = model.val()
    print(f"   mAP50: {metrics.box.map50:.3f}")
    print(f"   mAP50-95: {metrics.box.map:.3f}")

    # Copy model
    os.makedirs('models', exist_ok=True)
    best_model = 'runs/basketball/basketball_detector_yolo11l/weights/best.pt'
    output_model = 'models/basketball_detector_yolo11l.pt'

    if os.path.exists(best_model):
        shutil.copy2(best_model, output_model)
        print(f"\n✓ Model saved: {output_model}")

    return output_model


def main():
    print("=" * 70)
    print("YOLO11 Basketball Detector Training (Simple)")
    print("=" * 70)

    # Find datasets
    print("\n[Step 1/3] Looking for datasets in data/basketball_training/...")
    datasets = find_datasets('data/basketball_training')

    if not datasets:
        print("\n❌ No datasets found!")
        print("\n💡 Soluciones:")
        print("\n1. Descarga con script automático:")
        print("   python scripts/download_roboflow_dataset.py --api-key YOUR_KEY --download-all")
        print("\n2. Descarga manual:")
        print("   a. Ve a: https://universe.roboflow.com/roboflow-100/basketball-detection")
        print("   b. Click 'Download Dataset'")
        print("   c. Selecciona formato: YOLOv8")
        print("   d. Descarga y extrae a: data/basketball_training/")
        print("\n3. Diagnostica el problema:")
        print("   python scripts/diagnose_datasets.py")
        print("\n4. Busca más datasets:")
        print("   https://universe.roboflow.com/search?q=basketball")
        return 1

    print(f"   ✓ Found {len(datasets)} dataset(s):")
    for ds in datasets:
        print(f"     - {Path(ds).name}")

    # Combine datasets
    print("\n[Step 2/3] Combining datasets...")
    data_yaml = combine_datasets(datasets)

    if not data_yaml:
        return 1

    # Train
    print("\n[Step 3/3] Training model...")
    model_path = train_model(data_yaml, epochs=50, batch_size=16)

    print("\n" + "=" * 70)
    print("🎉 Training complete!")
    print("=" * 70)
    print(f"\nModel ready: {model_path}")
    print("\nNext steps:")
    print("1. Test: python scripts/test_basketball_model.py")
    print("2. Use: python scripts/pipeline.py")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
