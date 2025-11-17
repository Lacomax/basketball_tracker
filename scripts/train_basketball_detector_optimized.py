#!/usr/bin/env python3
"""
OPTIMIZED YOLO11 Basketball Detector Training
Optimized for: Windows + Conda + AMD Ryzen 9 + NVIDIA RTX GPU

This script provides maximum performance using:
- Mixed precision training (FP16) for RTX tensor cores
- Optimized batch sizes and workers for Windows
- Smart caching and memory management
- GPU-optimized data loading

Usage:
    python scripts/train_basketball_detector_optimized.py [--preset balanced|max_performance|memory_efficient]
"""

import os
import sys
import yaml
import shutil
import torch
import psutil
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime


def print_gpu_info():
    """Print GPU information and capabilities."""
    if not torch.cuda.is_available():
        print("❌ CUDA not available! Please install PyTorch with CUDA support:")
        print("   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia")
        return False

    print("\n" + "="*70)
    print("GPU INFORMATION")
    print("="*70)
    print(f"CUDA Available: ✓")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1024**3
        print(f"\nGPU {i}: {props.name}")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Total Memory: {memory_gb:.1f} GB")
        print(f"  Multi-Processors: {props.multi_processor_count}")

        # Check for tensor cores (Compute Capability >= 7.0 for Volta, >= 7.5 for Turing/Ampere)
        if props.major >= 7:
            print(f"  Tensor Cores: ✓ (Mixed Precision Supported)")
        else:
            print(f"  Tensor Cores: ✗ (FP16 training not recommended)")

    # Memory info
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    cached = torch.cuda.memory_reserved(0) / 1024**3
    print(f"\nCurrent GPU 0 Memory:")
    print(f"  Allocated: {allocated:.2f} GB")
    print(f"  Cached: {cached:.2f} GB")

    return True


def print_system_info():
    """Print system information."""
    print("\n" + "="*70)
    print("SYSTEM INFORMATION")
    print("="*70)
    print(f"CPU: {psutil.cpu_count(logical=False)} cores ({psutil.cpu_count(logical=True)} threads)")
    print(f"RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    print(f"Available RAM: {psutil.virtual_memory().available / 1024**3:.1f} GB")
    print(f"Python: {sys.version.split()[0]}")


def load_config(preset='balanced'):
    """Load performance configuration."""
    config_path = 'config_performance.yaml'

    if not os.path.exists(config_path):
        print(f"⚠️  Config file not found: {config_path}")
        print("Using default settings...")
        return get_default_config(preset)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Apply preset if specified
    if preset and preset in config.get('presets', {}):
        preset_config = config['presets'][preset]
        config['training'].update(preset_config)
        print(f"✓ Using preset: {preset}")

    return config


def get_default_config(preset='balanced'):
    """Get default configuration if config file doesn't exist."""
    presets = {
        'max_performance': {
            'batch_size': 48,
            'workers': 8,
            'cache': True,
            'imgsz': 640,
        },
        'balanced': {
            'batch_size': 24,
            'workers': 6,
            'cache': True,
            'imgsz': 640,
        },
        'memory_efficient': {
            'batch_size': 12,
            'workers': 4,
            'cache': 'disk',
            'imgsz': 512,
        }
    }

    return {
        'training': presets.get(preset, presets['balanced']),
        'gpu': {'use_fp16': True, 'amp': True},
        'models': {'training_model': 'yolo11l.pt'}
    }


def find_datasets(base_dir='data/basketball_training'):
    """Find all YOLO datasets in the training directory."""
    datasets = []

    if not os.path.exists(base_dir):
        return datasets

    for item in os.listdir(base_dir):
        dataset_path = os.path.join(base_dir, item)

        if os.path.isdir(dataset_path):
            data_yaml = os.path.join(dataset_path, 'data.yaml')
            train_dir = os.path.join(dataset_path, 'train', 'images')

            if os.path.exists(data_yaml) or os.path.exists(train_dir):
                datasets.append(dataset_path)

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
        print(f"   [{idx+1}/{len(dataset_paths)}] {dataset_name}")

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

    train_count = len(os.listdir(train_images)) if os.path.exists(train_images) else 0
    val_count = len(os.listdir(val_images)) if os.path.exists(val_images) else 0

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


def train_model(data_yaml, config, epochs=100, model_name='basketball_detector_rtx'):
    """Train YOLO model with optimized settings."""
    training_config = config['training']
    gpu_config = config.get('gpu', {})
    model_config = config.get('models', {})

    batch_size = training_config.get('batch_size', 24)
    workers = training_config.get('workers', 6)
    imgsz = training_config.get('imgsz', 640)
    cache = training_config.get('cache', True)
    persistent_workers = training_config.get('persistent_workers', True)

    model_path = model_config.get('training_model', 'yolo11l.pt')

    print("\n" + "="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Image Size: {imgsz}")
    print(f"Workers: {workers}")
    print(f"Cache: {cache}")
    print(f"Persistent Workers: {persistent_workers}")
    print(f"Mixed Precision (AMP): {gpu_config.get('amp', True)}")
    print(f"Half Precision (FP16): {gpu_config.get('use_fp16', True)}")
    print("="*70)

    # Estimate VRAM usage
    # Rough estimate: batch_size * imgsz^2 * 4 bytes * 3 (image + gradients + optimizer)
    estimated_vram = (batch_size * imgsz * imgsz * 4 * 3) / 1024**3
    print(f"\nEstimated VRAM usage: {estimated_vram:.1f} GB")

    if torch.cuda.is_available():
        available_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Available VRAM: {available_vram:.1f} GB")

        if estimated_vram > available_vram * 0.9:
            print("⚠️  Warning: Estimated VRAM usage is close to maximum!")
            print("   Consider reducing batch_size if you encounter OOM errors.")

    input("\nPress ENTER to start training...")

    # Load model
    print(f"\n📦 Loading model: {model_path}")
    model = YOLO(model_path)

    # Enable cuDNN benchmarking for faster training
    if gpu_config.get('cudnn_benchmark', True):
        torch.backends.cudnn.benchmark = True

    # Training arguments
    train_args = {
        'data': data_yaml,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': imgsz,
        'name': model_name,
        'patience': 20,  # Early stopping patience
        'save': True,
        'save_period': 10,  # Save checkpoint every 10 epochs
        'device': 0,  # GPU 0
        'workers': workers,
        'project': 'runs/basketball',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'AdamW',  # AdamW is generally better than SGD
        'verbose': True,
        'seed': 42,  # Reproducibility
        'deterministic': False,  # True = reproducible but slower
        'single_cls': True,  # Single class (basketball)
        'rect': False,  # Rectangular training (can be faster but may reduce accuracy)
        'cos_lr': True,  # Cosine learning rate scheduler
        'close_mosaic': training_config.get('close_mosaic', 10),
        'amp': gpu_config.get('amp', True),  # Automatic Mixed Precision
        'fraction': 1.0,  # Use 100% of dataset
        'profile': False,  # Enable for profiling (slower)
        'freeze': None,  # Layers to freeze (None = train all)
        'lr0': 0.01,  # Initial learning rate
        'lrf': 0.01,  # Final learning rate (lr0 * lrf)
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,  # Box loss gain
        'cls': 0.5,  # Class loss gain
        'dfl': 1.5,  # DFL loss gain
        # Data augmentation (tuned for basketball)
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 10.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,  # No vertical flip
        'fliplr': 0.5,  # Horizontal flip
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0,
    }

    # Add cache setting
    if cache:
        train_args['cache'] = cache

    # Add persistent_workers (requires workers > 0)
    if persistent_workers and workers > 0:
        train_args['persistent_workers'] = True

    # Start training
    start_time = datetime.now()
    print(f"\n🚀 Training started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    try:
        results = model.train(**train_args)

        end_time = datetime.now()
        duration = end_time - start_time

        print("\n" + "="*70)
        print("✅ TRAINING COMPLETED!")
        print("="*70)
        print(f"Duration: {duration}")
        print(f"Best model: runs/basketball/{model_name}/weights/best.pt")
        print(f"Last model: runs/basketball/{model_name}/weights/last.pt")

        # Validate
        print("\n📊 Validating best model...")
        best_model = YOLO(f'runs/basketball/{model_name}/weights/best.pt')
        metrics = best_model.val()

        print(f"\nValidation Results:")
        print(f"  mAP50: {metrics.box.map50:.4f}")
        print(f"  mAP50-95: {metrics.box.map:.4f}")
        print(f"  Precision: {metrics.box.mp:.4f}")
        print(f"  Recall: {metrics.box.mr:.4f}")

        # Copy best model to models directory
        os.makedirs('models', exist_ok=True)
        best_model_src = f'runs/basketball/{model_name}/weights/best.pt'
        output_model = f'models/{model_name}.pt'

        if os.path.exists(best_model_src):
            shutil.copy2(best_model_src, output_model)
            print(f"\n✓ Model saved to: {output_model}")

            # Export to TensorRT for faster inference (if requested)
            export_tensorrt = input("\nExport to TensorRT for faster inference? (requires TensorRT installed) [y/N]: ").strip().lower()
            if export_tensorrt == 'y':
                try:
                    print("\n🔧 Exporting to TensorRT...")
                    best_model.export(format='engine', half=True)
                    print("✓ TensorRT export complete!")
                except Exception as e:
                    print(f"❌ TensorRT export failed: {e}")
                    print("   Install TensorRT: pip install tensorrt")

        return output_model

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        return None
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main training pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description='Optimized YOLO11 Basketball Detector Training')
    parser.add_argument('--preset', choices=['balanced', 'max_performance', 'memory_efficient'],
                       default='balanced', help='Performance preset to use')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--name', default=None, help='Model name (default: auto-generated)')
    args = parser.parse_args()

    print("="*70)
    print("YOLO11 BASKETBALL DETECTOR - OPTIMIZED TRAINING")
    print("Optimized for: Windows + Conda + RTX GPU + Ryzen 9")
    print("="*70)

    # Print system info
    print_system_info()

    # Check GPU
    if not print_gpu_info():
        return 1

    # Load configuration
    config = load_config(args.preset)

    # Find datasets
    print("\n" + "="*70)
    print("[Step 1/3] FINDING DATASETS")
    print("="*70)
    datasets = find_datasets('data/basketball_training')

    if not datasets:
        print("\n❌ No datasets found in data/basketball_training/")
        print("\n📥 Download datasets manually:")
        print("1. Go to: https://universe.roboflow.com/search?q=basketball")
        print("2. Find datasets with 'basketball' annotations")
        print("3. Download in YOLOv8 format")
        print("4. Extract to: data/basketball_training/dataset_1/")
        print("\nRecommended datasets:")
        print("- https://universe.roboflow.com/roboflow-100/basketball-detection")
        print("- https://universe.roboflow.com/basketball-project/basketball-ball-detection")
        return 1

    print(f"\n✓ Found {len(datasets)} dataset(s):")
    for ds in datasets:
        print(f"  - {Path(ds).name}")

    # Combine datasets
    print("\n" + "="*70)
    print("[Step 2/3] COMBINING DATASETS")
    print("="*70)
    data_yaml = combine_datasets(datasets)

    if not data_yaml:
        return 1

    # Train model
    print("\n" + "="*70)
    print("[Step 3/3] TRAINING MODEL")
    print("="*70)

    model_name = args.name or f"basketball_detector_rtx_{args.preset}"
    model_path = train_model(data_yaml, config, epochs=args.epochs, model_name=model_name)

    if model_path:
        print("\n" + "="*70)
        print("🎉 ALL DONE!")
        print("="*70)
        print(f"\nYour optimized basketball detector: {model_path}")
        print("\nNext steps:")
        print("1. Test: python scripts/test_basketball_model.py")
        print("2. Use in pipeline: Update config_performance.yaml with your model path")
        print("3. For maximum inference speed, export to TensorRT")
        return 0
    else:
        return 1


if __name__ == '__main__':
    sys.exit(main())
