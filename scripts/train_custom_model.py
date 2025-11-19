#!/usr/bin/env python3
"""
Train custom YOLO model with your manual annotations + filtered dataset.

Optimized for RTX GPU with:
- Batch size tuned for GPU memory
- Transfer learning from pre-trained weights
- Early stopping
- Best model checkpointing
"""

import os
import sys
from pathlib import Path

# Fix OpenMP
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.logger import get_logger
from ultralytics import YOLO

def main():
    """Main training function."""
    logger = get_logger(__name__)

    logger.info("=" * 70)
    logger.info("CUSTOM BASKETBALL DETECTOR TRAINING")
    logger.info("=" * 70)
    logger.info("")

    # Configuration
    DATASET_PATH = "data/basketball_custom/data.yaml"
    MODEL_SIZE = "yolo11n"  # nano - fast and efficient
    EPOCHS = 100
    BATCH_SIZE = 16  # Good for RTX
    IMAGE_SIZE = 640
    PATIENCE = 20  # Early stopping after 20 epochs without improvement

    # Output
    OUTPUT_DIR = "runs/train/basketball_custom"
    MODEL_OUTPUT = "models/basketball_detector_custom.pt"

    # Check dataset
    if not os.path.exists(DATASET_PATH):
        logger.error(f"Dataset not found: {DATASET_PATH}")
        logger.error("Run: python scripts/prepare_custom_dataset.py first")
        sys.exit(1)

    logger.info(f"Dataset: {DATASET_PATH}")
    logger.info(f"Model: {MODEL_SIZE}")
    logger.info(f"Epochs: {EPOCHS}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Image size: {IMAGE_SIZE}")
    logger.info(f"Early stopping patience: {PATIENCE} epochs")
    logger.info("")

    logger.info("-" * 70)
    logger.info("Starting training...")
    logger.info("-" * 70)
    logger.info("")

    try:
        # Load pre-trained YOLO model (transfer learning)
        model = YOLO(f"{MODEL_SIZE}.pt")

        # Train
        results = model.train(
            data=DATASET_PATH,
            epochs=EPOCHS,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            name="basketball_custom",
            patience=PATIENCE,
            device=0,  # Use GPU 0
            workers=8,  # Parallel data loading
            save=True,
            save_period=10,  # Save checkpoint every 10 epochs
            val=True,
            plots=True,
            verbose=True,
            # Augmentation (moderate for small dataset)
            hsv_h=0.015,  # Hue augmentation
            hsv_s=0.7,    # Saturation
            hsv_v=0.4,    # Value
            degrees=10,   # Rotation
            translate=0.1, # Translation
            scale=0.5,    # Scaling
            flipud=0.0,   # No vertical flip (ball physics)
            fliplr=0.5,   # Horizontal flip ok
            mosaic=1.0,   # Mosaic augmentation
            # Optimization
            optimizer='AdamW',
            lr0=0.01,     # Initial learning rate
            lrf=0.01,     # Final learning rate
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            # Loss weights
            box=7.5,      # Box loss weight
            cls=0.5,      # Class loss weight
            dfl=1.5,      # Distribution focal loss
        )

        logger.info("")
        logger.info("=" * 70)
        logger.info("TRAINING COMPLETE!")
        logger.info("=" * 70)
        logger.info("")

        # Copy best model
        best_model_path = Path(f"runs/train/basketball_custom/weights/best.pt")

        if best_model_path.exists():
            import shutil

            # Create models directory
            Path("models").mkdir(exist_ok=True)

            # Copy best model
            shutil.copy(best_model_path, MODEL_OUTPUT)

            logger.info(f"[+] Best model saved to: {MODEL_OUTPUT}")
            logger.info("")

            # Show metrics
            logger.info("Training Results:")
            logger.info(f"  Final metrics saved in: runs/train/basketball_custom/")
            logger.info(f"  Plots available in: runs/train/basketball_custom/")
            logger.info("")

            logger.info("Next steps:")
            logger.info("  1. Test the model:")
            logger.info(f"     python scripts/test_custom_model.py")
            logger.info("")
            logger.info("  2. Use in pipeline:")
            logger.info(f"     Update config.yaml to use: {MODEL_OUTPUT}")
            logger.info("")
        else:
            logger.error("Best model not found after training")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)

    logger.info("=" * 70)

if __name__ == '__main__':
    main()
