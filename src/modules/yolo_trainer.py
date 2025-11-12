"""
YOLO model trainer for basketball detection.

This module handles dataset preparation, model training, and inference
for basketball detection using YOLOv8/YOLOv11.
"""

import os
import json
import yaml
import cv2
import numpy as np
import argparse
import random
import logging

from ultralytics import YOLO
from ..config import (
    setup_logging,
    DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_IMG_SIZE,
    YOLO_MODEL_NAME,
    ROTATION_RANGE,
    BRIGHTNESS_MIN,
    BRIGHTNESS_MAX,
    BLUR_KERNEL_SIZES,
    RANDOM_SEED,
)

logger = setup_logging(__name__)

# Set random seed for reproducibility
if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)


class UltraYOLOBallTrainer:
    """YOLO model trainer for basketball detection."""

    def __init__(self, video_path: str, annotations: str, output_dir: str = 'outputs/yolo_dataset',
                 model: str = 'model/yolov8n.pt', sports_model: str = None):
        """
        Initialize the YOLO trainer.

        Args:
            video_path: Path to input video file
            annotations: Path to JSON annotations file
            output_dir: Directory for dataset and training outputs
            model: Path to base YOLO model
            sports_model: Optional path to sport-specific pre-trained model
        """
        self.video = video_path
        with open(annotations, 'r') as f:
            self.annotations = json.load(f)
        self.output = output_dir
        # Use sport-specific pre-trained model if provided and exists
        if sports_model and os.path.exists(sports_model):
            self.model = os.path.normpath(sports_model)
        else:
            self.model = os.path.normpath(model)
        if not os.path.exists(self.model):
            raise FileNotFoundError(f"YOLO model not found at: {self.model}")
        # Create necessary dataset directories
        for sub in ['images/train', 'images/val', 'labels/train', 'labels/val']:
            os.makedirs(os.path.join(output_dir, sub), exist_ok=True)
    
    def augment_image(self, img, ann, w, h):
        """
        Generate augmented versions of an image with corresponding labels.

        Applies three types of augmentations: rotation, brightness, and blur.

        Args:
            img: Input image as numpy array
            ann: Annotation dictionary with 'center' and 'radius'
            w: Image width
            h: Image height

        Returns:
            Tuple of (augmented_images, augmented_labels) lists
        """
        aug_imgs = []
        aug_labels = []
        cx, cy = ann['center']
        r = ann['radius']
        box_w = box_h = 2 * r

        # Augmentation 1: Rotation
        angle = random.uniform(-ROTATION_RANGE, ROTATION_RANGE)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h))
        center_pt = np.array([cx, cy, 1])
        new_center = M.dot(center_pt)
        new_label = [0, new_center[0] / w, new_center[1] / h, box_w / w, box_h / h]
        aug_imgs.append(rotated)
        aug_labels.append(new_label)

        # Augmentation 2: Brightness
        brightness = random.uniform(BRIGHTNESS_MIN, BRIGHTNESS_MAX)
        brightened = cv2.convertScaleAbs(img, alpha=brightness, beta=0)
        new_label2 = [0, cx / w, cy / h, box_w / w, box_h / h]
        aug_imgs.append(brightened)
        aug_labels.append(new_label2)

        # Augmentation 3: Blur
        ksize = random.choice(BLUR_KERNEL_SIZES)
        # Ensure ksize is odd
        ksize = ksize if ksize % 2 == 1 else ksize + 1
        blurred = cv2.GaussianBlur(img, (ksize, ksize), 0)
        new_label3 = [0, cx / w, cy / h, box_w / w, box_h / h]  # Keep same label
        aug_imgs.append(blurred)
        aug_labels.append(new_label3)

        return aug_imgs, aug_labels

    def extract_frames(self, max_frames: int = 500, val_split: float = 0.2, augment: bool = True, batch_size: int = 32):
        """
        Extract frames from video and prepare YOLO dataset.

        Extracts annotated frames, splits into train/val, applies augmentations,
        and generates dataset.yaml for YOLO training. Optimized with batch processing.

        Args:
            max_frames: Maximum number of frames to extract
            val_split: Fraction of frames to use for validation
            augment: Whether to apply data augmentation
            batch_size: Batch size for frame extraction

        Returns:
            Path to generated dataset.yaml file
        """
        cap = cv2.VideoCapture(self.video)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_indices = sorted(int(k) for k in self.annotations.keys())
        if not frame_indices:
            cap.release()
            raise ValueError("No annotations found to extract frames.")
        use_frames = np.random.choice(frame_indices, size=min(max_frames, len(frame_indices)), replace=False)
        # Split frames for validation
        val_frames = set(np.random.choice(use_frames, size=int(len(use_frames) * val_split), replace=False))

        # Process frames in batches for better performance
        for batch_start in range(0, len(use_frames), batch_size):
            batch_end = min(batch_start + batch_size, len(use_frames))
            batch_frames = use_frames[batch_start:batch_end]

            for idx in batch_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                split = 'val' if idx in val_frames else 'train'
                base_name = f'{idx:08d}'
                # Save original image
                img_path = os.path.join(self.output, 'images', split, base_name + '.jpg')
                cv2.imwrite(img_path, frame)
                ann = self.annotations[str(idx)]
                cx = ann['center'][0] / w
                cy = ann['center'][1] / h
                bw = (ann['radius'] * 2) / w
                bh = (ann['radius'] * 2) / h
                label_txt = f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"
                label_path = os.path.join(self.output, 'labels', split, base_name + '.txt')
                with open(label_path, 'w') as f:
                    f.write(label_txt)
                # Generate augmentations only for training set
                if split == 'train' and augment:
                    aug_imgs, aug_labels = self.augment_image(frame, ann, w, h)
                    for i, (a_img, a_label) in enumerate(zip(aug_imgs, aug_labels)):
                        aug_name = f'{base_name}_aug{i}.jpg'
                        cv2.imwrite(os.path.join(self.output, 'images', split, aug_name), a_img)
                        with open(os.path.join(self.output, 'labels', split, f'{base_name}_aug{i}.txt'), 'w') as f:
                            f.write(f"{a_label[0]} {a_label[1]:.6f} {a_label[2]:.6f} {a_label[3]:.6f} {a_label[4]:.6f}\n")
        cap.release()
        dataset_yaml = {
            'path': os.path.abspath(self.output).replace("\\", "/"),
            'train': 'images/train',
            'val': 'images/val',
            'names': {0: 'basketball'}
        }
        yaml_path = os.path.join(self.output, 'dataset.yaml')
        with open(yaml_path, 'w') as yf:
            yaml.dump(dataset_yaml, yf, default_flow_style=False, sort_keys=False)
        logger.info(f"Dataset prepared at {yaml_path}")
        return yaml_path

    def train(self, epochs: int = 50, batch_size: int = 16, img_size: int = 640, patience: int = 10):
        """
        Train YOLO model on prepared dataset.

        Args:
            epochs: Number of training epochs
            batch_size: Training batch size
            img_size: Input image size for YOLO
            patience: Early stopping patience

        Returns:
            Training results from YOLO model
        """
        data_config = self.extract_frames()
        model = YOLO(self.model)
        logger.info(f"Starting training: epochs={epochs}, batch_size={batch_size}, img_size={img_size}")
        # Early stopping is supported via patience parameter
        return model.train(data=data_config, epochs=epochs, imgsz=img_size, batch=batch_size,
                           name='basketball_detector', patience=patience)

    @staticmethod
    def detect(video_path: str, model_path: str, output_path: str = None, conf: float = 0.3):
        """
        Run inference on a video using a trained YOLO model.

        Args:
            video_path: Path to input video
            model_path: Path to trained YOLO model
            output_path: Optional path for output video
            conf: Confidence threshold for detections

        Returns:
            YOLO prediction results
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")
        logger.info(f"Running inference on {video_path}")
        return YOLO(model_path).predict(source=video_path, conf=conf, save=output_path is not None)

    def optimize_hyperparameters(self, param_grid: dict):
        """
        Grid search for hyperparameter optimization (basic implementation).

        Note: This is a placeholder. Extend with actual validation metrics.

        Args:
            param_grid: Dictionary with parameter names and lists of values

        Returns:
            Dictionary with best parameters found
        """
        best_params = None
        best_metric = float('inf')
        for epochs in param_grid.get('epochs', [30]):
            for batch in param_grid.get('batch_size', [16]):
                logger.info(f"Training with epochs={epochs}, batch_size={batch}")
                # This simulates a validation metric (extend with actual validation)
                metric = random.uniform(0, 1)
                if metric < best_metric:
                    best_metric = metric
                    best_params = {'epochs': epochs, 'batch_size': batch}
        logger.info(f"Best parameters found: {best_params} with metric {best_metric}")
        return best_params

def main():
    """Main entry point for YOLO training pipeline."""
    parser = argparse.ArgumentParser(description='YOLO training pipeline for basketball detection')
    parser.add_argument('--video', default='data/input_video.mp4', help='Path to input video')
    parser.add_argument('--annotations', default='outputs/verified.json', help='JSON file with verified annotations')
    parser.add_argument('--output', default='outputs/yolo_dataset', help='Directory for dataset and YOLO results')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--detect', action='store_true', help='Run inference after training')
    parser.add_argument('--sports_model', default=None, help='Path to sport-specific pre-trained model')
    args = parser.parse_args()

    trainer = UltraYOLOBallTrainer(args.video, args.annotations, args.output,
                                   model='model/yolov8n.pt', sports_model=args.sports_model)

    # Example of hyperparameter optimization (commented, extend as needed)
    # param_grid = {'epochs': [20, 30], 'batch_size': [8, 16]}
    # best_params = trainer.optimize_hyperparameters(param_grid)
    # if best_params:
    #     args.epochs = best_params['epochs']
    #     batch = best_params['batch_size']
    # else:
    #     batch = 16

    trainer.train(epochs=args.epochs, batch_size=16, img_size=640, patience=10)
    if args.detect:
        best_weights = os.path.join(args.output, 'weights', 'best.pt')
        if os.path.exists(best_weights):
            UltraYOLOBallTrainer.detect(args.video, best_weights,
                                        output_path='outputs/detected_basketball.mp4', conf=0.3)
        else:
            logger.warning("No trained weights found for detection.")

if __name__ == '__main__':
    main()
