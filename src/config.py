"""
Configuration module for Basketball Tracker.

This module centralizes all magic numbers and configurable parameters
to make the pipeline easier to adjust without modifying core logic.
"""

import logging

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


def setup_logging(name: str, level=LOG_LEVEL):
    """
    Setup logging for a module.

    Args:
        name: Module name for the logger
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(LOG_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# ============================================================================
# BALL DETECTION PARAMETERS (Hough Circle Detection)
# ============================================================================

# Hough circle detection thresholds
HOUGH_PARAM1 = 50  # Canny edge detection threshold
HOUGH_PARAM2_STRICT = 30  # Circle detection threshold (strict)
HOUGH_PARAM2_LOOSE = 15  # Circle detection threshold (loose/fallback)

# Ball radius constraints (in pixels)
MIN_RADIUS = 10
MAX_RADIUS = 50
DEFAULT_RADIUS = 15

# Region of Interest (ROI) offset for ball search
ROI_OFFSET = 30


# ============================================================================
# ANOMALY DETECTION PARAMETERS
# ============================================================================

# Distance threshold for anomaly detection (in pixels)
# Detections beyond this distance from smooth trajectory are flagged
ANOMALY_THRESHOLD = 50


# ============================================================================
# TRAJECTORY DETECTION PARAMETERS (Kalman Filter)
# ============================================================================

# Kalman filter parameters
# State: [x, y, vx, vy] (position and velocity)
KALMAN_PROCESS_NOISE = 0.01  # Process noise covariance
KALMAN_MEASUREMENT_NOISE = 10.0  # Measurement noise covariance

# Trajectory smoothing window
TRAJECTORY_WINDOW = 90  # Number of frames for anomaly detection window
CONNECTION_THRESHOLD = 30  # Frames within which to connect trajectory segments


# ============================================================================
# DATA AUGMENTATION PARAMETERS (YOLO Training)
# ============================================================================

# Rotation augmentation
ROTATION_RANGE = 15  # degrees (will be applied as ±ROTATION_RANGE)

# Brightness augmentation
BRIGHTNESS_MIN = 0.8  # Darkest multiplier
BRIGHTNESS_MAX = 1.2  # Brightest multiplier

# Blur augmentation
BLUR_KERNEL_SIZES = [3, 5, 7, 9]  # Gaussian blur kernel sizes


# ============================================================================
# YOLO TRAINING PARAMETERS
# ============================================================================

# Model training parameters
DEFAULT_EPOCHS = 50  # Number of training epochs
DEFAULT_BATCH_SIZE = 16  # Training batch size
DEFAULT_IMG_SIZE = 640  # Input image size for YOLO

# Model names and paths
YOLO_MODEL_NAME = "yolov8s"  # YOLOv8 small model (balance of speed/accuracy)
YOLO_OUTPUT_DIR = "runs/detect/basketball_detector"  # Training output directory
BEST_WEIGHTS_PATH = f"{YOLO_OUTPUT_DIR}/weights/best.pt"

# Data augmentation flags
USE_MOSAIC = True  # Use mosaic augmentation during training
MOSAIC_PROB = 0.8  # Probability of mosaic augmentation
FLIPUD_PROB = 0.5  # Probability of vertical flip
FLIPLR_PROB = 0.5  # Probability of horizontal flip


# ============================================================================
# INFERENCE PARAMETERS
# ============================================================================

# Confidence threshold for detections
INFERENCE_CONFIDENCE = 0.5  # Minimum confidence score

# IoU threshold for NMS (Non-Maximum Suppression)
INFERENCE_IOU = 0.45


# ============================================================================
# VIDEO PROCESSING PARAMETERS
# ============================================================================

# Video codec for output videos
VIDEO_CODEC = "mp4v"  # FourCC codec
VIDEO_FPS = 30  # Frames per second for output video
VIDEO_QUALITY = 0.85  # Compression quality (0-1)


# ============================================================================
# FILE PATHS
# ============================================================================

# Default file names
ANNOTATIONS_FILE = "annotations.json"
DETECTIONS_FILE = "detections.json"
VERIFIED_FILE = "verified.json"


# ============================================================================
# RANDOM SEED (for reproducibility)
# ============================================================================

RANDOM_SEED = 42  # Set to None to disable seeding for non-deterministic results
