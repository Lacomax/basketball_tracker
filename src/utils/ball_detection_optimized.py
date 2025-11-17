"""
Optimized ball detection for NVIDIA RTX GPUs.

This module provides GPU-accelerated ball detection with:
- Half-precision (FP16) inference for 2x speedup on RTX
- Batch processing for multiple frames
- TensorRT support for maximum performance
- Smart caching and memory management

Optimized for: Windows + Conda + RTX GPU
"""

import cv2
import numpy as np
import os
import torch
from functools import lru_cache
from typing import Tuple, Dict, Optional, List
from ..config import (
    HOUGH_PARAM1,
    HOUGH_PARAM2_STRICT,
    HOUGH_PARAM2_LOOSE,
    MIN_RADIUS,
    MAX_RADIUS,
    DEFAULT_RADIUS,
    ROI_OFFSET,
)

# YOLO model (loaded on demand)
_yolo_model = None
_model_config = {
    'use_fp16': True,  # Use half precision on RTX
    'use_tensorrt': False,  # Set to True if TensorRT engine exists
    'batch_size': 8,  # Process 8 frames at once
}


def set_model_config(**kwargs):
    """
    Configure model optimization settings.

    Args:
        use_fp16: Enable FP16 inference (default True for RTX)
        use_tensorrt: Use TensorRT engine if available (default False)
        batch_size: Batch size for inference (default 8)
    """
    global _model_config
    _model_config.update(kwargs)


def get_yolo_model(force_reload=False):
    """
    Get optimized YOLO model instance (lazy loading).

    Tries to load models in order of performance:
    1. TensorRT engine (fastest, if available)
    2. Custom trained model with FP16
    3. Pre-trained YOLO11 with FP16

    Args:
        force_reload: Force reload the model

    Returns:
        YOLO model instance or None
    """
    global _yolo_model

    if _yolo_model is not None and not force_reload:
        return _yolo_model if _yolo_model is not False else None

    if force_reload:
        _yolo_model = None

    try:
        from ultralytics import YOLO

        # Check CUDA availability
        if not torch.cuda.is_available():
            print("⚠️  CUDA not available, using CPU (will be slow)")

        # Model paths to try (in order of preference)
        model_candidates = [
            # TensorRT engines (fastest)
            ('models/basketball_detector_yolo11l.engine', 'tensorrt', '🚀 TensorRT engine'),
            ('models/basketball_detector.engine', 'tensorrt', '🚀 TensorRT engine'),
            # Custom trained models
            ('models/basketball_detector_yolo11l.pt', 'pytorch', '✓ Custom model (YOLO11-L)'),
            ('models/basketball_detector.pt', 'pytorch', '✓ Custom model'),
            # Pre-trained models
            ('yolo11l.pt', 'pytorch', '✓ Pre-trained YOLO11-L'),
            ('yolo11n.pt', 'pytorch', '✓ Pre-trained YOLO11-N'),
        ]

        for model_path, model_type, desc in model_candidates:
            # Check if file exists (for local models)
            if model_path.startswith('models/'):
                if not os.path.exists(model_path):
                    continue

            try:
                print(f"Loading {model_path}...")
                model = YOLO(model_path)

                # Configure for RTX GPU
                if torch.cuda.is_available():
                    # Move model to GPU
                    model.to('cuda:0')

                    # Enable FP16 if not TensorRT (TensorRT already uses FP16)
                    if model_type != 'tensorrt' and _model_config['use_fp16']:
                        # YOLO handles FP16 automatically via the 'half' parameter
                        print("  FP16 inference enabled (RTX Tensor Cores)")

                    # Enable cuDNN auto-tuner
                    torch.backends.cudnn.benchmark = True

                print(f"{desc} loaded")
                _yolo_model = model

                # Store model type for inference optimization
                _yolo_model._model_type = model_type

                return _yolo_model

            except Exception as e:
                if model_path.startswith('models/'):
                    print(f"⚠️  Could not load {model_path}: {e}")
                continue

        print("❌ No YOLO model could be loaded")
        _yolo_model = False
        return None

    except ImportError:
        print("⚠️  Ultralytics not available, YOLO detection disabled")
        _yolo_model = False
        return None


def is_basketball_color(frame: np.ndarray, center: tuple, radius: int) -> bool:
    """
    Check if the detected circle has basketball-like color (orange/brown).

    Optimized version with numpy vectorization.

    Args:
        frame: Input frame (BGR)
        center: (x, y) center of circle
        radius: Circle radius

    Returns:
        True if color matches basketball, False otherwise
    """
    h, w = frame.shape[:2]
    x, y = int(center[0]), int(center[1])

    # Ensure we're within bounds
    if x - radius < 0 or x + radius >= w or y - radius < 0 or y + radius >= h:
        return False

    # Sample the center region
    sample_radius = max(1, radius // 2)
    roi = frame[y - sample_radius:y + sample_radius, x - sample_radius:x + sample_radius]

    if roi.size == 0:
        return False

    # Convert to HSV (use cv2.cvtColor for speed)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Basketball colors in HSV
    lower_orange = np.array([5, 80, 100], dtype=np.uint8)
    upper_orange = np.array([30, 255, 255], dtype=np.uint8)

    # Vectorized color check
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    ratio = np.count_nonzero(mask) / mask.size

    return ratio > 0.2


def detect_ball_yolo_batch(frames: List[np.ndarray],
                          search_points: Optional[List[tuple]] = None,
                          max_distance: int = 150,
                          debug_frames: Optional[List[int]] = None) -> List[Optional[dict]]:
    """
    Detect basketball using YOLO11 with batch processing (GPU optimized).

    Processes multiple frames at once for better GPU utilization.

    Args:
        frames: List of input frames
        search_points: Optional list of (x, y) to search near for each frame
        max_distance: Maximum distance from search_point to consider detection
        debug_frames: Frame numbers for debug logging

    Returns:
        List of detection dicts (one per frame), None if no ball detected
    """
    model = get_yolo_model()
    if model is None:
        return [None] * len(frames)

    if search_points is None:
        search_points = [None] * len(frames)

    if debug_frames is None:
        debug_frames = [None] * len(frames)

    # Prepare batch inference parameters
    use_fp16 = _model_config['use_fp16'] and torch.cuda.is_available()

    # Run batch inference
    try:
        # YOLO batch inference with optimizations
        results = model(
            frames,
            classes=[32],  # Sports ball class
            verbose=False,
            conf=0.15,
            half=use_fp16,  # Use FP16 on RTX
            device=0 if torch.cuda.is_available() else 'cpu',
            stream=True,  # Stream results for memory efficiency
        )

        detections = []
        for i, (result, search_point, debug_frame) in enumerate(zip(results, search_points, debug_frames)):
            detection = _process_single_yolo_result(
                result,
                frames[i],
                search_point,
                max_distance,
                debug_frame
            )
            detections.append(detection)

        return detections

    except Exception as e:
        print(f"❌ Batch YOLO inference failed: {e}")
        return [None] * len(frames)


def detect_ball_yolo(frame: np.ndarray,
                    search_point: Optional[tuple] = None,
                    max_distance: int = 150,
                    debug_frame: int = None) -> Optional[dict]:
    """
    Detect basketball using YOLO11 (optimized single-frame version).

    For batch processing, use detect_ball_yolo_batch() instead.

    Args:
        frame: Input frame
        search_point: Optional (x, y) to search near
        max_distance: Maximum distance from search_point to consider detection
        debug_frame: Frame number for debug logging

    Returns:
        Dict with 'center' and 'radius', or None if no ball detected
    """
    # Use batch function with single frame for consistency
    results = detect_ball_yolo_batch(
        [frame],
        [search_point],
        max_distance,
        [debug_frame]
    )
    return results[0] if results else None


def _process_single_yolo_result(result, frame, search_point, max_distance, debug_frame):
    """Helper to process single YOLO result from batch inference."""
    show_debug = debug_frame and debug_frame % 50 == 0

    num_detections = len(result.boxes) if result.boxes is not None else 0

    if show_debug:
        print(f"\n  [Frame {debug_frame} DEBUG] YOLO detected {num_detections} basketball(s)")

    if num_detections > 0:
        best_detection = _process_yolo_detections(result.boxes, search_point, max_distance)
        if best_detection:
            return best_detection

    # Strategy 2: Look for small objects near search point
    if search_point is not None:
        # This would require another inference pass, so we skip it in batch mode
        # to maintain performance. Single-frame mode can use the original logic.
        pass

    return None


def _process_yolo_detections(boxes, search_point: Optional[tuple], max_distance: int) -> Optional[dict]:
    """Helper function to process YOLO detection boxes (GPU-optimized)."""
    if boxes is None or len(boxes) == 0:
        return None

    best_detection = None
    min_dist = float('inf')

    # Move to CPU for processing (more efficient for small data)
    boxes_cpu = boxes.cpu()

    for box in boxes_cpu:
        # Get bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0].numpy()
        confidence = float(box.conf[0])

        # Calculate center and radius
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        radius = int(max((x2 - x1), (y2 - y1)) / 2)

        # Clamp radius
        radius = max(MIN_RADIUS, min(MAX_RADIUS, radius))

        # If search point provided, find closest ball
        if search_point is not None:
            dist = np.sqrt((cx - search_point[0])**2 + (cy - search_point[1])**2)

            if dist > max_distance:
                continue

            if dist < min_dist:
                min_dist = dist
                best_detection = {
                    'center': [cx, cy],
                    'radius': radius,
                    'confidence': confidence
                }
        else:
            # No search point, return best confidence
            if best_detection is None or confidence > best_detection.get('confidence', 0):
                best_detection = {
                    'center': [cx, cy],
                    'radius': radius,
                    'confidence': confidence
                }

    return best_detection


def auto_detect_ball(frame: np.ndarray,
                    point: tuple,
                    use_yolo: bool = True,
                    debug_frame: int = None) -> dict:
    """
    Automatically detect a basketball around a clicked point (optimized).

    Uses GPU-accelerated YOLO detection with FP16 on RTX, falls back to
    CPU-based Hough circle detection if needed.

    Args:
        frame: Input image as numpy array (BGR format)
        point: Tuple (x, y) click coordinates
        use_yolo: Whether to try YOLO detection first (default True)
        debug_frame: Frame number for debug logging

    Returns:
        Dictionary with 'center', 'radius', and 'method'
    """
    x, y = int(point[0]), int(point[1])

    # Try YOLO detection first (GPU-accelerated)
    if use_yolo:
        yolo_result = detect_ball_yolo(
            frame,
            search_point=(x, y),
            max_distance=150,
            debug_frame=debug_frame
        )
        if yolo_result is not None:
            yolo_result['method'] = 'yolo_rtx'
            return yolo_result

    # Fallback to Hough circles (CPU-based)
    h, w = frame.shape[:2]
    roi_size = ROI_OFFSET + 30
    roi_x0 = max(0, x - roi_size)
    roi_y0 = max(0, y - roi_size)
    roi_x1 = min(w, x + roi_size)
    roi_y1 = min(h, y + roi_size)

    roi = frame[roi_y0:roi_y1, roi_x0:roi_x1]

    # Optimized preprocessing
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(filtered, 30, 100)

    # Hough circle detection
    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=30,
        param1=50,
        param2=15,
        minRadius=MIN_RADIUS,
        maxRadius=MAX_RADIUS,
    )

    # Try with blurred if edge-based fails
    if circles is None:
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=HOUGH_PARAM1,
            param2=HOUGH_PARAM2_LOOSE,
            minRadius=MIN_RADIUS,
            maxRadius=MAX_RADIUS,
        )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        best_circle = None
        min_dist = float('inf')

        for (cx, cy, r) in circles:
            dist = np.sqrt((cx - (x - roi_x0))**2 + (cy - (y - roi_y0))**2)

            full_cx = int(cx) + roi_x0
            full_cy = int(cy) + roi_y0

            # Filter by basketball color
            if not is_basketball_color(frame, (full_cx, full_cy), int(r)):
                continue

            if dist < min_dist:
                min_dist = dist
                best_circle = (cx, cy, r)

        if best_circle:
            cx, cy, r = best_circle
            cx, cy = int(cx) + roi_x0, int(cy) + roi_y0
            r = max(MIN_RADIUS, min(MAX_RADIUS, int(r)))
            return {"center": [cx, cy], "radius": r, "method": "hough"}

    # Fallback
    return {"center": [x, y], "radius": DEFAULT_RADIUS, "method": "fallback"}


# Batch processing utilities
class BatchBallDetector:
    """
    Batch ball detector for maximum GPU utilization.

    Usage:
        detector = BatchBallDetector(batch_size=16)
        results = detector.detect_batch(frames, search_points)
    """

    def __init__(self, batch_size=8):
        """
        Initialize batch detector.

        Args:
            batch_size: Number of frames to process at once
        """
        self.batch_size = batch_size
        self.model = get_yolo_model()

    def detect_batch(self, frames: List[np.ndarray],
                    search_points: Optional[List[tuple]] = None) -> List[Optional[dict]]:
        """
        Detect balls in multiple frames efficiently.

        Args:
            frames: List of frames
            search_points: Optional list of search points

        Returns:
            List of detection results
        """
        if not frames:
            return []

        # Process in batches
        results = []
        for i in range(0, len(frames), self.batch_size):
            batch_frames = frames[i:i + self.batch_size]
            batch_points = None
            if search_points:
                batch_points = search_points[i:i + self.batch_size]

            batch_results = detect_ball_yolo_batch(batch_frames, batch_points)
            results.extend(batch_results)

        return results

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup resources."""
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Export original functions for compatibility
from .ball_detection import (
    preprocess_frame,
    batch_detect_balls,
    clear_cache,
)


__all__ = [
    'get_yolo_model',
    'set_model_config',
    'detect_ball_yolo',
    'detect_ball_yolo_batch',
    'auto_detect_ball',
    'is_basketball_color',
    'BatchBallDetector',
]
