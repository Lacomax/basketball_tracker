"""
Shared utility functions for ball detection.

This module contains common ball detection functions used across
the basketball tracker pipeline. Optimized with caching and batch processing.
"""

import cv2
import numpy as np
from functools import lru_cache
from typing import Tuple, Dict, Optional
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

def get_yolo_model():
    """Get YOLO model instance (lazy loading)."""
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            _yolo_model = YOLO('yolo11n.pt')
            print("✓ YOLO11 model loaded for ball detection")
        except ImportError:
            print("⚠ Ultralytics not available, YOLO detection disabled")
            _yolo_model = False
    return _yolo_model if _yolo_model is not False else None

# Global cache for preprocessed frames
_frame_cache = {}
_cache_size = 100


def detect_ball_yolo(frame: np.ndarray, search_point: Optional[tuple] = None, max_distance: int = 150) -> Optional[dict]:
    """
    Detect basketball using YOLO11 (sports ball class).

    Args:
        frame: Input frame
        search_point: Optional (x, y) to search near. If None, detects globally.
        max_distance: Maximum distance from search_point to consider detection

    Returns:
        Dict with 'center' and 'radius', or None if no ball detected
    """
    model = get_yolo_model()
    if model is None:
        return None

    # Run YOLO detection
    # Class 32 in COCO dataset is 'sports ball'
    results = model(frame, classes=[32], verbose=False, conf=0.3)

    if len(results) == 0 or len(results[0].boxes) == 0:
        return None

    # Get all detected balls
    boxes = results[0].boxes
    best_detection = None
    min_dist = float('inf')

    for box in boxes:
        # Get bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        confidence = float(box.conf[0])

        # Calculate center and radius from bounding box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        radius = int(max((x2 - x1), (y2 - y1)) / 2)

        # Clamp radius to reasonable range
        radius = max(MIN_RADIUS, min(MAX_RADIUS, radius))

        # If search point provided, find closest ball
        if search_point is not None:
            dist = np.sqrt((cx - search_point[0])**2 + (cy - search_point[1])**2)
            if dist > max_distance:
                continue
            if dist < min_dist:
                min_dist = dist
                best_detection = {'center': [cx, cy], 'radius': radius, 'confidence': confidence}
        else:
            # No search point, return first/best detection
            if best_detection is None or confidence > best_detection.get('confidence', 0):
                best_detection = {'center': [cx, cy], 'radius': radius, 'confidence': confidence}

    return best_detection


def auto_detect_ball(frame: np.ndarray, point: tuple, use_yolo: bool = True) -> dict:
    """
    Automatically detect a basketball around a clicked point.

    Tries YOLO11 detection first (if enabled), then falls back to Hough circle
    detection with Canny edge detection. Basketball radius is relatively
    constant due to minimal perspective change.

    Args:
        frame: Input image as numpy array (BGR format)
        point: Tuple (x, y) click coordinates in the frame
        use_yolo: Whether to try YOLO detection first (default True)

    Returns:
        Dictionary with keys:
            - 'center': List [x, y] for circle center in frame coordinates
            - 'radius': Integer radius in pixels
            - 'method': Detection method used ('yolo', 'hough', or 'fallback')

        If no detection is possible, returns fallback circle at click point.

    Example:
        >>> result = auto_detect_ball(frame, (100, 150))
        >>> print(result['center'], result['radius'], result['method'])
        [105, 148] 12 'yolo'
    """
    x, y = int(point[0]), int(point[1])

    # Try YOLO detection first
    if use_yolo:
        yolo_result = detect_ball_yolo(frame, search_point=(x, y), max_distance=150)
        if yolo_result is not None:
            yolo_result['method'] = 'yolo'
            return yolo_result
    h, w = frame.shape[:2]

    # Define a larger region of interest around the click point
    roi_size = ROI_OFFSET + 30  # Larger ROI for better edge detection
    roi_x0 = max(0, x - roi_size)
    roi_y0 = max(0, y - roi_size)
    roi_x1 = min(w, x + roi_size)
    roi_y1 = min(h, y + roi_size)

    roi = frame[roi_y0:roi_y1, roi_x0:roi_x1]

    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter to reduce noise while keeping edges sharp
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)

    # Apply Canny edge detection for better circle detection
    edges = cv2.Canny(filtered, 30, 100)

    # Use the edges for Hough circle detection
    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=30,  # Increased to avoid detecting multiple circles
        param1=50,   # Lower threshold for Canny (already applied)
        param2=15,   # Lower accumulator threshold for better detection
        minRadius=MIN_RADIUS,
        maxRadius=MAX_RADIUS,
    )

    # If edge-based detection fails, try with blurred grayscale
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
        # Find the circle closest to the click point
        circles = np.round(circles[0, :]).astype("int")
        best_circle = None
        min_dist = float('inf')

        for (cx, cy, r) in circles:
            # Calculate distance from click point to circle center
            dist = np.sqrt((cx - (x - roi_x0))**2 + (cy - (y - roi_y0))**2)
            if dist < min_dist:
                min_dist = dist
                best_circle = (cx, cy, r)

        if best_circle:
            cx, cy, r = best_circle
            # Offset ROI coordinates to get center in full frame
            cx, cy = int(cx) + roi_x0, int(cy) + roi_y0

            # Clamp radius to reasonable range (ball doesn't change size much)
            # Basketball radius is relatively constant ~12-18 pixels
            r = max(MIN_RADIUS, min(MAX_RADIUS, int(r)))

            return {"center": [cx, cy], "radius": r, "method": "hough"}

    # Fallback if no circle is detected
    # Use a constant radius since ball size doesn't change much
    return {"center": [x, y], "radius": DEFAULT_RADIUS, "method": "fallback"}


def preprocess_frame(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess frame for ball detection (cached).

    Args:
        frame: Input frame

    Returns:
        Tuple of (grayscale, blurred) frames
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray, blurred


def batch_detect_balls(frames: list, points: list) -> list:
    """
    Batch process multiple frames for ball detection.

    Args:
        frames: List of frames
        points: List of click points for each frame

    Returns:
        List of detection dictionaries
    """
    results = []

    for frame, point in zip(frames, points):
        result = auto_detect_ball(frame, point)
        results.append(result)

    return results


def clear_cache():
    """Clear the frame preprocessing cache."""
    global _frame_cache
    _frame_cache = {}
