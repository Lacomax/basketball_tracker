"""
Shared utility functions for ball detection.

This module contains common ball detection functions used across
the basketball tracker pipeline.
"""

import cv2
import numpy as np
from ..config import (
    HOUGH_PARAM1,
    HOUGH_PARAM2_STRICT,
    HOUGH_PARAM2_LOOSE,
    MIN_RADIUS,
    MAX_RADIUS,
    DEFAULT_RADIUS,
    ROI_OFFSET,
)


def auto_detect_ball(frame: np.ndarray, point: tuple) -> dict:
    """
    Automatically detect a basketball around a clicked point.

    Uses Hough circle detection on a region of interest (ROI) around
    the provided point. Falls back to a loose threshold if strict
    detection fails.

    Args:
        frame: Input image as numpy array (BGR format)
        point: Tuple (x, y) click coordinates in the frame

    Returns:
        Dictionary with keys:
            - 'center': List [x, y] for circle center in frame coordinates
            - 'radius': Integer radius in pixels

        If no circle is detected, returns fallback circle at click point.

    Example:
        >>> result = auto_detect_ball(frame, (100, 150))
        >>> print(result['center'], result['radius'])
        [105, 148] 12
    """
    x, y = int(point[0]), int(point[1])
    h, w = frame.shape[:2]

    # Define a region of interest around the click point
    roi_x0 = max(0, x - ROI_OFFSET)
    roi_y0 = max(0, y - ROI_OFFSET)
    roi_x1 = min(w, x + ROI_OFFSET)
    roi_y1 = min(h, y + ROI_OFFSET)

    roi = frame[roi_y0:roi_y1, roi_x0:roi_x1]

    # Convert to grayscale and apply Gaussian blur for circle detection
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Try detecting a circle (ball) in the ROI with strict threshold
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2_STRICT,
        minRadius=MIN_RADIUS,
        maxRadius=MAX_RADIUS,
    )

    # Retry with loose threshold if strict detection fails
    if circles is None:
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
        # Extract the first (best) detected circle
        cx, cy, r = circles[0][0]
        # Offset ROI coordinates to get center in full frame
        cx, cy = int(cx) + roi_x0, int(cy) + roi_y0
        return {"center": [cx, cy], "radius": int(r)}

    # Fallback if no circle is detected
    return {"center": [x, y], "radius": DEFAULT_RADIUS}
