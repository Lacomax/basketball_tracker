"""
Trajectory detection module using Kalman filtering.

This module generates smooth basketball trajectories by interpolating
between manual annotations using a constant-velocity Kalman filter.
"""

import cv2
import numpy as np
import json
import os
import logging
from filterpy.kalman import KalmanFilter

from ..config import setup_logging, KALMAN_PROCESS_NOISE, KALMAN_MEASUREMENT_NOISE
from ..utils.ball_detection import auto_detect_ball
from ..utils.video_utils import open_video_robust

logger = setup_logging(__name__)


def clamp(val, min_val, max_val):
    """Clamp a value between min and max bounds."""
    return max(min(val, max_val), min_val)

def create_kalman_filter(initial_pos):
    """
    Create and initialize a constant-velocity Kalman filter.

    Args:
        initial_pos: Tuple (x, y) with initial position

    Returns:
        Initialized KalmanFilter instance with state [x, y, vx, vy]
    """
    # State: [x, y, vx, vy] (position and velocity)
    kf = KalmanFilter(dim_x=4, dim_z=2)
    x, y = initial_pos
    kf.x = np.array([x, y, 0, 0], dtype=float)
    dt = 1.0  # Assuming 1 frame per unit time
    kf.F = np.array([[1, 0, dt, 0],
                     [0, 1, 0, dt],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]], dtype=float)
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]], dtype=float)
    kf.P *= 500.0
    kf.R = np.eye(2) * 5.0
    kf.Q = np.eye(4) * KALMAN_PROCESS_NOISE
    return kf

def process_trajectory_video(video_path: str, annotations_path: str, output_path: str):
    """
    Generate smooth basketball trajectory detections using Kalman filtering.

    Interpolates between manual annotations using a constant-velocity
    Kalman filter to produce detections for all frames between keyframes.
    Enhanced with improved occlusion detection.

    Args:
        video_path: Path to input video file
        annotations_path: Path to JSON file with manual annotations
        output_path: Path where detection results will be saved

    Returns:
        Dictionary mapping frame indices to detection dictionaries
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Try to use converted video first (to avoid codec issues)
    original_video = video_path
    if os.path.exists("input_video_converted.mp4"):
        video_path = "input_video_converted.mp4"
        logger.info("Using converted video: input_video_converted.mp4")
    elif os.path.exists(video_path.replace(".mp4", "_converted.mp4")):
        video_path = video_path.replace(".mp4", "_converted.mp4")
        logger.info(f"Using converted video: {video_path}")
    else:
        logger.warning(f"Converted video not found. Using original: {video_path}")
        logger.warning("You may see codec warnings. To fix this:")
        logger.warning("  1. Run: python convert_video.py")
        logger.warning("  2. Or convert video online and save as input_video_converted.mp4")

    # Load manual annotations
    try:
        with open(annotations_path, 'r') as f:
            annotations = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        annotations = {}
    if not annotations:
        with open(output_path, 'w') as f:
            json.dump({}, f, indent=2)
        logger.warning(f"No annotations found in {annotations_path}, created empty output")
        return {}

    try:
        cap = open_video_robust(video_path)
    except IOError as e:
        raise IOError(f"Cannot open video: {video_path}. {e}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detection_points = {}
    # Use manual annotations as keyframes
    manual_frames = sorted(int(k) for k in annotations.keys())

    # Calculate average radius from manual annotations (ball size is constant)
    radii = [annotations[str(f)]['radius'] for f in manual_frames]
    avg_radius = int(np.mean(radii))
    median_radius = int(np.median(radii))

    # Use median radius (more robust to outliers)
    constant_radius = median_radius
    logger.info(f"Using constant ball radius: {constant_radius}px (median of {len(radii)} annotations)")
    logger.info(f"  Radius range in annotations: {min(radii)}-{max(radii)}px")

    # Enhanced occlusion detection thresholds
    VELOCITY_THRESHOLD = 50.0  # High velocity suggests fast movement or occlusion
    ACCELERATION_THRESHOLD = 30.0  # Sudden acceleration suggests bounce or occlusion
    CONFIDENCE_DECAY = 0.95  # Confidence decay during prediction without measurement

    # Process segments between manual annotations
    auto_detections = 0
    failed_detections = 0

    for idx in range(len(manual_frames) - 1):
        start_f = manual_frames[idx]
        end_f = manual_frames[idx + 1]
        start_ann = annotations[str(start_f)]
        end_ann = annotations[str(end_f)]
        detection_points[start_f] = {
            'center': start_ann['center'],
            'radius': constant_radius,  # Use constant radius
            'confidence': 1.0,
            'method': 'manual'
        }

        # Initialize Kalman filter with starting keyframe
        kf = create_kalman_filter(start_ann['center'])
        prev_velocity = 0.0
        confidence = 1.0

        for f in range(start_f + 1, end_f):
            kf.predict()
            pred_center = [int(kf.x[0]), int(kf.x[1])]

            # Try to auto-detect ball near predicted position
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ret, frame = cap.read()

            detected_center = None
            detection_method = 'kalman'

            if ret:
                # Try to detect ball near predicted position (within 50px radius)
                try:
                    detected_ball = auto_detect_ball(frame, tuple(pred_center))
                    detected_center_candidate = detected_ball['center']

                    # Check if detected ball is reasonably close to prediction
                    dist = np.sqrt((detected_center_candidate[0] - pred_center[0])**2 +
                                 (detected_center_candidate[1] - pred_center[1])**2)

                    if dist < 50:  # Within 50 pixels of prediction
                        # Update Kalman filter with detection
                        kf.update(np.array(detected_center_candidate))
                        detected_center = detected_center_candidate
                        detection_method = 'auto-detected'
                        auto_detections += 1
                        confidence = min(1.0, confidence * 1.1)  # Boost confidence
                    else:
                        failed_detections += 1
                except:
                    failed_detections += 1

            # Use detected or predicted center
            if detected_center:
                final_center = detected_center
            else:
                final_center = pred_center

            # Clamp position to frame boundaries
            final_center[0] = int(clamp(final_center[0], 0, frame_width - 1))
            final_center[1] = int(clamp(final_center[1], 0, frame_height - 1))

            # Enhanced occlusion detection
            velocity = np.sqrt(kf.x[2] ** 2 + kf.x[3] ** 2)
            acceleration = abs(velocity - prev_velocity)
            prev_velocity = velocity

            # Decay confidence during long predictions (unless we detected the ball)
            if not detected_center:
                confidence *= CONFIDENCE_DECAY

            detection = {
                'center': final_center,
                'radius': constant_radius,  # Use constant radius
                'confidence': confidence,
                'velocity': float(velocity),
                'method': detection_method
            }

            # Mark as occluded if high velocity or acceleration
            if velocity > VELOCITY_THRESHOLD or acceleration > ACCELERATION_THRESHOLD:
                detection['occluded'] = True
                detection['occlusion_reason'] = 'high_velocity' if velocity > VELOCITY_THRESHOLD else 'high_acceleration'

            # Mark low confidence detections
            if confidence < 0.5:
                detection['low_confidence'] = True

            detection_points[f] = detection

        # At final keyframe, use manual annotation directly
        detection_points[end_f] = {
            'center': end_ann['center'],
            'radius': constant_radius,  # Use constant radius
            'confidence': 1.0,
            'method': 'manual'
        }

    # Log auto-detection statistics
    total_interpolated = sum(1 for d in detection_points.values() if d.get('method') != 'manual')
    if total_interpolated > 0:
        auto_rate = (auto_detections / total_interpolated) * 100
        logger.info(f"Auto-detection results:")
        logger.info(f"  Successful: {auto_detections}/{total_interpolated} ({auto_rate:.1f}%)")
        logger.info(f"  Failed: {failed_detections}/{total_interpolated}")

    # Process frames after last annotation
    last_frame = manual_frames[-1]
    last_ann = annotations[str(last_frame)]
    detection_points[last_frame] = {
        'center': last_ann['center'],
        'radius': constant_radius,  # Use constant radius
        'confidence': 1.0
    }
    if last_frame < total_frames - 1:
        kf = create_kalman_filter(last_ann['center'])
        confidence = 1.0
        for f in range(last_frame + 1, total_frames):
            kf.predict()
            pred_center = [kf.x[0], kf.x[1]]
            pred_center[0] = int(clamp(pred_center[0], 0, frame_width - 1))
            pred_center[1] = int(clamp(pred_center[1], 0, frame_height - 1))
            confidence *= CONFIDENCE_DECAY
            # Keep radius constant in final section
            detection_points[f] = {
                'center': pred_center,
                'radius': constant_radius,  # Use constant radius
                'confidence': confidence
            }

    cap.release()
    with open(output_path, 'w') as f:
        json.dump(detection_points, f, indent=2)
    logger.info(f"Generated {len(detection_points)} detections -> {output_path}")
    return detection_points

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Trajectory detector using Kalman filter")
    parser.add_argument("--video", default="data/input_video.mp4", help="Path to input video")
    parser.add_argument("--annotations", default="outputs/annotations.json", help="JSON file with manual annotations")
    parser.add_argument("--output", default="outputs/detections.json", help="Path for detection output")
    args = parser.parse_args()
    process_trajectory_video(args.video, args.annotations, args.output)
