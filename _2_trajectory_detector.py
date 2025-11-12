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

from config import setup_logging, KALMAN_PROCESS_NOISE, KALMAN_MEASUREMENT_NOISE

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

    Args:
        video_path: Path to input video file
        annotations_path: Path to JSON file with manual annotations
        output_path: Path where detection results will be saved

    Returns:
        Dictionary mapping frame indices to detection dictionaries
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

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

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detection_points = {}
    # Use manual annotations as keyframes
    manual_frames = sorted(int(k) for k in annotations.keys())

    # Threshold to detect abrupt movement (possible occlusion or bounce)
    VELOCITY_THRESHOLD = 50.0

    # Process segments between manual annotations
    for idx in range(len(manual_frames) - 1):
        start_f = manual_frames[idx]
        end_f = manual_frames[idx + 1]
        start_ann = annotations[str(start_f)]
        end_ann = annotations[str(end_f)]
        detection_points[start_f] = {'center': start_ann['center'], 'radius': start_ann['radius']}

        # Linear interpolation for radius between keyframes
        def interp_radius(f):
            ratio = (f - start_f) / (end_f - start_f)
            return int(start_ann['radius'] + ratio * (end_ann['radius'] - start_ann['radius']))

        # Initialize Kalman filter with starting keyframe
        kf = create_kalman_filter(start_ann['center'])

        for f in range(start_f + 1, end_f):
            kf.predict()
            pred_center = [kf.x[0], kf.x[1]]
            # Clamp position to frame boundaries
            pred_center[0] = int(clamp(pred_center[0], 0, frame_width - 1))
            pred_center[1] = int(clamp(pred_center[1], 0, frame_height - 1))
            radius = interp_radius(f)
            # Detect occlusion/bounce by excessive velocity
            velocity = np.sqrt(kf.x[2] ** 2 + kf.x[3] ** 2)
            detection = {'center': pred_center, 'radius': radius}
            if velocity > VELOCITY_THRESHOLD:
                detection['occluded'] = True
            detection_points[f] = detection

        # At final keyframe, use manual annotation directly
        detection_points[end_f] = {'center': end_ann['center'], 'radius': end_ann['radius']}

    # Process frames after last annotation
    last_frame = manual_frames[-1]
    last_ann = annotations[str(last_frame)]
    detection_points[last_frame] = {'center': last_ann['center'], 'radius': last_ann['radius']}
    if last_frame < total_frames - 1:
        kf = create_kalman_filter(last_ann['center'])
        for f in range(last_frame + 1, total_frames):
            kf.predict()
            pred_center = [kf.x[0], kf.x[1]]
            pred_center[0] = int(clamp(pred_center[0], 0, frame_width - 1))
            pred_center[1] = int(clamp(pred_center[1], 0, frame_height - 1))
            # Keep radius constant in final section
            detection_points[f] = {'center': pred_center, 'radius': last_ann['radius']}

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
