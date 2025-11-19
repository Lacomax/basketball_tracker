"""
Improved trajectory detection with physics-based interpolation.

This module uses polynomial interpolation and physics-based prediction
to generate smooth, realistic basketball trajectories.
"""

import cv2
import numpy as np
import json
import os
import logging
from typing import Dict, List, Tuple
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline

from ..config import setup_logging
from ..utils.ball_detection import auto_detect_ball
from ..utils.video_utils import open_video_robust

logger = setup_logging(__name__)

# Physics constants (in pixels, assuming 30 fps)
GRAVITY = 0.98  # pixels per frame² (approximation for basketball)
MAX_BOUNCE_FRAMES = 5  # Max frames for ball to be at ground during bounce


class KalmanFilter:
    """Simple Kalman filter for 2D position tracking."""

    def __init__(self, process_variance=1.0, measurement_variance=5.0):
        # State: [x, y, vx, vy]
        self.state = np.zeros(4)
        self.covariance = np.eye(4) * 100

        # Process noise
        self.process_variance = process_variance
        self.Q = np.eye(4) * process_variance

        # Measurement noise
        self.measurement_variance = measurement_variance
        self.R = np.eye(2) * measurement_variance

        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, 1, 0],  # x = x + vx
            [0, 1, 0, 1],  # y = y + vy
            [0, 0, 1, 0],  # vx = vx
            [0, 0, 0, 1]   # vy = vy
        ])

        # Measurement matrix (we only observe position)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

    def predict(self):
        """Predict next state."""
        self.state = self.F @ self.state
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q
        return self.state[:2]  # Return predicted position

    def update(self, measurement):
        """Update state with measurement."""
        measurement = np.array(measurement)

        # Innovation
        y = measurement - self.H @ self.state
        S = self.H @ self.covariance @ self.H.T + self.R

        # Kalman gain
        K = self.covariance @ self.H.T @ np.linalg.inv(S)

        # Update state and covariance
        self.state = self.state + K @ y
        self.covariance = (np.eye(4) - K @ self.H) @ self.covariance

        return self.state[:2]  # Return filtered position


def apply_kalman_smoothing(detections: Dict, process_var=1.0, measurement_var=5.0) -> Dict:
    """
    Apply Kalman filter smoothing to trajectory.

    Args:
        detections: Dict mapping frame_number -> {'center': [x, y], ...}
        process_var: Process noise variance (lower = smoother but less responsive)
        measurement_var: Measurement noise variance (higher = more smoothing)

    Returns:
        Smoothed detections dict
    """
    if len(detections) < 2:
        return detections

    frames = sorted(detections.keys())
    kf = KalmanFilter(process_variance=process_var, measurement_variance=measurement_var)

    # Initialize with first position
    first_frame = frames[0]
    kf.state[:2] = detections[first_frame]['center']

    smoothed = {}
    for frame in frames:
        # Predict
        predicted = kf.predict()

        # Update with measurement
        measurement = detections[frame]['center']
        filtered = kf.update(measurement)

        # Store smoothed position
        smoothed[frame] = detections[frame].copy()
        smoothed[frame]['center'] = [int(filtered[0]), int(filtered[1])]
        smoothed[frame]['method'] = smoothed[frame].get('method', 'unknown') + '-kalman-smoothed'

    return smoothed


def apply_moving_average(detections: Dict, window_size: int = 5) -> Dict:
    """
    Apply moving average smoothing to trajectory.

    Args:
        detections: Dict mapping frame_number -> {'center': [x, y], ...}
        window_size: Size of moving average window (must be odd)

    Returns:
        Smoothed detections dict
    """
    if len(detections) < window_size:
        return detections

    frames = sorted(detections.keys())
    positions = np.array([detections[f]['center'] for f in frames])

    # Ensure window size is odd
    if window_size % 2 == 0:
        window_size += 1

    half_window = window_size // 2

    smoothed = {}
    for i, frame in enumerate(frames):
        # Calculate window boundaries
        start = max(0, i - half_window)
        end = min(len(frames), i + half_window + 1)

        # Average positions in window
        avg_pos = np.mean(positions[start:end], axis=0)

        smoothed[frame] = detections[frame].copy()
        smoothed[frame]['center'] = [int(avg_pos[0]), int(avg_pos[1])]
        smoothed[frame]['method'] = smoothed[frame].get('method', 'unknown') + '-mavg-smoothed'

    return smoothed


def apply_savgol_smoothing(detections: Dict, window_length: int = 11, polyorder: int = 3) -> Dict:
    """
    Apply Savitzky-Golay filter smoothing to trajectory.
    This preserves features better than simple moving average.

    Args:
        detections: Dict mapping frame_number -> {'center': [x, y], ...}
        window_length: Length of filter window (must be odd, >= polyorder+1)
        polyorder: Order of polynomial used to fit samples

    Returns:
        Smoothed detections dict
    """
    if len(detections) < window_length:
        return detections

    frames = sorted(detections.keys())
    positions = np.array([detections[f]['center'] for f in frames])

    # Ensure window_length is odd and valid
    if window_length % 2 == 0:
        window_length += 1
    if window_length < polyorder + 1:
        window_length = polyorder + 2
        if window_length % 2 == 0:
            window_length += 1

    try:
        # Apply Savitzky-Golay filter separately to x and y
        smoothed_x = savgol_filter(positions[:, 0], window_length, polyorder)
        smoothed_y = savgol_filter(positions[:, 1], window_length, polyorder)

        smoothed = {}
        for i, frame in enumerate(frames):
            smoothed[frame] = detections[frame].copy()
            smoothed[frame]['center'] = [int(smoothed_x[i]), int(smoothed_y[i])]
            smoothed[frame]['method'] = smoothed[frame].get('method', 'unknown') + '-savgol-smoothed'

        return smoothed
    except Exception as e:
        logger.warning(f"Savitzky-Golay smoothing failed: {e}, returning original")
        return detections


def apply_spline_smoothing(detections: Dict, smoothing_factor: float = 0.5) -> Dict:
    """
    Apply spline interpolation for ultra-smooth trajectory.

    Args:
        detections: Dict mapping frame_number -> {'center': [x, y], ...}
        smoothing_factor: Smoothness parameter (0=interpolate exactly, higher=smoother)

    Returns:
        Smoothed detections dict
    """
    if len(detections) < 4:
        return detections

    frames = sorted(detections.keys())
    positions = np.array([detections[f]['center'] for f in frames])

    try:
        # Fit splines to x and y coordinates
        k = min(3, len(frames) - 1)  # Degree of spline (max 3, cubic)
        spline_x = UnivariateSpline(frames, positions[:, 0], k=k, s=smoothing_factor * len(frames))
        spline_y = UnivariateSpline(frames, positions[:, 1], k=k, s=smoothing_factor * len(frames))

        smoothed = {}
        for frame in frames:
            smoothed[frame] = detections[frame].copy()
            smoothed[frame]['center'] = [int(spline_x(frame)), int(spline_y(frame))]
            smoothed[frame]['method'] = smoothed[frame].get('method', 'unknown') + '-spline-smoothed'

        return smoothed
    except Exception as e:
        logger.warning(f"Spline smoothing failed: {e}, returning original")
        return detections


def interpolate_parabolic(keyframes: Dict, total_frames: int, max_gap: int = 50, min_parabolic_gap: int = 40) -> Dict:
    """
    Interpolate ball positions using physics-based parabolic motion (SMART VERSION).

    IMPORTANT: This function only applies parabolic physics to segments that are:
    1. Long enough (>= min_parabolic_gap frames, default 40)
    2. Have significant vertical motion (indicating a throw/arc)

    For shorter segments, it uses smooth interpolation to avoid creating
    incorrect "mini-parabolas" between closely-spaced manual annotations.

    Args:
        keyframes: Dict mapping frame_number -> {'center': [x, y]}
        total_frames: Total number of frames in video
        max_gap: Maximum gap to interpolate across
        min_parabolic_gap: Minimum gap to apply parabolic physics (default 40 frames)

    Returns:
        Dict mapping frame_number -> {'center': [x, y], 'radius': r, 'confidence': c}
    """
    frames = sorted(keyframes.keys())
    if len(frames) < 2:
        logger.warning("Need at least 2 keyframes for interpolation")
        return keyframes

    result = {}

    # Process each segment between keyframes
    for i in range(len(frames) - 1):
        start_f = frames[i]
        end_f = frames[i + 1]
        gap = end_f - start_f

        # Add start keyframe
        result[start_f] = keyframes[start_f].copy()
        result[start_f]['method'] = 'manual'

        # Only interpolate if gap is reasonable
        if gap <= max_gap and gap > 1:
            start_pos = np.array(keyframes[start_f]['center'], dtype=float)
            end_pos = np.array(keyframes[end_f]['center'], dtype=float)

            # Calculate vertical movement
            vertical_movement = abs(end_pos[1] - start_pos[1])

            # Only use parabolic physics for LONG segments with SIGNIFICANT vertical movement
            # This avoids creating mini-parabolas between close annotations
            use_parabolic = (gap >= min_parabolic_gap and vertical_movement > 50)

            if gap <= 3 or not use_parabolic:
                # Use linear interpolation for short segments or segments without arc motion
                for j in range(1, gap):
                    t = j / gap
                    pos = start_pos + t * (end_pos - start_pos)
                    f = start_f + j

                    result[f] = {
                        'center': [int(pos[0]), int(pos[1])],
                        'radius': keyframes[start_f].get('radius', 15),
                        'confidence': 0.95,
                        'method': 'linear-interpolated'
                    }
            else:
                # Use parabolic physics for long segments with vertical arc
                dt = gap

                # Horizontal motion: constant velocity
                vx = (end_pos[0] - start_pos[0]) / dt

                # Vertical motion: solve for initial velocity given start, end, and gravity
                # y_end = y_start + vy*t - 0.5*g*t²
                # vy = (y_end - y_start + 0.5*g*t²) / t
                vy = (end_pos[1] - start_pos[1] + 0.5 * GRAVITY * dt * dt) / dt

                # Generate intermediate positions using physics
                for j in range(1, gap):
                    t = j  # Time in frames

                    # Horizontal position (linear)
                    x = start_pos[0] + vx * t

                    # Vertical position (parabolic with gravity)
                    y = start_pos[1] + vy * t - 0.5 * GRAVITY * t * t

                    f = start_f + j

                    # Confidence decreases in the middle of long segments
                    t_norm = j / gap  # Normalized time [0, 1]
                    confidence = max(0.65, 1.0 - abs(0.5 - t_norm) * 0.3)

                    result[f] = {
                        'center': [int(x), int(y)],
                        'radius': keyframes[start_f].get('radius', 15),
                        'confidence': float(confidence),
                        'velocity': float(np.sqrt(vx**2 + (vy - GRAVITY*t)**2)),
                        'method': 'physics-parabolic'
                    }

        elif gap > max_gap:
            logger.warning(f"Gap too large between frames {start_f}-{end_f} ({gap} frames), skipping interpolation")

    # Add final keyframe
    result[frames[-1]] = keyframes[frames[-1]].copy()
    result[frames[-1]]['method'] = 'manual'

    return result


def interpolate_smooth(keyframes, total_frames, max_gap=50):
    """
    Interpolate ball positions using piecewise polynomial for smooth, physics-like motion.

    Args:
        keyframes: Dict mapping frame_number -> {'center': [x, y]}
        total_frames: Total number of frames in video
        max_gap: Maximum gap to interpolate across (frames with larger gaps are not filled)

    Returns:
        Dict mapping frame_number -> {'center': [x, y], 'radius': r, 'confidence': c}
    """
    # Extract keyframe data
    frames = sorted(keyframes.keys())
    if len(frames) < 2:
        logger.warning("Need at least 2 keyframes for interpolation")
        return keyframes

    result = {}

    # Process each segment between keyframes
    for i in range(len(frames) - 1):
        start_f = frames[i]
        end_f = frames[i + 1]
        gap = end_f - start_f

        # Add start keyframe
        result[start_f] = keyframes[start_f].copy()
        result[start_f]['method'] = 'manual'

        # Only interpolate if gap is reasonable
        if gap <= max_gap and gap > 1:
            start_pos = np.array(keyframes[start_f]['center'], dtype=float)
            end_pos = np.array(keyframes[end_f]['center'], dtype=float)

            # Simple smooth interpolation
            # For short segments, use linear or quadratic
            if gap <= 5:
                # Linear interpolation for very short segments
                for j in range(1, gap):
                    t = j / gap  # Interpolation parameter [0, 1]
                    pos = start_pos + t * (end_pos - start_pos)
                    f = start_f + j

                    velocity = float(np.linalg.norm(end_pos - start_pos) / gap)

                    result[f] = {
                        'center': [int(pos[0]), int(pos[1])],
                        'radius': keyframes[start_f].get('radius', 15),
                        'confidence': 0.9,
                        'velocity': velocity,
                        'method': 'linear-interpolated'
                    }
            else:
                # Smooth interpolation using cosine easing for longer segments
                for j in range(1, gap):
                    t = j / gap  # Linear parameter [0, 1]

                    # Apply cosine easing for smoother acceleration/deceleration
                    t_smooth = 0.5 - 0.5 * np.cos(t * np.pi)

                    pos = start_pos + t_smooth * (end_pos - start_pos)
                    f = start_f + j

                    # Calculate velocity (derivative of position)
                    if f > 0 and (f-1) in result:
                        prev = np.array(result[f-1]['center'])
                        velocity = float(np.linalg.norm(pos - prev))
                    else:
                        velocity = 0.0

                    # Confidence is lower in the middle of long segments
                    confidence = max(0.6, 1.0 - abs(0.5 - t) * 0.4)

                    result[f] = {
                        'center': [int(pos[0]), int(pos[1])],
                        'radius': keyframes[start_f].get('radius', 15),
                        'confidence': float(confidence),
                        'velocity': velocity,
                        'method': 'smooth-interpolated'
                    }
        elif gap > max_gap:
            logger.warning(f"Gap too large between frames {start_f}-{end_f} ({gap} frames), skipping interpolation")

    # Add final keyframe
    result[frames[-1]] = keyframes[frames[-1]].copy()
    result[frames[-1]]['method'] = 'manual'

    return result


def process_trajectory_video(video_path: str, annotations_path: str, output_path: str):
    """
    Generate smooth basketball trajectory using improved interpolation.

    Args:
        video_path: Path to input video file
        annotations_path: Path to JSON file with manual annotations
        output_path: Path where detection results will be saved

    Returns:
        Dictionary mapping frame indices to detection dictionaries
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Try to use converted video first
    if os.path.exists("input_video_converted.mp4"):
        video_path = "input_video_converted.mp4"
        logger.info("Using converted video: input_video_converted.mp4")
    elif os.path.exists(video_path.replace(".mp4", "_converted.mp4")):
        video_path = video_path.replace(".mp4", "_converted.mp4")
        logger.info(f"Using converted video: {video_path}")

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

    # Calculate constant radius
    radii = [annotations[str(f)]['radius'] for f in annotations.keys()]
    constant_radius = int(np.median(radii))
    logger.info(f"Using constant ball radius: {constant_radius}px (median of {len(radii)} annotations)")
    logger.info(f"  Radius range in annotations: {min(radii)}-{max(radii)}px")

    # Prepare keyframes
    keyframes = {}
    for frame_str, ann in annotations.items():
        frame = int(frame_str)
        keyframes[frame] = {
            'center': ann['center'],
            'radius': constant_radius,
            'confidence': 1.0
        }

    # Interpolate using smooth interpolation (better than blind parabolic physics)
    logger.info("Interpolating trajectory with smooth piecewise interpolation...")
    detection_points = interpolate_smooth(keyframes, total_frames, max_gap=50)

    # Try auto-detection to refine interpolated positions
    logger.info("Refining with auto-detection...")
    refined_count = 0
    for frame_num in sorted(detection_points.keys()):
        det = detection_points[frame_num]

        # Skip manual annotations
        if det.get('method') == 'manual':
            continue

        # Try auto-detection
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            try:
                predicted_center = tuple(det['center'])
                ball_velocity = det.get('velocity', 0.0)  # Get velocity for adaptive threshold
                detected = auto_detect_ball(frame, predicted_center, use_yolo=True, debug_frame=frame_num, ball_velocity=ball_velocity)
                detected_center = detected['center']
                detection_method = detected.get('method', 'unknown')

                # Log YOLO detections
                if detection_method == 'yolo':
                    logger.info(f"Frame {frame_num}: YOLO detected ball at {detected_center}")

                # Check if detection is reasonable
                dist = np.sqrt((detected_center[0] - predicted_center[0])**2 +
                              (detected_center[1] - predicted_center[1])**2)

                # Adaptive acceptance threshold based on velocity
                # YOLO is more accurate than interpolation, so trust it even if far from prediction
                acceptance_threshold = 100  # Base threshold (was 50px)
                if ball_velocity > 15:  # Fast moving ball
                    acceptance_threshold = 200  # Accept larger deviations
                elif ball_velocity > 8:  # Medium speed
                    acceptance_threshold = 150

                if dist < acceptance_threshold:
                    # Use detected position (more accurate)
                    det['center'] = list(detected_center)
                    det['method'] = f'auto-refined-{detection_method}'
                    det['confidence'] = detected.get('confidence', 1.0)
                    refined_count += 1

                    if detection_method == 'yolo':
                        print(f"  ✓ Frame {frame_num}: YOLO detected ball at ({detected_center[0]}, {detected_center[1]}) [dist={dist:.1f}px, threshold={acceptance_threshold}px]")
                else:
                    # Detection too far from prediction, likely false positive or very bad interpolation
                    # For YOLO detections with high confidence, log this as it might indicate interpolation error
                    if detection_method == 'yolo' and detected.get('confidence', 0) > 0.5:
                        logger.warning(f"Frame {frame_num}: YOLO detected ball at {detected_center} but {dist:.1f}px from prediction {predicted_center} (threshold={acceptance_threshold}px, velocity={ball_velocity:.1f}px/frame)")
            except Exception as e:
                logger.debug(f"Frame {frame_num}: Detection failed - {e}")
                pass  # Keep interpolated position

    logger.info(f"Refined {refined_count} positions with auto-detection")

    # Don't extend beyond last annotation (avoids static ball problem)
    last_frame = max(keyframes.keys())
    detection_points = {f: det for f, det in detection_points.items() if f <= last_frame}

    # Apply smoothing to make trajectory more fluid
    logger.info("Applying Kalman filter for smooth trajectory...")
    detection_points = apply_kalman_smoothing(detection_points, process_var=0.5, measurement_var=3.0)

    # Optional: Apply additional Savitzky-Golay smoothing for extra smoothness
    logger.info("Applying Savitzky-Golay filter for additional smoothing...")
    detection_points = apply_savgol_smoothing(detection_points, window_length=9, polyorder=3)

    cap.release()

    # Save results
    with open(output_path, 'w') as f:
        json.dump(detection_points, f, indent=2)

    logger.info(f"Generated {len(detection_points)} detections -> {output_path}")

    # Statistics
    methods = {}
    for det in detection_points.values():
        method = det.get('method', 'unknown')
        methods[method] = methods.get(method, 0) + 1

    logger.info("Detection methods:")
    for method, count in sorted(methods.items(), key=lambda x: x[1], reverse=True):
        pct = (count / len(detection_points)) * 100
        logger.info(f"  {method}: {count} ({pct:.1f}%)")

    return detection_points


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Improved trajectory detector")
    parser.add_argument("--video", default="data/input_video.mp4", help="Path to input video")
    parser.add_argument("--annotations", default="outputs/annotations.json", help="JSON file with manual annotations")
    parser.add_argument("--output", default="outputs/detections.json", help="Path for detection output")
    args = parser.parse_args()
    process_trajectory_video(args.video, args.annotations, args.output)
