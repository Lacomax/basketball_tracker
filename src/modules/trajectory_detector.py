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

from ..config import setup_logging
from ..utils.ball_detection import auto_detect_ball
from ..utils.video_utils import open_video_robust

logger = setup_logging(__name__)

# Physics constants (in pixels, assuming 30 fps)
GRAVITY = 0.98  # pixels per frame² (approximation for basketball)
MAX_BOUNCE_FRAMES = 5  # Max frames for ball to be at ground during bounce


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
                detected = auto_detect_ball(frame, predicted_center, use_yolo=True)
                detected_center = detected['center']
                detection_method = detected.get('method', 'unknown')

                # Log YOLO detections
                if detection_method == 'yolo':
                    logger.info(f"Frame {frame_num}: YOLO detected ball at {detected_center}")

                # Check if detection is reasonable
                dist = np.sqrt((detected_center[0] - predicted_center[0])**2 +
                              (detected_center[1] - predicted_center[1])**2)

                if dist < 50:  # Within 50px of interpolation (increased for YOLO)
                    # Use detected position (more accurate)
                    det['center'] = list(detected_center)
                    det['method'] = f'auto-refined-{detection_method}'
                    det['confidence'] = detected.get('confidence', 1.0)
                    refined_count += 1

                    if detection_method == 'yolo':
                        print(f"  ✓ Frame {frame_num}: YOLO detected ball at ({detected_center[0]}, {detected_center[1]})")
            except Exception as e:
                logger.debug(f"Frame {frame_num}: Detection failed - {e}")
                pass  # Keep interpolated position

    logger.info(f"Refined {refined_count} positions with auto-detection")

    # Don't extend beyond last annotation (avoids static ball problem)
    last_frame = max(keyframes.keys())
    detection_points = {f: det for f, det in detection_points.items() if f <= last_frame}

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
