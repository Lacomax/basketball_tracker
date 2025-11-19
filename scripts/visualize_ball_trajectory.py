#!/usr/bin/env python3
"""Quick visualization of ball trajectory."""

import cv2
import json
import sys
from pathlib import Path

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.config_loader import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)
config = get_config()

def main():
    """Visualize ball trajectory on video."""
    # Load detections
    detections_file = f"{config.get_output_dir()}/detections.json"

    if not Path(detections_file).exists():
        logger.error(f"Detections file not found: {detections_file}")
        logger.info("Run trajectory_detector first")
        return 1

    with open(detections_file, 'r') as f:
        detections = json.load(f)

    logger.info(f"Loaded {len(detections)} detections")

    # Open video
    video_paths = config.get_video_paths()
    video_path = None
    for path in video_paths:
        if Path(path).exists():
            video_path = path
            break

    if not video_path:
        logger.error("Video file not found")
        return 1

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return 1

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"Video: {width}x{height}, {fps} FPS, {total_frames} frames")

    # Setup output
    output_path = f"{config.get_output_dir()}/ball_trajectory_visualization.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        logger.error("Could not create output video")
        return 1

    logger.info(f"Creating visualization: {output_path}")

    # Store trajectory history for trail effect
    trajectory = []
    max_trail = 30  # frames

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get detection for current frame
        frame_key = str(frame_num)
        if frame_key in detections:
            detection = detections[frame_key]
            center = detection.get('center', [0, 0])
            radius = detection.get('radius', 15)
            method = detection.get('method', 'unknown')

            # Add to trajectory
            trajectory.append(center)
            if len(trajectory) > max_trail:
                trajectory.pop(0)

            # Draw trail
            for i in range(1, len(trajectory)):
                # Fade effect
                alpha = i / len(trajectory)
                color = (int(255 * alpha), int(165 * alpha), 0)  # Orange fade
                thickness = max(1, int(3 * alpha))

                pt1 = tuple(map(int, trajectory[i-1]))
                pt2 = tuple(map(int, trajectory[i]))
                cv2.line(frame, pt1, pt2, color, thickness)

            # Draw current ball position
            cx, cy = int(center[0]), int(center[1])

            # Color based on detection method
            if 'yolo' in method:
                color = (0, 255, 0)  # Green for YOLO
            elif 'hough' in method:
                color = (255, 0, 0)  # Blue for Hough
            elif 'manual' in method:
                color = (0, 0, 255)  # Red for manual
            else:
                color = (128, 128, 128)  # Gray for fallback

            # Draw circle
            cv2.circle(frame, (cx, cy), radius, color, 2)
            cv2.circle(frame, (cx, cy), 3, color, -1)

            # Draw method label
            label = method.split('-')[0] if '-' in method else method
            cv2.putText(frame, label, (cx + radius + 5, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Draw frame number and detection info
        info_text = f"Frame: {frame_num}/{total_frames}"
        cv2.putText(frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        detection_count = len([k for k in detections.keys() if int(k) <= frame_num])
        det_text = f"Detections: {detection_count}/{len(detections)}"
        cv2.putText(frame, det_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Legend
        cv2.putText(frame, "YOLO", (10, height - 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, "Hough", (10, height - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(frame, "Manual", (10, height - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        out.write(frame)

        # Progress
        if frame_num % 50 == 0:
            logger.info(f"Processing frame {frame_num}/{total_frames} ({frame_num*100/total_frames:.1f}%)")

        frame_num += 1

    cap.release()
    out.release()

    logger.info(f"[+] Visualization saved: {output_path}")
    logger.info(f"[+] Total detections: {len(detections)}")

    # Stats by method
    methods = {}
    for det in detections.values():
        method = det.get('method', 'unknown').split('-')[0]
        methods[method] = methods.get(method, 0) + 1

    logger.info("\nDetection methods:")
    for method, count in sorted(methods.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {method}: {count} ({count*100/len(detections):.1f}%)")

    return 0

if __name__ == '__main__':
    sys.exit(main())
