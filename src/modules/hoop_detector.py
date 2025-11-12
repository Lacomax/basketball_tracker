"""
Basketball hoop detection module.

This module detects the basketball hoop (rim) in video frames
to determine made vs missed shots.
"""

import cv2
import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

from ..config import setup_logging

logger = setup_logging(__name__)


@dataclass
class Hoop:
    """Data class representing a detected basketball hoop."""
    center: List[int]  # [x, y]
    radius: int
    confidence: float
    frame_number: int


class HoopDetector:
    """Detect basketball hoops in video frames."""

    def __init__(self):
        """Initialize hoop detector."""
        # Hoop detection parameters
        self.HOOP_RADIUS_MIN = 15  # Minimum rim radius in pixels
        self.HOOP_RADIUS_MAX = 80  # Maximum rim radius in pixels
        self.HOOP_COLOR_LOWER = np.array([0, 50, 50])  # Orange/red lower bound (HSV)
        self.HOOP_COLOR_UPPER = np.array([20, 255, 255])  # Orange/red upper bound
        self.BASKET_THRESHOLD = 20  # Pixels - ball must pass within this distance

        # Cached hoop position (usually static)
        self.cached_hoop_position = None
        self.position_confidence = 0.0

    def detect_hoop(self, frame: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None) -> Optional[Hoop]:
        """
        Detect basketball hoop in a single frame.

        Args:
            frame: Input frame as numpy array
            roi: Optional region of interest (x1, y1, x2, y2)

        Returns:
            Hoop object if detected, None otherwise
        """
        # Apply ROI if specified
        if roi:
            x1, y1, x2, y2 = roi
            search_frame = frame[y1:y2, x1:x2]
            offset_x, offset_y = x1, y1
        else:
            search_frame = frame
            offset_x, offset_y = 0, 0

        # Convert to HSV for color detection
        hsv = cv2.cvtColor(search_frame, cv2.COLOR_BGR2HSV)

        # Detect orange/red color (typical rim color)
        mask_orange = cv2.inRange(hsv, self.HOOP_COLOR_LOWER, self.HOOP_COLOR_UPPER)

        # Also try to detect rim by edge detection
        gray = cv2.cvtColor(search_frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # Detect circles using Hough transform
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=100,
            param1=50,
            param2=30,
            minRadius=self.HOOP_RADIUS_MIN,
            maxRadius=self.HOOP_RADIUS_MAX
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))

            # Find the best circle (most likely to be the hoop)
            best_circle = None
            best_score = 0

            for circle in circles[0, :]:
                cx, cy, r = circle

                # Score based on:
                # 1. Orange color presence in the circle area
                # 2. Circle is in upper part of frame (hoops are usually high)
                # 3. Circle size is reasonable

                # Check color score
                circle_mask = np.zeros(mask_orange.shape, dtype=np.uint8)
                cv2.circle(circle_mask, (cx, cy), r, 255, -1)
                color_score = np.sum(mask_orange & circle_mask) / (np.pi * r * r * 255)

                # Position score (hoops are usually in upper 2/3 of frame)
                height_ratio = cy / search_frame.shape[0]
                position_score = 1.0 - height_ratio if height_ratio < 0.7 else 0.5

                # Size score (prefer medium-sized circles)
                size_score = 1.0 - abs(r - 40) / 40.0

                total_score = color_score * 0.5 + position_score * 0.3 + size_score * 0.2

                if total_score > best_score:
                    best_score = total_score
                    best_circle = (cx + offset_x, cy + offset_y, r)

            if best_circle and best_score > 0.3:
                cx, cy, r = best_circle
                return Hoop(
                    center=[int(cx), int(cy)],
                    radius=int(r),
                    confidence=float(best_score),
                    frame_number=0
                )

        return None

    def detect_hoop_in_video(self, video_path: str, sample_frames: int = 30,
                            output_path: str = None) -> Optional[Hoop]:
        """
        Detect hoop position by sampling multiple frames.

        Args:
            video_path: Path to video file
            sample_frames: Number of frames to sample
            output_path: Optional path to save hoop position

        Returns:
            Most confident Hoop detection
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Sample frames uniformly across the video
        frame_indices = np.linspace(0, total_frames - 1, sample_frames, dtype=int)

        detections = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            hoop = self.detect_hoop(frame)
            if hoop:
                hoop.frame_number = idx
                detections.append(hoop)

        cap.release()

        if not detections:
            logger.warning("No hoop detected in video")
            return None

        # Find most confident detection
        best_hoop = max(detections, key=lambda h: h.confidence)

        # Cache the position
        self.cached_hoop_position = best_hoop.center
        self.position_confidence = best_hoop.confidence

        # Save if requested
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(asdict(best_hoop), f, indent=2)
            logger.info(f"Hoop position saved to {output_path}")

        logger.info(f"Hoop detected at {best_hoop.center} (confidence: {best_hoop.confidence:.2f})")
        return best_hoop

    def is_basket_made(self, ball_trajectory: List[List[int]],
                      hoop_position: List[int],
                      hoop_radius: int = 40) -> Tuple[bool, float]:
        """
        Determine if a shot was made based on ball trajectory.

        Args:
            ball_trajectory: List of [x, y] ball positions
            hoop_position: [x, y] position of the hoop
            hoop_radius: Radius of the hoop in pixels

        Returns:
            Tuple of (is_made, confidence)
        """
        if len(ball_trajectory) < 5:
            return False, 0.0

        hx, hy = hoop_position
        threshold = hoop_radius + self.BASKET_THRESHOLD

        # Check if ball passes through or near the hoop
        min_distance = float('inf')
        passed_through_height = False
        approach_from_above = False

        for i, pos in enumerate(ball_trajectory):
            bx, by = pos

            # Calculate distance to hoop
            distance = np.sqrt((bx - hx)**2 + (by - hy)**2)
            min_distance = min(min_distance, distance)

            # Check if ball is at approximately hoop height
            if abs(by - hy) < 30:  # Within 30 pixels of hoop height
                passed_through_height = True

            # Check if ball approached from above (characteristic of a made shot)
            if i > 0 and by < hy and ball_trajectory[i-1][1] >= hy:
                approach_from_above = True

        # Made basket criteria:
        # 1. Ball passed very close to hoop center
        # 2. Ball was at hoop height
        # 3. Ball approached from above (downward motion)

        is_made = (min_distance < threshold and
                  passed_through_height and
                  approach_from_above)

        # Confidence based on how close the ball got
        confidence = max(0.0, 1.0 - (min_distance / threshold))

        return is_made, confidence

    def classify_shots(self, events: List[Dict], hoop_position: List[int] = None) -> List[Dict]:
        """
        Classify shot events as made or missed.

        Args:
            events: List of shot events with trajectories
            hoop_position: Optional hoop position (uses cached if not provided)

        Returns:
            Updated events with 'made' and 'confidence' fields
        """
        if hoop_position is None:
            if self.cached_hoop_position is None:
                logger.warning("No hoop position available, cannot classify shots")
                return events
            hoop_position = self.cached_hoop_position

        classified_events = []

        for event in events:
            if event.get('event_type') != 'shot':
                classified_events.append(event)
                continue

            trajectory = event.get('ball_trajectory', [])
            if trajectory:
                is_made, confidence = self.is_basket_made(trajectory, hoop_position)
                event['made'] = is_made
                event['basket_confidence'] = confidence
                event['event_type'] = 'made_basket' if is_made else 'missed_basket'

            classified_events.append(event)

        made_count = sum(1 for e in classified_events if e.get('event_type') == 'made_basket')
        missed_count = sum(1 for e in classified_events if e.get('event_type') == 'missed_basket')

        logger.info(f"Shot classification: {made_count} made, {missed_count} missed")
        return classified_events

    def manual_hoop_selection(self, frame: np.ndarray) -> Optional[Hoop]:
        """
        Allow user to manually select hoop position.

        Args:
            frame: Frame to display for selection

        Returns:
            Manually selected Hoop object
        """
        logger.info("Click on the center of the basketball hoop, then press any key")

        selected_point = [None, None]

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                selected_point[0] = x
                selected_point[1] = y
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                cv2.circle(frame, (x, y), 40, (0, 255, 0), 2)
                cv2.imshow("Select Hoop", frame)

        cv2.imshow("Select Hoop", frame)
        cv2.setMouseCallback("Select Hoop", mouse_callback)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if selected_point[0] is not None:
            hoop = Hoop(
                center=selected_point,
                radius=40,  # Default radius
                confidence=1.0,
                frame_number=0
            )
            self.cached_hoop_position = selected_point
            self.position_confidence = 1.0
            return hoop

        return None


def main():
    """Example usage of hoop detector."""
    import argparse
    parser = argparse.ArgumentParser(description='Detect basketball hoop')
    parser.add_argument('--video', required=True, help='Path to input video')
    parser.add_argument('--output', default='outputs/hoop.json', help='Output JSON file')
    parser.add_argument('--manual', action='store_true', help='Manual selection mode')
    args = parser.parse_args()

    detector = HoopDetector()

    if args.manual:
        # Manual selection
        cap = cv2.VideoCapture(args.video)
        ret, frame = cap.read()
        cap.release()

        if ret:
            hoop = detector.manual_hoop_selection(frame)
            if hoop:
                with open(args.output, 'w') as f:
                    json.dump(asdict(hoop), f, indent=2)
                print(f"Hoop position saved to {args.output}")
    else:
        # Automatic detection
        hoop = detector.detect_hoop_in_video(args.video, output_path=args.output)
        if hoop:
            print(f"Hoop detected at {hoop.center} with confidence {hoop.confidence:.2f}")
        else:
            print("No hoop detected. Try manual selection with --manual flag")


if __name__ == '__main__':
    main()
