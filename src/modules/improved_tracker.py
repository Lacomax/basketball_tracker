"""
Improved player tracking module with DeepSORT integration.

This module provides robust player tracking across frames using
DeepSORT algorithm with fallback to simple IoU tracking.
"""

import cv2
import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

from ultralytics import YOLO

# Try to import DeepSORT, fallback to simple tracking if not available
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    DEEPSORT_AVAILABLE = True
except ImportError:
    DEEPSORT_AVAILABLE = False
    logging.warning("DeepSORT not available. Install with: pip install deep-sort-realtime")

from ..config import setup_logging

logger = setup_logging(__name__)


@dataclass
class TrackedPlayer:
    """Data class representing a tracked player."""
    track_id: int
    bbox: List[int]  # [x1, y1, x2, y2]
    confidence: float
    center: List[int]  # [x, y]
    team: Optional[str] = None
    keypoints: Optional[List[List[float]]] = None
    velocity: Optional[List[float]] = None  # [vx, vy]
    frames_tracked: int = 0


class ImprovedPlayerTracker:
    """Robust player tracker with DeepSORT integration."""

    def __init__(self, model: str = 'yolov8n.pt', pose_model: str = 'yolov8n-pose.pt',
                 use_deepsort: bool = True, max_age: int = 30):
        """
        Initialize improved player tracker.

        Args:
            model: Path to YOLO detection model
            pose_model: Path to YOLO pose model
            use_deepsort: Whether to use DeepSORT (if available)
            max_age: Maximum frames to keep track alive without detection
        """
        self.detection_model = YOLO(model)
        self.pose_model = YOLO(pose_model)

        self.use_deepsort = use_deepsort and DEEPSORT_AVAILABLE

        if self.use_deepsort:
            self.tracker = DeepSort(
                max_age=max_age,
                n_init=3,
                nms_max_overlap=1.0,
                max_cosine_distance=0.3,
                nn_budget=None,
                embedder="mobilenet",
                half=True,
                embedder_gpu=True
            )
            logger.info("Using DeepSORT for player tracking")
        else:
            self.tracker = SimpleIOUTracker(max_age=max_age)
            logger.info("Using simple IoU tracker (DeepSORT not available)")

        self.track_history = defaultdict(list)
        self.team_assignments = {}

    def detect_and_track(self, frame: np.ndarray, frame_number: int,
                        use_pose: bool = False) -> List[TrackedPlayer]:
        """
        Detect and track players in a single frame.

        Args:
            frame: Input frame
            frame_number: Frame number for tracking
            use_pose: Whether to use pose estimation

        Returns:
            List of TrackedPlayer objects
        """
        # Run YOLO detection
        if use_pose:
            results = self.pose_model(frame, classes=[0], verbose=False)
        else:
            results = self.detection_model(frame, classes=[0], verbose=False)

        if len(results) == 0 or results[0].boxes is None:
            return []

        # Prepare detections for tracker
        detections = []
        boxes = results[0].boxes

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])

            # DeepSORT expects [x1, y1, w, h, conf]
            w, h = x2 - x1, y2 - y1
            detections.append(([x1, y1, w, h], conf, 'person'))

        # Update tracker
        if self.use_deepsort:
            tracks = self.tracker.update_tracks(detections, frame=frame)
        else:
            tracks = self.tracker.update(detections, frame_number)

        # Convert to TrackedPlayer objects
        tracked_players = []

        for track in tracks:
            if self.use_deepsort:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                bbox = track.to_ltrb()  # [x1, y1, x2, y2]
                bbox = [int(x) for x in bbox]
            else:
                track_id = track['track_id']
                bbox = track['bbox']

            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Calculate velocity from history
            velocity = None
            if track_id in self.track_history and len(self.track_history[track_id]) > 0:
                prev_center = self.track_history[track_id][-1]
                velocity = [center_x - prev_center[0], center_y - prev_center[1]]

            # Get team assignment if available
            team = self.team_assignments.get(track_id)

            player = TrackedPlayer(
                track_id=track_id,
                bbox=bbox,
                confidence=0.9 if self.use_deepsort else 0.8,
                center=[center_x, center_y],
                team=team,
                velocity=velocity,
                frames_tracked=len(self.track_history[track_id])
            )

            # Update track history
            self.track_history[track_id].append([center_x, center_y])

            tracked_players.append(player)

        return tracked_players

    def assign_teams(self, frame: np.ndarray, players: List[TrackedPlayer]) -> List[TrackedPlayer]:
        """
        Assign team labels based on jersey color.

        Args:
            frame: Input frame
            players: List of tracked players

        Returns:
            Players with team assignments
        """
        if len(players) < 2:
            return players

        colors = []
        for player in players:
            x1, y1, x2, y2 = player.bbox
            roi = frame[y1:y2, x1:x2]

            if roi.size == 0:
                colors.append([0, 0, 0])
                continue

            # Get upper portion (jersey area)
            h = y2 - y1
            jersey_roi = roi[:h//2, :]

            pixels = jersey_roi.reshape(-1, 3)
            if len(pixels) > 0:
                dominant_color = np.median(pixels, axis=0)
                colors.append(dominant_color.tolist())
            else:
                colors.append([0, 0, 0])

        # K-means clustering for team assignment
        colors_array = np.array(colors, dtype=np.float32)

        if len(colors_array) >= 2:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, _ = cv2.kmeans(colors_array, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

            for i, player in enumerate(players):
                team = f"Team_{labels[i][0]}"
                player.team = team
                # Update persistent team assignment
                self.team_assignments[player.track_id] = team

        return players

    def process_video(self, video_path: str, output_path: str,
                     use_pose: bool = False, detect_teams: bool = True) -> Dict:
        """
        Process entire video with improved tracking.

        Args:
            video_path: Path to input video
            output_path: Path to save tracking results
            use_pose: Whether to use pose estimation
            detect_teams: Whether to assign teams

        Returns:
            Dictionary with frame-by-frame tracking data
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        tracking_data = {}
        frame_idx = 0

        logger.info(f"Processing video with improved tracking: {video_path}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect and track players
            players = self.detect_and_track(frame, frame_idx, use_pose=use_pose)

            # Assign teams
            if detect_teams and len(players) > 0:
                players = self.assign_teams(frame, players)

            # Store tracking data
            tracking_data[frame_idx] = [asdict(p) for p in players]

            frame_idx += 1
            if frame_idx % 100 == 0:
                logger.info(f"Processed {frame_idx} frames...")

        cap.release()

        # Save results
        with open(output_path, 'w') as f:
            json.dump(tracking_data, f, indent=2)

        logger.info(f"Tracking data saved to {output_path}")
        logger.info(f"Total unique tracks: {len(self.track_history)}")

        return tracking_data


class SimpleIOUTracker:
    """Simple IoU-based tracker as fallback when DeepSORT is not available."""

    def __init__(self, max_age: int = 30, min_iou: float = 0.3):
        """
        Initialize simple tracker.

        Args:
            max_age: Maximum frames to keep track alive
            min_iou: Minimum IoU for matching
        """
        self.max_age = max_age
        self.min_iou = min_iou
        self.tracks = {}
        self.next_id = 1

    def iou(self, box1: List[int], box2: List[int]) -> float:
        """Calculate IoU between two boxes."""
        x1_1, y1_1, w1, h1 = box1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1

        x1_2, y1_2, w2, h2 = box2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2

        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i < x1_i or y2_i < y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def update(self, detections: List[Tuple], frame_number: int) -> List[Dict]:
        """
        Update tracker with new detections.

        Args:
            detections: List of ([x, y, w, h], conf, class) tuples
            frame_number: Current frame number

        Returns:
            List of active tracks
        """
        # Match detections to existing tracks
        matched_tracks = set()
        matched_detections = set()

        for track_id, track in self.tracks.items():
            if frame_number - track['last_frame'] > self.max_age:
                continue

            best_iou = 0
            best_detection_idx = -1

            for det_idx, (bbox, conf, cls) in enumerate(detections):
                if det_idx in matched_detections:
                    continue

                iou = self.iou(track['bbox'][:4], bbox)
                if iou > best_iou and iou > self.min_iou:
                    best_iou = iou
                    best_detection_idx = det_idx

            if best_detection_idx >= 0:
                # Update track
                bbox, conf, cls = detections[best_detection_idx]
                x, y, w, h = bbox
                track['bbox'] = [x, y, x + w, y + h]
                track['last_frame'] = frame_number
                track['age'] = 0
                matched_tracks.add(track_id)
                matched_detections.add(best_detection_idx)

        # Create new tracks for unmatched detections
        for det_idx, (bbox, conf, cls) in enumerate(detections):
            if det_idx not in matched_detections:
                x, y, w, h = bbox
                self.tracks[self.next_id] = {
                    'track_id': self.next_id,
                    'bbox': [x, y, x + w, y + h],
                    'last_frame': frame_number,
                    'age': 0
                }
                self.next_id += 1

        # Age out old tracks
        to_delete = []
        for track_id, track in self.tracks.items():
            if frame_number - track['last_frame'] > self.max_age:
                to_delete.append(track_id)
            else:
                track['age'] += 1

        for track_id in to_delete:
            del self.tracks[track_id]

        # Return active tracks
        return [track for track in self.tracks.values()
                if frame_number - track['last_frame'] <= self.max_age]


def main():
    """Example usage of improved tracker."""
    import argparse
    parser = argparse.ArgumentParser(description='Improved player tracking with DeepSORT')
    parser.add_argument('--video', required=True, help='Path to input video')
    parser.add_argument('--output', default='outputs/tracked_players.json', help='Output JSON file')
    parser.add_argument('--pose', action='store_true', help='Use pose estimation')
    parser.add_argument('--no-teams', action='store_true', help='Disable team detection')
    parser.add_argument('--no-deepsort', action='store_true', help='Disable DeepSORT')
    args = parser.parse_args()

    tracker = ImprovedPlayerTracker(use_deepsort=not args.no_deepsort)
    tracker.process_video(
        video_path=args.video,
        output_path=args.output,
        use_pose=args.pose,
        detect_teams=not args.no_teams
    )


if __name__ == '__main__':
    main()
