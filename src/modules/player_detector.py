"""
Player detection and tracking module.

This module detects and tracks basketball players in video frames
using YOLOv8 person detection and pose estimation.
"""

import cv2
import numpy as np
import json
import logging
from ultralytics import YOLO
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

from ..config import setup_logging

logger = setup_logging(__name__)


@dataclass
class Player:
    """Data class representing a detected player."""
    player_id: int
    bbox: List[int]  # [x1, y1, x2, y2]
    confidence: float
    center: List[int]  # [x, y]
    keypoints: Optional[List[List[float]]] = None  # Pose keypoints if available
    team: Optional[str] = None  # Team assignment based on jersey color


class PlayerDetector:
    """Detect and track players in basketball videos."""

    def __init__(self, model: str = 'yolov8n.pt', pose_model: str = 'yolov8n-pose.pt'):
        """
        Initialize player detector.

        Args:
            model: Path to YOLO detection model
            pose_model: Path to YOLO pose estimation model
        """
        self.detection_model = YOLO(model)
        self.pose_model = YOLO(pose_model)
        self.trackers = {}  # Dictionary to store player trackers
        self.next_player_id = 1

    def detect_players(self, frame: np.ndarray, conf_threshold: float = 0.5) -> List[Player]:
        """
        Detect players in a single frame.

        Args:
            frame: Input frame as numpy array
            conf_threshold: Confidence threshold for detections

        Returns:
            List of Player objects detected in the frame
        """
        # Run YOLO detection for persons
        results = self.detection_model(frame, classes=[0], conf=conf_threshold, verbose=False)

        players = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                player = Player(
                    player_id=self.next_player_id + i,
                    bbox=[x1, y1, x2, y2],
                    confidence=conf,
                    center=[center_x, center_y]
                )
                players.append(player)

        return players

    def detect_poses(self, frame: np.ndarray, conf_threshold: float = 0.5) -> List[Player]:
        """
        Detect players with pose estimation.

        Args:
            frame: Input frame as numpy array
            conf_threshold: Confidence threshold for detections

        Returns:
            List of Player objects with pose keypoints
        """
        # Run YOLO pose estimation
        results = self.pose_model(frame, conf=conf_threshold, verbose=False)

        players = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            keypoints = results[0].keypoints if hasattr(results[0], 'keypoints') else None

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Extract keypoints if available
                kpts = None
                if keypoints is not None and i < len(keypoints):
                    kpts = keypoints[i].xy[0].tolist()

                player = Player(
                    player_id=self.next_player_id + i,
                    bbox=[x1, y1, x2, y2],
                    confidence=conf,
                    center=[center_x, center_y],
                    keypoints=kpts
                )
                players.append(player)

        return players

    def assign_teams(self, frame: np.ndarray, players: List[Player]) -> List[Player]:
        """
        Assign team labels to players based on jersey color.

        Uses k-means clustering on dominant colors in player bounding boxes.

        Args:
            frame: Input frame
            players: List of detected players

        Returns:
            List of players with team assignments
        """
        if len(players) < 2:
            return players

        # Extract dominant colors from each player's bbox
        colors = []
        for player in players:
            x1, y1, x2, y2 = player.bbox
            roi = frame[y1:y2, x1:x2]

            if roi.size == 0:
                colors.append([0, 0, 0])
                continue

            # Get upper portion of bbox (jersey area)
            h = y2 - y1
            jersey_roi = roi[:h//2, :]

            # Calculate dominant color
            pixels = jersey_roi.reshape(-1, 3)
            if len(pixels) > 0:
                dominant_color = np.median(pixels, axis=0)
                colors.append(dominant_color.tolist())
            else:
                colors.append([0, 0, 0])

        # Simple k-means clustering to assign teams (2 clusters)
        colors_array = np.array(colors, dtype=np.float32)

        if len(colors_array) >= 2:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, _ = cv2.kmeans(colors_array, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

            for i, player in enumerate(players):
                player.team = f"Team_{labels[i][0]}"

        return players

    def process_video(self, video_path: str, output_path: str,
                     use_pose: bool = False, detect_teams: bool = True) -> Dict:
        """
        Process entire video to detect and track players.

        Args:
            video_path: Path to input video
            output_path: Path to save detection results
            use_pose: Whether to use pose estimation
            detect_teams: Whether to assign team labels

        Returns:
            Dictionary with frame-by-frame player detections
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        detections = {}
        frame_idx = 0

        logger.info(f"Processing video: {video_path}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect players (with or without pose)
            if use_pose:
                players = self.detect_poses(frame)
            else:
                players = self.detect_players(frame)

            # Assign teams if requested
            if detect_teams and len(players) > 0:
                players = self.assign_teams(frame, players)

            # Store detections
            detections[frame_idx] = [asdict(p) for p in players]
            frame_idx += 1

            if frame_idx % 100 == 0:
                logger.info(f"Processed {frame_idx} frames...")

        cap.release()

        # Save detections to JSON
        with open(output_path, 'w') as f:
            json.dump(detections, f, indent=2)

        logger.info(f"Player detections saved to {output_path}")
        return detections


def main():
    """Main entry point for player detection."""
    import argparse
    parser = argparse.ArgumentParser(description='Detect and track basketball players')
    parser.add_argument('--video', required=True, help='Path to input video')
    parser.add_argument('--output', default='outputs/players.json', help='Output JSON file')
    parser.add_argument('--pose', action='store_true', help='Use pose estimation')
    parser.add_argument('--no-teams', action='store_true', help='Disable team detection')
    args = parser.parse_args()

    detector = PlayerDetector()
    detector.process_video(
        video_path=args.video,
        output_path=args.output,
        use_pose=args.pose,
        detect_teams=not args.no_teams
    )


if __name__ == '__main__':
    main()
