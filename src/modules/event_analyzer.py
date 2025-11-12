"""
Basketball event analysis module.

This module analyzes basketball game events such as shots, passes,
dribbles, rebounds, and more using ball and player detections.
"""

import cv2
import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum

from ..config import setup_logging

logger = setup_logging(__name__)


class EventType(Enum):
    """Types of basketball events."""
    SHOT = "shot"
    MADE_BASKET = "made_basket"
    MISSED_BASKET = "missed_basket"
    PASS = "pass"
    DRIBBLE = "dribble"
    REBOUND = "rebound"
    STEAL = "steal"
    TURNOVER = "turnover"


@dataclass
class GameEvent:
    """Data class representing a basketball game event."""
    event_type: str
    frame_start: int
    frame_end: int
    player_id: Optional[int] = None
    ball_trajectory: Optional[List[List[int]]] = None
    confidence: float = 0.0
    metadata: Optional[Dict] = None


class EventAnalyzer:
    """Analyze basketball game events from ball and player detections."""

    def __init__(self, ball_detections: Dict, player_detections: Dict = None):
        """
        Initialize event analyzer.

        Args:
            ball_detections: Dictionary of ball detections per frame
            player_detections: Optional dictionary of player detections per frame
        """
        self.ball_detections = ball_detections
        self.player_detections = player_detections or {}
        self.events = []

        # Event detection thresholds
        self.SHOT_MIN_HEIGHT_CHANGE = 100  # Minimum vertical distance for shot detection
        self.SHOT_ARC_THRESHOLD = 0.7  # Parabolic trajectory confidence
        self.PASS_MIN_DISTANCE = 150  # Minimum distance for pass detection
        self.DRIBBLE_BOUNCE_THRESHOLD = 50  # Vertical movement for dribble bounce
        self.PROXIMITY_THRESHOLD = 100  # Pixels to associate ball with player

    def _get_ball_position(self, frame_idx: int) -> Optional[Tuple[int, int]]:
        """Get ball position at specific frame."""
        frame_key = str(frame_idx)
        if frame_key in self.ball_detections:
            center = self.ball_detections[frame_key].get('center')
            if center:
                return tuple(center)
        return None

    def _get_ball_velocity(self, frame_idx: int) -> float:
        """Get ball velocity at specific frame."""
        frame_key = str(frame_idx)
        if frame_key in self.ball_detections:
            return self.ball_detections[frame_key].get('velocity', 0.0)
        return 0.0

    def _find_nearest_player(self, ball_pos: Tuple[int, int], frame_idx: int) -> Optional[int]:
        """Find the nearest player to the ball at a given frame."""
        if not self.player_detections or str(frame_idx) not in self.player_detections:
            return None

        players = self.player_detections[str(frame_idx)]
        if not players:
            return None

        min_dist = float('inf')
        nearest_player = None

        for player in players:
            player_center = player.get('center')
            if player_center:
                dist = np.sqrt((ball_pos[0] - player_center[0])**2 +
                             (ball_pos[1] - player_center[1])**2)
                if dist < min_dist and dist < self.PROXIMITY_THRESHOLD:
                    min_dist = dist
                    nearest_player = player.get('player_id')

        return nearest_player

    def _is_parabolic_trajectory(self, positions: List[Tuple[int, int]]) -> float:
        """
        Check if trajectory follows parabolic arc (shot trajectory).

        Returns:
            Confidence score (0-1) for parabolic fit
        """
        if len(positions) < 5:
            return 0.0

        # Extract y-coordinates (vertical position)
        y_coords = np.array([p[1] for p in positions])
        x_coords = np.arange(len(y_coords))

        # Fit parabola (quadratic polynomial)
        try:
            coeffs = np.polyfit(x_coords, y_coords, 2)
            poly = np.poly1d(coeffs)
            fitted = poly(x_coords)

            # Calculate R-squared for goodness of fit
            ss_res = np.sum((y_coords - fitted) ** 2)
            ss_tot = np.sum((y_coords - np.mean(y_coords)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # Check if parabola opens downward (positive coefficient means upward arc)
            if coeffs[0] > 0:  # Ball goes up then down
                return max(0, r_squared)
        except:
            pass

        return 0.0

    def detect_shots(self, window_size: int = 60) -> List[GameEvent]:
        """
        Detect shooting events (shots at basket).

        Args:
            window_size: Number of frames to analyze for shot trajectory

        Returns:
            List of shot events
        """
        shots = []
        frames = sorted([int(k) for k in self.ball_detections.keys()])

        i = 0
        while i < len(frames) - window_size:
            start_frame = frames[i]
            end_frame = frames[min(i + window_size, len(frames) - 1)]

            # Extract ball trajectory in window
            trajectory = []
            for f in range(start_frame, end_frame + 1):
                pos = self._get_ball_position(f)
                if pos:
                    trajectory.append(pos)

            if len(trajectory) < 10:
                i += 1
                continue

            # Check for vertical movement (shot indicator)
            y_positions = [p[1] for p in trajectory]
            height_change = max(y_positions) - min(y_positions)

            if height_change > self.SHOT_MIN_HEIGHT_CHANGE:
                # Check if trajectory is parabolic
                arc_confidence = self._is_parabolic_trajectory(trajectory)

                if arc_confidence > self.SHOT_ARC_THRESHOLD:
                    # Find player who took the shot
                    player_id = self._find_nearest_player(trajectory[0], start_frame)

                    shot_event = GameEvent(
                        event_type=EventType.SHOT.value,
                        frame_start=start_frame,
                        frame_end=end_frame,
                        player_id=player_id,
                        ball_trajectory=trajectory,
                        confidence=arc_confidence,
                        metadata={'height_change': height_change}
                    )
                    shots.append(shot_event)
                    i += window_size  # Skip processed window

            i += 1

        logger.info(f"Detected {len(shots)} shot events")
        return shots

    def detect_passes(self, min_frames: int = 5, max_frames: int = 30) -> List[GameEvent]:
        """
        Detect passing events between players.

        Args:
            min_frames: Minimum frames for a pass
            max_frames: Maximum frames for a pass

        Returns:
            List of pass events
        """
        passes = []
        frames = sorted([int(k) for k in self.ball_detections.keys()])

        i = 0
        while i < len(frames) - min_frames:
            start_frame = frames[i]

            # Look for fast horizontal movement (characteristic of passes)
            trajectory = []
            for j in range(min_frames, max_frames):
                if i + j >= len(frames):
                    break
                pos = self._get_ball_position(frames[i + j])
                if pos:
                    trajectory.append(pos)

            if len(trajectory) < min_frames:
                i += 1
                continue

            # Calculate horizontal displacement
            x_displacement = abs(trajectory[-1][0] - trajectory[0][0])
            y_displacement = abs(trajectory[-1][1] - trajectory[0][1])

            # Pass typically has more horizontal than vertical movement
            if x_displacement > self.PASS_MIN_DISTANCE and x_displacement > y_displacement * 1.5:
                # Find passer and receiver
                passer = self._find_nearest_player(trajectory[0], start_frame)
                receiver = self._find_nearest_player(trajectory[-1], frames[i + len(trajectory)])

                if passer and receiver and passer != receiver:
                    pass_event = GameEvent(
                        event_type=EventType.PASS.value,
                        frame_start=start_frame,
                        frame_end=frames[i + len(trajectory)],
                        player_id=passer,
                        ball_trajectory=trajectory,
                        confidence=0.8,
                        metadata={'receiver': receiver, 'distance': x_displacement}
                    )
                    passes.append(pass_event)
                    i += len(trajectory)

            i += 1

        logger.info(f"Detected {len(passes)} pass events")
        return passes

    def detect_dribbles(self, window_size: int = 20) -> List[GameEvent]:
        """
        Detect dribbling events (ball bouncing).

        Args:
            window_size: Number of frames to analyze

        Returns:
            List of dribble events
        """
        dribbles = []
        frames = sorted([int(k) for k in self.ball_detections.keys()])

        i = 0
        while i < len(frames) - window_size:
            start_frame = frames[i]
            trajectory = []

            for j in range(window_size):
                if i + j >= len(frames):
                    break
                pos = self._get_ball_position(frames[i + j])
                if pos:
                    trajectory.append(pos)

            if len(trajectory) < window_size // 2:
                i += 1
                continue

            # Detect bouncing pattern (periodic vertical movement)
            y_positions = np.array([p[1] for p in trajectory])

            # Look for local minima (bounces)
            bounces = 0
            for k in range(1, len(y_positions) - 1):
                if y_positions[k] < y_positions[k-1] and y_positions[k] < y_positions[k+1]:
                    if abs(y_positions[k-1] - y_positions[k]) > self.DRIBBLE_BOUNCE_THRESHOLD:
                        bounces += 1

            if bounces >= 2:  # At least 2 bounces for dribble
                player_id = self._find_nearest_player(trajectory[0], start_frame)

                dribble_event = GameEvent(
                    event_type=EventType.DRIBBLE.value,
                    frame_start=start_frame,
                    frame_end=frames[i + window_size],
                    player_id=player_id,
                    ball_trajectory=trajectory,
                    confidence=0.7,
                    metadata={'bounces': bounces}
                )
                dribbles.append(dribble_event)
                i += window_size

            i += 1

        logger.info(f"Detected {len(dribbles)} dribble events")
        return dribbles

    def analyze_all_events(self) -> List[GameEvent]:
        """
        Analyze all event types in the game.

        Returns:
            Complete list of detected events sorted by frame
        """
        logger.info("Analyzing all game events...")

        # Detect all event types
        self.events = []
        self.events.extend(self.detect_shots())
        self.events.extend(self.detect_passes())
        self.events.extend(self.detect_dribbles())

        # Sort events by start frame
        self.events.sort(key=lambda e: e.frame_start)

        logger.info(f"Total events detected: {len(self.events)}")
        return self.events

    def save_events(self, output_path: str):
        """Save detected events to JSON file."""
        events_dict = [asdict(event) for event in self.events]
        with open(output_path, 'w') as f:
            json.dump(events_dict, f, indent=2)
        logger.info(f"Events saved to {output_path}")


def main():
    """Main entry point for event analysis."""
    import argparse
    parser = argparse.ArgumentParser(description='Analyze basketball game events')
    parser.add_argument('--ball', required=True, help='Path to ball detections JSON')
    parser.add_argument('--players', help='Path to player detections JSON')
    parser.add_argument('--output', default='outputs/events.json', help='Output JSON file')
    args = parser.parse_args()

    # Load detections
    with open(args.ball, 'r') as f:
        ball_detections = json.load(f)

    player_detections = None
    if args.players:
        with open(args.players, 'r') as f:
            player_detections = json.load(f)

    # Analyze events
    analyzer = EventAnalyzer(ball_detections, player_detections)
    analyzer.analyze_all_events()
    analyzer.save_events(args.output)


if __name__ == '__main__':
    main()
