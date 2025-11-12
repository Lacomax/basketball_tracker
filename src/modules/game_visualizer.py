"""
Game visualization module.

This module creates annotated videos with player tracking, ball trajectory,
events, and real-time statistics overlays.
"""

import cv2
import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Optional
from collections import deque, defaultdict

from ..config import setup_logging

logger = setup_logging(__name__)


class GameVisualizer:
    """Create visualized videos with game analytics."""

    def __init__(self, video_path: str,
                 ball_detections: Dict = None,
                 player_detections: Dict = None,
                 events: List[Dict] = None,
                 statistics: Dict = None,
                 possession_data: Dict = None,
                 hoop_position: List[int] = None):
        """
        Initialize game visualizer.

        Args:
            video_path: Path to input video
            ball_detections: Ball detection data
            player_detections: Player tracking data
            events: List of game events
            statistics: Player statistics
            possession_data: Possession analysis data
            hoop_position: [x, y] position of basketball hoop
        """
        self.video_path = video_path
        self.ball_detections = ball_detections or {}
        self.player_detections = player_detections or {}
        self.events = events or []
        self.statistics = statistics or {}
        self.possession_data = possession_data or {}
        self.hoop_position = hoop_position

        # Visualization settings
        self.ball_trail_length = 30
        self.show_player_boxes = True
        self.show_player_ids = True
        self.show_ball_trail = True
        self.show_stats_panel = True
        self.show_events = True

        # Colors
        self.team_colors = {
            'Team_0': (255, 0, 0),    # Blue
            'Team_1': (0, 0, 255),    # Red
            'Unknown': (128, 128, 128)  # Gray
        }
        self.ball_color = (0, 165, 255)  # Orange
        self.hoop_color = (0, 255, 255)  # Yellow

        # Ball trail
        self.ball_trail = deque(maxlen=self.ball_trail_length)

        # Event display
        self.active_events = []
        self.event_display_frames = 60  # Show events for 60 frames (2 seconds @ 30fps)

        # Real-time statistics
        self.frame_stats = defaultdict(lambda: {
            'shots': 0, 'made': 0, 'passes': 0, 'possessions': 0
        })

    def _get_team_color(self, team: str) -> Tuple[int, int, int]:
        """Get color for team."""
        return self.team_colors.get(team, self.team_colors['Unknown'])

    def _draw_player_box(self, frame: np.ndarray, player: Dict):
        """Draw bounding box and info for player."""
        x1, y1, x2, y2 = player['bbox']
        team = player.get('team', 'Unknown')
        player_id = player.get('player_id') or player.get('track_id', '?')

        color = self._get_team_color(team)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw player ID
        if self.show_player_ids:
            label = f"#{player_id}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

            # Background for text
            cv2.rectangle(frame,
                         (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0] + 10, y1),
                         color, -1)

            # Text
            cv2.putText(frame, label,
                       (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       (255, 255, 255), 2)

        # Draw velocity arrow if available
        velocity = player.get('velocity')
        if velocity and np.linalg.norm(velocity) > 2:
            center = player.get('center', [(x1 + x2) // 2, (y1 + y2) // 2])
            vx, vy = velocity
            end_point = (int(center[0] + vx * 3), int(center[1] + vy * 3))
            cv2.arrowedLine(frame, tuple(center), end_point, color, 2, tipLength=0.3)

    def _draw_ball(self, frame: np.ndarray, frame_idx: int):
        """Draw ball and its trail."""
        frame_key = str(frame_idx)
        if frame_key not in self.ball_detections:
            return

        detection = self.ball_detections[frame_key]
        center = detection.get('center')
        radius = detection.get('radius', 10)

        if not center:
            return

        center = tuple(center)

        # Add to trail
        self.ball_trail.append(center)

        # Draw trail
        if self.show_ball_trail and len(self.ball_trail) > 1:
            trail_points = list(self.ball_trail)
            for i in range(1, len(trail_points)):
                # Fade trail (older = more transparent)
                alpha = i / len(trail_points)
                thickness = int(2 * alpha) + 1
                cv2.line(frame, trail_points[i-1], trail_points[i],
                        self.ball_color, thickness)

        # Draw current ball position
        cv2.circle(frame, center, radius, self.ball_color, 2)
        cv2.circle(frame, center, 3, self.ball_color, -1)

        # Show occlusion warning
        if detection.get('occluded'):
            cv2.putText(frame, "OCCLUDED",
                       (center[0] - 40, center[1] - radius - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (0, 0, 255), 2)

    def _draw_hoop(self, frame: np.ndarray):
        """Draw basketball hoop indicator."""
        if not self.hoop_position:
            return

        x, y = self.hoop_position
        radius = 40

        # Draw hoop circle
        cv2.circle(frame, (x, y), radius, self.hoop_color, 3)

        # Draw backboard
        cv2.line(frame, (x - 60, y), (x + 60, y), self.hoop_color, 2)

        # Label
        cv2.putText(frame, "HOOP", (x - 20, y - radius - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                   self.hoop_color, 2)

    def _draw_event_notification(self, frame: np.ndarray, event: Dict, age: int):
        """Draw event notification overlay."""
        event_type = event.get('event_type', 'event')
        player_id = event.get('player_id', '?')

        # Event messages
        messages = {
            'shot': f"🏀 SHOT by Player #{player_id}",
            'made_basket': f"✓ BASKET by Player #{player_id}!",
            'missed_basket': f"✗ MISS by Player #{player_id}",
            'pass': f"➜ PASS by Player #{player_id}",
            'dribble': f"⬇ DRIBBLE by Player #{player_id}"
        }

        message = messages.get(event_type, f"Event by Player #{player_id}")

        # Position (top center)
        h, w = frame.shape[:2]
        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)[0]
        x = (w - text_size[0]) // 2
        y = 80 + age * 40  # Stack multiple events

        # Fade out effect
        alpha = max(0, 1.0 - (age / self.event_display_frames))

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay,
                     (x - 10, y - text_size[1] - 10),
                     (x + text_size[0] + 10, y + 10),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, alpha * 0.7, frame, 1 - alpha * 0.7, 0, frame)

        # Text color based on event
        color = (0, 255, 0) if 'made' in event_type else (255, 255, 255)

        cv2.putText(frame, message,
                   (x, y),
                   cv2.FONT_HERSHEY_DUPLEX, 1.0,
                   color, 2)

    def _draw_stats_panel(self, frame: np.ndarray, frame_idx: int):
        """Draw statistics panel."""
        h, w = frame.shape[:2]

        # Panel dimensions
        panel_width = 300
        panel_height = 400
        panel_x = w - panel_width - 20
        panel_y = 20

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay,
                     (panel_x, panel_y),
                     (panel_x + panel_width, panel_y + panel_height),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Border
        cv2.rectangle(frame,
                     (panel_x, panel_y),
                     (panel_x + panel_width, panel_y + panel_height),
                     (255, 255, 255), 2)

        # Title
        cv2.putText(frame, "GAME STATS",
                   (panel_x + 10, panel_y + 30),
                   cv2.FONT_HERSHEY_DUPLEX, 0.8,
                   (255, 255, 255), 2)

        y_offset = panel_y + 60

        # Team statistics
        if 'team_statistics' in self.possession_data:
            team_stats = self.possession_data['team_statistics']

            for team, stats in sorted(team_stats.items()):
                color = self._get_team_color(team)
                pct = stats.get('possession_percentage', 0)

                # Team name
                cv2.putText(frame, team,
                           (panel_x + 10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                           color, 2)

                # Possession bar
                bar_width = int((panel_width - 40) * (pct / 100))
                cv2.rectangle(frame,
                             (panel_x + 10, y_offset + 10),
                             (panel_x + 10 + bar_width, y_offset + 25),
                             color, -1)

                # Percentage
                cv2.putText(frame, f"{pct:.1f}%",
                           (panel_x + panel_width - 70, y_offset + 22),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                           (255, 255, 255), 1)

                y_offset += 45

        # Event counts (updated through frame)
        y_offset += 20
        cv2.line(frame,
                (panel_x + 10, y_offset),
                (panel_x + panel_width - 10, y_offset),
                (255, 255, 255), 1)
        y_offset += 25

        # Count events up to current frame
        shot_count = sum(1 for e in self.events
                        if e.get('event_type') in ['shot', 'made_basket', 'missed_basket']
                        and e.get('frame_start', 0) <= frame_idx)
        made_count = sum(1 for e in self.events
                        if e.get('event_type') == 'made_basket'
                        and e.get('frame_start', 0) <= frame_idx)
        pass_count = sum(1 for e in self.events
                        if e.get('event_type') == 'pass'
                        and e.get('frame_start', 0) <= frame_idx)

        # Display counts
        stats_text = [
            f"Shots: {shot_count}",
            f"Made: {made_count}",
            f"FG%: {(made_count/shot_count*100) if shot_count > 0 else 0:.1f}%",
            f"Passes: {pass_count}"
        ]

        for text in stats_text:
            cv2.putText(frame, text,
                       (panel_x + 10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       (255, 255, 255), 1)
            y_offset += 30

    def create_visualization(self, output_path: str, fps: Optional[int] = None):
        """
        Create fully annotated video.

        Args:
            output_path: Path for output video
            fps: Optional output fps (uses input fps if not specified)
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {self.video_path}")

        # Get video properties
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        output_fps = fps or input_fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))

        logger.info(f"Creating visualization: {output_path}")
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Draw all elements
            if self.show_player_boxes:
                players = self.player_detections.get(str(frame_idx), [])
                for player in players:
                    self._draw_player_box(frame, player)

            if self.show_ball_trail:
                self._draw_ball(frame, frame_idx)

            self._draw_hoop(frame)

            # Handle event notifications
            if self.show_events:
                # Add new events
                for event in self.events:
                    if event.get('frame_start') == frame_idx:
                        self.active_events.append({'event': event, 'age': 0})

                # Draw and age events
                events_to_remove = []
                for i, active_event in enumerate(self.active_events):
                    self._draw_event_notification(frame, active_event['event'],
                                                  active_event['age'])
                    active_event['age'] += 1

                    if active_event['age'] > self.event_display_frames:
                        events_to_remove.append(i)

                # Remove expired events
                for i in reversed(events_to_remove):
                    del self.active_events[i]

            if self.show_stats_panel:
                self._draw_stats_panel(frame, frame_idx)

            # Write frame
            out.write(frame)

            frame_idx += 1
            if frame_idx % 100 == 0:
                logger.info(f"Processed {frame_idx} frames...")

        cap.release()
        out.release()

        logger.info(f"Visualization complete: {output_path}")
        logger.info(f"Total frames: {frame_idx}")


def main():
    """Example usage of game visualizer."""
    import argparse
    parser = argparse.ArgumentParser(description='Create visualized basketball game video')
    parser.add_argument('--video', required=True, help='Path to input video')
    parser.add_argument('--ball', help='Path to ball detections JSON')
    parser.add_argument('--players', help='Path to player detections JSON')
    parser.add_argument('--events', help='Path to events JSON')
    parser.add_argument('--possession', help='Path to possession JSON')
    parser.add_argument('--hoop', help='Path to hoop JSON')
    parser.add_argument('--output', default='outputs/visualized_game.mp4',
                       help='Output video path')
    parser.add_argument('--fps', type=int, help='Output FPS')
    args = parser.parse_args()

    # Load data files
    ball_detections = {}
    if args.ball:
        with open(args.ball, 'r') as f:
            ball_detections = json.load(f)

    player_detections = {}
    if args.players:
        with open(args.players, 'r') as f:
            player_detections = json.load(f)

    events = []
    if args.events:
        with open(args.events, 'r') as f:
            events = json.load(f)

    possession_data = {}
    if args.possession:
        with open(args.possession, 'r') as f:
            possession_data = json.load(f)

    hoop_position = None
    if args.hoop:
        with open(args.hoop, 'r') as f:
            hoop_data = json.load(f)
            hoop_position = hoop_data.get('center')

    # Create visualization
    visualizer = GameVisualizer(
        video_path=args.video,
        ball_detections=ball_detections,
        player_detections=player_detections,
        events=events,
        possession_data=possession_data,
        hoop_position=hoop_position
    )

    visualizer.create_visualization(args.output, fps=args.fps)


if __name__ == '__main__':
    main()
