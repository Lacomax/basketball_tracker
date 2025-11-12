"""
Advanced metrics calculator for player statistics.

This module calculates speed, distance, acceleration, and other
performance metrics from player tracking data.
"""

import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

from ..config import setup_logging

logger = setup_logging(__name__)


@dataclass
class PlayerMetrics:
    """Data class for player performance metrics."""
    player_id: int
    total_distance_m: float  # Total distance in meters
    average_speed_kmh: float  # Average speed in km/h
    max_speed_kmh: float  # Maximum speed in km/h
    sprint_count: int  # Number of sprints detected
    sprint_distance_m: float  # Distance covered while sprinting
    acceleration_events: int  # Number of significant accelerations
    active_time_s: float  # Time player was actively tracked
    distance_per_minute: float  # Distance per minute


class MetricsCalculator:
    """Calculate advanced player metrics from tracking data."""

    def __init__(self, fps: float = 30.0, pixels_per_meter: float = 50.0,
                 sprint_threshold_kmh: float = 18.0,
                 acceleration_threshold: float = 3.0):
        """
        Initialize metrics calculator.

        Args:
            fps: Video frame rate
            pixels_per_meter: Conversion factor from pixels to meters
            sprint_threshold_kmh: Speed threshold for sprint detection (default: 18 km/h)
            acceleration_threshold: Threshold for significant acceleration (m/s²)
        """
        self.fps = fps
        self.pixels_per_meter = pixels_per_meter
        self.sprint_threshold_kmh = sprint_threshold_kmh
        self.sprint_threshold_ms = sprint_threshold_kmh / 3.6  # Convert to m/s
        self.acceleration_threshold = acceleration_threshold

        # Time between frames
        self.dt = 1.0 / fps

        logger.info(f"MetricsCalculator initialized (fps={fps}, scale={pixels_per_meter}px/m)")

    def calculate_distance(self, pos1: Tuple[float, float],
                          pos2: Tuple[float, float]) -> float:
        """
        Calculate Euclidean distance between two positions.

        Args:
            pos1: First position (x, y) in pixels
            pos2: Second position (x, y) in pixels

        Returns:
            Distance in meters
        """
        x1, y1 = pos1
        x2, y2 = pos2
        distance_px = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        distance_m = distance_px / self.pixels_per_meter
        return distance_m

    def calculate_speed(self, pos1: Tuple[float, float],
                       pos2: Tuple[float, float],
                       dt: Optional[float] = None) -> float:
        """
        Calculate instantaneous speed between two positions.

        Args:
            pos1: First position (x, y)
            pos2: Second position (x, y)
            dt: Time difference (seconds). If None, uses frame time.

        Returns:
            Speed in km/h
        """
        if dt is None:
            dt = self.dt

        distance_m = self.calculate_distance(pos1, pos2)
        speed_ms = distance_m / dt if dt > 0 else 0.0
        speed_kmh = speed_ms * 3.6  # Convert m/s to km/h

        return speed_kmh

    def calculate_acceleration(self, speeds: List[float]) -> List[float]:
        """
        Calculate acceleration from speed sequence.

        Args:
            speeds: List of speeds in m/s

        Returns:
            List of accelerations in m/s²
        """
        if len(speeds) < 2:
            return []

        accelerations = []
        for i in range(1, len(speeds)):
            acc = (speeds[i] - speeds[i-1]) / self.dt
            accelerations.append(acc)

        return accelerations

    def detect_sprints(self, trajectory: List[Tuple[float, float]],
                      timestamps: Optional[List[float]] = None) -> Tuple[int, float]:
        """
        Detect sprint events in trajectory.

        Args:
            trajectory: List of positions [(x, y), ...]
            timestamps: Optional timestamps for each position

        Returns:
            Tuple of (sprint_count, sprint_distance_m)
        """
        if len(trajectory) < 2:
            return 0, 0.0

        sprint_count = 0
        sprint_distance = 0.0
        in_sprint = False

        for i in range(1, len(trajectory)):
            speed_kmh = self.calculate_speed(trajectory[i-1], trajectory[i])

            if speed_kmh >= self.sprint_threshold_kmh:
                if not in_sprint:
                    sprint_count += 1
                    in_sprint = True

                distance_m = self.calculate_distance(trajectory[i-1], trajectory[i])
                sprint_distance += distance_m
            else:
                in_sprint = False

        return sprint_count, sprint_distance

    def calculate_player_metrics(self, trajectory: List[Tuple[float, float]],
                                 player_id: int,
                                 frame_indices: Optional[List[int]] = None) -> PlayerMetrics:
        """
        Calculate comprehensive metrics for a player.

        Args:
            trajectory: List of player positions [(x, y), ...]
            player_id: Player ID
            frame_indices: Optional list of frame indices for each position

        Returns:
            PlayerMetrics object
        """
        if len(trajectory) < 2:
            return PlayerMetrics(
                player_id=player_id,
                total_distance_m=0.0,
                average_speed_kmh=0.0,
                max_speed_kmh=0.0,
                sprint_count=0,
                sprint_distance_m=0.0,
                acceleration_events=0,
                active_time_s=0.0,
                distance_per_minute=0.0
            )

        # Calculate total distance
        total_distance = 0.0
        speeds = []

        for i in range(1, len(trajectory)):
            distance_m = self.calculate_distance(trajectory[i-1], trajectory[i])
            total_distance += distance_m

            speed_kmh = self.calculate_speed(trajectory[i-1], trajectory[i])
            speeds.append(speed_kmh)

        # Calculate speed statistics
        average_speed = np.mean(speeds) if speeds else 0.0
        max_speed = np.max(speeds) if speeds else 0.0

        # Detect sprints
        sprint_count, sprint_distance = self.detect_sprints(trajectory)

        # Calculate accelerations
        speeds_ms = [s / 3.6 for s in speeds]  # Convert to m/s
        accelerations = self.calculate_acceleration(speeds_ms)
        acceleration_events = sum(1 for acc in accelerations if abs(acc) > self.acceleration_threshold)

        # Calculate active time
        if frame_indices:
            active_frames = len(frame_indices)
        else:
            active_frames = len(trajectory)

        active_time_s = active_frames / self.fps

        # Distance per minute
        distance_per_minute = (total_distance / active_time_s) * 60.0 if active_time_s > 0 else 0.0

        return PlayerMetrics(
            player_id=player_id,
            total_distance_m=round(total_distance, 2),
            average_speed_kmh=round(average_speed, 2),
            max_speed_kmh=round(max_speed, 2),
            sprint_count=sprint_count,
            sprint_distance_m=round(sprint_distance, 2),
            acceleration_events=acceleration_events,
            active_time_s=round(active_time_s, 2),
            distance_per_minute=round(distance_per_minute, 2)
        )

    def calculate_team_metrics(self, player_metrics: List[PlayerMetrics]) -> Dict:
        """
        Calculate aggregate team metrics.

        Args:
            player_metrics: List of PlayerMetrics for team players

        Returns:
            Dictionary with team-level metrics
        """
        if not player_metrics:
            return {}

        return {
            'total_distance_m': sum(m.total_distance_m for m in player_metrics),
            'average_speed_kmh': np.mean([m.average_speed_kmh for m in player_metrics]),
            'max_speed_kmh': max(m.max_speed_kmh for m in player_metrics),
            'total_sprints': sum(m.sprint_count for m in player_metrics),
            'total_sprint_distance_m': sum(m.sprint_distance_m for m in player_metrics),
            'total_acceleration_events': sum(m.acceleration_events for m in player_metrics),
            'average_distance_per_minute': np.mean([m.distance_per_minute for m in player_metrics]),
            'player_count': len(player_metrics)
        }

    def process_tracking_data(self, tracking_data: Dict,
                             use_court_positions: bool = False) -> Dict[int, PlayerMetrics]:
        """
        Process complete tracking data to calculate metrics for all players.

        Args:
            tracking_data: Dictionary with frame-by-frame tracking data
            use_court_positions: Whether to use 'court_position' or 'center'/'bbox'

        Returns:
            Dictionary mapping player_id to PlayerMetrics
        """
        # Collect trajectories for each player
        player_trajectories = defaultdict(list)
        player_frame_indices = defaultdict(list)

        for frame_idx_str, detections in tracking_data.items():
            frame_idx = int(frame_idx_str)

            for detection in detections:
                player_id = detection.get('player_id') or detection.get('track_id')

                if player_id is None:
                    continue

                # Get position
                if use_court_positions and 'court_position' in detection:
                    position = tuple(detection['court_position'])
                elif 'center' in detection:
                    position = tuple(detection['center'])
                elif 'bbox' in detection:
                    bbox = detection['bbox']
                    position = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
                else:
                    continue

                player_trajectories[player_id].append(position)
                player_frame_indices[player_id].append(frame_idx)

        # Calculate metrics for each player
        all_metrics = {}

        for player_id, trajectory in player_trajectories.items():
            frame_indices = player_frame_indices[player_id]
            metrics = self.calculate_player_metrics(trajectory, player_id, frame_indices)
            all_metrics[player_id] = metrics

            logger.debug(f"Player {player_id}: {metrics.total_distance_m}m, "
                        f"{metrics.average_speed_kmh} km/h avg, "
                        f"{metrics.sprint_count} sprints")

        logger.info(f"Calculated metrics for {len(all_metrics)} players")

        return all_metrics

    def save_metrics(self, metrics: Dict[int, PlayerMetrics], output_path: str):
        """Save metrics to JSON file."""
        metrics_dict = {
            str(player_id): asdict(m) for player_id, m in metrics.items()
        }

        with open(output_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)

        logger.info(f"Metrics saved to {output_path}")

    def generate_metrics_report(self, metrics: Dict[int, PlayerMetrics],
                               team_assignments: Optional[Dict[int, str]] = None) -> str:
        """
        Generate a human-readable metrics report.

        Args:
            metrics: Dictionary of player metrics
            team_assignments: Optional dictionary mapping player_id to team

        Returns:
            Formatted report string
        """
        report = "=" * 60 + "\n"
        report += "PLAYER PERFORMANCE METRICS\n"
        report += "=" * 60 + "\n\n"

        # Sort by total distance
        sorted_players = sorted(metrics.items(),
                              key=lambda x: x[1].total_distance_m,
                              reverse=True)

        for player_id, m in sorted_players:
            team = team_assignments.get(player_id, "Unknown") if team_assignments else "Unknown"

            report += f"Player {player_id} ({team})\n"
            report += f"  Total Distance: {m.total_distance_m} m\n"
            report += f"  Avg Speed: {m.average_speed_kmh} km/h\n"
            report += f"  Max Speed: {m.max_speed_kmh} km/h\n"
            report += f"  Sprints: {m.sprint_count} ({m.sprint_distance_m} m)\n"
            report += f"  Acceleration Events: {m.acceleration_events}\n"
            report += f"  Active Time: {m.active_time_s} s\n"
            report += f"  Distance/Minute: {m.distance_per_minute} m/min\n"
            report += "-" * 60 + "\n"

        # Team summaries if available
        if team_assignments:
            teams = set(team_assignments.values())
            for team in teams:
                team_players = [m for pid, m in metrics.items()
                              if team_assignments.get(pid) == team]
                if team_players:
                    team_metrics = self.calculate_team_metrics(team_players)
                    report += f"\n{team} Team Summary\n"
                    report += f"  Total Distance: {team_metrics['total_distance_m']:.2f} m\n"
                    report += f"  Avg Speed: {team_metrics['average_speed_kmh']:.2f} km/h\n"
                    report += f"  Max Speed: {team_metrics['max_speed_kmh']:.2f} km/h\n"
                    report += f"  Total Sprints: {team_metrics['total_sprints']}\n"
                    report += "-" * 60 + "\n"

        return report


def main():
    """Example usage of metrics calculator."""
    import argparse
    parser = argparse.ArgumentParser(description='Calculate player performance metrics')
    parser.add_argument('--tracking', required=True, help='Path to tracking data JSON')
    parser.add_argument('--output', default='outputs/player_metrics.json',
                       help='Output JSON file')
    parser.add_argument('--fps', type=float, default=30.0, help='Video FPS')
    parser.add_argument('--scale', type=float, default=50.0,
                       help='Pixels per meter (default: 50)')
    parser.add_argument('--court-positions', action='store_true',
                       help='Use court positions instead of video positions')
    parser.add_argument('--report', action='store_true',
                       help='Generate text report')
    args = parser.parse_args()

    # Load tracking data
    with open(args.tracking, 'r') as f:
        tracking_data = json.load(f)

    # Initialize calculator
    calculator = MetricsCalculator(fps=args.fps, pixels_per_meter=args.scale)

    # Calculate metrics
    metrics = calculator.process_tracking_data(
        tracking_data,
        use_court_positions=args.court_positions
    )

    # Save metrics
    calculator.save_metrics(metrics, args.output)

    # Generate report if requested
    if args.report:
        report = calculator.generate_metrics_report(metrics)
        print(report)

        # Save report to file
        report_path = args.output.replace('.json', '_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"\nReport saved to {report_path}")


if __name__ == '__main__':
    main()
