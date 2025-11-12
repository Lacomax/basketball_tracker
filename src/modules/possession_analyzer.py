"""
Ball possession analysis module.

This module determines which player has possession of the ball
at any given time during the game.
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
class PossessionEvent:
    """Data class representing a possession event."""
    player_id: int
    team: Optional[str]
    frame_start: int
    frame_end: int
    duration: int  # frames
    ball_touches: int = 0


class PossessionAnalyzer:
    """Analyze ball possession throughout the game."""

    def __init__(self, ball_detections: Dict, player_detections: Dict,
                 proximity_threshold: int = 80,
                 temporal_smoothing: int = 5):
        """
        Initialize possession analyzer.

        Args:
            ball_detections: Dictionary of ball detections per frame
            player_detections: Dictionary of player detections per frame
            proximity_threshold: Distance threshold for possession (pixels)
            temporal_smoothing: Frames to smooth possession changes
        """
        self.ball_detections = ball_detections
        self.player_detections = player_detections
        self.proximity_threshold = proximity_threshold
        self.temporal_smoothing = temporal_smoothing

        self.possession_events = []
        self.player_possession_time = defaultdict(int)
        self.team_possession_time = defaultdict(int)

    def _get_ball_position(self, frame_idx: int) -> Optional[Tuple[int, int]]:
        """Get ball position at specific frame."""
        frame_key = str(frame_idx)
        if frame_key in self.ball_detections:
            center = self.ball_detections[frame_key].get('center')
            if center:
                return tuple(center)
        return None

    def _get_players_at_frame(self, frame_idx: int) -> List[Dict]:
        """Get all players at specific frame."""
        frame_key = str(frame_idx)
        if frame_key in self.player_detections:
            return self.player_detections[frame_key]
        return []

    def _calculate_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two positions."""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def detect_possession(self, frame_idx: int) -> Optional[Dict]:
        """
        Detect which player has possession at a given frame.

        Args:
            frame_idx: Frame number

        Returns:
            Dictionary with player_id, team, distance, confidence
        """
        ball_pos = self._get_ball_position(frame_idx)
        if not ball_pos:
            return None

        players = self._get_players_at_frame(frame_idx)
        if not players:
            return None

        # Find nearest player to ball
        min_distance = float('inf')
        nearest_player = None

        for player in players:
            player_center = player.get('center')
            if not player_center:
                continue

            distance = self._calculate_distance(ball_pos, tuple(player_center))

            if distance < min_distance:
                min_distance = distance
                nearest_player = player

        # Determine if player has possession
        if nearest_player and min_distance < self.proximity_threshold:
            # Confidence based on proximity
            confidence = 1.0 - (min_distance / self.proximity_threshold)

            # Additional confidence from ball velocity (low velocity = more likely possession)
            ball_velocity = self.ball_detections[str(frame_idx)].get('velocity', 0)
            velocity_factor = max(0, 1.0 - (ball_velocity / 50.0))  # Normalize by typical velocity

            final_confidence = confidence * 0.7 + velocity_factor * 0.3

            return {
                'player_id': nearest_player.get('player_id') or nearest_player.get('track_id'),
                'team': nearest_player.get('team'),
                'distance': min_distance,
                'confidence': final_confidence
            }

        return None

    def analyze_possessions(self) -> List[PossessionEvent]:
        """
        Analyze possession throughout the entire game.

        Returns:
            List of PossessionEvent objects
        """
        frames = sorted([int(k) for k in self.ball_detections.keys()])

        if not frames:
            logger.warning("No ball detections available")
            return []

        # Detect possession for each frame
        frame_possessions = {}
        for frame_idx in frames:
            possession = self.detect_possession(frame_idx)
            frame_possessions[frame_idx] = possession

        # Smooth possession changes (reduce jitter)
        smoothed_possessions = self._smooth_possessions(frame_possessions)

        # Create possession events
        current_event = None

        for frame_idx in frames:
            possession = smoothed_possessions.get(frame_idx)

            if possession:
                player_id = possession['player_id']
                team = possession['team']

                if current_event is None:
                    # Start new possession event
                    current_event = PossessionEvent(
                        player_id=player_id,
                        team=team,
                        frame_start=frame_idx,
                        frame_end=frame_idx,
                        duration=1,
                        ball_touches=1
                    )
                elif current_event.player_id == player_id:
                    # Continue current possession
                    current_event.frame_end = frame_idx
                    current_event.duration += 1
                else:
                    # Change of possession
                    self.possession_events.append(current_event)

                    # Update statistics
                    self.player_possession_time[current_event.player_id] += current_event.duration
                    if current_event.team:
                        self.team_possession_time[current_event.team] += current_event.duration

                    # Start new possession
                    current_event = PossessionEvent(
                        player_id=player_id,
                        team=team,
                        frame_start=frame_idx,
                        frame_end=frame_idx,
                        duration=1,
                        ball_touches=1
                    )
            else:
                # No possession detected
                if current_event:
                    # End current possession
                    self.possession_events.append(current_event)
                    self.player_possession_time[current_event.player_id] += current_event.duration
                    if current_event.team:
                        self.team_possession_time[current_event.team] += current_event.duration
                    current_event = None

        # Add final event if exists
        if current_event:
            self.possession_events.append(current_event)
            self.player_possession_time[current_event.player_id] += current_event.duration
            if current_event.team:
                self.team_possession_time[current_event.team] += current_event.duration

        logger.info(f"Detected {len(self.possession_events)} possession events")
        return self.possession_events

    def _smooth_possessions(self, frame_possessions: Dict) -> Dict:
        """
        Smooth possession changes using temporal filtering.

        Args:
            frame_possessions: Raw frame-by-frame possessions

        Returns:
            Smoothed possessions
        """
        frames = sorted(frame_possessions.keys())
        smoothed = {}

        for i, frame_idx in enumerate(frames):
            # Look at surrounding frames
            start_idx = max(0, i - self.temporal_smoothing)
            end_idx = min(len(frames), i + self.temporal_smoothing + 1)

            window_frames = frames[start_idx:end_idx]

            # Count player occurrences in window
            player_counts = defaultdict(int)
            for window_frame in window_frames:
                possession = frame_possessions.get(window_frame)
                if possession:
                    player_id = possession['player_id']
                    player_counts[player_id] += 1

            # Most common player in window
            if player_counts:
                most_common_player = max(player_counts, key=player_counts.get)
                # Use original data for that player
                for window_frame in window_frames:
                    possession = frame_possessions.get(window_frame)
                    if possession and possession['player_id'] == most_common_player:
                        smoothed[frame_idx] = possession
                        break
            else:
                smoothed[frame_idx] = None

        return smoothed

    def get_player_statistics(self) -> Dict:
        """
        Get possession statistics per player.

        Returns:
            Dictionary with player possession stats
        """
        stats = {}

        for player_id, total_frames in self.player_possession_time.items():
            # Find team for this player (from any possession event)
            team = None
            for event in self.possession_events:
                if event.player_id == player_id:
                    team = event.team
                    break

            # Count possessions
            possession_count = sum(1 for e in self.possession_events if e.player_id == player_id)

            # Average possession duration
            avg_duration = total_frames / possession_count if possession_count > 0 else 0

            stats[player_id] = {
                'total_frames': total_frames,
                'total_possessions': possession_count,
                'average_duration': avg_duration,
                'team': team
            }

        return stats

    def get_team_statistics(self) -> Dict:
        """
        Get possession statistics per team.

        Returns:
            Dictionary with team possession stats
        """
        total_frames = sum(self.team_possession_time.values())

        stats = {}
        for team, frames in self.team_possession_time.items():
            percentage = (frames / total_frames * 100) if total_frames > 0 else 0

            stats[team] = {
                'total_frames': frames,
                'possession_percentage': percentage
            }

        return stats

    def save_results(self, output_path: str):
        """Save possession analysis results."""
        results = {
            'possession_events': [asdict(e) for e in self.possession_events],
            'player_statistics': self.get_player_statistics(),
            'team_statistics': self.get_team_statistics()
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Possession analysis saved to {output_path}")

    def generate_report(self) -> str:
        """Generate human-readable possession report."""
        report = []
        report.append("=" * 60)
        report.append("BALL POSSESSION ANALYSIS")
        report.append("=" * 60)
        report.append("")

        # Team statistics
        team_stats = self.get_team_statistics()
        if team_stats:
            report.append("Team Possession:")
            report.append("-" * 60)
            for team, stats in sorted(team_stats.items()):
                report.append(f"{team}: {stats['possession_percentage']:.1f}% "
                            f"({stats['total_frames']} frames)")
            report.append("")

        # Player statistics
        player_stats = self.get_player_statistics()
        if player_stats:
            report.append("Player Possession:")
            report.append("-" * 60)

            # Group by team
            by_team = defaultdict(list)
            for player_id, stats in player_stats.items():
                team = stats['team'] or 'Unknown'
                by_team[team].append((player_id, stats))

            for team in sorted(by_team.keys()):
                report.append(f"\n{team}:")
                players = sorted(by_team[team], key=lambda x: x[1]['total_frames'], reverse=True)

                for player_id, stats in players:
                    report.append(f"  Player #{player_id}:")
                    report.append(f"    Possessions: {stats['total_possessions']}")
                    report.append(f"    Total time: {stats['total_frames']} frames")
                    report.append(f"    Avg duration: {stats['average_duration']:.1f} frames")

        report.append("\n" + "=" * 60)

        return "\n".join(report)


def main():
    """Example usage of possession analyzer."""
    import argparse
    parser = argparse.ArgumentParser(description='Analyze ball possession')
    parser.add_argument('--ball', required=True, help='Path to ball detections JSON')
    parser.add_argument('--players', required=True, help='Path to player detections JSON')
    parser.add_argument('--output', default='outputs/possession.json', help='Output JSON file')
    parser.add_argument('--threshold', type=int, default=80, help='Proximity threshold')
    args = parser.parse_args()

    # Load detections
    with open(args.ball, 'r') as f:
        ball_detections = json.load(f)

    with open(args.players, 'r') as f:
        player_detections = json.load(f)

    # Analyze possessions
    analyzer = PossessionAnalyzer(
        ball_detections,
        player_detections,
        proximity_threshold=args.threshold
    )

    analyzer.analyze_possessions()
    analyzer.save_results(args.output)

    # Print report
    print(analyzer.generate_report())


if __name__ == '__main__':
    main()
