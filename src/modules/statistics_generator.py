"""
Basketball statistics generation module.

This module generates comprehensive statistics per player from
detected events and game data.
"""

import json
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict, field
from collections import defaultdict

from ..config import setup_logging

logger = setup_logging(__name__)


@dataclass
class PlayerStatistics:
    """Comprehensive statistics for a single player."""
    player_id: int
    team: Optional[str] = None

    # Shooting statistics
    shots_attempted: int = 0
    shots_made: int = 0
    shooting_percentage: float = 0.0

    # Passing statistics
    passes: int = 0
    assists: int = 0
    turnovers: int = 0

    # Dribbling statistics
    dribbles: int = 0
    total_bounces: int = 0

    # Defensive statistics
    rebounds: int = 0
    steals: int = 0
    blocks: int = 0

    # Time statistics
    time_on_court: int = 0  # Frames with ball or active
    time_with_ball: int = 0  # Frames in possession

    # Distance covered
    distance_traveled: float = 0.0  # Pixels

    # Additional metadata
    metadata: Dict = field(default_factory=dict)


class StatisticsGenerator:
    """Generate player statistics from game events and detections."""

    def __init__(self, events_file: str, players_file: str):
        """
        Initialize statistics generator.

        Args:
            events_file: Path to JSON file with detected events
            players_file: Path to JSON file with player detections
        """
        with open(events_file, 'r') as f:
            self.events = json.load(f)

        with open(players_file, 'r') as f:
            self.player_detections = json.load(f)

        self.player_stats = {}

    def _get_or_create_stats(self, player_id: int, team: str = None) -> PlayerStatistics:
        """Get existing player stats or create new entry."""
        if player_id not in self.player_stats:
            self.player_stats[player_id] = PlayerStatistics(
                player_id=player_id,
                team=team
            )
        return self.player_stats[player_id]

    def _calculate_distance(self, pos1: List[int], pos2: List[int]) -> float:
        """Calculate Euclidean distance between two positions."""
        return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5

    def process_shots(self):
        """Process shooting events and update statistics."""
        for event in self.events:
            if event.get('event_type') in ['shot', 'made_basket', 'missed_basket']:
                player_id = event.get('player_id')
                if player_id:
                    stats = self._get_or_create_stats(player_id)
                    stats.shots_attempted += 1

                    if event.get('event_type') == 'made_basket':
                        stats.shots_made += 1

        # Calculate shooting percentages
        for stats in self.player_stats.values():
            if stats.shots_attempted > 0:
                stats.shooting_percentage = (stats.shots_made / stats.shots_attempted) * 100

        logger.info(f"Processed shooting statistics for {len(self.player_stats)} players")

    def process_passes(self):
        """Process passing events and update statistics."""
        for event in self.events:
            if event.get('event_type') == 'pass':
                passer_id = event.get('player_id')
                receiver_id = event.get('metadata', {}).get('receiver')

                if passer_id:
                    stats = self._get_or_create_stats(passer_id)
                    stats.passes += 1

                # Track assists (pass followed by made basket)
                # This would require more sophisticated event sequencing
                # For now, increment assists based on successful passes
                if receiver_id and passer_id:
                    # Simplified: assume some passes lead to assists
                    # Real implementation would check for subsequent shot
                    pass

        logger.info("Processed passing statistics")

    def process_dribbles(self):
        """Process dribbling events and update statistics."""
        for event in self.events:
            if event.get('event_type') == 'dribble':
                player_id = event.get('player_id')
                bounces = event.get('metadata', {}).get('bounces', 0)

                if player_id:
                    stats = self._get_or_create_stats(player_id)
                    stats.dribbles += 1
                    stats.total_bounces += bounces

        logger.info("Processed dribbling statistics")

    def calculate_time_statistics(self):
        """Calculate time-based statistics for each player."""
        # Track frames where each player is visible
        player_frames = defaultdict(set)

        for frame_idx, players in self.player_detections.items():
            for player in players:
                player_id = player.get('player_id')
                if player_id:
                    player_frames[player_id].add(int(frame_idx))

        # Update time on court for each player
        for player_id, frames in player_frames.items():
            stats = self._get_or_create_stats(player_id)
            stats.time_on_court = len(frames)

        logger.info("Calculated time statistics")

    def calculate_distance_traveled(self):
        """Calculate total distance traveled by each player."""
        player_positions = defaultdict(list)

        # Collect all positions for each player
        for frame_idx in sorted(self.player_detections.keys(), key=int):
            players = self.player_detections[frame_idx]
            for player in players:
                player_id = player.get('player_id')
                center = player.get('center')
                if player_id and center:
                    player_positions[player_id].append(center)

        # Calculate cumulative distance
        for player_id, positions in player_positions.items():
            stats = self._get_or_create_stats(player_id)
            total_distance = 0.0

            for i in range(1, len(positions)):
                distance = self._calculate_distance(positions[i-1], positions[i])
                total_distance += distance

            stats.distance_traveled = total_distance

        logger.info("Calculated distance statistics")

    def assign_teams(self):
        """Assign team information to player statistics."""
        # Use team info from player detections
        for frame_idx, players in self.player_detections.items():
            for player in players:
                player_id = player.get('player_id')
                team = player.get('team')
                if player_id and team:
                    stats = self._get_or_create_stats(player_id)
                    if stats.team is None:
                        stats.team = team

        logger.info("Assigned team information")

    def generate_all_statistics(self) -> Dict[int, PlayerStatistics]:
        """
        Generate comprehensive statistics for all players.

        Returns:
            Dictionary mapping player IDs to PlayerStatistics objects
        """
        logger.info("Generating comprehensive player statistics...")

        self.assign_teams()
        self.process_shots()
        self.process_passes()
        self.process_dribbles()
        self.calculate_time_statistics()
        self.calculate_distance_traveled()

        logger.info(f"Statistics generated for {len(self.player_stats)} players")
        return self.player_stats

    def save_statistics(self, output_path: str):
        """Save player statistics to JSON file."""
        stats_dict = {
            player_id: asdict(stats)
            for player_id, stats in self.player_stats.items()
        }

        with open(output_path, 'w') as f:
            json.dump(stats_dict, f, indent=2)

        logger.info(f"Statistics saved to {output_path}")

    def generate_summary_report(self) -> str:
        """
        Generate human-readable summary report.

        Returns:
            Formatted string with statistics summary
        """
        report = []
        report.append("=" * 60)
        report.append("BASKETBALL GAME STATISTICS SUMMARY")
        report.append("=" * 60)
        report.append("")

        # Sort players by team
        by_team = defaultdict(list)
        for player_id, stats in self.player_stats.items():
            team = stats.team or "Unknown"
            by_team[team].append((player_id, stats))

        for team, players in sorted(by_team.items()):
            report.append(f"\n{team}")
            report.append("-" * 60)

            for player_id, stats in sorted(players, key=lambda x: x[0]):
                report.append(f"\nPlayer #{player_id}:")
                report.append(f"  Shots: {stats.shots_made}/{stats.shots_attempted} "
                            f"({stats.shooting_percentage:.1f}%)")
                report.append(f"  Passes: {stats.passes}")
                report.append(f"  Dribbles: {stats.dribbles} ({stats.total_bounces} bounces)")
                report.append(f"  Time on court: {stats.time_on_court} frames")
                report.append(f"  Distance traveled: {stats.distance_traveled:.1f} pixels")

        report.append("\n" + "=" * 60)

        return "\n".join(report)

    def save_summary_report(self, output_path: str):
        """Save summary report to text file."""
        report = self.generate_summary_report()
        with open(output_path, 'w') as f:
            f.write(report)
        logger.info(f"Summary report saved to {output_path}")
        print(report)


def main():
    """Main entry point for statistics generation."""
    import argparse
    parser = argparse.ArgumentParser(description='Generate basketball player statistics')
    parser.add_argument('--events', required=True, help='Path to events JSON')
    parser.add_argument('--players', required=True, help='Path to players JSON')
    parser.add_argument('--output', default='outputs/statistics.json', help='Output JSON file')
    parser.add_argument('--report', default='outputs/statistics_report.txt', help='Output report file')
    args = parser.parse_args()

    # Generate statistics
    generator = StatisticsGenerator(args.events, args.players)
    generator.generate_all_statistics()
    generator.save_statistics(args.output)
    generator.save_summary_report(args.report)


if __name__ == '__main__':
    main()
