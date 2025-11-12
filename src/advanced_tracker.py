"""
Advanced Basketball Tracker with comprehensive analytics.

This module orchestrates the complete basketball analysis pipeline
including ball tracking, player detection, event analysis, and statistics generation.
"""

import os
import cv2
import logging
from pathlib import Path
from typing import Optional

from .config import setup_logging
from .modules.trajectory_detector import process_trajectory_video
from .modules.player_detector import PlayerDetector
from .modules.event_analyzer import EventAnalyzer
from .modules.statistics_generator import StatisticsGenerator
from .utils.database import BasketballDatabase

logger = setup_logging(__name__)


class AdvancedBasketballTracker:
    """
    Advanced basketball tracker with full game analytics.

    This orchestrator runs the complete analysis pipeline:
    1. Ball detection and trajectory tracking (with occlusion handling)
    2. Player detection and team assignment
    3. Event detection (shots, passes, dribbles, etc.)
    4. Statistics generation per player
    5. Database persistence
    """

    def __init__(self, video_path: str, output_dir: str = 'outputs',
                 db_path: str = 'data/basketball_stats.db'):
        """
        Initialize advanced tracker.

        Args:
            video_path: Path to input video file
            output_dir: Directory for output files
            db_path: Path to SQLite database
        """
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.db_path = db_path

        # File paths for intermediate outputs
        self.files = {
            'ball_detections': self.output_dir / 'ball_detections.json',
            'players': self.output_dir / 'players.json',
            'events': self.output_dir / 'events.json',
            'statistics': self.output_dir / 'statistics.json',
            'report': self.output_dir / 'statistics_report.txt'
        }

        # Get video metadata
        self.video_metadata = self._get_video_metadata()

        logger.info(f"Initialized AdvancedBasketballTracker for {video_path}")

    def _get_video_metadata(self) -> dict:
        """Extract video metadata."""
        cap = cv2.VideoCapture(self.video_path)
        metadata = {
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration_seconds': 0.0
        }
        if metadata['fps'] > 0:
            metadata['duration_seconds'] = metadata['total_frames'] / metadata['fps']
        cap.release()
        return metadata

    def detect_ball(self, annotations_path: str) -> dict:
        """
        Run ball detection with enhanced occlusion handling.

        Args:
            annotations_path: Path to manual annotations JSON

        Returns:
            Dictionary of ball detections per frame
        """
        logger.info("Step 1: Detecting ball with enhanced occlusion handling...")

        detections = process_trajectory_video(
            video_path=self.video_path,
            annotations_path=annotations_path,
            output_path=str(self.files['ball_detections'])
        )

        logger.info(f"Ball detections saved to {self.files['ball_detections']}")
        return detections

    def detect_players(self, use_pose: bool = False, detect_teams: bool = True) -> dict:
        """
        Run player detection and tracking.

        Args:
            use_pose: Whether to use pose estimation
            detect_teams: Whether to assign team labels

        Returns:
            Dictionary of player detections per frame
        """
        logger.info("Step 2: Detecting and tracking players...")

        detector = PlayerDetector()
        detections = detector.process_video(
            video_path=self.video_path,
            output_path=str(self.files['players']),
            use_pose=use_pose,
            detect_teams=detect_teams
        )

        logger.info(f"Player detections saved to {self.files['players']}")
        return detections

    def analyze_events(self) -> list:
        """
        Analyze game events (shots, passes, dribbles, etc.).

        Returns:
            List of detected events
        """
        logger.info("Step 3: Analyzing game events...")

        import json
        with open(self.files['ball_detections'], 'r') as f:
            ball_detections = json.load(f)

        with open(self.files['players'], 'r') as f:
            player_detections = json.load(f)

        analyzer = EventAnalyzer(ball_detections, player_detections)
        events = analyzer.analyze_all_events()
        analyzer.save_events(str(self.files['events']))

        logger.info(f"Events saved to {self.files['events']}")
        return events

    def generate_statistics(self) -> dict:
        """
        Generate comprehensive player statistics.

        Returns:
            Dictionary of player statistics
        """
        logger.info("Step 4: Generating player statistics...")

        generator = StatisticsGenerator(
            events_file=str(self.files['events']),
            players_file=str(self.files['players'])
        )

        stats = generator.generate_all_statistics()
        generator.save_statistics(str(self.files['statistics']))
        generator.save_summary_report(str(self.files['report']))

        logger.info(f"Statistics saved to {self.files['statistics']}")
        return stats

    def persist_to_database(self, game_id: Optional[int] = None) -> int:
        """
        Persist all results to SQLite database.

        Args:
            game_id: Optional existing game ID to update

        Returns:
            Game ID in database
        """
        logger.info("Step 5: Persisting results to database...")

        import json

        with BasketballDatabase(self.db_path) as db:
            # Create or get game record
            if game_id is None:
                game_id = db.insert_game(
                    video_path=self.video_path,
                    total_frames=self.video_metadata['total_frames'],
                    duration_seconds=self.video_metadata['duration_seconds'],
                    metadata={
                        'fps': self.video_metadata['fps'],
                        'width': self.video_metadata['width'],
                        'height': self.video_metadata['height']
                    }
                )

            # Insert ball detections
            with open(self.files['ball_detections'], 'r') as f:
                ball_detections = json.load(f)
            db.bulk_insert_ball_detections(game_id, ball_detections)

            # Insert players
            with open(self.files['players'], 'r') as f:
                player_detections = json.load(f)
            unique_players = set()
            for frame_players in player_detections.values():
                for player in frame_players:
                    player_id = player.get('player_id')
                    team = player.get('team')
                    if player_id and player_id not in unique_players:
                        db.insert_player(game_id, player_id, team=team)
                        unique_players.add(player_id)

            # Insert events
            with open(self.files['events'], 'r') as f:
                events = json.load(f)
            for event in events:
                db.insert_event(game_id, event)

            # Insert player statistics
            with open(self.files['statistics'], 'r') as f:
                statistics = json.load(f)
            for player_id, stats in statistics.items():
                stats['player_id'] = int(player_id)
                db.insert_player_statistics(game_id, stats)

        logger.info(f"All results persisted to database (Game ID: {game_id})")
        return game_id

    def run_full_analysis(self, annotations_path: str, use_pose: bool = False,
                         detect_teams: bool = True, save_to_db: bool = True) -> int:
        """
        Run complete basketball game analysis pipeline.

        Args:
            annotations_path: Path to manual ball annotations
            use_pose: Whether to use player pose estimation
            detect_teams: Whether to detect and assign teams
            save_to_db: Whether to persist results to database

        Returns:
            Game ID if saved to database, else 0
        """
        logger.info("=" * 60)
        logger.info("STARTING COMPLETE BASKETBALL ANALYSIS PIPELINE")
        logger.info("=" * 60)

        # Step 1: Ball detection
        self.detect_ball(annotations_path)

        # Step 2: Player detection
        self.detect_players(use_pose=use_pose, detect_teams=detect_teams)

        # Step 3: Event analysis
        self.analyze_events()

        # Step 4: Statistics generation
        self.generate_statistics()

        # Step 5: Database persistence
        game_id = 0
        if save_to_db:
            game_id = self.persist_to_database()

        logger.info("=" * 60)
        logger.info("ANALYSIS COMPLETE!")
        logger.info(f"Results saved to: {self.output_dir}")
        if save_to_db:
            logger.info(f"Database: {self.db_path} (Game ID: {game_id})")
        logger.info("=" * 60)

        # Print summary report
        with open(self.files['report'], 'r') as f:
            print(f.read())

        return game_id


def main():
    """Main entry point for advanced tracker."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Advanced Basketball Tracker with comprehensive analytics'
    )
    parser.add_argument('--video', required=True, help='Path to input video')
    parser.add_argument('--annotations', required=True,
                       help='Path to ball annotations JSON')
    parser.add_argument('--output', default='outputs/advanced',
                       help='Output directory')
    parser.add_argument('--db', default='data/basketball_stats.db',
                       help='SQLite database path')
    parser.add_argument('--pose', action='store_true',
                       help='Use player pose estimation')
    parser.add_argument('--no-teams', action='store_true',
                       help='Disable team detection')
    parser.add_argument('--no-db', action='store_true',
                       help='Do not save to database')

    args = parser.parse_args()

    # Run analysis
    tracker = AdvancedBasketballTracker(
        video_path=args.video,
        output_dir=args.output,
        db_path=args.db
    )

    tracker.run_full_analysis(
        annotations_path=args.annotations,
        use_pose=args.pose,
        detect_teams=not args.no_teams,
        save_to_db=not args.no_db
    )


if __name__ == '__main__':
    main()
