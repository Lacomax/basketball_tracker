"""
Database module for persisting basketball statistics.

This module handles SQLite database operations for storing
game statistics, events, and player data.
"""

import sqlite3
import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

from ..config import setup_logging

logger = setup_logging(__name__)


class BasketballDatabase:
    """SQLite database manager for basketball statistics."""

    def __init__(self, db_path: str = "data/basketball_stats.db"):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self.cursor = None
        self._connect()
        self._create_tables()

    def _connect(self):
        """Establish database connection."""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        logger.info(f"Connected to database: {self.db_path}")

    def _create_tables(self):
        """Create database tables if they don't exist."""
        # Games table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS games (
                game_id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_path TEXT NOT NULL,
                date_processed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_frames INTEGER,
                duration_seconds REAL,
                metadata TEXT
            )
        ''')

        # Players table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS players (
                player_id INTEGER PRIMARY KEY,
                game_id INTEGER,
                team TEXT,
                jersey_number INTEGER,
                name TEXT,
                FOREIGN KEY (game_id) REFERENCES games (game_id)
            )
        ''')

        # Player statistics table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS player_statistics (
                stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id INTEGER,
                player_id INTEGER,
                shots_attempted INTEGER DEFAULT 0,
                shots_made INTEGER DEFAULT 0,
                shooting_percentage REAL DEFAULT 0.0,
                passes INTEGER DEFAULT 0,
                assists INTEGER DEFAULT 0,
                turnovers INTEGER DEFAULT 0,
                dribbles INTEGER DEFAULT 0,
                total_bounces INTEGER DEFAULT 0,
                rebounds INTEGER DEFAULT 0,
                steals INTEGER DEFAULT 0,
                blocks INTEGER DEFAULT 0,
                time_on_court INTEGER DEFAULT 0,
                time_with_ball INTEGER DEFAULT 0,
                distance_traveled REAL DEFAULT 0.0,
                FOREIGN KEY (game_id) REFERENCES games (game_id),
                FOREIGN KEY (player_id) REFERENCES players (player_id)
            )
        ''')

        # Events table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id INTEGER,
                event_type TEXT NOT NULL,
                frame_start INTEGER,
                frame_end INTEGER,
                player_id INTEGER,
                confidence REAL,
                trajectory TEXT,
                metadata TEXT,
                FOREIGN KEY (game_id) REFERENCES games (game_id),
                FOREIGN KEY (player_id) REFERENCES players (player_id)
            )
        ''')

        # Ball detections table (for caching)
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS ball_detections (
                detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id INTEGER,
                frame_number INTEGER,
                center_x INTEGER,
                center_y INTEGER,
                radius INTEGER,
                confidence REAL,
                occluded BOOLEAN DEFAULT 0,
                velocity REAL,
                FOREIGN KEY (game_id) REFERENCES games (game_id)
            )
        ''')

        self.conn.commit()
        logger.info("Database tables created/verified")

    def insert_game(self, video_path: str, total_frames: int = 0,
                   duration_seconds: float = 0.0, metadata: Dict = None) -> int:
        """
        Insert a new game record.

        Args:
            video_path: Path to video file
            total_frames: Total number of frames in video
            duration_seconds: Duration of video in seconds
            metadata: Additional metadata as dictionary

        Returns:
            Game ID of inserted record
        """
        metadata_json = json.dumps(metadata) if metadata else None
        self.cursor.execute('''
            INSERT INTO games (video_path, total_frames, duration_seconds, metadata)
            VALUES (?, ?, ?, ?)
        ''', (video_path, total_frames, duration_seconds, metadata_json))
        self.conn.commit()
        game_id = self.cursor.lastrowid
        logger.info(f"Inserted game record with ID: {game_id}")
        return game_id

    def insert_player(self, game_id: int, player_id: int, team: str = None,
                     jersey_number: int = None, name: str = None):
        """Insert or update player record."""
        self.cursor.execute('''
            INSERT OR REPLACE INTO players (player_id, game_id, team, jersey_number, name)
            VALUES (?, ?, ?, ?, ?)
        ''', (player_id, game_id, team, jersey_number, name))
        self.conn.commit()
        logger.debug(f"Inserted/updated player {player_id} for game {game_id}")

    def insert_player_statistics(self, game_id: int, stats: Dict):
        """
        Insert player statistics.

        Args:
            game_id: Game ID
            stats: Dictionary with player statistics
        """
        self.cursor.execute('''
            INSERT INTO player_statistics (
                game_id, player_id, shots_attempted, shots_made, shooting_percentage,
                passes, assists, turnovers, dribbles, total_bounces,
                rebounds, steals, blocks, time_on_court, time_with_ball, distance_traveled
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            game_id,
            stats.get('player_id'),
            stats.get('shots_attempted', 0),
            stats.get('shots_made', 0),
            stats.get('shooting_percentage', 0.0),
            stats.get('passes', 0),
            stats.get('assists', 0),
            stats.get('turnovers', 0),
            stats.get('dribbles', 0),
            stats.get('total_bounces', 0),
            stats.get('rebounds', 0),
            stats.get('steals', 0),
            stats.get('blocks', 0),
            stats.get('time_on_court', 0),
            stats.get('time_with_ball', 0),
            stats.get('distance_traveled', 0.0)
        ))
        self.conn.commit()

    def insert_event(self, game_id: int, event: Dict):
        """Insert game event."""
        trajectory = json.dumps(event.get('ball_trajectory')) if event.get('ball_trajectory') else None
        metadata = json.dumps(event.get('metadata')) if event.get('metadata') else None

        self.cursor.execute('''
            INSERT INTO events (game_id, event_type, frame_start, frame_end,
                              player_id, confidence, trajectory, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            game_id,
            event.get('event_type'),
            event.get('frame_start'),
            event.get('frame_end'),
            event.get('player_id'),
            event.get('confidence', 0.0),
            trajectory,
            metadata
        ))
        self.conn.commit()

    def insert_ball_detection(self, game_id: int, frame_number: int, detection: Dict):
        """Insert ball detection data."""
        center = detection.get('center', [0, 0])
        self.cursor.execute('''
            INSERT INTO ball_detections (game_id, frame_number, center_x, center_y,
                                        radius, confidence, occluded, velocity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            game_id,
            frame_number,
            center[0],
            center[1],
            detection.get('radius', 0),
            detection.get('confidence', 1.0),
            1 if detection.get('occluded', False) else 0,
            detection.get('velocity', 0.0)
        ))

    def bulk_insert_ball_detections(self, game_id: int, detections: Dict):
        """Bulk insert ball detections for better performance."""
        data = []
        for frame_num, detection in detections.items():
            center = detection.get('center', [0, 0])
            data.append((
                game_id,
                int(frame_num),
                center[0],
                center[1],
                detection.get('radius', 0),
                detection.get('confidence', 1.0),
                1 if detection.get('occluded', False) else 0,
                detection.get('velocity', 0.0)
            ))

        self.cursor.executemany('''
            INSERT INTO ball_detections (game_id, frame_number, center_x, center_y,
                                        radius, confidence, occluded, velocity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', data)
        self.conn.commit()
        logger.info(f"Bulk inserted {len(data)} ball detections")

    def get_player_statistics(self, game_id: int, player_id: int = None) -> List[Dict]:
        """
        Retrieve player statistics.

        Args:
            game_id: Game ID
            player_id: Optional player ID to filter

        Returns:
            List of statistics dictionaries
        """
        if player_id:
            query = '''
                SELECT * FROM player_statistics
                WHERE game_id = ? AND player_id = ?
            '''
            self.cursor.execute(query, (game_id, player_id))
        else:
            query = '''
                SELECT * FROM player_statistics
                WHERE game_id = ?
            '''
            self.cursor.execute(query, (game_id,))

        columns = [desc[0] for desc in self.cursor.description]
        results = []
        for row in self.cursor.fetchall():
            results.append(dict(zip(columns, row)))

        return results

    def get_events(self, game_id: int, event_type: str = None) -> List[Dict]:
        """
        Retrieve game events.

        Args:
            game_id: Game ID
            event_type: Optional event type filter

        Returns:
            List of event dictionaries
        """
        if event_type:
            query = '''
                SELECT * FROM events
                WHERE game_id = ? AND event_type = ?
                ORDER BY frame_start
            '''
            self.cursor.execute(query, (game_id, event_type))
        else:
            query = '''
                SELECT * FROM events
                WHERE game_id = ?
                ORDER BY frame_start
            '''
            self.cursor.execute(query, (game_id,))

        columns = [desc[0] for desc in self.cursor.description]
        results = []
        for row in self.cursor.fetchall():
            event = dict(zip(columns, row))
            # Parse JSON fields
            if event.get('trajectory'):
                event['trajectory'] = json.loads(event['trajectory'])
            if event.get('metadata'):
                event['metadata'] = json.loads(event['metadata'])
            results.append(event)

        return results

    def get_all_games(self) -> List[Dict]:
        """Retrieve all game records."""
        self.cursor.execute('SELECT * FROM games ORDER BY date_processed DESC')
        columns = [desc[0] for desc in self.cursor.description]
        results = []
        for row in self.cursor.fetchall():
            game = dict(zip(columns, row))
            if game.get('metadata'):
                game['metadata'] = json.loads(game['metadata'])
            results.append(game)
        return results

    def delete_game(self, game_id: int):
        """Delete a game and all associated data."""
        self.cursor.execute('DELETE FROM ball_detections WHERE game_id = ?', (game_id,))
        self.cursor.execute('DELETE FROM events WHERE game_id = ?', (game_id,))
        self.cursor.execute('DELETE FROM player_statistics WHERE game_id = ?', (game_id,))
        self.cursor.execute('DELETE FROM players WHERE game_id = ?', (game_id,))
        self.cursor.execute('DELETE FROM games WHERE game_id = ?', (game_id,))
        self.conn.commit()
        logger.info(f"Deleted game {game_id} and all associated data")

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def main():
    """Example usage of database module."""
    with BasketballDatabase() as db:
        # Example: Insert a game
        game_id = db.insert_game(
            video_path="data/raw/game1.mp4",
            total_frames=5000,
            duration_seconds=180.0,
            metadata={"team_a": "Lakers", "team_b": "Warriors"}
        )
        print(f"Created game with ID: {game_id}")

        # Example: Insert player
        db.insert_player(game_id, player_id=23, team="Lakers", jersey_number=23, name="Player 23")

        # Example: Query games
        games = db.get_all_games()
        print(f"Total games in database: {len(games)}")


if __name__ == '__main__':
    main()
