#!/usr/bin/env python3
"""
Configuration loader for basketball tracker.

Loads configuration from config.yaml and provides easy access to settings.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional


class Config:
    """Configuration manager for basketball tracker."""

    _instance = None
    _config = None

    def __new__(cls):
        """Singleton pattern to ensure only one config instance."""
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize configuration."""
        if self._config is None:
            self.load()

    def load(self, config_path: Optional[str] = None):
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to config file. If None, searches for config.yaml
                        in standard locations.
        """
        if config_path is None:
            # Search for config.yaml in standard locations
            search_paths = [
                Path.cwd() / "config.yaml",
                Path.cwd() / "config" / "config.yaml",
                Path(__file__).parent.parent.parent / "config.yaml",
            ]

            for path in search_paths:
                if path.exists():
                    config_path = str(path)
                    break

            if config_path is None:
                raise FileNotFoundError(
                    "config.yaml not found. Searched locations:\n" +
                    "\n".join(f"  - {p}" for p in search_paths)
                )

        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key: Configuration key in dot notation (e.g., 'video.output_dir')
            default: Default value if key not found

        Returns:
            Configuration value or default

        Example:
            >>> config = Config()
            >>> output_dir = config.get('video.output_dir')
            >>> max_players = config.get('tracking.max_players_per_frame', 10)
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_video_paths(self) -> List[str]:
        """Get list of input video paths to search."""
        return self.get('video.input_paths', [])

    def get_output_dir(self) -> str:
        """Get output directory path."""
        return self.get('video.output_dir', 'outputs')

    def get_log_dir(self) -> str:
        """Get log directory path."""
        return self.get('logging.log_dir', 'logs')

    def get_max_players_per_frame(self) -> int:
        """Get maximum players per frame."""
        return self.get('tracking.max_players_per_frame', 10)

    def get_confidence_threshold(self) -> float:
        """Get confidence threshold for tracking."""
        return self.get('tracking.confidence_threshold', 0.7)

    def get_trail_length(self) -> int:
        """Get player trail length in frames."""
        return self.get('trails.max_length', 30)

    def get_ball_trajectory_history(self) -> int:
        """Get ball trajectory history in frames."""
        return self.get('ball.trajectory_history', 20)

    def get_team_colors(self) -> Dict[str, List[int]]:
        """Get team colors configuration."""
        return self.get('team_colors', {})

    def get_num_workers(self) -> int:
        """Get number of parallel workers."""
        return self.get('video_processing.num_workers', 4)

    def get_batch_size(self) -> int:
        """Get batch size for parallel processing."""
        return self.get('video_processing.batch_size', 32)

    def use_gpu(self) -> bool:
        """Check if GPU processing is enabled."""
        return self.get('video_processing.use_gpu', True)

    def get_codecs(self) -> List[str]:
        """Get list of video codecs to try."""
        primary = self.get('video_processing.codec', 'mp4v')
        fallbacks = self.get('video_processing.fallback_codecs', [])
        return [primary] + fallbacks

    def ensure_directories(self):
        """Ensure all required directories exist."""
        dirs = [
            self.get_output_dir(),
            self.get_log_dir(),
        ]

        for directory in dirs:
            Path(directory).mkdir(parents=True, exist_ok=True)


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get global configuration instance."""
    return config


def reload_config(config_path: Optional[str] = None):
    """Reload configuration from file."""
    config.load(config_path)


if __name__ == '__main__':
    # Test configuration loading
    cfg = get_config()
    print("Configuration loaded successfully!")
    print(f"Output directory: {cfg.get_output_dir()}")
    print(f"Max players per frame: {cfg.get_max_players_per_frame()}")
    print(f"Confidence threshold: {cfg.get_confidence_threshold()}")
    print(f"Use GPU: {cfg.use_gpu()}")
    print(f"Number of workers: {cfg.get_num_workers()}")
