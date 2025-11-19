#!/usr/bin/env python3
"""
Professional logging system for basketball tracker.

Replaces print statements with structured logging.
"""

import logging
import sys
import os
from pathlib import Path
from typing import Optional
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }

    # Use ASCII characters for Windows compatibility
    ICONS_UNICODE = {
        'DEBUG': '🔍',
        'INFO': '✓',
        'WARNING': '⚠',
        'ERROR': '❌',
        'CRITICAL': '🔥'
    }

    ICONS_ASCII = {
        'DEBUG': '[D]',
        'INFO': '[+]',
        'WARNING': '[!]',
        'ERROR': '[X]',
        'CRITICAL': '[!!]'
    }

    # Detect Windows console encoding
    @staticmethod
    def supports_unicode():
        """Check if console supports unicode."""
        try:
            # Try to encode unicode character
            sys.stdout.encoding
            '✓'.encode(sys.stdout.encoding or 'utf-8')
            return True
        except (UnicodeEncodeError, AttributeError, LookupError):
            return False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Choose icon set based on platform/encoding
        if os.name == 'nt' or not self.supports_unicode():
            self.ICONS = self.ICONS_ASCII
        else:
            self.ICONS = self.ICONS_UNICODE

    def format(self, record):
        """Format log record with colors and icons."""
        # Add color
        if record.levelname in self.COLORS:
            record.levelname_colored = (
                f"{self.COLORS[record.levelname]}"
                f"{self.ICONS.get(record.levelname, '')} {record.levelname}"
                f"{self.COLORS['RESET']}"
            )
        else:
            record.levelname_colored = record.levelname

        # Format message
        log_message = super().format(record)
        return log_message


class LoggerManager:
    """Manages logger configuration and setup."""

    _loggers = {}

    @classmethod
    def get_logger(
        cls,
        name: str,
        level: Optional[str] = None,
        log_file: Optional[str] = None,
        console_output: bool = True,
        file_output: bool = True
    ) -> logging.Logger:
        """
        Get or create a logger with specified configuration.

        Args:
            name: Logger name (usually __name__)
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Path to log file
            console_output: Enable console output
            file_output: Enable file output

        Returns:
            Configured logger instance
        """
        # Return existing logger if already configured
        if name in cls._loggers:
            return cls._loggers[name]

        # Create logger
        logger = logging.getLogger(name)

        # Set level
        if level is None:
            # Try to load from config
            try:
                from .config_loader import get_config
                config = get_config()
                level = config.get('logging.level', 'INFO')
            except:
                level = 'INFO'

        logger.setLevel(getattr(logging, level.upper()))

        # Remove existing handlers
        logger.handlers.clear()

        # Console handler with colors
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)

            # Colored format for console
            console_format = '%(levelname_colored)s %(message)s'
            console_formatter = ColoredFormatter(console_format)
            console_handler.setFormatter(console_formatter)

            logger.addHandler(console_handler)

        # File handler
        if file_output:
            # Get log file path from config or use default
            if log_file is None:
                try:
                    from .config_loader import get_config
                    config = get_config()
                    log_dir = Path(config.get_log_dir())
                    log_dir.mkdir(parents=True, exist_ok=True)

                    # Create timestamped log file
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    log_file = log_dir / f"basketball_tracker_{timestamp}.log"
                except:
                    log_dir = Path("logs")
                    log_dir.mkdir(parents=True, exist_ok=True)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    log_file = log_dir / f"basketball_tracker_{timestamp}.log"

            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)

            # Detailed format for file
            file_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            file_formatter = logging.Formatter(
                file_format,
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)

            logger.addHandler(file_handler)

            # Log the log file location
            logger.debug(f"Logging to file: {log_file}")

        # Prevent propagation to root logger
        logger.propagate = False

        # Store logger
        cls._loggers[name] = logger

        return logger

    @classmethod
    def setup_logger_from_config(cls, name: str) -> logging.Logger:
        """
        Setup logger using configuration from config.yaml.

        Args:
            name: Logger name

        Returns:
            Configured logger
        """
        try:
            from .config_loader import get_config
            config = get_config()

            return cls.get_logger(
                name=name,
                level=config.get('logging.level', 'INFO'),
                console_output=config.get('logging.console_output', True),
                file_output=config.get('logging.file_output', True)
            )
        except Exception as e:
            # Fallback to default configuration
            print(f"Warning: Could not load config, using default logging: {e}")
            return cls.get_logger(name)


def get_logger(name: str) -> logging.Logger:
    """
    Convenience function to get a logger.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance

    Example:
        >>> from src.utils.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
        >>> logger.error("Something went wrong")
    """
    return LoggerManager.setup_logger_from_config(name)


class ProgressLogger:
    """Helper class for logging progress with consistent formatting."""

    def __init__(self, logger: logging.Logger, total: int, desc: str = "Progress"):
        """
        Initialize progress logger.

        Args:
            logger: Logger instance
            total: Total number of items
            desc: Description of the task
        """
        self.logger = logger
        self.total = total
        self.desc = desc
        self.current = 0
        self.last_percentage = -1

    def update(self, n: int = 1):
        """
        Update progress.

        Args:
            n: Number of items completed
        """
        self.current += n
        percentage = int((self.current / self.total) * 100)

        # Only log at percentage milestones to avoid spam
        if percentage != self.last_percentage and percentage % 10 == 0:
            self.logger.info(
                f"{self.desc}: {percentage}% ({self.current}/{self.total})"
            )
            self.last_percentage = percentage

    def finish(self):
        """Log completion."""
        self.logger.info(f"{self.desc}: Complete! ({self.total}/{self.total})")


# Convenience functions for quick logging without logger setup
def info(message: str):
    """Quick info log."""
    logger = get_logger('basketball_tracker')
    logger.info(message)


def warning(message: str):
    """Quick warning log."""
    logger = get_logger('basketball_tracker')
    logger.warning(message)


def error(message: str):
    """Quick error log."""
    logger = get_logger('basketball_tracker')
    logger.error(message)


def debug(message: str):
    """Quick debug log."""
    logger = get_logger('basketball_tracker')
    logger.debug(message)


if __name__ == '__main__':
    # Test logging
    logger = get_logger(__name__)

    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    # Test progress logger
    progress = ProgressLogger(logger, 100, "Processing frames")
    for i in range(100):
        progress.update(1)
    progress.finish()
