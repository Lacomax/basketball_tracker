"""
Video utilities for robust video opening across different platforms.
"""

import cv2
import logging

logger = logging.getLogger(__name__)


def open_video_robust(video_path: str):
    """
    Open video with multiple fallback methods for cross-platform compatibility.

    Tries in order:
    1. FFMPEG backend (best, avoids GStreamer warnings)
    2. Default backend
    3. Any available backend

    Args:
        video_path: Path to video file

    Returns:
        cv2.VideoCapture object or None if all methods fail

    Raises:
        IOError: If video cannot be opened with any method
    """
    # Method 1: Try FFMPEG backend (Linux/Mac best)
    try:
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if cap.isOpened():
            logger.info(f"Opened video with FFMPEG backend: {video_path}")
            return cap
        cap.release()
    except Exception as e:
        logger.debug(f"FFMPEG backend failed: {e}")

    # Method 2: Try default backend
    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            logger.info(f"Opened video with default backend: {video_path}")
            return cap
        cap.release()
    except Exception as e:
        logger.debug(f"Default backend failed: {e}")

    # Method 3: Try MSMF backend (Windows best)
    try:
        cap = cv2.VideoCapture(video_path, cv2.CAP_MSMF)
        if cap.isOpened():
            logger.info(f"Opened video with MSMF backend: {video_path}")
            return cap
        cap.release()
    except Exception as e:
        logger.debug(f"MSMF backend failed: {e}")

    # Method 4: Try DSHOW backend (Windows alternative)
    try:
        cap = cv2.VideoCapture(video_path, cv2.CAP_DSHOW)
        if cap.isOpened():
            logger.info(f"Opened video with DSHOW backend: {video_path}")
            return cap
        cap.release()
    except Exception as e:
        logger.debug(f"DSHOW backend failed: {e}")

    # Method 5: Try ANY backend
    try:
        cap = cv2.VideoCapture(video_path, cv2.CAP_ANY)
        if cap.isOpened():
            logger.info(f"Opened video with ANY backend: {video_path}")
            return cap
        cap.release()
    except Exception as e:
        logger.debug(f"ANY backend failed: {e}")

    # All methods failed
    raise IOError(f"Cannot open video: {video_path}. Tried all available backends.")


def create_video_writer_robust(output_path: str, fps: int, width: int, height: int):
    """
    Create VideoWriter with multiple fallback codecs for cross-platform compatibility.

    Tries in order:
    1. H.264 (avc1) - best quality and compatibility
    2. mp4v - widely supported
    3. XVID - fallback

    Args:
        output_path: Output video path
        fps: Frames per second
        width: Video width
        height: Video height

    Returns:
        cv2.VideoWriter object

    Raises:
        IOError: If video writer cannot be created with any codec
    """
    codecs = [
        ('avc1', cv2.CAP_FFMPEG, 'H.264'),
        ('mp4v', cv2.CAP_FFMPEG, 'MP4V'),
        ('XVID', None, 'XVID'),
        ('MJPG', None, 'MJPEG'),
    ]

    for codec_str, backend, name in codecs:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec_str)

            if backend is not None:
                writer = cv2.VideoWriter(output_path, backend, fourcc, fps, (width, height))
            else:
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            if writer.isOpened():
                logger.info(f"Created video writer with {name} codec")
                return writer

            writer.release()
        except Exception as e:
            logger.debug(f"{name} codec failed: {e}")

    # Fallback: try without specifying backend
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if writer.isOpened():
            logger.info("Created video writer with default mp4v codec")
            return writer
    except Exception as e:
        logger.debug(f"Default mp4v failed: {e}")

    raise IOError(f"Cannot create video writer: {output_path}. Tried all available codecs.")
