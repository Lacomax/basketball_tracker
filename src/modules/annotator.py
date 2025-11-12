import cv2
import numpy as np
import json
import os
import logging

from ..utils.ball_detection import auto_detect_ball
from ..config import setup_logging

logger = setup_logging(__name__)


class BallAnnotator:
    """Interactive annotation tool for manual basketball detection in videos."""

    def __init__(self, video: str = 'data/input_video.mp4', output: str = 'outputs/annotations.json'):
        """
        Initialize the annotation tool.

        Args:
            video: Path to input video file
            output: Path for saving annotations JSON
        """
        os.makedirs(os.path.dirname(output), exist_ok=True)
        self.cap = cv2.VideoCapture(video)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {video}")
        self.output = output
        # Load existing annotations if available
        try:
            with open(self.output, 'r') as f:
                self.annotations = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.annotations = {}
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_idx = 0
        self.current_frame = None
        self.selected_ball = None

    def _draw_annotations(self, frame: np.ndarray) -> np.ndarray:
        """Draw the current frame’s annotation (if any) and info text."""
        frame_key = str(self.frame_idx)
        if frame_key in self.annotations:
            ann = self.annotations[frame_key]
            center = tuple(map(int, ann['center']))
            cv2.circle(frame, center, ann['radius'], (0, 255, 0), 2)
        # Overlay frame index and usage instructions
        cv2.putText(frame, f"Frame: {self.frame_idx+1}/{self.total_frames}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "A/D: Prev/Next  S: Save  Q: Quit", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return frame

    def _mouse_handler(self, event, x, y, flags, param):
        """Handle mouse events for annotation tool."""
        frame_key = str(self.frame_idx)
        if event == cv2.EVENT_LBUTTONDOWN:
            # On click, detect ball and start dragging
            self.annotations[frame_key] = auto_detect_ball(self.current_frame, (x, y))
            self.selected_ball = frame_key
            cv2.imshow('Annotator', self._draw_annotations(self.current_frame.copy()))
        elif event == cv2.EVENT_MOUSEMOVE and self.selected_ball:
            # Update center position while dragging
            self.annotations[self.selected_ball]['center'] = [x, y]
            cv2.imshow('Annotator', self._draw_annotations(self.current_frame.copy()))
        elif event == cv2.EVENT_LBUTTONUP:
            # Stop dragging
            self.selected_ball = None

    def run(self):
        """Run the interactive annotation tool."""
        cv2.namedWindow('Annotator')
        cv2.setMouseCallback('Annotator', self._mouse_handler)
        while self.frame_idx < self.total_frames:
            # Load and display the current frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx)
            ret, self.current_frame = self.cap.read()
            if not ret:
                break
            cv2.imshow('Annotator', self._draw_annotations(self.current_frame.copy()))
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save annotations to JSON
                with open(self.output, 'w') as f:
                    json.dump(self.annotations, f, indent=2)
            # Frame navigation (A/← for prev, D/→ for next)
            elif (key == ord('a') or key == 81) and self.frame_idx > 0:
                self.frame_idx -= 1
            elif (key == ord('d') or key == 83) and self.frame_idx < self.total_frames - 1:
                self.frame_idx += 1
        cv2.destroyAllWindows()
        self.cap.release()

def main():
    BallAnnotator().run()

if __name__ == '__main__':
    main()
