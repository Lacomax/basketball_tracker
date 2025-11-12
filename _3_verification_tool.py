import cv2
import numpy as np
import json
import os
import argparse
import logging

from ball_detection_utils import auto_detect_ball
from config import setup_logging, ANOMALY_THRESHOLD, TRAJECTORY_WINDOW, CONNECTION_THRESHOLD

logger = setup_logging(__name__)


class CompactBallVerifier:
    """Interactive verification tool for basketball detections."""

    def __init__(self, video_path: str, detection_file: str = None, output_file: str = "outputs/verified.json"):
        """
        Initialize the verification tool.

        Args:
            video_path: Path to input video file
            detection_file: Path to detection JSON file (loaded if output doesn't exist)
            output_file: Path for saving verified detections
        """
        self.video_path = video_path
        self.output_file = output_file
        # Load previous verifications if they exist; otherwise load detections
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r') as f:
                    self.verified = json.load(f)
            except json.JSONDecodeError:
                self.verified = {}
        else:
            try:
                if detection_file and os.path.exists(detection_file):
                    with open(detection_file, 'r') as f:
                        self.verified = json.load(f)
                else:
                    self.verified = {}
            except (json.JSONDecodeError, FileNotFoundError):
                self.verified = {}
        # Ensure all keys are strings
        self.verified = {str(k): v for k, v in self.verified.items()}
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_idx = 0
        self.dirty = False
        self.current_frame = None
        self.full_traj = False  # Toggle to preview full trajectory
        self.anomaly_threshold = ANOMALY_THRESHOLD

    def _get_anomaly(self):
        """Detect anomalies by comparing with previous detection."""
        current_key = str(self.frame_idx)
        sorted_keys = sorted([int(k) for k in self.verified.keys()])
        prev_frame = None
        for k in sorted_keys:
            if k < self.frame_idx:
                prev_frame = k
            else:
                break
        if prev_frame is not None and current_key in self.verified:
            prev_center = np.array(self.verified[str(prev_frame)]['center'])
            curr_center = np.array(self.verified[current_key]['center'])
            if np.linalg.norm(curr_center - prev_center) > self.anomaly_threshold:
                return True
        return False

    def _compute_anomaly_frames(self):
        """Return sorted list of frames with anomalies."""
        anomaly_frames = []
        sorted_keys = sorted([int(k) for k in self.verified.keys()])
        for i in range(1, len(sorted_keys)):
            prev = sorted_keys[i-1]
            curr = sorted_keys[i]
            prev_center = np.array(self.verified[str(prev)]['center'])
            curr_center = np.array(self.verified[str(curr)]['center'])
            if np.linalg.norm(curr_center - prev_center) > self.anomaly_threshold:
                anomaly_frames.append(curr)
        return anomaly_frames

    def _mouse_handler(self, event: int, x: int, y: int, flags: int, param: any):
        """Handle mouse events for verification tool."""
        frame_key = str(self.frame_idx)
        if event == cv2.EVENT_LBUTTONDOWN:
            if frame_key in self.verified:
                existing = self.verified[frame_key]
                # Si clic cerca de la anotación existente, mover centro
                if np.hypot(x - existing['center'][0], y - existing['center'][1]) <= existing['radius']:
                    self.verified[frame_key]['center'] = [x, y]
                else:
                    self.verified[frame_key] = auto_detect_ball(self.current_frame, (x, y))
            else:
                self.verified[frame_key] = auto_detect_ball(self.current_frame, (x, y))
            self.dirty = True
            self._update_display()

    def _update_display(self):
        """Update the display with current frame and annotations."""
        if self.current_frame is None:
            return
        vis_frame = self.current_frame.copy()
        frame_key = str(self.frame_idx)
        # Select points to display: full trajectory or local window
        if self.full_traj:
            points = [(int(k), v) for k, v in self.verified.items()]
            points.sort(key=lambda x: x[0])
            line_color = (150, 150, 150)
        else:
            points = [(int(k), v) for k, v in self.verified.items() if abs(int(k) - self.frame_idx) <= TRAJECTORY_WINDOW]
            points.sort(key=lambda x: x[0])
            line_color = (0, 100, 255)
        for (prev_idx, prev_pt), (curr_idx, curr_pt) in zip(points[:-1], points[1:]):
            if abs(curr_idx - prev_idx) <= CONNECTION_THRESHOLD:
                cv2.line(vis_frame,
                         tuple(map(int, prev_pt['center'])),
                         tuple(map(int, curr_pt['center'])),
                         line_color, 1)
        # Draw current frame annotation
        if frame_key in self.verified:
            center = tuple(map(int, self.verified[frame_key]['center']))
            radius = self.verified[frame_key]['radius']
            if self.verified[frame_key].get('hidden', False):
                cv2.circle(vis_frame, center, radius, (0, 0, 255), 2)
                cv2.putText(vis_frame, "HIDDEN", (center[0] - 10, center[1] - radius - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.circle(vis_frame, center, radius, (0, 255, 0), 2)
                cv2.circle(vis_frame, center, 3, (0, 0, 255), -1)
        # Show anomaly indicator
        if self._get_anomaly():
            cv2.putText(vis_frame, "ANOMALY", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        # On-screen information
        cv2.putText(vis_frame, f"Frame: {self.frame_idx}/{self.total_frames}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        info_text = "a/d: Prev/Next   +/-: Radius   S: Save   Q: Quit   t: Toggle Traj   h: Toggle Hidden"
        cv2.putText(vis_frame, info_text, (10, vis_frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(vis_frame, "p: Prev anomaly   n: Next anomaly", (10, vis_frame.shape[0]-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.imshow("Ball Verifier", vis_frame)

    def run(self):
        """Run the interactive verification tool."""
        cv2.namedWindow("Ball Verifier")
        cv2.setMouseCallback("Ball Verifier", self._mouse_handler)
        while self.frame_idx < self.total_frames:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx)
            ret, self.current_frame = self.cap.read()
            if not ret:
                break
            self._update_display()
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                with open(self.output_file, 'w') as f:
                    json.dump(self.verified, f, indent=2)
                self.dirty = False
            elif key in (ord('a'), 81):
                self.frame_idx = max(0, self.frame_idx - 1)
            elif key in (ord('d'), 83):
                self.frame_idx = min(self.total_frames - 1, self.frame_idx + 1)
            elif key in (ord('+'), 82):
                fk = str(self.frame_idx)
                if fk in self.verified:
                    self.verified[fk]['radius'] += 1
                    self.dirty = True
            elif key in (ord('-'), 84):
                fk = str(self.frame_idx)
                if fk in self.verified:
                    self.verified[fk]['radius'] = max(1, self.verified[fk]['radius'] - 1)
                    self.dirty = True
            elif key == ord('t'):
                self.full_traj = not self.full_traj
            elif key == ord('h'):
                fk = str(self.frame_idx)
                if fk in self.verified:
                    self.verified[fk]['hidden'] = not self.verified[fk].get('hidden', False)
                    self.dirty = True
            elif key == ord('p'):
                # Navigate to previous anomaly
                anomaly_frames = self._compute_anomaly_frames()
                prev_anom = [f for f in anomaly_frames if f < self.frame_idx]
                if prev_anom:
                    self.frame_idx = max(prev_anom)
            elif key == ord('n'):
                # Navigate to next anomaly
                anomaly_frames = self._compute_anomaly_frames()
                next_anom = [f for f in anomaly_frames if f > self.frame_idx]
                if next_anom:
                    self.frame_idx = min(next_anom)
            # If no navigation key pressed, stay on same frame for adjustments
        cv2.destroyAllWindows()
        if self.dirty:
            with open(self.output_file, 'w') as f:
                json.dump(self.verified, f, indent=2)
        self.cap.release()


def main():
    """Main entry point for the verification tool."""
    parser = argparse.ArgumentParser(description="Interactive basketball detection verification tool")
    parser.add_argument("--video", default="data/input_video.mp4", help="Path to input video")
    parser.add_argument("--detections", default="outputs/detections.json", help="JSON file with detections")
    parser.add_argument("--output", default="outputs/verified.json", help="Path for verification output")
    args = parser.parse_args()
    CompactBallVerifier(args.video, args.detections, output_file=args.output).run()

if __name__ == '__main__':
    main()
