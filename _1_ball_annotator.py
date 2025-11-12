import cv2, numpy as np, json, os

class BallAnnotator:
    def __init__(self, video: str = 'data/input_video.mp4', output: str = 'outputs/annotations.json'):
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

    def _auto_detect_ball(self, frame: np.ndarray, point: tuple) -> dict:
        """Detect the ball around a clicked point, returning center and radius."""
        x, y = int(point[0]), int(point[1])
        h, w = frame.shape[:2]
        # Define a region of interest around the click
        roi_x0, roi_y0 = max(0, x - 30), max(0, y - 30)
        roi = frame[roi_y0: min(h, y + 30), roi_x0: min(w, x + 30)]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Try detecting a circle (ball) in the ROI
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                   param1=50, param2=30, minRadius=10, maxRadius=50)
        if circles is None:
            # Retry with a more sensitive threshold if not found
            circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                       param1=50, param2=15, minRadius=5, maxRadius=50)
        if circles is not None:
            cx, cy, r = circles[0][0]
            # Offset ROI coordinates to get center in full frame
            cx, cy = int(cx) + roi_x0, int(cy) + roi_y0
            return {'center': [cx, cy], 'radius': int(r)}
        # Fallback if no circle is detected
        return {'center': [x, y], 'radius': 15}

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
        frame_key = str(self.frame_idx)
        if event == cv2.EVENT_LBUTTONDOWN:
            # On click, detect ball and start dragging
            self.annotations[frame_key] = self._auto_detect_ball(self.current_frame, (x, y))
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
