"""
Tactical top-down view using homography transformation.

This module detects court lines and transforms player positions
to a standardized bird's-eye view for tactical analysis.
"""

import cv2
import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

from ..config import setup_logging

logger = setup_logging(__name__)


@dataclass
class CourtMapping:
    """Data class for court homography mapping."""
    homography_matrix: np.ndarray
    court_corners_video: List[List[int]]  # 4 corners in video coordinates
    court_corners_standard: List[List[int]]  # 4 corners in standard court coordinates
    confidence: float


class TacticalView:
    """Transform player positions to tactical top-down view."""

    def __init__(self, court_width: int = 28, court_height: int = 15,
                 output_scale: int = 50):
        """
        Initialize tactical view transformer.

        Args:
            court_width: Standard basketball court width in meters (NBA: 28.65m, FIBA: 28m)
            court_height: Standard basketball court height in meters (NBA: 15.24m, FIBA: 15m)
            output_scale: Pixels per meter for output court (default: 50)
        """
        self.court_width_m = court_width
        self.court_height_m = court_height
        self.output_scale = output_scale

        # Output court dimensions in pixels
        self.court_width_px = int(court_width * output_scale)
        self.court_height_px = int(court_height * output_scale)

        # Standard court corners (top-down view)
        self.standard_court_corners = np.array([
            [0, 0],  # Top-left
            [self.court_width_px, 0],  # Top-right
            [self.court_width_px, self.court_height_px],  # Bottom-right
            [0, self.court_height_px]  # Bottom-left
        ], dtype=np.float32)

        # Homography matrix (to be computed)
        self.homography_matrix = None
        self.court_corners_video = None

        logger.info(f"Initialized TacticalView ({court_width}m x {court_height}m, scale={output_scale}px/m)")

    def detect_court_lines(self, frame: np.ndarray) -> Optional[List[List[int]]]:
        """
        Detect basketball court lines using edge detection and Hough transform.

        Args:
            frame: Input frame

        Returns:
            List of 4 corner points [[x, y], ...] or None if detection fails
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=100,
            minLineLength=100,
            maxLineGap=10
        )

        if lines is None:
            logger.warning("No court lines detected")
            return None

        # Find the largest quadrilateral (court boundary)
        # For simplicity, use manual selection or predefined corners
        # In production, use more sophisticated court detection

        # Fallback: Use frame corners (assumes court fills most of frame)
        h, w = frame.shape[:2]
        margin = int(w * 0.1)  # 10% margin

        corners = [
            [margin, margin],  # Top-left
            [w - margin, margin],  # Top-right
            [w - margin, h - margin],  # Bottom-right
            [margin, h - margin]  # Bottom-left
        ]

        logger.info("Using default court corners (manual calibration recommended)")
        return corners

    def manual_court_selection(self, frame: np.ndarray) -> Optional[List[List[int]]]:
        """
        Allow user to manually select court corners.

        Args:
            frame: Frame to display for selection

        Returns:
            List of 4 corner points or None
        """
        logger.info("Select 4 court corners (top-left, top-right, bottom-right, bottom-left)")
        logger.info("Click on each corner, then press any key when done")

        selected_points = []
        display_frame = frame.copy()

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(selected_points) < 4:
                selected_points.append([x, y])
                cv2.circle(display_frame, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(display_frame, f"{len(selected_points)}", (x+10, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imshow("Select Court Corners", display_frame)

        cv2.imshow("Select Court Corners", display_frame)
        cv2.setMouseCallback("Select Court Corners", mouse_callback)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if len(selected_points) == 4:
            return selected_points

        logger.warning("Manual selection incomplete")
        return None

    def compute_homography(self, frame: np.ndarray,
                          court_corners: Optional[List[List[int]]] = None,
                          manual: bool = False) -> CourtMapping:
        """
        Compute homography matrix from video coordinates to standard court.

        Args:
            frame: Sample frame from video
            court_corners: Optional predefined court corners
            manual: Whether to use manual selection

        Returns:
            CourtMapping object
        """
        if court_corners is None:
            if manual:
                court_corners = self.manual_court_selection(frame)
            else:
                court_corners = self.detect_court_lines(frame)

        if court_corners is None:
            raise ValueError("Failed to detect court corners. Use manual=True for manual selection")

        # Convert to numpy array
        video_corners = np.array(court_corners, dtype=np.float32)

        # Compute homography matrix
        homography_matrix, status = cv2.findHomography(
            video_corners,
            self.standard_court_corners,
            method=cv2.RANSAC
        )

        if homography_matrix is None:
            raise ValueError("Failed to compute homography matrix")

        # Cache for future use
        self.homography_matrix = homography_matrix
        self.court_corners_video = court_corners

        # Calculate confidence based on RANSAC inliers
        confidence = float(np.sum(status)) / len(status) if status is not None else 0.0

        mapping = CourtMapping(
            homography_matrix=homography_matrix,
            court_corners_video=court_corners,
            court_corners_standard=self.standard_court_corners.tolist(),
            confidence=confidence
        )

        logger.info(f"Homography computed with confidence: {confidence:.2f}")
        return mapping

    def transform_point(self, point: Tuple[int, int]) -> Tuple[int, int]:
        """
        Transform a point from video coordinates to court coordinates.

        Args:
            point: Point in video coordinates (x, y)

        Returns:
            Point in court coordinates (x, y)
        """
        if self.homography_matrix is None:
            raise ValueError("Homography not computed. Call compute_homography() first")

        # Convert point to homogeneous coordinates
        point_h = np.array([[point[0], point[1]]], dtype=np.float32)
        point_h = point_h.reshape(-1, 1, 2)

        # Apply perspective transform
        transformed = cv2.perspectiveTransform(point_h, self.homography_matrix)

        x, y = transformed[0][0]
        return (int(x), int(y))

    def transform_players(self, player_detections: List[Dict]) -> List[Dict]:
        """
        Transform player positions to court coordinates.

        Args:
            player_detections: List of player detections with 'bbox' or 'center'

        Returns:
            List of detections with added 'court_position' field
        """
        if self.homography_matrix is None:
            raise ValueError("Homography not computed. Call compute_homography() first")

        transformed_detections = []

        for detection in player_detections:
            # Get player position (use center of bbox)
            if 'center' in detection:
                video_pos = tuple(detection['center'])
            elif 'bbox' in detection:
                bbox = detection['bbox']
                video_pos = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
            else:
                continue

            # Transform to court coordinates
            court_pos = self.transform_point(video_pos)

            # Add court position to detection
            detection_copy = detection.copy()
            detection_copy['court_position'] = list(court_pos)
            detection_copy['video_position'] = list(video_pos)

            transformed_detections.append(detection_copy)

        return transformed_detections

    def create_tactical_visualization(self, player_detections: List[Dict],
                                     ball_position: Optional[Tuple[int, int]] = None,
                                     show_trails: bool = True,
                                     trail_history: Optional[Dict] = None) -> np.ndarray:
        """
        Create tactical top-down visualization of player positions.

        Args:
            player_detections: List of player detections with 'court_position'
            ball_position: Optional ball position in court coordinates
            show_trails: Whether to show player movement trails
            trail_history: Optional dict of player_id -> list of positions

        Returns:
            Tactical view image
        """
        # Create blank court
        court_img = self._draw_basketball_court()

        # Draw player trails
        if show_trails and trail_history:
            for player_id, positions in trail_history.items():
                if len(positions) > 1:
                    points = np.array(positions, dtype=np.int32)
                    cv2.polylines(court_img, [points], False, (200, 200, 200), 2)

        # Draw players
        for detection in player_detections:
            if 'court_position' not in detection:
                continue

            court_pos = detection['court_position']
            player_id = detection.get('player_id') or detection.get('track_id')
            team = detection.get('team', 'Unknown')

            # Team colors
            if 'Team_0' in team or 'red' in team.lower():
                color = (0, 0, 255)  # Red
            elif 'Team_1' in team or 'blue' in team.lower():
                color = (255, 0, 0)  # Blue
            else:
                color = (128, 128, 128)  # Gray

            # Draw player circle
            cv2.circle(court_img, tuple(court_pos), 15, color, -1)
            cv2.circle(court_img, tuple(court_pos), 15, (255, 255, 255), 2)

            # Draw player ID
            if player_id is not None:
                cv2.putText(court_img, str(player_id),
                           (court_pos[0] - 10, court_pos[1] + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Draw ball
        if ball_position:
            cv2.circle(court_img, ball_position, 8, (0, 165, 255), -1)  # Orange

        return court_img

    def _draw_basketball_court(self) -> np.ndarray:
        """
        Draw a standard basketball court (top-down view).

        Returns:
            Court image
        """
        # Create blank court
        court = np.ones((self.court_height_px, self.court_width_px, 3), dtype=np.uint8) * 139

        # Court color (wood)
        cv2.rectangle(court, (0, 0), (self.court_width_px, self.court_height_px),
                     (139, 69, 19), -1)

        # Court boundary (white)
        cv2.rectangle(court, (0, 0), (self.court_width_px, self.court_height_px),
                     (255, 255, 255), 3)

        # Center line
        center_x = self.court_width_px // 2
        cv2.line(court, (center_x, 0), (center_x, self.court_height_px),
                (255, 255, 255), 2)

        # Center circle
        center_y = self.court_height_px // 2
        radius = int(1.8 * self.output_scale)  # 1.8m radius
        cv2.circle(court, (center_x, center_y), radius, (255, 255, 255), 2)

        # Three-point lines (simplified)
        three_point_radius = int(6.75 * self.output_scale)  # FIBA: 6.75m
        cv2.ellipse(court, (0, center_y), (three_point_radius, three_point_radius),
                   0, -90, 90, (255, 255, 255), 2)
        cv2.ellipse(court, (self.court_width_px, center_y),
                   (three_point_radius, three_point_radius),
                   0, 90, 270, (255, 255, 255), 2)

        # Free-throw lanes
        lane_width = int(4.9 * self.output_scale)  # FIBA: 4.9m
        lane_length = int(5.8 * self.output_scale)  # FIBA: 5.8m

        # Left lane
        lane_top_left = (0, center_y - lane_width // 2)
        lane_bottom_left = (lane_length, lane_top_left[1] + lane_width)
        cv2.rectangle(court, lane_top_left, lane_bottom_left, (255, 255, 255), 2)

        # Right lane
        lane_top_right = (self.court_width_px - lane_length, center_y - lane_width // 2)
        lane_bottom_right = (self.court_width_px, lane_top_right[1] + lane_width)
        cv2.rectangle(court, lane_top_right, lane_bottom_right, (255, 255, 255), 2)

        return court

    def save_homography(self, output_path: str):
        """Save homography matrix to file."""
        if self.homography_matrix is None:
            raise ValueError("Homography not computed")

        data = {
            'homography_matrix': self.homography_matrix.tolist(),
            'court_corners_video': self.court_corners_video,
            'court_corners_standard': self.standard_court_corners.tolist(),
            'court_width_px': self.court_width_px,
            'court_height_px': self.court_height_px
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Homography saved to {output_path}")

    def load_homography(self, input_path: str):
        """Load homography matrix from file."""
        with open(input_path, 'r') as f:
            data = json.load(f)

        self.homography_matrix = np.array(data['homography_matrix'], dtype=np.float32)
        self.court_corners_video = data['court_corners_video']

        logger.info(f"Homography loaded from {input_path}")


def main():
    """Example usage of tactical view."""
    import argparse
    parser = argparse.ArgumentParser(description='Tactical top-down view with homography')
    parser.add_argument('--video', required=True, help='Path to input video')
    parser.add_argument('--output', default='outputs/homography.json', help='Output JSON file')
    parser.add_argument('--manual', action='store_true', help='Manual court selection')
    parser.add_argument('--visualize', action='store_true', help='Show tactical view')
    args = parser.parse_args()

    # Initialize tactical view
    tactical = TacticalView()

    # Read first frame
    cap = cv2.VideoCapture(args.video)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Failed to read video")
        return

    # Compute homography
    try:
        mapping = tactical.compute_homography(frame, manual=args.manual)
        tactical.save_homography(args.output)
        print(f"Homography computed and saved to {args.output}")
        print(f"Confidence: {mapping.confidence:.2f}")

        # Visualize if requested
        if args.visualize:
            # Create sample tactical view
            sample_players = [
                {'court_position': [400, 300], 'player_id': 1, 'team': 'Team_0'},
                {'court_position': [800, 400], 'player_id': 2, 'team': 'Team_1'},
            ]
            tactical_img = tactical.create_tactical_visualization(sample_players)
            cv2.imshow("Tactical View", tactical_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
