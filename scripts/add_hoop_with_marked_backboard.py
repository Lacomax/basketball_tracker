#!/usr/bin/env python3
"""
Add basketball hoop using manually marked backboard corners.
Uses exact perspective from user's markings.
"""

import sys
import os
import cv2
import json
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config_loader import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    config = get_config()
    output_dir = config.get_output_dir()

    # Load backboard data
    backboard_file = f"{output_dir}/backboard.json"
    if not os.path.exists(backboard_file):
        logger.error(f"Backboard data not found: {backboard_file}")
        logger.error("Run: python scripts/mark_backboard_corners.py first")
        return 1

    with open(backboard_file, 'r') as f:
        backboard_data = json.load(f)

    corners = backboard_data['corners']
    hoop_center = tuple(backboard_data['hoop_center'])
    hoop_radius = backboard_data['hoop_radius']

    # Convert corners to numpy array
    backboard_corners = np.array(corners, dtype=np.int32)

    logger.info("Backboard corners:")
    for i, (corner, label) in enumerate(zip(corners, backboard_data['corner_labels'])):
        logger.info(f"  {i+1}. {label}: {corner}")
    logger.info(f"Hoop center: {hoop_center}, radius: {hoop_radius}")

    # Open input video
    input_video = f"{output_dir}/final_video_clean.mp4"
    if not os.path.exists(input_video):
        logger.error(f"Input video not found: {input_video}")
        return 1

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        logger.error("Cannot open video")
        return 1

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"Video: {width}x{height}, {fps} FPS, {total_frames} frames")

    # Create output video
    output_video = f"{output_dir}/final_video_COMPLETO.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    if not out.isOpened():
        logger.error("Cannot create output video")
        return 1

    logger.info(f"Creating final video with exact backboard perspective...")

    # Progress bar
    try:
        from tqdm import tqdm
        progress_bar = tqdm(total=total_frames, desc="Processing frames", unit="frame")
    except ImportError:
        progress_bar = None
        logger.info("Install tqdm for progress bar: pip install tqdm")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ========== DRAW BACKBOARD ==========
        # Semi-transparent fill
        overlay = frame.copy()
        cv2.fillPoly(overlay, [backboard_corners], (245, 245, 245))  # Very light gray
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        # Backboard border (white with shadow)
        # Shadow (black, offset)
        shadow_corners = backboard_corners + 2
        cv2.polylines(frame, [shadow_corners], True, (50, 50, 50), 4)
        # Main border (white)
        cv2.polylines(frame, [backboard_corners], True, (255, 255, 255), 3)

        # Inner rectangle (target square) - use manually marked corners if available
        if 'inner_box_corners' in backboard_data:
            # Use manually marked inner box
            inner_rect = np.array(backboard_data['inner_box_corners'], dtype=np.int32)
            logger.info("Using manually marked inner box") if frame_count == 0 else None
        else:
            # Fallback: calculate proportional inset
            inset_ratio_horizontal = 0.30
            inset_ratio_vertical = 0.30

            inner_top_left_x = int(corners[0][0] + (corners[1][0] - corners[0][0]) * inset_ratio_horizontal)
            inner_top_left_y = int(corners[0][1] + (corners[3][1] - corners[0][1]) * inset_ratio_vertical)

            inner_top_right_x = int(corners[1][0] - (corners[1][0] - corners[0][0]) * inset_ratio_horizontal)
            inner_top_right_y = int(corners[1][1] + (corners[2][1] - corners[1][1]) * inset_ratio_vertical)

            inner_bottom_right_x = int(corners[2][0] - (corners[2][0] - corners[3][0]) * inset_ratio_horizontal)
            inner_bottom_right_y = int(corners[2][1] - (corners[2][1] - corners[1][1]) * inset_ratio_vertical)

            inner_bottom_left_x = int(corners[3][0] + (corners[2][0] - corners[3][0]) * inset_ratio_horizontal)
            inner_bottom_left_y = int(corners[3][1] - (corners[3][1] - corners[0][1]) * inset_ratio_vertical)

            inner_rect = np.array([
                [inner_top_left_x, inner_top_left_y],
                [inner_top_right_x, inner_top_right_y],
                [inner_bottom_right_x, inner_bottom_right_y],
                [inner_bottom_left_x, inner_bottom_left_y]
            ], dtype=np.int32)

        cv2.polylines(frame, [inner_rect], True, (0, 100, 200), 3)  # Orange/red target box

        # ========== DRAW RIM ==========
        # Rim attachment (bracket)
        bracket_width = hoop_radius // 2
        bracket_height = 8
        bracket_top = hoop_center[1] - 4
        bracket_bottom = hoop_center[1] + 4
        cv2.rectangle(frame,
                     (hoop_center[0] - bracket_width, bracket_top),
                     (hoop_center[0] + bracket_width, bracket_bottom),
                     (80, 80, 80), -1)  # Dark gray bracket

        # Main rim (bright orange) with 3D effect
        # Shadow (darker orange, offset down-right)
        shadow_center = (hoop_center[0] + 2, hoop_center[1] + 2)
        cv2.circle(frame, shadow_center, hoop_radius, (0, 100, 180), 3)

        # Main rim
        cv2.circle(frame, hoop_center, hoop_radius, (0, 140, 255), 4)

        # Inner rim (lighter orange)
        cv2.circle(frame, hoop_center, hoop_radius - 3, (0, 180, 255), 2)

        # Rim highlight on top (lighter, to show 3D)
        cv2.ellipse(frame, hoop_center, (hoop_radius, hoop_radius//3),
                   180, 0, 180, (50, 200, 255), 2)

        # Center dot
        cv2.circle(frame, hoop_center, 3, (0, 0, 255), -1)

        # ========== DRAW NET ==========
        # Calculate net dimensions
        net_bottom_width = int(hoop_radius * 1.3)
        net_height = int(hoop_radius * 2.8)
        net_bottom_center = (hoop_center[0], hoop_center[1] + net_height)

        # Net outline (trapezoid)
        # Top of net follows the rim
        num_rim_points = 12
        rim_arc_points = []
        for i in range(num_rim_points + 1):
            angle = np.pi * i / num_rim_points  # 0 to 180 degrees
            x = int(hoop_center[0] + hoop_radius * np.cos(angle + np.pi))
            y = hoop_center[1]
            rim_arc_points.append([x, y])

        # Bottom of net
        net_bottom_points = [
            [net_bottom_center[0] - net_bottom_width//2, net_bottom_center[1]],
            [net_bottom_center[0] + net_bottom_width//2, net_bottom_center[1]]
        ]

        # Combine into full net outline
        net_outline = rim_arc_points + [net_bottom_points[1]] + [net_bottom_points[0]]
        net_outline_np = np.array(net_outline, dtype=np.int32)

        # Semi-transparent white fill for net
        overlay = frame.copy()
        cv2.fillPoly(overlay, [net_outline_np], (255, 255, 255))
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

        # Net border
        cv2.polylines(frame, [net_outline_np], True, (230, 230, 230), 2)

        # Net mesh pattern
        # Vertical lines
        num_vertical_lines = 8
        for i in range(num_vertical_lines + 1):
            t = i / num_vertical_lines
            # Top point (on rim arc)
            rim_idx = int(t * num_rim_points)
            if rim_idx < len(rim_arc_points):
                top_point = rim_arc_points[rim_idx]
            else:
                top_point = rim_arc_points[-1]

            # Bottom point
            bottom_x = int(net_bottom_center[0] - net_bottom_width//2 + net_bottom_width * t)
            bottom_point = [bottom_x, net_bottom_center[1]]

            cv2.line(frame, tuple(top_point), tuple(bottom_point), (210, 210, 210), 1)

        # Horizontal lines
        for i in range(1, 5):
            y_ratio = i / 5
            y_pos = int(hoop_center[1] + net_height * y_ratio)

            # Width at this y position (trapezoid interpolation)
            width_at_y = int(hoop_radius * 2 + (net_bottom_width - hoop_radius * 2) * y_ratio)

            left_x = hoop_center[0] - width_at_y // 2
            right_x = hoop_center[0] + width_at_y // 2

            cv2.line(frame, (left_x, y_pos), (right_x, y_pos), (210, 210, 210), 1)

        # ========== LABEL ==========
        # Label above backboard with background
        label_text = "CANASTA"
        label_pos = (corners[0][0] + 5, corners[0][1] - 15)

        # Text background (semi-transparent black)
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        bg_x1 = label_pos[0] - 5
        bg_y1 = label_pos[1] - text_size[1] - 5
        bg_x2 = label_pos[0] + text_size[0] + 5
        bg_y2 = label_pos[1] + 5

        overlay = frame.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # Text (green)
        cv2.putText(frame, label_text, label_pos,
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        out.write(frame)
        frame_count += 1

        if progress_bar:
            progress_bar.update(1)
        else:
            # Only log every 50 frames if no progress bar
            if frame_count % 50 == 0:
                logger.info(f"Processed {frame_count}/{total_frames} frames ({frame_count*100//total_frames}%)")

    cap.release()
    out.release()

    if progress_bar:
        progress_bar.close()

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"VIDEO FINAL CREADO: {output_video}")
    logger.info("=" * 70)
    logger.info("")
    logger.info("El video incluye:")
    logger.info("  - Tablero con perspectiva EXACTA (marcado por ti)")
    logger.info("  - Caja interior roja (target)")
    logger.info("  - Aro naranja con efecto 3D")
    logger.info("  - Red blanca con malla detallada")
    logger.info("  - Jugadores rastreados con nombres")
    logger.info("  - Trayectoria del balon suavizada")
    logger.info("")
    logger.info("Este es tu VIDEO FINAL DEFINITIVO!")
    logger.info("")

    return 0

if __name__ == '__main__':
    sys.exit(main())
