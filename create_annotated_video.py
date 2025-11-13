#!/usr/bin/env python3
"""
Create annotated video with tracking visualizations.

This script reads tracking data and creates a video with:
- Player bounding boxes with IDs
- Player trails
- Team colors
- Statistics overlay
"""

import sys
import os
import cv2
import json
import numpy as np
from collections import defaultdict, deque
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.utils.video_utils import open_video_robust, create_video_writer_robust

print("=" * 60)
print("BASKETBALL TRACKER - VIDEO VISUALIZATION")
print("=" * 60)
print()

# Check for required files
input_video = None
if os.path.exists("input_video_converted.mp4"):
    input_video = "input_video_converted.mp4"
elif os.path.exists("input_video.mp4"):
    input_video = "input_video.mp4"
else:
    print("❌ Video not found")
    sys.exit(1)

# Use the best available tracking data
tracking_file = None
if os.path.exists("outputs/tracked_players_named_teams.json"):
    tracking_file = "outputs/tracked_players_named_teams.json"
    print("✓ Using named tracking data with teams (BEST)")
elif os.path.exists("outputs/tracked_players_filtered_teams.json"):
    tracking_file = "outputs/tracked_players_filtered_teams.json"
    print("✓ Using filtered tracking data with teams")
elif os.path.exists("outputs/tracked_players_teams.json"):
    tracking_file = "outputs/tracked_players_teams.json"
    print("✓ Using tracking data with teams")
elif os.path.exists("outputs/tracked_players_named.json"):
    tracking_file = "outputs/tracked_players_named.json"
    print("✓ Using named tracking data (with player names)")
elif os.path.exists("outputs/tracked_players_filtered.json"):
    tracking_file = "outputs/tracked_players_filtered.json"
    print("✓ Using filtered tracking data (court ROI only)")
elif os.path.exists("outputs/tracked_players.json"):
    tracking_file = "outputs/tracked_players.json"
    print("⚠ Using raw tracking data (may include bench/crowd)")
else:
    print("❌ No tracking data found")
    print("Run player tracking first:")
    print("  python -m src.modules.improved_tracker --video input_video.mp4")
    sys.exit(1)

print(f"✓ Video: {input_video}")
print(f"✓ Tracking data: {tracking_file}")
print()

# Load tracking data
with open(tracking_file, 'r') as f:
    tracking_data = json.load(f)

print(f"✓ Loaded tracking data for {len(tracking_data)} frames")
print()

# Load ball trajectory if available
ball_trajectory = {}
if os.path.exists("outputs/detections.json"):
    try:
        with open("outputs/detections.json", 'r') as f:
            ball_trajectory = json.load(f)
        # Convert string keys to int
        ball_trajectory = {int(k): v for k, v in ball_trajectory.items()}
        print(f"✓ Loaded ball trajectory for {len(ball_trajectory)} frames")
    except (FileNotFoundError, json.JSONDecodeError):
        print("⚠ No ball trajectory found")
else:
    print("⚠ No ball trajectory found (outputs/detections.json)")
print()

# Team colors (cycling through colors for different track IDs)
COLORS = [
    (255, 0, 0),    # Blue
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 255),  # Purple
    (255, 128, 0),  # Orange
    (0, 128, 255),  # Light Blue
    (128, 255, 0),  # Light Green
]

def get_color(track_id):
    """Get consistent color for track ID."""
    return COLORS[track_id % len(COLORS)]

# Player trails
player_trails = defaultdict(lambda: deque(maxlen=30))

# Open video with robust method (tries multiple backends)
try:
    cap = open_video_robust(input_video)
except IOError as e:
    print(f"❌ {e}")
    sys.exit(1)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video info:")
print(f"  - Resolution: {width}x{height}")
print(f"  - FPS: {fps}")
print(f"  - Total frames: {total_frames}")
print()

# Output video (use robust video writer)
output_video = "outputs/annotated_video.mp4"
try:
    out = create_video_writer_robust(output_video, fps, width, height)
except IOError as e:
    print(f"❌ {e}")
    cap.release()
    sys.exit(1)

print("Creating annotated video...")
print("This may take several minutes...")
print()

frame_idx = 0
active_tracks = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get tracking data for this frame
    frame_key = str(frame_idx)
    if frame_key in tracking_data:
        players = tracking_data[frame_key]

        # Draw players
        for player in players:
            track_id = player.get('track_id')
            if track_id is None:
                continue

            # Skip players with name "public" or team "Public" (crowd/bench)
            name = player.get('name', '')
            team = player.get('team', 'Unknown')

            if name.lower() == 'public' or team.lower() == 'public':
                continue

            active_tracks.add(track_id)
            bbox = player.get('bbox')
            center = player.get('center')

            if bbox:
                x1, y1, x2, y2 = bbox
                color = get_color(track_id)

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw track ID and/or name
                if name:
                    label = name
                else:
                    label = f"ID:{track_id}"

                if team and team != 'Unknown':
                    label += f" ({team})"

                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            if center:
                # Add to trail
                player_trails[track_id].append(tuple(center))

                # Draw trail
                if len(player_trails[track_id]) > 1:
                    points = np.array(list(player_trails[track_id]), dtype=np.int32)
                    for i in range(1, len(points)):
                        # Fade trail (older points more transparent)
                        alpha = i / len(points)
                        thickness = max(1, int(3 * alpha))
                        cv2.line(frame, tuple(points[i-1]), tuple(points[i]),
                                color, thickness)

                # Draw center point
                cv2.circle(frame, tuple(center), 4, color, -1)

    # Draw ball trajectory if available
    if frame_idx in ball_trajectory:
        ball_data = ball_trajectory[frame_idx]
        ball_center = ball_data.get('center')
        ball_radius = ball_data.get('radius', 12)

        if ball_center:
            # Convert center to tuple of ints
            ball_x, ball_y = int(ball_center[0]), int(ball_center[1])

            # Draw ball with orange color (easy to see)
            cv2.circle(frame, (ball_x, ball_y), ball_radius, (0, 165, 255), 2)  # Orange
            cv2.circle(frame, (ball_x, ball_y), 3, (0, 165, 255), -1)  # Center dot

            # Add "BALL" label
            cv2.putText(frame, "BALL", (ball_x + 15, ball_y - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

    # Draw ball trajectory trail (last 20 frames)
    if ball_trajectory:
        trajectory_points = []
        for i in range(max(0, frame_idx - 20), frame_idx + 1):
            if i in ball_trajectory:
                ball_data = ball_trajectory[i]
                ball_center = ball_data.get('center')
                if ball_center:
                    trajectory_points.append((int(ball_center[0]), int(ball_center[1])))

        # Draw trajectory line
        if len(trajectory_points) > 1:
            for i in range(1, len(trajectory_points)):
                alpha = i / len(trajectory_points)
                thickness = max(1, int(2 * alpha))
                cv2.line(frame, trajectory_points[i-1], trajectory_points[i],
                        (0, 165, 255), thickness)  # Orange trail

    # Draw statistics overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Statistics text
    stats_text = [
        f"Frame: {frame_idx}/{total_frames}",
        f"Active Players: {len(active_tracks)}",
        f"Current Frame Players: {len(tracking_data.get(frame_key, []))}",
    ]

    y_offset = 30
    for text in stats_text:
        cv2.putText(frame, text, (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30

    # Write frame
    out.write(frame)

    # Progress update
    if frame_idx % 100 == 0:
        progress = (frame_idx / total_frames) * 100
        print(f"  Progress: {progress:.1f}% ({frame_idx}/{total_frames} frames)")

    frame_idx += 1

cap.release()
out.release()

print()
print("=" * 60)
print("SUCCESS!")
print("=" * 60)
print()
print(f"✅ Annotated video created: {output_video}")
print()
print("The video includes:")
print("  ✓ Player bounding boxes with IDs")
print("  ✓ Player movement trails (30 frame history)")
print("  ✓ Team assignments (if detected)")
print("  ✓ Real-time statistics overlay")
print()
print(f"Total unique players tracked: {len(active_tracks)}")
print()
print("Next steps:")
print("  - Open the video with a media player")
print("  - Add ball tracking: python -m src.modules.trajectory_detector")
print("  - Add shot detection: python test_with_video.py")
print()
