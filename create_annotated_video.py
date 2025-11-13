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
if os.path.exists("outputs/tracked_players_named.json"):
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

# Open video
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    print("❌ Cannot open video")
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

# Output video
output_video = "outputs/annotated_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

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

            active_tracks.add(track_id)
            bbox = player.get('bbox')
            center = player.get('center')
            team = player.get('team', 'Unknown')

            if bbox:
                x1, y1, x2, y2 = bbox
                color = get_color(track_id)

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw track ID and/or name
                name = player.get('name')
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
