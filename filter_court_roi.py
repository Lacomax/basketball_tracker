#!/usr/bin/env python3
"""
Filter tracking data to only include players in the court area.

This removes detections of people sitting on benches, in the crowd, etc.
Allows manual ROI selection or automatic detection.
"""

import sys
import os
import cv2
import json
import numpy as np

print("=" * 60)
print("COURT ROI SELECTOR & TRACKING FILTER")
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

tracking_file = "outputs/tracked_players.json"
if not os.path.exists(tracking_file):
    print(f"❌ Tracking data not found: {tracking_file}")
    sys.exit(1)

print(f"✓ Video: {input_video}")
print(f"✓ Tracking data: {tracking_file}")
print()

# Load tracking data
with open(tracking_file, 'r') as f:
    tracking_data = json.load(f)

print(f"✓ Loaded tracking data for {len(tracking_data)} frames")
print()

# Open video
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    print("❌ Cannot open video")
    sys.exit(1)

# Read first frame
ret, first_frame = cap.read()
cap.release()

if not ret:
    print("❌ Cannot read first frame")
    sys.exit(1)

print("=" * 60)
print("STEP 1: SELECT COURT AREA (ROI)")
print("=" * 60)
print()
print("Instructions:")
print("1. A window will open showing the first frame")
print("2. Click to define the corners of the COURT AREA")
print("3. Click at least 4 points (corners of the court)")
print("4. Press ENTER when done")
print("5. Players outside this area will be filtered out")
print()
print("Waiting for your selection...")

# Global variables for ROI selection
roi_points = []
drawing = False

def mouse_callback(event, x, y, flags, param):
    global roi_points, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        roi_points.append([x, y])
        drawing = True

        # Draw on frame
        frame_copy = param.copy()

        # Draw all points
        for i, pt in enumerate(roi_points):
            cv2.circle(frame_copy, tuple(pt), 5, (0, 255, 0), -1)
            cv2.putText(frame_copy, str(i+1), (pt[0]+10, pt[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw lines between points
        if len(roi_points) > 1:
            for i in range(len(roi_points) - 1):
                cv2.line(frame_copy, tuple(roi_points[i]), tuple(roi_points[i+1]),
                        (0, 255, 0), 2)

        # Close polygon if 4+ points
        if len(roi_points) >= 4:
            cv2.line(frame_copy, tuple(roi_points[-1]), tuple(roi_points[0]),
                    (0, 255, 0), 2)

            # Fill polygon semi-transparent
            overlay = frame_copy.copy()
            pts = np.array(roi_points, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            cv2.addWeighted(overlay, 0.2, frame_copy, 0.8, 0, frame_copy)

        cv2.imshow("Select Court ROI", frame_copy)

# Show frame for selection
display_frame = first_frame.copy()
cv2.putText(display_frame, "Click corners of court area, then press ENTER",
           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.imshow("Select Court ROI", display_frame)
cv2.setMouseCallback("Select Court ROI", mouse_callback, first_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(roi_points) < 4:
    print("❌ Need at least 4 points to define ROI")
    sys.exit(1)

print(f"✓ ROI defined with {len(roi_points)} points")

# Save ROI
roi_data = {
    'points': roi_points,
    'num_points': len(roi_points)
}

with open('outputs/court_roi.json', 'w') as f:
    json.dump(roi_data, f, indent=2)

print(f"✓ ROI saved to outputs/court_roi.json")
print()

# Create ROI mask
print("=" * 60)
print("STEP 2: FILTER TRACKING DATA")
print("=" * 60)
print()

h, w = first_frame.shape[:2]
roi_mask = np.zeros((h, w), dtype=np.uint8)
pts = np.array(roi_points, dtype=np.int32)
cv2.fillPoly(roi_mask, [pts], 255)

print("Filtering players outside court area...")

def is_inside_roi(center, bbox, mask, margin=50):
    """
    Check if player is inside ROI mask.

    Uses multiple criteria:
    1. Center point is inside ROI
    2. Bottom-center of bbox is inside ROI (feet position)
    3. Any part of bbox overlaps with ROI (with margin)
    """
    x, y = center

    # Check center point with margin
    if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
        if mask[y, x] > 0:
            return True

    # Check bottom-center (feet position) - more reliable for court area
    if bbox:
        x1, y1, x2, y2 = bbox
        bottom_center_x = int((x1 + x2) / 2)
        bottom_center_y = int(y2)  # Bottom of bbox

        if 0 <= bottom_center_x < mask.shape[1] and 0 <= bottom_center_y < mask.shape[0]:
            if mask[bottom_center_y, bottom_center_x] > 0:
                return True

    # Check if any corner of bbox is inside ROI (with margin)
    if bbox:
        x1, y1, x2, y2 = bbox
        corners = [
            (x1, y1), (x2, y1),  # Top corners
            (x1, y2), (x2, y2),  # Bottom corners
        ]

        for cx, cy in corners:
            if 0 <= cx < mask.shape[1] and 0 <= cy < mask.shape[0]:
                if mask[cy, cx] > 0:
                    return True

    # Check with margin (dilate ROI mask)
    kernel = np.ones((margin, margin), np.uint8)
    expanded_mask = cv2.dilate(roi_mask, kernel, iterations=1)

    if 0 <= x < expanded_mask.shape[1] and 0 <= y < expanded_mask.shape[0]:
        if expanded_mask[y, x] > 0:
            return True

    return False

# Filter tracking data
filtered_data = {}
players_filtered_out = 0
players_kept = 0

for frame_idx, players in tracking_data.items():
    filtered_players = []

    for player in players:
        center = player.get('center')
        bbox = player.get('bbox')

        if center and is_inside_roi(center, bbox, roi_mask):
            filtered_players.append(player)
            players_kept += 1
        else:
            players_filtered_out += 1

    if filtered_players:  # Only keep frames with at least one player
        filtered_data[frame_idx] = filtered_players

print(f"✓ Players kept (inside court): {players_kept}")
print(f"✓ Players filtered out: {players_filtered_out}")
print(f"✓ Frames with players: {len(filtered_data)}/{len(tracking_data)}")
print()

# Save filtered data
output_file = "outputs/tracked_players_filtered.json"
with open(output_file, 'w') as f:
    json.dump(filtered_data, f, indent=2)

print(f"✓ Filtered tracking data saved to {output_file}")
print()

# Get unique player IDs
unique_ids = set()
for players in filtered_data.values():
    for player in players:
        track_id = player.get('track_id')
        if track_id is not None:
            unique_ids.add(track_id)

print(f"✓ Unique players in court: {len(unique_ids)}")
print(f"  IDs: {sorted(unique_ids)}")
print()

print("=" * 60)
print("SUCCESS!")
print("=" * 60)
print()
print("Next steps:")
print("  1. Create annotated video with filtered data:")
print("     python create_annotated_video.py")
print("     (Edit script to use 'tracked_players_filtered.json')")
print()
print("  2. Or apply player names:")
print("     python assign_player_names.py")
print()
