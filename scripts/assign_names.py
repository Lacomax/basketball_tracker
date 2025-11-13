#!/usr/bin/env python3
"""
Assign names to tracked players and improve ID consistency with ReID.

This script:
1. Shows you each unique player
2. Lets you assign names
3. Uses ReID to merge inconsistent IDs (same player with different IDs)
4. Creates final tracking data with names
"""

import sys
import os
import cv2
import json
import numpy as np
from collections import defaultdict
# Add parent directory to path so we can import from src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.video_utils import open_video_robust

print("=" * 60)
print("PLAYER NAME ASSIGNMENT & ID CONSOLIDATION")
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

# Use filtered data if available
tracking_file = None
if os.path.exists("outputs/tracked_players_filtered.json"):
    tracking_file = "outputs/tracked_players_filtered.json"
    print("✓ Using filtered tracking data (court ROI)")
elif os.path.exists("outputs/tracked_players.json"):
    tracking_file = "outputs/tracked_players.json"
    print("⚠ Using unfiltered data (may include bench/crowd)")
else:
    print("❌ No tracking data found")
    sys.exit(1)

print(f"✓ Video: {input_video}")
print(f"✓ Tracking data: {tracking_file}")
print()

# Load tracking data
with open(tracking_file, 'r') as f:
    tracking_data = json.load(f)

print(f"✓ Loaded tracking data for {len(tracking_data)} frames")
print()

# Open video with robust method (tries multiple backends)
try:
    cap = open_video_robust(input_video)
except IOError as e:
    print(f"❌ {e}")
    sys.exit(1)

# Collect player images for each unique ID
print("Collecting player appearances...")
player_images = defaultdict(list)  # {track_id: [(frame, bbox, center), ...]}

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_key = str(frame_idx)
    if frame_key in tracking_data:
        for player in tracking_data[frame_key]:
            track_id = player.get('track_id')
            bbox = player.get('bbox')
            center = player.get('center')

            if track_id is not None and bbox:
                player_images[track_id].append((frame.copy(), bbox, center))

    frame_idx += 1

cap.release()

print(f"✓ Found {len(player_images)} unique player IDs")
print()

# Load previous player names if available
previous_names = {}
if os.path.exists('outputs/player_names.json'):
    try:
        with open('outputs/player_names.json', 'r') as f:
            # Convert string keys to int
            previous_names_raw = json.load(f)
            previous_names = {int(k): v for k, v in previous_names_raw.items()}
        print(f"✓ Loaded {len(previous_names)} previous player names")
        print()
    except (FileNotFoundError, json.JSONDecodeError):
        pass

# Show each player and get name
print("=" * 60)
print("STEP 1: ASSIGN PLAYER NAMES")
print("=" * 60)
print()
print("For each player, you'll see a cropped image.")
print("Type the player's name (or press ENTER to keep current/skip)")
if previous_names:
    print("Previous names will be shown - you can edit or keep them")
print()

player_names = {}

for track_id in sorted(player_images.keys()):
    appearances = player_images[track_id]

    # Get best quality image (middle of sequence, largest bbox)
    best_idx = len(appearances) // 2
    frame, bbox, center = appearances[best_idx]

    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]

    # Create display with much larger context around player
    # Show player with surrounding area (2x bbox size in each direction)
    bbox_width = x2 - x1
    bbox_height = y2 - y1

    # Calculate expanded region (show more context)
    context_margin_x = int(bbox_width * 1.5)
    context_margin_y = int(bbox_height * 1.5)

    # Expanded region coordinates
    region_x1 = max(0, x1 - context_margin_x)
    region_y1 = max(0, y1 - context_margin_y)
    region_x2 = min(w, x2 + context_margin_x)
    region_y2 = min(h, y2 + context_margin_y)

    # Extract region
    region = frame[region_y1:region_y2, region_x1:region_x2].copy()

    if region.size == 0:
        continue

    # Draw bounding box on player in the region
    # Adjust bbox coordinates relative to region
    box_x1 = x1 - region_x1
    box_y1 = y1 - region_y1
    box_x2 = x2 - region_x1
    box_y2 = y2 - region_y1

    # Draw thick green box around player
    cv2.rectangle(region, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), 4)

    # Add arrow pointing to player
    center_x = (box_x1 + box_x2) // 2
    arrow_y = max(0, box_y1 - 30)
    cv2.arrowedLine(region, (center_x, arrow_y), (center_x, box_y1),
                    (0, 255, 0), 3, tipLength=0.3)

    # Resize for display (larger for better visibility)
    max_display_height = 600
    max_display_width = 800

    # Calculate resize factor
    scale = min(max_display_width / region.shape[1],
                max_display_height / region.shape[0])

    if scale < 1.0:
        display_width = int(region.shape[1] * scale)
        display_height = int(region.shape[0] * scale)
        player_display = cv2.resize(region, (display_width, display_height))
    else:
        player_display = region

    # Add info banner at top
    banner_height = 60
    banner = np.zeros((banner_height, player_display.shape[1], 3), dtype=np.uint8)
    banner[:] = (40, 40, 40)  # Dark gray background

    info_text = f"Player ID: {track_id}"
    cv2.putText(banner, info_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    appearances_text = f"Appearances: {len(appearances)} frames"
    cv2.putText(banner, appearances_text, (10, 55),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Combine banner and image
    player_display = np.vstack([banner, player_display])

    # Show image
    cv2.imshow("Assign Player Name", player_display)
    cv2.waitKey(1)  # Refresh window

    # Get name from user (show previous name if exists)
    previous_name = previous_names.get(track_id, "")
    if previous_name:
        prompt = f"Player ID {track_id} ({len(appearances)} frames) - Current: '{previous_name}' (ENTER to keep, or type new name): "
    else:
        prompt = f"Player ID {track_id} ({len(appearances)} frames) - Enter name (or ENTER to skip): "

    name = input(prompt).strip()

    if name:
        # User entered a new name
        player_names[track_id] = name
        print(f"  ✓ Assigned: ID {track_id} → {name}")
    elif previous_name:
        # Keep previous name
        player_names[track_id] = previous_name
        print(f"  ✓ Kept: ID {track_id} → {previous_name}")
    else:
        # No name, use default
        player_names[track_id] = f"Player {track_id}"
        print(f"  ⚠ Skipped: Using 'Player {track_id}'")

    print()

cv2.destroyAllWindows()

print(f"✓ Assigned names to {len(player_names)} players")
print()

# Save player names
with open('outputs/player_names.json', 'w') as f:
    json.dump(player_names, f, indent=2)

print("✓ Player names saved to outputs/player_names.json")
print()

# Auto-detect IDs with same name
print("=" * 60)
print("STEP 2: MERGE DUPLICATE IDs")
print("=" * 60)
print()

# Group IDs by name (auto-detect duplicates)
name_to_ids = defaultdict(list)
for track_id, name in player_names.items():
    name_to_ids[name].append(track_id)

# Find duplicates (same name, multiple IDs)
auto_merges = []
for name, ids in name_to_ids.items():
    if len(ids) > 1 and name.lower() != "public":  # Don't auto-merge "public"
        auto_merges.append((name, ids))

if auto_merges:
    print("🔍 Detected players with same name but different IDs:")
    print()
    for name, ids in auto_merges:
        print(f"  '{name}': IDs {sorted(ids)}")
    print()

    merge_auto = input("Auto-merge these IDs? (Y/n): ").strip().lower()
    if merge_auto != 'n':
        id_merges = [ids for name, ids in auto_merges]
        print(f"✓ Will auto-merge {len(id_merges)} groups")
    else:
        print("⚠ Skipping auto-merge")
        id_merges = []
else:
    print("✓ No duplicate names detected")
    id_merges = []

print()

# Manual merge option
print("Manual merge (optional):")
print("If you want to merge additional IDs, enter them now.")
print("Format: 'id1,id2,id3' (comma-separated IDs to merge)")
print("Press ENTER when done")
print()

while True:
    merge_input = input("Merge IDs (e.g., '3,15,22' or ENTER to finish): ").strip()
    if not merge_input:
        break

    try:
        ids_to_merge = [int(x.strip()) for x in merge_input.split(',')]
        if len(ids_to_merge) < 2:
            print("  ⚠ Need at least 2 IDs to merge")
            continue

        # Verify IDs exist
        valid_ids = [id for id in ids_to_merge if id in player_names]
        if len(valid_ids) != len(ids_to_merge):
            print(f"  ⚠ Some IDs don't exist: {set(ids_to_merge) - set(valid_ids)}")
            continue

        id_merges.append(ids_to_merge)
        print(f"  ✓ Will merge: {ids_to_merge} → {ids_to_merge[0]}")
    except ValueError:
        print("  ⚠ Invalid format. Use comma-separated numbers")

print()

# Apply ID merges
if id_merges:
    print("Applying ID merges...")

    # Create ID mapping
    id_mapping = {}
    for merge_group in id_merges:
        target_id = merge_group[0]  # First ID is the target
        for source_id in merge_group[1:]:
            id_mapping[source_id] = target_id

    # Update tracking data
    for frame_idx, players in tracking_data.items():
        for player in players:
            old_id = player.get('track_id')
            if old_id in id_mapping:
                player['track_id'] = id_mapping[old_id]
                # Also update name
                if id_mapping[old_id] in player_names:
                    player['name'] = player_names[id_mapping[old_id]]

    # Update player_names (remove merged IDs)
    for source_id, target_id in id_mapping.items():
        if source_id in player_names:
            del player_names[source_id]

    print(f"✓ Merged {len(id_mapping)} IDs")
    print()

# Add names to tracking data
print("Adding names to tracking data...")
for frame_idx, players in tracking_data.items():
    for player in players:
        track_id = player.get('track_id')
        if track_id in player_names:
            player['name'] = player_names[track_id]
        else:
            player['name'] = f"Player {track_id}"

print("✓ Names added to tracking data")
print()

# Save final data
output_file = "outputs/tracked_players_named.json"
with open(output_file, 'w') as f:
    json.dump(tracking_data, f, indent=2)

print(f"✓ Final tracking data saved to {output_file}")
print()

print("=" * 60)
print("SUCCESS!")
print("=" * 60)
print()
print("Player roster:")
for track_id, name in sorted(player_names.items()):
    print(f"  #{track_id}: {name}")
print()
print("Next step:")
print("  Create final video with names:")
print("  python create_annotated_video.py")
print("  (Edit script to use 'tracked_players_named.json')")
print()
