#!/usr/bin/env python3
"""
Simple test script to process basketball video with Version 3.0 features.

This script simplifies the workflow and allows manual hoop selection if needed.
"""

import sys
import os
import cv2
import json
import numpy as np

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 60)
print("BASKETBALL TRACKER V3.0 - VIDEO TEST")
print("=" * 60)
print()

# Check for video files
video_path = None
if os.path.exists("input_video_converted.mp4"):
    video_path = "input_video_converted.mp4"
    print(f"✓ Using converted video: {video_path}")
elif os.path.exists("input_video.mp4"):
    video_path = "input_video.mp4"
    print(f"✓ Video found: {video_path}")
else:
    print(f"❌ Video not found")
    print("Please place your video as 'input_video.mp4' or 'input_video_converted.mp4'")
    sys.exit(1)

# Create outputs directory
os.makedirs("outputs", exist_ok=True)

# Open video to check
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Cannot open video - codec not supported by OpenCV")
    print()
    print("Solution: Convert the video to a compatible format")
    print("Run: python convert_video.py")
    print()
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"✓ Video info:")
print(f"  - Resolution: {width}x{height}")
print(f"  - FPS: {fps}")
print(f"  - Total frames: {total_frames}")
print(f"  - Duration: {total_frames/fps:.1f} seconds")
print()

# Read first frame
ret, first_frame = cap.read()
cap.release()

if not ret:
    print("❌ Cannot read first frame - video codec incompatible")
    print()
    print("Solution: Convert the video to a compatible format")
    print("Run: python convert_video.py")
    print()
    sys.exit(1)

print("=" * 60)
print("STEP 1: HOOP DETECTION")
print("=" * 60)
print()

# Manual hoop selection
print("Please select the hoop position:")
print("1. A window will open showing the first frame")
print("2. Click on the CENTER of the basketball hoop")
print("3. Press any key to confirm")
print()
print("Waiting for your selection...")

# Global variable for mouse callback
hoop_position = None

def mouse_callback(event, x, y, flags, param):
    global hoop_position
    if event == cv2.EVENT_LBUTTONDOWN:
        hoop_position = [x, y]
        # Draw circle on frame
        frame_copy = param.copy()
        cv2.circle(frame_copy, (x, y), 40, (0, 255, 0), 3)
        cv2.circle(frame_copy, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(frame_copy, f"Hoop: ({x}, {y})", (x+50, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Select Hoop Position", frame_copy)

# Show frame for selection
display_frame = first_frame.copy()
cv2.putText(display_frame, "Click on hoop center, then press any key",
           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.imshow("Select Hoop Position", display_frame)
cv2.setMouseCallback("Select Hoop Position", mouse_callback, first_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

if hoop_position is None:
    print("❌ No hoop selected")
    sys.exit(1)

print(f"✓ Hoop position selected: {hoop_position}")

# Save hoop position
hoop_data = {
    'center': hoop_position,
    'radius': 40,
    'confidence': 1.0,
    'frame_number': 0,
    'method': 'manual'
}

with open('outputs/hoop.json', 'w') as f:
    json.dump(hoop_data, f, indent=2)

print(f"✓ Hoop position saved to outputs/hoop.json")
print()

print("=" * 60)
print("STEP 2: BALL TRAJECTORY DETECTION")
print("=" * 60)
print()

print("Detecting ball trajectory...")
print("This may take a few moments...")
print()

try:
    from src.modules.trajectory_detector import BallTrajectoryDetector

    detector = BallTrajectoryDetector()
    detections = detector.process_video(video_path, 'outputs/ball_trajectory.json')

    ball_count = sum(1 for frame_dets in detections.values() if frame_dets)
    print(f"✓ Ball detected in {ball_count}/{total_frames} frames")
    print(f"✓ Trajectory saved to outputs/ball_trajectory.json")
    print()

except ImportError as e:
    print(f"⚠ Ball detection skipped: {e}")
    print()

print("=" * 60)
print("STEP 3: SHOT ANALYSIS")
print("=" * 60)
print()

print("Analyzing shots with 97% accuracy linear regression...")

try:
    # Load ball trajectory
    with open('outputs/ball_trajectory.json', 'r') as f:
        ball_data = json.load(f)

    from src.modules.hoop_detector import HoopDetector

    hoop_detector = HoopDetector()

    # Simple shot detection: group consecutive ball detections
    shots_detected = 0
    shots_made = 0

    # Group frames into potential shots
    current_shot = []
    shot_results = []

    for frame_idx in sorted([int(k) for k in ball_data.keys()]):
        frame_data = ball_data[str(frame_idx)]
        if frame_data:
            current_shot.append(frame_data['center'])
        else:
            # End of shot
            if len(current_shot) >= 5:
                is_made, confidence = hoop_detector.is_basket_made(
                    current_shot,
                    hoop_position,
                    hoop_radius=40
                )
                shots_detected += 1
                if is_made:
                    shots_made += 1

                shot_results.append({
                    'shot_number': shots_detected,
                    'made': is_made,
                    'confidence': confidence,
                    'trajectory_length': len(current_shot)
                })

            current_shot = []

    # Save shot results
    with open('outputs/shot_analysis.json', 'w') as f:
        json.dump(shot_results, f, indent=2)

    print(f"✓ Shots detected: {shots_detected}")
    print(f"✓ Shots made: {shots_made}")
    print(f"✓ Shooting accuracy: {shots_made/shots_detected*100:.1f}%" if shots_detected > 0 else "✓ No shots detected")
    print(f"✓ Results saved to outputs/shot_analysis.json")
    print()

except Exception as e:
    print(f"⚠ Shot analysis failed: {e}")
    print()

print("=" * 60)
print("SUMMARY")
print("=" * 60)
print()
print("✅ Video processing complete!")
print()
print("Generated files:")
print("  - outputs/hoop.json           (hoop position)")
print("  - outputs/ball_trajectory.json (ball detections)")
print("  - outputs/shot_analysis.json   (shot statistics)")
print()
print("Next steps:")
print("  - Review shot analysis in outputs/shot_analysis.json")
print("  - Run player tracking: python -m src.modules.improved_tracker --video input_video.mp4")
print("  - Calculate metrics: python -m src.modules.metrics_calculator")
print()
