#!/usr/bin/env python3
"""
Annotate basketball hoop position manually.

For videos with static camera, the hoop position doesn't change.
Click on the center of the hoop to mark its position.
"""

# Fix OpenMP error
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import json
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.video_utils import open_video_robust
from src.utils.config_loader import get_config

print("=" * 70)
print("BASKETBALL HOOP ANNOTATION")
print("=" * 70)
print()

# Get video from config
config = get_config()
video_paths = config.get_video_paths()

video_file = None
for path in video_paths:
    if os.path.exists(path):
        video_file = path
        break

if not video_file:
    print(f"[X] Video not found")
    print("   Searched in:")
    for path in video_paths:
        print(f"     - {path}")
    sys.exit(1)

print(f"[+] Video: {video_file}")
print()

# Load video
try:
    cap = open_video_robust(video_file)
except IOError as e:
    print(f"[X] {e}")
    sys.exit(1)

# Read first frame
ret, frame = cap.read()
if not ret:
    print("[X] Cannot read video frame")
    sys.exit(1)

cap.release()

print("=" * 70)
print("INSTRUCTIONS")
print("=" * 70)
print()
print("1. A window will open showing the first frame")
print("2. Click on the CENTER of the basketball hoop")
print("3. Click again to adjust if needed")
print("4. Press ENTER when done, ESC to cancel")
print()

# Global variables for mouse callback
hoop_center = None
hoop_radius = 25  # Default radius for hoop visualization


def mouse_callback(event, x, y, flags, param):
    """Handle mouse click to mark hoop center."""
    global hoop_center, frame_display

    if event == cv2.EVENT_LBUTTONDOWN:
        hoop_center = [x, y]
        # Redraw frame with hoop marker
        frame_display = frame.copy()
        cv2.circle(frame_display, (x, y), hoop_radius, (0, 0, 255), 3)
        cv2.circle(frame_display, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(frame_display, f"Hoop: ({x}, {y})", (x + 30, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Annotate Hoop - Click center, ENTER when done", frame_display)


# Create window and set mouse callback
frame_display = frame.copy()
cv2.namedWindow("Annotate Hoop - Click center, ENTER when done")
cv2.setMouseCallback("Annotate Hoop - Click center, ENTER when done", mouse_callback)

# Show frame
cv2.imshow("Annotate Hoop - Click center, ENTER when done", frame_display)

print("Waiting for your click...")
print()

# Wait for user input
while True:
    key = cv2.waitKey(1) & 0xFF

    if key == 13:  # ENTER
        if hoop_center:
            break
        else:
            print("[!] Please click on the hoop center first")
    elif key == 27:  # ESC
        print("[X] Cancelled by user")
        cv2.destroyAllWindows()
        sys.exit(0)

cv2.destroyAllWindows()

# Save hoop annotation
hoop_data = {
    'center': hoop_center,
    'radius': hoop_radius,
    'type': 'basketball_hoop',
    'note': 'Manually annotated hoop position (static camera)'
}

os.makedirs('outputs', exist_ok=True)
with open('outputs/hoop.json', 'w') as f:
    json.dump(hoop_data, f, indent=2)

print("[+] Hoop position saved to outputs/hoop.json")
print()
print(f"  Center: {hoop_center}")
print(f"  Radius: {hoop_radius}px (for visualization)")
print()

print("=" * 70)
print("SUCCESS!")
print("=" * 70)
print()
print("Next steps:")
print("  - The hoop position will be used for shot detection")
print("  - Run the pipeline to see ball-hoop interactions")
print()
