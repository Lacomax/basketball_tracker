#!/usr/bin/env python3
"""
Analyze ball trajectory and detect problems.
"""

import json
import sys
import math
from collections import defaultdict

def load_json(path):
    """Load JSON file."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {path}")
        sys.exit(1)

def distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def analyze_trajectory(detections, annotations):
    """Analyze trajectory for problems."""
    print("=" * 70)
    print("BALL TRAJECTORY ANALYSIS")
    print("=" * 70)
    print()

    # Convert to sorted list
    frames = sorted([int(f) for f in detections.keys()])
    manual_frames = set(int(f) for f in annotations.keys())

    print(f"Total frames: {len(frames)}")
    print(f"Manual annotations: {len(manual_frames)}")
    print(f"Auto-detected: {len(frames) - len(manual_frames)}")
    print()

    # Detect problems
    problems = defaultdict(list)

    prev_frame = None
    prev_center = None
    static_count = 0
    static_start = None

    for i, frame in enumerate(frames):
        det = detections[str(frame)]
        center = det['center']
        method = det.get('method', 'unknown')
        velocity = det.get('velocity', 0)

        if prev_center:
            dist = distance(center, prev_center)
            frame_gap = frame - prev_frame

            # Problem 1: Static ball (same position for multiple frames)
            if dist < 1.0 and frame_gap == 1:
                if static_count == 0:
                    static_start = prev_frame
                static_count += 1
            else:
                if static_count >= 5:  # 5+ frames static
                    problems['static'].append({
                        'frames': f"{static_start}-{prev_frame}",
                        'count': static_count,
                        'position': prev_center
                    })
                static_count = 0

            # Problem 2: Large jumps (erratic movement)
            if dist > 50 and frame_gap == 1:
                problems['jump'].append({
                    'frame': frame,
                    'distance': dist,
                    'velocity': velocity,
                    'from': prev_center,
                    'to': center
                })

            # Problem 3: High velocity
            if velocity > 25:
                problems['high_velocity'].append({
                    'frame': frame,
                    'velocity': velocity,
                    'distance': dist
                })

        prev_frame = frame
        prev_center = center

    # Check final static count
    if static_count >= 5:
        problems['static'].append({
            'frames': f"{static_start}-{prev_frame}",
            'count': static_count,
            'position': prev_center
        })

    # Report problems
    print("=" * 70)
    print("DETECTED PROBLEMS")
    print("=" * 70)
    print()

    if problems['static']:
        print(f"⚠ STATIC BALL ({len(problems['static'])} instances):")
        print("   Ball stays in same position for multiple frames (physically impossible)")
        print()
        for p in problems['static']:
            print(f"   Frames {p['frames']}: {p['count']} frames static at {p['position']}")
        print()

    if problems['jump']:
        print(f"⚠ ERRATIC JUMPS ({len(problems['jump'])} instances):")
        print("   Ball teleports >50px in one frame")
        print()
        for p in sorted(problems['jump'], key=lambda x: x['distance'], reverse=True)[:10]:
            print(f"   Frame {p['frame']}: {p['distance']:.1f}px jump (velocity={p['velocity']:.1f})")
            print(f"      From {p['from']} → {p['to']}")
        print()

    if problems['high_velocity']:
        print(f"⚠ HIGH VELOCITY ({len(problems['high_velocity'])} instances):")
        print("   Unrealistic ball speed")
        print()
        for p in sorted(problems['high_velocity'], key=lambda x: x['velocity'], reverse=True)[:10]:
            print(f"   Frame {p['frame']}: velocity={p['velocity']:.1f} (distance={p['distance']:.1f}px)")
        print()

    if not any(problems.values()):
        print("✓ No major problems detected!")
        print()

    # Statistics
    print("=" * 70)
    print("STATISTICS")
    print("=" * 70)
    print()

    velocities = [d.get('velocity', 0) for d in detections.values() if 'velocity' in d]
    if velocities:
        print(f"Velocity range: {min(velocities):.1f} - {max(velocities):.1f}")
        print(f"Average velocity: {sum(velocities)/len(velocities):.1f}")
        print()

    methods = defaultdict(int)
    for d in detections.values():
        methods[d.get('method', 'unknown')] += 1

    print("Detection methods:")
    for method, count in sorted(methods.items(), key=lambda x: x[1], reverse=True):
        pct = (count / len(detections)) * 100
        print(f"  {method}: {count} ({pct:.1f}%)")
    print()

    # Recommendations
    print("=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print()

    if problems['static']:
        print("1. Fix static ball problem:")
        print("   - Improve auto-detection between keyframes")
        print("   - Use physics-based prediction (gravity) instead of static position")
        print()

    if problems['jump'] or problems['high_velocity']:
        print("2. Reduce erratic jumps:")
        print("   - Smooth trajectory with better Kalman filtering")
        print("   - Add velocity/acceleration constraints")
        print("   - Use constant-acceleration model (not constant-velocity)")
        print()

    if max(frames) - min(manual_frames) > 100:
        print("3. Add more manual annotations:")
        print("   - Ball trajectory after last annotation is unreliable")
        print("   - Consider hiding ball if confidence < threshold")
        print()

if __name__ == '__main__':
    detections = load_json('outputs/detections.json')
    annotations = load_json('outputs/annotations.json')

    analyze_trajectory(detections, annotations)
