#!/usr/bin/env python3
"""
Test custom basketball detector model on sample frames.

This script loads the trained model and tests it on frames from your video
to verify detection quality before using it in the full pipeline.

Usage:
    python scripts/test_basketball_model.py [--model MODEL_PATH] [--video VIDEO_PATH]
"""

import argparse
import cv2
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ultralytics import YOLO


def test_model_on_video(model_path: str, video_path: str, num_frames: int = 10):
    """
    Test basketball detector on random frames from video.

    Args:
        model_path: Path to trained model
        video_path: Path to video file
        num_frames: Number of random frames to test
    """
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video has {total_frames} frames")

    # Test on evenly spaced frames
    test_frames = [int(i * total_frames / num_frames) for i in range(num_frames)]

    detections_count = 0
    total_confidence = 0.0

    print(f"\n{'Frame':<8} {'Detections':<12} {'Confidence':<12} {'Size (px)':<12}")
    print("-" * 50)

    for frame_num in test_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if not ret:
            continue

        # Run detection
        results = model(frame, verbose=False, conf=0.15)

        num_detections = len(results[0].boxes)

        if num_detections > 0:
            # Get best detection
            boxes = results[0].boxes
            best_conf = 0
            best_size = 0

            for box in boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                size = max(x2 - x1, y2 - y1)

                if conf > best_conf:
                    best_conf = conf
                    best_size = size

            detections_count += 1
            total_confidence += best_conf

            print(f"{frame_num:<8} {num_detections:<12} {best_conf:<12.3f} {best_size:<12.1f}")
        else:
            print(f"{frame_num:<8} 0            -            -")

    cap.release()

    print("\n" + "=" * 50)
    print(f"Detection rate: {detections_count}/{num_frames} frames ({100*detections_count/num_frames:.1f}%)")

    if detections_count > 0:
        avg_confidence = total_confidence / detections_count
        print(f"Average confidence: {avg_confidence:.3f}")

        if detections_count >= 7:
            print("\n✅ Model looks good! Detection rate >= 70%")
        elif detections_count >= 4:
            print("\n⚠️  Model is okay but could be better (40-70% detection rate)")
            print("   Consider training for more epochs or using more data")
        else:
            print("\n❌ Model needs improvement (< 40% detection rate)")
            print("   Consider:")
            print("   - Adding more diverse training data")
            print("   - Training for more epochs")
            print("   - Using data augmentation")
    else:
        print("\n❌ Model detected nothing!")
        print("   The model may not be trained correctly.")
        print("   Check training logs and dataset quality.")


def test_model_with_visualization(model_path: str, video_path: str, output_dir: str = 'outputs/model_test'):
    """
    Test model and save annotated frames for visual inspection.

    Args:
        model_path: Path to trained model
        video_path: Path to video file
        output_dir: Directory to save annotated frames
    """
    print(f"\nSaving annotated test frames to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Test 10 frames
    test_frames = [int(i * total_frames / 10) for i in range(10)]

    for idx, frame_num in enumerate(test_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if not ret:
            continue

        # Run detection
        results = model(frame, verbose=False, conf=0.15)

        # Draw detections
        annotated_frame = frame.copy()

        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0])

                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw confidence
                label = f"ball {conf:.2f}"
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        # Add frame number
        cv2.putText(
            annotated_frame,
            f"Frame: {frame_num}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        # Save frame
        output_path = os.path.join(output_dir, f'frame_{idx:03d}_{frame_num:05d}.jpg')
        cv2.imwrite(output_path, annotated_frame)

    cap.release()
    print(f"✓ Saved 10 annotated frames to {output_dir}")
    print(f"  Review these images to assess model quality")


def main():
    parser = argparse.ArgumentParser(description="Test custom basketball detector")
    parser.add_argument(
        '--model',
        default='models/basketball_detector_yolo11l.pt',
        help='Path to trained model'
    )
    parser.add_argument(
        '--video',
        default='input_video.mp4',
        help='Path to test video'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Save annotated frames for visual inspection'
    )

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        print("\nTrain a model first:")
        print("  python scripts/train_basketball_detector.py")
        return 1

    if not os.path.exists(args.video):
        print(f"❌ Video not found: {args.video}")
        return 1

    print("=" * 50)
    print("Basketball Detector Model Test")
    print("=" * 50)

    # Run quantitative test
    test_model_on_video(args.model, args.video, num_frames=10)

    # Run visualization test if requested
    if args.visualize:
        test_model_with_visualization(args.model, args.video)

    print("\n" + "=" * 50)
    print("Testing complete!")
    print("=" * 50)

    return 0


if __name__ == '__main__':
    sys.exit(main())
