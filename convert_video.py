#!/usr/bin/env python3
"""
Convert video to OpenCV-compatible format.

This script converts videos to a format that OpenCV can read reliably.
Uses ffmpeg-python or moviepy.
"""

import sys
import os
import subprocess

print("=" * 60)
print("VIDEO CONVERTER FOR BASKETBALL TRACKER")
print("=" * 60)
print()

input_video = "input_video.mp4"
output_video = "input_video_converted.mp4"

if not os.path.exists(input_video):
    print(f"❌ Video not found: {input_video}")
    sys.exit(1)

print(f"✓ Video found: {input_video}")
print()

# Try different conversion methods
conversion_successful = False

# Method 1: Try ffmpeg directly
print("Method 1: Trying ffmpeg...")
try:
    result = subprocess.run(
        ["ffmpeg", "-version"],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print("✓ FFmpeg found")
        print()
        print("Converting video to OpenCV-compatible format...")
        print("This may take a few minutes...")
        print()

        cmd = [
            "ffmpeg",
            "-i", input_video,
            "-c:v", "libx264",      # H.264 codec
            "-preset", "fast",       # Fast encoding
            "-crf", "23",            # Good quality
            "-pix_fmt", "yuv420p",   # Compatible pixel format
            "-vf", "scale=1568:880", # Keep original resolution
            "-r", "30",              # Reduce to 30 fps (easier to process)
            "-y",                    # Overwrite output
            output_video
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0 and os.path.exists(output_video):
            print("✅ Video converted successfully!")
            conversion_successful = True
        else:
            print("⚠ FFmpeg conversion failed")
            print(f"Error: {result.stderr[:200]}")
    else:
        print("⚠ FFmpeg not working properly")

except FileNotFoundError:
    print("⚠ FFmpeg not found in PATH")

print()

# Method 2: Try moviepy
if not conversion_successful:
    print("Method 2: Trying moviepy...")
    try:
        from moviepy.editor import VideoFileClip

        print("✓ MoviePy found")
        print()
        print("Converting video...")

        clip = VideoFileClip(input_video)

        # Reduce fps to 30 for easier processing
        clip_resized = clip.set_fps(30)

        clip_resized.write_videofile(
            output_video,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True,
            preset='fast',
            ffmpeg_params=['-pix_fmt', 'yuv420p']
        )

        clip.close()
        clip_resized.close()

        if os.path.exists(output_video):
            print()
            print("✅ Video converted successfully!")
            conversion_successful = True

    except ImportError:
        print("⚠ MoviePy not installed")
        print("   Install with: pip install moviepy")
    except Exception as e:
        print(f"⚠ MoviePy conversion failed: {e}")

print()

if conversion_successful:
    # Test if OpenCV can read the converted video
    print("Testing converted video with OpenCV...")

    try:
        import cv2

        cap = cv2.VideoCapture(output_video)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()

            if ret:
                print("✅ OpenCV can read the converted video!")
                print()
                print("=" * 60)
                print("SUCCESS!")
                print("=" * 60)
                print()
                print(f"✓ Original video: {input_video}")
                print(f"✓ Converted video: {output_video}")
                print()
                print("Next steps:")
                print(f"  1. Use '{output_video}' instead of '{input_video}'")
                print(f"  2. Run: python test_with_video.py")
                print(f"     (Edit the script to use '{output_video}')")
                print()
            else:
                print("⚠ OpenCV can read but cannot decode frames")
        else:
            print("⚠ OpenCV still cannot open the converted video")

    except Exception as e:
        print(f"⚠ Error testing with OpenCV: {e}")
else:
    print("=" * 60)
    print("CONVERSION FAILED")
    print("=" * 60)
    print()
    print("Neither ffmpeg nor moviepy worked.")
    print()
    print("Manual solutions:")
    print()
    print("1. Install FFmpeg:")
    print("   - Download from: https://ffmpeg.org/download.html")
    print("   - Add to PATH")
    print("   - Run this script again")
    print()
    print("2. Or use VLC Media Player:")
    print("   - Media → Convert/Save")
    print("   - Select your video")
    print("   - Profile: Video - H.264 + MP3 (MP4)")
    print("   - Save as 'input_video_converted.mp4'")
    print()
    print("3. Or use online converter:")
    print("   - https://cloudconvert.com/mp4-converter")
    print("   - Convert to MP4 (H.264)")
    print()
