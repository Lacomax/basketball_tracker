# Tools

Utility tools for video conversion and testing.

## `convert_video.py`
**Convert video to OpenCV-compatible format**

Converts videos that OpenCV cannot open natively (QuickTime, certain codecs, etc.) to a standard H.264/MP4 format.

**Usage:**
```bash
python tools/convert_video.py
```

**Methods (tries in order):**
1. FFmpeg command-line (fastest, best quality)
2. MoviePy library (Python-based)

**Input:** `input_video.mp4`
**Output:** `input_video_converted.mp4`

**When to use:**
- OpenCV shows "Cannot open video" errors
- GStreamer warnings appear
- Video has incompatible codec (QuickTime, AV1, etc.)

**Alternative:** Convert online at cloudconvert.com and save as `input_video_converted.mp4`

---

## `test_features.py`
**Test v3.0 features without dependencies**

Validates that all v3.0 modules are correctly structured and can be imported.

**Usage:**
```bash
python tools/test_features.py
```

**Tests:**
- Module imports
- Function signatures
- Class structure
- No actual video processing (safe to run anytime)

**Output:**
```
Testing linear regression shot detection...
✓ Linear regression implementation exists

Testing metrics calculation...
✓ MetricsCalculator class exists

Testing homography transformation...
✓ TacticalView class exists

✅ ALL CODE TESTS PASSED!
```

---

## `analyze_trajectory.py`
**Analyze ball trajectory for problems**

Detects and reports problems in ball trajectory such as static frames, erratic jumps, and unrealistic velocities.

**Usage:**
```bash
python tools/analyze_trajectory.py
```

**Analyzes:**
- Static ball (same position for multiple frames)
- Erratic jumps (>50px teleports)
- High velocity (unrealistic speed)
- Detection method distribution
- Velocity statistics

**Example Output:**
```
⚠ STATIC BALL (2 instances):
   Frames 42-53: 11 frames static at [1004, 279]
   Frames 229-322: 93 frames static at [1184, 647]

⚠ ERRATIC JUMPS (13 instances):
   Frame 180: 244.4px jump (velocity=0.0)
   Frame 42: 160.3px jump (velocity=0.0)

RECOMMENDATIONS:
1. Add more manual annotations
2. Use improved trajectory detector
3. Check auto-detection parameters
```

**When to use:**
- After generating ball trajectory
- When trajectory looks erratic in video
- To validate trajectory quality before creating final video
- To diagnose interpolation problems

**Required files:**
- `outputs/detections.json` - Generated trajectory
- `outputs/annotations.json` - Manual annotations

---

## Tips

### Video Conversion

If `convert_video.py` fails:

1. **Install FFmpeg** (best option):
   ```bash
   # Ubuntu/Debian
   sudo apt install ffmpeg

   # Mac
   brew install ffmpeg

   # Windows
   # Download from ffmpeg.org
   ```

2. **Install MoviePy**:
   ```bash
   pip install moviepy
   ```

3. **Use online converter**:
   - Go to cloudconvert.com
   - Convert to MP4 (H.264 codec, yuv420p)
   - Save as `input_video_converted.mp4`

### Testing

Run `test_features.py` after:
- Updating code
- Installing new dependencies
- Pulling from git
- Before processing important videos
