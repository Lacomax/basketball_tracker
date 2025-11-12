# Basketball Tracker 🏀

A computer vision pipeline for tracking basketballs in video using manual annotation, Kalman filtering, and YOLOv8 deep learning.

## Project Overview

The Basketball Tracker implements a complete pipeline for detecting and tracking basketballs across video frames:

1. **Manual Annotation** - Mark basketball positions in key frames
2. **Trajectory Detection** - Interpolate positions between annotations using Kalman filtering
3. **Verification** - Review and correct detections with anomaly detection
4. **YOLO Training** - Train a custom YOLOv8 model on annotated frames
5. **Inference** - Deploy the trained model on new videos

## Architecture

```
Input Video
    ↓
[1] Manual Annotation (_1_ball_annotator.py)
    ↓
[2] Trajectory Detection (_2_trajectory_detector.py)
    ↓
[3] Verification & Correction (_3_verification_tool.py)
    ↓
[4] YOLO Model Training (_4_yolo_trainer.py)
    ↓
[5] Inference & Tracking (_5_yolo_tracker.py)
    ↓
Output: Tracked video with bounding boxes
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd basketball_tracker

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start with Full Pipeline

```python
from basketball_tracker import UltraBasketballTracker

tracker = UltraBasketballTracker()
tracker.full_pipeline(
    input_video="input_video.mp4",
    max_annotation_frames=5
)
```

### Step-by-Step Usage

#### 1. Annotate Frames Manually

```python
from _1_ball_annotator import BallAnnotator

annotator = BallAnnotator()
annotator.annotate_frames(
    video_path="input_video.mp4",
    max_frames=5
)
# Saves annotations to: annotations.json
```

#### 2. Generate Trajectory via Kalman Filtering

```python
from _2_trajectory_detector import TrajectoryDetector

detector = TrajectoryDetector()
detections = detector.detect_trajectory(
    video_path="input_video.mp4",
    annotation_file="annotations.json"
)
# Saves to: detections.json
```

#### 3. Verify and Correct Detections

```python
from _3_verification_tool import VerificationTool

verifier = VerificationTool()
verified = verifier.verify_detections(
    video_path="input_video.mp4",
    detection_file="detections.json"
)
# Saves to: verified.json
```

#### 4. Train YOLOv8 Model

```python
from _4_yolo_trainer import YOLOTrainer

trainer = YOLOTrainer()
trainer.train_model(
    video_path="input_video.mp4",
    verification_file="verified.json",
    epochs=50
)
# Saves model to: runs/detect/basketball_detector/
```

#### 5. Run Inference on New Video

```python
from _5_yolo_tracker import YOLOTracker

tracker = YOLOTracker()
tracker.track_video(
    video_path="new_video.mp4",
    model_path="runs/detect/basketball_detector/weights/best.pt"
)
# Outputs: new_video_tracked.mp4
```

## File Structure

```
basketball_tracker/
├── basketball_tracker.py              # Main orchestrator class
├── _1_ball_annotator.py               # Manual annotation tool
├── _2_trajectory_detector.py          # Kalman filter-based interpolation
├── _3_verification_tool.py            # Verification & correction UI
├── _4_yolo_trainer.py                 # YOLO model training
├── _5_yolo_tracker.py                 # Inference on new videos
├── config.py                          # Configuration constants
├── ball_detection_utils.py            # Shared utility functions
├── requirements.txt                   # Python dependencies
├── annotations.json                   # Manual annotations (output from step 1)
├── detections.json                    # Kalman-filtered detections (output from step 2)
├── verified.json                      # Verified detections (output from step 3)
├── input_video.mp4                    # Example input video
├── runs/detect/basketball_detector/   # YOLO training outputs
└── README.md                          # This file
```

## Configuration

All magic numbers and configurable parameters are centralized in `config.py`:

```python
# Ball detection parameters
HOUGH_PARAM1 = 50
HOUGH_PARAM2 = 30
MIN_RADIUS = 10
MAX_RADIUS = 50
DEFAULT_RADIUS = 15
ROI_OFFSET = 30

# Anomaly detection parameters
ANOMALY_THRESHOLD = 50

# Trajectory parameters
TRAJECTORY_WINDOW = 90
CONNECTION_THRESHOLD = 30

# YOLO training parameters
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 16
DEFAULT_IMG_SIZE = 640
```

Edit `config.py` to adjust these parameters without modifying pipeline code.

## Technologies

| Technology | Version | Purpose |
|-----------|---------|---------|
| OpenCV | 4.8+ | Video processing & frame extraction |
| NumPy | 1.24+ | Numerical operations & Kalman filtering |
| YOLOv8 (Ultralytics) | 8.0+ | Object detection & training |
| FilterPy | 1.4+ | Kalman filter implementation |
| PyTorch | 2.0+ | Deep learning backend |
| PyYAML | 6.0+ | Configuration file handling |

## Algorithm Details

### Kalman Filter Trajectory Detection

The trajectory detector uses a constant-velocity Kalman filter to interpolate ball positions between manual annotations:

- **State**: [x, y, vx, vy] (position and velocity)
- **Process Model**: Constant velocity motion
- **Measurement Model**: Observed ball positions from annotation

This generates smooth trajectory estimates even with sparse manual annotations.

### YOLO Training with Augmentation

The training pipeline applies data augmentation to increase robustness:

- **Rotation**: ±15 degrees
- **Brightness**: 0.8x to 1.2x
- **Blur**: Gaussian blur with varying kernels

### Verification with Anomaly Detection

The verification tool identifies outliers using a sliding window approach:

- **Window size**: 90 frames
- **Outlier detection**: Distance threshold (default: 50 pixels)
- **Correction**: Interactive UI for manual fixes

## Known Limitations

- Manual annotation is required for the initial training set
- Performance depends on video quality and basketball visibility
- Small or occluded basketballs may be missed
- GPU recommended for real-time inference on high-resolution videos

## Future Improvements

- [ ] Multi-object tracking (multiple basketballs)
- [ ] 3D trajectory reconstruction with camera calibration
- [ ] Real-time inference optimization
- [ ] Web UI for annotation and verification
- [ ] Support for different ball sports (soccer, volleyball, etc.)

## Troubleshooting

### "Circle not detected" in annotation

- Ensure basketball is clearly visible in the frame
- Adjust `HOUGH_PARAM1` and `HOUGH_PARAM2` in `config.py`
- Try manual click-to-annotate feature

### Low detection accuracy

- Increase annotation frames (`max_annotation_frames`)
- Ensure verified.json has clean, correct annotations
- Increase YOLO training epochs (`DEFAULT_EPOCHS` in config.py)

### Memory issues during training

- Reduce batch size: `DEFAULT_BATCH_SIZE` in config.py
- Reduce image size: `DEFAULT_IMG_SIZE` in config.py
- Process shorter video clips

## Contributing

1. Follow PEP 8 style guidelines
2. Add docstrings to all functions
3. Use `config.py` for any magic numbers
4. Test changes on sample video data
5. Keep documentation updated

## License

[Add your license here]

## Contact

[Add contact information]

---

**Last Updated**: November 2024
