# Basketball Tracker Architecture

## Project Overview

Basketball Tracker is a modular computer vision pipeline designed to detect, track, and analyze basketballs in video footage. It combines manual annotation, statistical filtering, and deep learning techniques to achieve robust ball detection.

## Directory Structure

```
basketball_tracker/
├── src/                           # Source code
│   ├── __init__.py               # Package initialization
│   ├── basketball_tracker.py     # Main orchestrator
│   ├── config.py                 # Centralized configuration
│   ├── modules/                  # Core pipeline modules
│   │   ├── __init__.py
│   │   ├── annotator.py          # Manual annotation tool
│   │   ├── trajectory_detector.py # Kalman-based interpolation
│   │   ├── verifier.py           # Verification and correction UI
│   │   └── yolo_trainer.py       # YOLO training pipeline
│   └── utils/                    # Shared utilities
│       ├── __init__.py
│       └── ball_detection.py     # Shared detection functions
│
├── data/                         # Data organization
│   ├── raw/                      # Original video files
│   ├── annotations/              # Manual annotations
│   ├── detections/               # Kalman-filtered detections
│   └── verified/                 # Verified/corrected detections
│
├── models/                       # Model management
│   ├── pretrained/               # Pre-trained YOLO weights
│   ├── trained/                  # Trained custom models
│   └── README.md                 # Model documentation
│
├── configs/                      # Configuration files
│   ├── default.yaml              # Default configuration
│   └── custom_config.yaml        # Example custom config
│
├── outputs/                      # Training outputs and results
│
├── docs/                         # Documentation
│   ├── ARCHITECTURE.md           # This file
│   └── USAGE.md                  # Usage guide
│
├── tests/                        # Unit and integration tests
│
├── requirements.txt              # Python dependencies
├── setup.py                      # Package installation
├── README.md                     # Project overview
└── .gitignore                    # Git ignore rules
```

## Core Components

### 1. **Main Orchestrator** (`src/basketball_tracker.py`)

The `UltraBasketballTracker` class coordinates the entire pipeline:

```python
tracker = UltraBasketballTracker(video_path="video.mp4")
tracker.full_pipeline()  # Runs all stages sequentially
```

**Key Methods:**
- `annotate()` - Launch manual annotation tool
- `detect()` - Run trajectory detection
- `verify()` - Launch verification UI
- `train_yolo()` - Train YOLO model
- `predict()` - Run inference
- `full_pipeline()` - Execute all stages

### 2. **Manual Annotation** (`src/modules/annotator.py`)

**Purpose:** Interactive tool for marking basketball positions in key frames

**Features:**
- Click to detect ball automatically (Hough circles)
- Drag to adjust detection
- Keyboard controls (A/D for navigation, S to save, Q to quit)

**Output:** `annotations.json`
```json
{
  "0": {"center": [640, 360], "radius": 12},
  "50": {"center": [600, 380], "radius": 13}
}
```

### 3. **Trajectory Detection** (`src/modules/trajectory_detector.py`)

**Purpose:** Interpolate smooth trajectories between manual annotations

**Algorithm:** Constant-velocity Kalman filter
- **State:** [x, y, vx, vy] (position and velocity)
- **Process:** Smooth interpolation between keyframes
- **Features:**
  - Handles occlusions (velocity thresholds)
  - Linear radius interpolation
  - Clamps positions to frame boundaries

**Output:** `detections.json`
```json
{
  "0": {"center": [640, 360], "radius": 12},
  "1": {"center": [642, 362], "radius": 12},
  ...
}
```

### 4. **Verification Tool** (`src/modules/verifier.py`)

**Purpose:** Review and correct auto-detected positions

**Features:**
- Interactive visualization with OpenCV
- Trajectory line display (local or global)
- Anomaly detection with flagging
- Radius adjustment (+/- keys)
- Hide/show detections
- Jump to anomalies (P/N keys)

**Output:** `verified.json` (same format as detections)

### 5. **YOLO Training** (`src/modules/yolo_trainer.py`)

**Purpose:** Train custom YOLOv8/YOLOv11 model for basketball detection

**Pipeline:**
1. Extract frames from video
2. Split into train/validation sets
3. Apply data augmentation
4. Train YOLO model
5. Save best weights

**Data Augmentations:**
- Rotation (±15°)
- Brightness (0.8x - 1.2x)
- Gaussian blur (3-9 kernel)

**Output:** `models/trained/weights/best.pt`

### 6. **Shared Utilities** (`src/utils/ball_detection.py`)

**Purpose:** Reusable ball detection function used by multiple modules

**Main Function:** `auto_detect_ball(frame, point)`
- Takes click point
- Applies Hough circle detection on ROI
- Returns circle center and radius
- Fallback: default radius if detection fails

## Configuration System

Configuration is centralized in `src/config.py` with three levels:

1. **Code Constants** (immutable defaults)
2. **YAML Configuration** (user-customizable in `configs/`)
3. **Runtime Parameters** (passed to functions)

**Example:**
```python
# src/config.py
HOUGH_PARAM1 = 50
ANOMALY_THRESHOLD = 50
```

```yaml
# configs/custom.yaml
ball_detection:
  hough_param1: 60
anomaly_threshold: 40
```

## Data Flow

```
Input Video
    ↓
[1] Manual Annotation → annotations.json
    ↓
[2] Trajectory Detection → detections.json
    ↓
[3] Verification/Correction → verified.json
    ↓
[4] YOLO Training ← verified.json
    ↓
    [Create Dataset]
    [Train Model]
    ↓ weights/best.pt
    ↓
[5] Inference
    ↓
Output: Tracked Video
```

## Module Dependencies

```
basketball_tracker.py (orchestrator)
├── annotator.py
│   └── ball_detection.py
├── trajectory_detector.py
├── verifier.py
│   └── ball_detection.py
└── yolo_trainer.py

All modules depend on:
├── config.py (logging, parameters)
└── External: OpenCV, YOLO, NumPy, filterpy
```

## Key Design Patterns

### 1. **Modularity**
Each stage is independent and can be:
- Run individually
- Reused in other projects
- Extended or modified

### 2. **Method Chaining**
All orchestrator methods return `self` for fluent API:
```python
tracker.annotate().detect().verify().train_yolo().predict()
```

### 3. **Dynamic Imports**
Modules imported at runtime to:
- Reduce startup time
- Allow flexible module loading
- Enable future plugin architecture

### 4. **Context Managers**
All file operations use `with` statements:
```python
with open(file_path, 'r') as f:
    data = json.load(f)
```

### 5. **Logging**
Structured logging instead of print() for:
- Production debugging
- Log level control
- Output redirection

## Extension Points

### Adding a New Pipeline Stage

1. Create module in `src/modules/new_stage.py`
2. Implement main function/class
3. Update `src/basketball_tracker.py`:
   ```python
   def new_stage(self):
       func = self._import_module('modules.new_stage').function
       func(...)
       return self
   ```
4. Add to `full_pipeline()` chain

### Supporting Additional Models

1. Place model weights in `models/pretrained/`
2. Update `YOLO_MODEL_NAME` in `src/config.py`
3. Create model-specific config in `configs/`

### Custom Configurations

1. Create `configs/my_config.yaml`
2. Load at runtime (future enhancement)
3. Override defaults per-project basis

## Performance Considerations

### Kalman Filter
- **Complexity:** O(n) where n = number of frames
- **Memory:** Minimal (state: 4 floats per filter)
- **Speed:** Real-time capable

### Hough Circle Detection
- **Complexity:** O(n·m) where n = pixels, m = radii
- **Optimization:** Only on ROI (reduce by ~90%)
- **Speed:** ~50-100ms per frame

### YOLO Training
- **Memory:** 8-16GB GPU recommended (batch_size=16)
- **Time:** ~30 minutes per 100 epochs on Tesla V100
- **Speed:** Real-time inference on GPU

## Debugging and Testing

### Enable Verbose Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Test Individual Modules
```python
from src.modules.annotator import BallAnnotator
annotator = BallAnnotator('test_video.mp4')
annotator.run()
```

### Unit Tests
```bash
pytest tests/
pytest --cov=src tests/
```

## Future Enhancements

- [ ] Multi-ball tracking
- [ ] 3D trajectory reconstruction
- [ ] Real-time processing
- [ ] Web UI for annotation
- [ ] Model ensemble voting
- [ ] Automatic hyperparameter tuning
- [ ] Video streaming support
- [ ] Distributed training support

## References

- **OpenCV:** https://docs.opencv.org/
- **YOLO:** https://docs.ultralytics.com/
- **FilterPy (Kalman):** https://filterpy.readthedocs.io/
- **NumPy:** https://numpy.org/doc/
