# Basketball Tracker 🏀

A modular computer vision pipeline for automated basketball detection and tracking using manual annotation, Kalman filtering, and YOLOv8 deep learning.

## 📋 Features

### Core Features
- ✅ **Interactive Manual Annotation** - Click-to-annotate UI for marking basketball positions
- ✅ **Intelligent Trajectory Detection** - Kalman filter-based smooth interpolation between frames
- ✅ **Verification Interface** - Interactive correction tool with anomaly detection
- ✅ **YOLO Model Training** - Custom YOLOv8 model training with data augmentation
- ✅ **Multi-Model Support** - Manage multiple trained models for different scenarios
- ✅ **Organized Data Structure** - Separate directories for raw data, annotations, and outputs
- ✅ **Production Ready** - Comprehensive logging, error handling, and documentation
- ✅ **Installable Package** - Install as Python package via `setup.py`

### 🆕 Advanced Analytics (NEW!)
- ✅ **Enhanced Occlusion Detection** - Detects when ball is hidden by players
- ✅ **Player Detection & Tracking** - Automatic player detection with team assignment
- ✅ **Event Analysis** - Detects shots, passes, dribbles, rebounds automatically
- ✅ **Player Statistics** - Comprehensive per-player stats (shots, assists, distance, etc.)
- ✅ **SQLite Database** - Persistent storage for historical game analysis
- ✅ **Performance Optimizations** - Batch processing and caching for faster analysis

### 🔥 Latest Features - Version 2.0 (NEWEST!)
- ✅ **Hoop Detection** - Automatic basket detection & shot classification (made/missed)
- ✅ **DeepSORT Tracking** - Robust player tracking with consistent IDs
- ✅ **Ball Possession Analysis** - Who has the ball at any moment
- ✅ **Game Visualizer** - Professional videos with real-time stats overlay
- ✅ **Player Re-ID** - Maintains IDs even when players leave/enter frame
- ✅ **YOLOv11 Support** - Latest YOLO models for better small object detection

📖 **See [ADVANCED_FEATURES.md](docs/ADVANCED_FEATURES.md) for complete guide!**
📖 **See [LATEST_FEATURES.md](docs/LATEST_FEATURES.md) for Version 2.0 features!**

## 📁 Project Structure

```
basketball_tracker/
├── src/                              # Source code (main package)
│   ├── basketball_tracker.py         # Main orchestrator class
│   ├── config.py                     # Centralized configuration
│   ├── modules/                      # Core pipeline modules
│   │   ├── annotator.py             # Manual annotation tool
│   │   ├── trajectory_detector.py   # Kalman-based interpolation
│   │   ├── verifier.py              # Interactive verification UI
│   │   └── yolo_trainer.py          # YOLO training pipeline
│   └── utils/                        # Shared utilities
│       └── ball_detection.py        # Common detection functions
│
├── data/                             # Data organization
│   ├── raw/                          # Original video files
│   ├── annotations/                  # Manual annotations (JSON)
│   ├── detections/                   # Kalman-filtered detections
│   └── verified/                     # Verified detections
│
├── models/                           # Model management
│   ├── pretrained/                   # Pre-trained YOLO weights
│   └── trained/                      # Your trained models
│
├── configs/                          # Configuration files
│   └── default.yaml                  # Default settings
│
├── outputs/                          # Training results
├── docs/                             # Documentation
│   └── ARCHITECTURE.md               # Detailed architecture guide
├── tests/                            # Unit tests
├── setup.py                          # Package installer
└── requirements.txt                  # Python dependencies
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA 11.8+ (optional, for GPU acceleration)

### Quick Install

#### Linux/Mac:

```bash
# Clone repository
git clone <repository-url>
cd basketball_tracker

# Install dependencies
pip install -r requirements.txt

# (Optional) Install as editable package
pip install -e .
```

#### Windows:

```powershell
# Clone repository
git clone <repository-url>
cd basketball_tracker

# Run Windows setup script (configures environment + creates directories)
.\setup_windows.ps1

# Install dependencies
pip install -r requirements.txt
```

**Windows Note:** If you get OpenMP errors, see [Windows Troubleshooting Guide](docs/SOLUCION_PROBLEMAS_WINDOWS.md)

**Quick fix for OpenMP errors:**
```powershell
.\fix_windows_openmp.ps1
```

## 📖 Usage

### Quick Start: Basic Pipeline

```python
from src.basketball_tracker import UltraBasketballTracker

# Initialize tracker with video path
tracker = UltraBasketballTracker(video_path="data/raw/your_video.mp4")

# Run complete pipeline
tracker.full_pipeline()
```

This executes all stages: annotate → detect → verify → train → predict

### 🆕 Quick Start: Advanced Analytics

```bash
# Complete game analysis with player stats and events
python -m src.advanced_tracker \
    --video data/raw/game.mp4 \
    --annotations data/annotations/game.json \
    --output outputs/game_analysis \
    --pose \
    --db data/stats.db
```

This executes: ball tracking → player detection → event analysis → statistics → database

### Step-by-Step: Individual Stages

#### 1. Manual Annotation

```python
from src.modules.annotator import BallAnnotator

annotator = BallAnnotator(
    video="data/raw/video.mp4",
    output="data/annotations/annotations.json"
)
annotator.run()
```

**Controls:**
- Click to detect ball (or add annotation)
- Drag to adjust position
- **A/D** - Previous/Next frame
- **S** - Save
- **Q** - Quit

#### 2. Trajectory Detection (Kalman Filter)

```python
from src.modules.trajectory_detector import process_trajectory_video

detections = process_trajectory_video(
    video_path="data/raw/video.mp4",
    annotations_path="data/annotations/annotations.json",
    output_path="data/detections/detections.json"
)
```

Interpolates smooth trajectories between manual annotations.

#### 3. Verification & Correction

```python
from src.modules.verifier import CompactBallVerifier

verifier = CompactBallVerifier(
    video_path="data/raw/video.mp4",
    detection_file="data/detections/detections.json",
    output_file="data/verified/verified.json"
)
verifier.run()
```

**Controls:**
- Click to adjust detections
- **A/D** - Previous/Next frame
- **+/-** - Increase/Decrease radius
- **T** - Toggle trajectory view
- **P/N** - Previous/Next anomaly
- **H** - Hide/Show detection
- **S** - Save
- **Q** - Quit

#### 4. YOLO Model Training

```python
from src.modules.yolo_trainer import UltraYOLOBallTrainer

trainer = UltraYOLOBallTrainer(
    video_path="data/raw/video.mp4",
    annotations="data/verified/verified.json",
    output_dir="models/trained/basketball_detector",
    model="yolov8s.pt"
)

trainer.train(epochs=50, batch_size=16, img_size=640)
```

#### 5. Inference & Detection

```python
from src.modules.yolo_trainer import UltraYOLOBallTrainer

UltraYOLOBallTrainer.detect(
    video_path="data/raw/new_video.mp4",
    model_path="models/trained/basketball_detector/weights/best.pt",
    output_path="outputs/detected_video.mp4",
    conf=0.5
)
```

## ⚙️ Configuration

### Default Configuration

Settings are in `src/config.py` and can be overridden via `configs/default.yaml`:

```yaml
ball_detection:
  hough_param1: 50
  hough_param2_strict: 30
  min_radius: 10
  max_radius: 50

trajectory:
  anomaly_threshold: 50
  window_size: 90
  connection_threshold: 30

yolo:
  epochs: 50
  batch_size: 16
  img_size: 640
```

### Multiple Configurations

Create different configs for different scenarios:

```bash
configs/
├── default.yaml         # Default settings
├── high_accuracy.yaml   # More epochs, larger model
└── real_time.yaml       # Faster inference, smaller model
```

## 📊 Data Format

### Annotations/Detections JSON

```json
{
  "frame_number": {
    "center": [x_pixel, y_pixel],
    "radius": radius_pixels
  }
}
```

**Example:**
```json
{
  "0": {"center": [640, 360], "radius": 12},
  "50": {"center": [600, 380], "radius": 13},
  "100": {"center": [580, 400], "radius": 14}
}
```

## 🎯 Workflow

```
1. Prepare Videos
   └─ Place videos in: data/raw/

2. Manual Annotation
   ├─ Run: src.modules.annotator
   └─ Output: data/annotations/*.json

3. Kalman Filtering
   ├─ Run: src.modules.trajectory_detector
   └─ Output: data/detections/*.json

4. Verification
   ├─ Run: src.modules.verifier
   └─ Output: data/verified/*.json

5. Model Training
   ├─ Run: src.modules.yolo_trainer.train()
   └─ Output: models/trained/*/weights/best.pt

6. Inference
   ├─ Run: src.modules.yolo_trainer.detect()
   └─ Output: outputs/*.mp4
```

## 🔧 Advanced Usage

### Custom Model Support

```python
# Train with different YOLO model
trainer = UltraYOLOBallTrainer(
    video_path="data/raw/video.mp4",
    annotations="data/verified/verified.json",
    model="yolov8m.pt"  # Medium model instead of small
)
trainer.train(epochs=100)
```

### Batch Processing Multiple Videos

```python
import os
from src.basketball_tracker import UltraBasketballTracker

for video_file in os.listdir("data/raw"):
    if video_file.endswith(".mp4"):
        tracker = UltraBasketballTracker(
            video_path=f"data/raw/{video_file}"
        )
        tracker.full_pipeline()
```

### Distributed Training (Future)

```python
# Train on multiple GPUs
trainer.train(epochs=50, device="cuda:0,cuda:1")
```

## 🆕 Quick Start with Pre-trained Models & Roboflow

### Use Pre-trained Model (No Training Required)

```bash
# Test detection with pre-trained YOLO model
python scripts/use_pretrained_model.py --video input_video.mp4

# Use custom trained model
python scripts/use_pretrained_model.py --video input_video.mp4 --model models/basketball_detector.pt
```

### Download Datasets from Roboflow

```bash
# List available verified datasets
python scripts/download_roboflow_dataset.py --list

# Download all recommended datasets
python scripts/download_roboflow_dataset.py --api-key YOUR_API_KEY --download-all

# Download specific dataset
python scripts/download_roboflow_dataset.py \
    --api-key YOUR_API_KEY \
    --workspace roboflow-100 \
    --project basketball-detection
```

### Extract Frames for Annotation

```bash
# Extract frames from video for annotation in Roboflow
python scripts/extract_frames_for_annotation.py --video input_video.mp4 --interval 10
```

**See full guides:**
- 🇪🇸 [Spanish Quick Start](GUIA_RAPIDA_ES.md)
- 📖 [Annotation Guide (Spanish)](docs/GUIA_ANOTACION_VIDEOS.md)
- 🔧 [Roboflow Setup Guide](docs/ROBOFLOW_SETUP_README.md)
- 🪟 [Windows Troubleshooting](docs/SOLUCION_PROBLEMAS_WINDOWS.md)

## 📚 Documentation

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Detailed system design and extension points
- **[API Reference](docs/API.md)** - Function and class documentation
- **[USAGE.md](docs/USAGE.md)** - Detailed usage examples

## 🐛 Troubleshooting

### Issue: "Cannot open video"
```python
# Ensure video file exists and path is correct
import os
assert os.path.exists("data/raw/video.mp4")
```

### Issue: Low detection accuracy
```python
# Increase annotations and training epochs
annotator.run()  # Add more key frames
trainer.train(epochs=100)  # More training
```

### Issue: Memory error during training
```python
# Reduce batch size
trainer.train(batch_size=8)  # Default is 16
```

### Issue: Slow inference
```python
# Use smaller model or CPU
UltraYOLOBallTrainer.detect(
    model_path="models/trained/best.pt",
    device="cpu"  # or "cuda"
)
```

## 🤝 Contributing

Contributions welcome! Please:
1. Follow PEP 8 style guide
2. Add docstrings to new functions
3. Update relevant documentation
4. Test changes on sample data

## 📝 License

[Specify your license here - MIT, Apache, etc.]

## 👥 Authors

Basketball Tracker Team

## 🙏 Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/yolov8)
- [FilterPy](https://github.com/rlabbe/filterpy)
- [OpenCV](https://opencv.org/)

## 📮 Contact & Support

For issues, questions, or suggestions:
- Create an issue on GitHub
- Contact: [your-email@example.com](mailto:your-email@example.com)

---

**Last Updated:** November 2024
**Version:** 1.0.0
