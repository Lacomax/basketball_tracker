# Project Structure

Clean and organized basketball tracking project structure.

## Directory Layout

```
basketball_tracker/
│
├── 📄 README.md                 # Main project documentation
├── 📄 QUICKSTART.md             # Quick start guide (START HERE)
├── 📄 setup.py                  # Package installation
├── 📄 requirements.txt          # Python dependencies
│
├── 📁 scripts/                  # Main user scripts ⭐
│   ├── README.md               # Script documentation
│   ├── pipeline.py             # Master pipeline (RUN THIS)
│   ├── filter_roi.py           # Filter court ROI
│   ├── assign_names.py         # Assign player names
│   ├── assign_teams.py         # Assign teams
│   └── create_video.py         # Create annotated video
│
├── 📁 tools/                    # Utility tools
│   ├── README.md               # Tool documentation
│   ├── convert_video.py        # Video format converter
│   └── test_features.py        # Feature testing
│
├── 📁 src/                      # Source code
│   ├── __init__.py
│   ├── config.py               # Configuration
│   │
│   ├── 📁 modules/             # Core modules
│   │   ├── annotator.py        # Ball annotation tool
│   │   ├── event_analyzer.py   # Event detection
│   │   ├── game_visualizer.py  # Game visualization
│   │   ├── hoop_detector.py    # Hoop detection
│   │   ├── improved_tracker.py # Player tracking (ByteTrack)
│   │   ├── metrics_calculator.py # Performance metrics
│   │   ├── player_detector.py  # Player detection (YOLO)
│   │   ├── player_reid.py      # Re-identification
│   │   ├── possession_analyzer.py # Possession analysis
│   │   ├── professional_visualizer.py # Pro viz
│   │   ├── statistics_generator.py # Statistics
│   │   ├── tactical_view.py    # Tactical view
│   │   ├── team_classifier.py  # Team classification
│   │   ├── trajectory_detector.py # Ball trajectory
│   │   ├── verifier.py         # Verification
│   │   └── yolo_trainer.py     # YOLO training
│   │
│   └── 📁 utils/               # Utilities
│       ├── ball_detection.py   # Ball detection utils
│       ├── database.py         # Database utils
│       └── video_utils.py      # Video I/O utils
│
├── 📁 docs/                     # Documentation
│   ├── ADVANCED_FEATURES.md    # Advanced features
│   ├── ARCHITECTURE.md         # System architecture
│   ├── FEATURES_V3.md          # v3.0 features
│   └── LATEST_FEATURES.md      # Latest updates
│
├── 📁 tests/                    # Unit tests
│   └── __init__.py
│
├── 📁 outputs/                  # Generated outputs
│   ├── tracked_players.json    # Raw tracking
│   ├── tracked_players_filtered.json  # Filtered
│   ├── tracked_players_named.json     # With names
│   ├── tracked_players_named_teams.json # With teams ⭐
│   ├── player_names.json       # Player names
│   ├── team_assignments.json   # Team data
│   ├── team_names.json         # Team names
│   ├── court_roi.json          # Court ROI
│   ├── annotations.json        # Ball annotations
│   ├── detections.json         # Ball trajectory
│   └── annotated_video.mp4     # Final video ⭐
│
├── 📁 data/                     # Input data
│   └── (training data, models)
│
└── 📁 documentation_from_other_projects/  # Reference docs
    └── (GitHub project docs)
```

---

## Key Files by Purpose

### For Users

| File | Purpose |
|------|---------|
| `QUICKSTART.md` | ⭐ Start here for quick setup |
| `scripts/pipeline.py` | ⭐ Run entire pipeline |
| `scripts/README.md` | Script documentation |
| `tools/README.md` | Tool documentation |
| `docs/FEATURES_V3.md` | Feature overview |

### For Developers

| File | Purpose |
|------|---------|
| `src/config.py` | Configuration settings |
| `src/modules/` | Core functionality |
| `src/utils/` | Utility functions |
| `docs/ARCHITECTURE.md` | System design |
| `tests/` | Unit tests |

---

## Module Overview

### Core Modules (`src/modules/`)

| Module | Description |
|--------|-------------|
| `improved_tracker.py` | ByteTrack player tracking |
| `player_detector.py` | YOLO person detection |
| `hoop_detector.py` | Hoop detection + shot analysis |
| `trajectory_detector.py` | Ball trajectory (Kalman + auto-detect) |
| `player_reid.py` | Player re-identification (MobileNetV3 + Faiss) |
| `team_classifier.py` | Team classification (Fashion CLIP) |
| `metrics_calculator.py` | Performance metrics |
| `tactical_view.py` | Tactical top-down view (homography) |
| `annotator.py` | Manual ball annotation |
| `game_visualizer.py` | Game visualization |
| `professional_visualizer.py` | Pro visualizations (mplbasketball) |

### Utilities (`src/utils/`)

| Utility | Description |
|---------|-------------|
| `video_utils.py` | Robust video I/O (multi-backend) |
| `ball_detection.py` | Ball detection helpers |
| `database.py` | Database operations |

---

## Data Flow

```
Input Video (input_video.mp4)
    ↓
[Convert if needed] → input_video_converted.mp4
    ↓
[Player Tracking] → tracked_players.json
    ↓
[Filter ROI] → tracked_players_filtered.json
    ↓
[Assign Names] → tracked_players_named.json
    ↓
[Assign Teams] → tracked_players_named_teams.json
    ↓
[Create Video] → annotated_video.mp4 ⭐
```

Optional ball tracking:
```
[Annotate Ball] → annotations.json
    ↓
[Generate Trajectory] → detections.json
    ↓
[Included in final video]
```

---

## Quick Navigation

### I want to...

**...get started quickly**
→ Read `QUICKSTART.md`

**...run the pipeline**
→ `python scripts/pipeline.py`

**...run a specific step**
→ See `scripts/README.md`

**...convert my video**
→ `python tools/convert_video.py`

**...understand the code**
→ See `docs/ARCHITECTURE.md`

**...see all features**
→ See `docs/FEATURES_V3.md`

**...develop/extend**
→ See `src/modules/` and `docs/`

---

## Recent Changes

### Reorganization (Latest)
- ✅ Moved scripts to `scripts/` folder
- ✅ Moved tools to `tools/` folder
- ✅ Removed obsolete files
- ✅ Added comprehensive README files
- ✅ Created QUICKSTART.md guide

### v3.0 Features
- ByteTrack tracking
- Auto ball detection in trajectory
- Robust video I/O (multi-backend)
- Team assignment system
- Public category for crowd
- Pipeline automation
- Cross-platform support

---

## Contributing

1. Keep scripts in `scripts/` folder
2. Keep tools in `tools/` folder
3. Core functionality goes in `src/modules/`
4. Utilities go in `src/utils/`
5. Documentation goes in `docs/`
6. Tests go in `tests/`

---

## Need Help?

- **Quick Start**: `QUICKSTART.md`
- **Scripts**: `scripts/README.md`
- **Tools**: `tools/README.md`
- **Features**: `docs/FEATURES_V3.md`
- **Architecture**: `docs/ARCHITECTURE.md`
