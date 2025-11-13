# Quick Start Guide

Get started with basketball tracking in 5 minutes.

## Prerequisites

1. **Video file**: Place your video as `input_video.mp4` or `input_video_converted.mp4`
2. **Player tracking**: Run player tracking first (or use existing `outputs/tracked_players.json`)

## One-Command Solution

```bash
python scripts/pipeline.py
```

This runs the complete pipeline with interactive prompts. You can skip optional steps or quit at any time.

---

## Step-by-Step Guide

### Step 1: Convert Video (if needed)

If you see "Cannot open video" errors:

```bash
python tools/convert_video.py
```

### Step 2: Track Players (if not done)

```bash
python -m src.modules.improved_tracker --video input_video_converted.mp4
```

### Step 3: Run Pipeline

```bash
python scripts/pipeline.py
```

Follow the interactive prompts:
- **Filter ROI**: Click court corners → ENTER
- **Assign Names**: Type names or ENTER to skip
- **Assign Teams**: Enter 1 (Team1), 2 (Team2), 3 (Referee), P (Public)
- **Ball Annotation**: Click ball in key frames (optional)
- **Create Video**: Generates final output

### Step 4: View Result

```bash
# Your video is ready!
outputs/annotated_video.mp4
```

---

## Common Commands

### Run Specific Step

```bash
# Just filter ROI
python scripts/filter_roi.py

# Just assign names
python scripts/assign_names.py

# Just assign teams
python scripts/assign_teams.py

# Just create video
python scripts/create_video.py
```

### Ball Tracking (Optional)

```bash
# Annotate ball manually
python -m src.modules.annotator --video input_video_converted.mp4

# Generate trajectory
python -m src.modules.trajectory_detector --video input_video_converted.mp4
```

---

## File Structure

```
basketball_tracker/
├── input_video.mp4              # Your input video
├── input_video_converted.mp4    # Converted video (if needed)
│
├── scripts/                     # Main user scripts
│   ├── pipeline.py             # ⭐ Run everything
│   ├── filter_roi.py           # Filter court area
│   ├── assign_names.py         # Name players
│   ├── assign_teams.py         # Assign teams
│   └── create_video.py         # Generate final video
│
├── tools/                       # Utilities
│   ├── convert_video.py        # Fix video codecs
│   └── test_features.py        # Test installation
│
├── outputs/                     # Generated files
│   ├── tracked_players*.json   # Tracking data
│   ├── player_names.json       # Player names
│   ├── team_assignments.json   # Team data
│   ├── detections.json         # Ball trajectory
│   └── annotated_video.mp4     # ⭐ Final result
│
└── src/                         # Source code
    ├── modules/                # Core modules
    └── utils/                  # Utilities
```

---

## Troubleshooting

### "Cannot open video"
```bash
python tools/convert_video.py
```

### "No tracking data found"
```bash
python -m src.modules.improved_tracker --video input_video_converted.mp4
```

### Want to re-run a step?
Just run the specific script again (e.g., `python scripts/assign_names.py`)

### Video has wrong players?
Re-run `python scripts/filter_roi.py` and define a better ROI

---

## Tips

1. **Use the pipeline**: `python scripts/pipeline.py` handles everything
2. **Skip optional steps**: Press `n` when asked
3. **Ball tracking is optional**: Video works great without it
4. **Public category**: Use "P" to hide crowd/bench from video
5. **Previous data**: Scripts remember your choices (names, teams, ROI)

---

## Need Help?

- **Full documentation**: See `docs/` folder
- **Script details**: See `scripts/README.md`
- **Tool details**: See `tools/README.md`
- **Issues**: Check GitHub issues page

---

## Example Session

```
$ python scripts/pipeline.py

============================================================
BASKETBALL TRACKING PIPELINE
============================================================

✓ Video found: input_video_converted.mp4

[1/6]
============================================================
STEP: Filter Court ROI
============================================================
Description: Define court area and filter out crowd/bench
Command: python scripts/filter_roi.py

Run this step? [Y/q]: Y

Running...
...
✓ Step completed successfully

[2/6]
============================================================
STEP: Assign Player Names
============================================================
...
Run this step? [Y/n/q]: Y

[3/6]
...

============================================================
PIPELINE COMPLETED!
============================================================

Generated files:
  ✓ outputs/tracked_players_named_teams.json
  ✓ outputs/annotated_video.mp4

Next steps:
  - Open outputs/annotated_video.mp4 to view results
```

Enjoy your basketball tracking! 🏀
