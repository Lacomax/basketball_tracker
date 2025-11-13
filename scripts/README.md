# Scripts

Main user-facing scripts for basketball tracking pipeline.

## Pipeline Script

### `pipeline.py`
**Master script that runs the complete tracking pipeline**

Executes all steps sequentially with interactive prompts:
1. Filter Court ROI
2. Assign Player Names
3. Assign Teams
4. Annotate Ball (optional)
5. Generate Ball Trajectory (optional)
6. Create Annotated Video

**Usage:**
```bash
python scripts/pipeline.py
```

**Features:**
- Interactive Y/n/q prompts for each step
- Skip optional steps
- Retry failed steps
- Quit at any time
- Progress tracking

---

## Individual Scripts

Use these if you want to run specific steps independently:

### `filter_roi.py`
**Filter tracking data to court area only**

- Define court region of interest (ROI) with mouse clicks
- Filter out bench/crowd detections
- Limit to max 10 players per frame
- Uses multiple criteria (center, feet, bbox corners)

**Usage:**
```bash
python scripts/filter_roi.py
```

**Requires:** `outputs/tracked_players.json`
**Creates:** `outputs/tracked_players_filtered.json`

---

### `assign_names.py`
**Assign names to tracked players**

- Shows each unique player with context
- Assign custom names
- Load previous names automatically
- Auto-detect and merge duplicate IDs

**Usage:**
```bash
python scripts/assign_names.py
```

**Requires:** `outputs/tracked_players_filtered.json` or `outputs/tracked_players.json`
**Creates:**
- `outputs/player_names.json`
- `outputs/tracked_players_named.json`

---

### `assign_teams.py`
**Assign players to teams**

- Define team names (e.g., "Red Team", "Yellow Team")
- Assign each player: 1=Team1, 2=Team2, 3=Referee, P=Public
- Public players are hidden in final video
- Load previous assignments

**Usage:**
```bash
python scripts/assign_teams.py
```

**Requires:** `outputs/tracked_players_named.json` (or filtered/raw)
**Creates:**
- `outputs/team_assignments.json`
- `outputs/team_names.json`
- `outputs/tracked_players_named_teams.json`

---

### `create_video.py`
**Create final annotated video**

- Player bounding boxes with names
- Team assignments
- Movement trails (30 frames)
- Ball trajectory (if available)
- Real-time statistics overlay
- Hides "Public" players

**Usage:**
```bash
python scripts/create_video.py
```

**Requires:** Any `outputs/tracked_players*.json`
**Creates:** `outputs/annotated_video.mp4`

**Priority order:**
1. `tracked_players_named_teams.json` ⭐ BEST
2. `tracked_players_filtered_teams.json`
3. `tracked_players_teams.json`
4. `tracked_players_named.json`
5. `tracked_players_filtered.json`
6. `tracked_players.json`

---

## Workflow Example

### Quick Start (Recommended)
```bash
# Run everything with one command
python scripts/pipeline.py
```

### Manual Step-by-Step
```bash
# Step 1: Filter court area
python scripts/filter_roi.py

# Step 2: Assign names (optional)
python scripts/assign_names.py

# Step 3: Assign teams (optional)
python scripts/assign_teams.py

# Step 4: Annotate ball (optional)
python -m src.modules.annotator --video input_video_converted.mp4

# Step 5: Generate trajectory (optional)
python -m src.modules.trajectory_detector --video input_video_converted.mp4

# Step 6: Create final video
python scripts/create_video.py
```

---

## Tips

1. **Always start with `pipeline.py`** unless you need to re-run a specific step

2. **ROI Selection**: Click corners of court area, press ENTER when done

3. **Player Names**: Press ENTER to keep previous names, type new name to change

4. **Teams**: Use numbers (1,2,3,P) to avoid conflicts

5. **Public Category**: Mark crowd/bench as "Public" to hide them in video

6. **Video Output**: Check `outputs/annotated_video.mp4` when complete
