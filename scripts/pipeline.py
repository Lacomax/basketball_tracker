#!/usr/bin/env python3
"""
Master pipeline script for basketball tracking.

Runs all steps sequentially with option to skip each step.
"""

import sys
import os
import subprocess

print("=" * 70)
print("BASKETBALL TRACKING PIPELINE")
print("=" * 70)
print()

# Check if video exists
video_file = None
if os.path.exists("input_video_converted.mp4"):
    video_file = "input_video_converted.mp4"
elif os.path.exists("input_video.mp4"):
    video_file = "input_video.mp4"
else:
    print("❌ No video found (input_video.mp4 or input_video_converted.mp4)")
    sys.exit(1)

print(f"✓ Video found: {video_file}")
print()

# Define pipeline steps
steps = [
    {
        'name': 'Filter Court ROI',
        'description': 'Define court area and filter out crowd/bench',
        'command': 'python scripts/filter_roi.py',
        'optional': False,
        'requires': ['outputs/tracked_players.json']
    },
    {
        'name': 'Assign Player Names',
        'description': 'Give names to players and merge duplicate IDs',
        'command': 'python scripts/assign_names.py',
        'optional': True,
        'requires': ['outputs/tracked_players_filtered.json', 'outputs/tracked_players.json']
    },
    {
        'name': 'Assign Teams',
        'description': 'Assign players to teams (Red, Yellow, Referee, Public)',
        'command': 'python scripts/assign_teams.py',
        'optional': True,
        'requires': ['outputs/tracked_players_named.json', 'outputs/tracked_players_filtered.json', 'outputs/tracked_players.json']
    },
    {
        'name': 'Annotate Ball',
        'description': 'Manually annotate ball positions in key frames',
        'command': f'python -m src.modules.annotator --video {video_file}',
        'optional': True,
        'requires': []
    },
    {
        'name': 'Annotate Hoop',
        'description': 'Manually annotate basketball hoop position (for static camera)',
        'command': 'python scripts/annotate_hoop.py',
        'optional': True,
        'requires': []
    },
    {
        'name': 'Generate Ball Trajectory',
        'description': 'Generate ball trajectory with auto-detection',
        'command': f'python -m src.modules.trajectory_detector --video {video_file}',
        'optional': True,
        'requires': ['outputs/annotations.json']
    },
    {
        'name': 'Create Annotated Video',
        'description': 'Create final video with all annotations',
        'command': 'python scripts/create_video.py',
        'optional': False,
        'requires': []
    }
]

def check_requirements(requires):
    """Check if any of the required files exist."""
    if not requires:
        return True

    for req in requires:
        if os.path.exists(req):
            return True

    return False

def run_step(step):
    """Run a pipeline step."""
    print()
    print("=" * 70)
    print(f"STEP: {step['name']}")
    print("=" * 70)
    print(f"Description: {step['description']}")
    print(f"Command: {step['command']}")
    print()

    # Check requirements
    if step['requires']:
        if not check_requirements(step['requires']):
            print("⚠ Required files not found:")
            for req in step['requires']:
                print(f"  - {req}")
            print()
            if step['optional']:
                print("This step is optional. Skipping...")
                return 'skip'
            else:
                print("❌ Cannot proceed without required files")
                return 'error'

    # Ask user
    response = input("Run this step? [Y/n/q]: ").strip().lower()

    if response == 'q':
        return 'quit'
    elif response == 'n':
        print("⏭ Skipped")
        return 'skip'
    elif response in ['', 'y', 'yes']:
        print()
        print("Running...")
        print("-" * 70)

        try:
            result = subprocess.run(step['command'], shell=True)

            if result.returncode == 0:
                print("-" * 70)
                print("✓ Step completed successfully")
                return 'success'
            else:
                print("-" * 70)
                print(f"⚠ Step exited with code {result.returncode}")

                retry = input("Retry this step? [y/N]: ").strip().lower()
                if retry in ['y', 'yes']:
                    return run_step(step)  # Recursive retry
                else:
                    if step['optional']:
                        cont = input("Continue to next step? [Y/n]: ").strip().lower()
                        if cont in ['', 'y', 'yes']:
                            return 'skip'
                        else:
                            return 'quit'
                    else:
                        return 'error'
        except KeyboardInterrupt:
            print()
            print("⚠ Interrupted by user")
            cont = input("Continue to next step? [y/N]: ").strip().lower()
            if cont in ['y', 'yes']:
                return 'skip'
            else:
                return 'quit'
    else:
        print("⏭ Skipped")
        return 'skip'

# Main pipeline execution
print("This pipeline will guide you through all tracking steps.")
print("You can skip optional steps or quit at any time.")
print()
input("Press ENTER to start...")

for i, step in enumerate(steps, 1):
    print()
    print(f"[{i}/{len(steps)}]")

    result = run_step(step)

    if result == 'quit':
        print()
        print("=" * 70)
        print("Pipeline stopped by user")
        print("=" * 70)
        sys.exit(0)
    elif result == 'error':
        print()
        print("=" * 70)
        print("Pipeline stopped due to error")
        print("=" * 70)
        sys.exit(1)

# Pipeline completed
print()
print("=" * 70)
print("PIPELINE COMPLETED!")
print("=" * 70)
print()

# Show results
print("Generated files:")
if os.path.exists("outputs/tracked_players_filtered.json"):
    print("  ✓ outputs/tracked_players_filtered.json")
if os.path.exists("outputs/tracked_players_named.json"):
    print("  ✓ outputs/tracked_players_named.json")
if os.path.exists("outputs/tracked_players_named_teams.json"):
    print("  ✓ outputs/tracked_players_named_teams.json")
if os.path.exists("outputs/annotations.json"):
    print("  ✓ outputs/annotations.json")
if os.path.exists("outputs/detections.json"):
    print("  ✓ outputs/detections.json")
if os.path.exists("outputs/annotated_video.mp4"):
    print("  ✓ outputs/annotated_video.mp4")

print()
print("Next steps:")
print("  - Open outputs/annotated_video.mp4 to view results")
print("  - Re-run specific steps if needed")
print()
