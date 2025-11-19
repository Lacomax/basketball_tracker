#!/usr/bin/env python3
"""
Improved master pipeline script for basketball tracking.

Features:
- Command-line arguments
- Configuration support
- Professional logging
- Better error handling
- Skip/resume functionality
"""

import sys
import os
import subprocess
import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_loader import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PipelineStep:
    """Represents a pipeline step."""

    def __init__(
        self,
        name: str,
        description: str,
        command: str,
        optional: bool = False,
        requires: Optional[List[str]] = None
    ):
        self.name = name
        self.description = description
        self.command = command
        self.optional = optional
        self.requires = requires or []

    def check_requirements(self) -> bool:
        """Check if required files exist."""
        if not self.requires:
            return True

        for req in self.requires:
            if os.path.exists(req):
                return True

        return False

    def __repr__(self):
        return f"PipelineStep(name={self.name}, optional={self.optional})"


class Pipeline:
    """Manages pipeline execution."""

    def __init__(self, config, interactive: bool = True, skip_steps: Optional[List[str]] = None):
        """
        Initialize pipeline.

        Args:
            config: Configuration object
            interactive: Enable interactive prompts
            skip_steps: List of step names to skip
        """
        self.config = config
        self.interactive = interactive
        self.skip_steps = skip_steps or []
        self.video_file = None
        self.steps = []

    def find_video(self) -> bool:
        """Find input video file."""
        video_paths = self.config.get_video_paths()

        for video_path in video_paths:
            if os.path.exists(video_path):
                self.video_file = video_path
                logger.info(f"Video found: {video_path}")
                return True

        logger.error("No video found. Searched paths:")
        for path in video_paths:
            logger.error(f"  - {path}")
        return False

    def setup_steps(self):
        """Setup pipeline steps."""
        output_dir = self.config.get_output_dir()

        self.steps = [
            PipelineStep(
                name='filter_roi',
                description='Define court area and filter out crowd/bench',
                command='python scripts/filter_roi.py',
                optional=False,
                requires=[f'{output_dir}/tracked_players.json']
            ),
            PipelineStep(
                name='assign_names',
                description='Give names to players and merge duplicate IDs',
                command='python scripts/assign_names.py',
                optional=True,
                requires=[
                    f'{output_dir}/tracked_players_filtered.json',
                    f'{output_dir}/tracked_players.json'
                ]
            ),
            PipelineStep(
                name='assign_teams',
                description='Assign players to teams (Red, Yellow, Referee, Public)',
                command='python scripts/assign_teams.py',
                optional=True,
                requires=[
                    f'{output_dir}/tracked_players_named.json',
                    f'{output_dir}/tracked_players_filtered.json',
                    f'{output_dir}/tracked_players.json'
                ]
            ),
            PipelineStep(
                name='annotate_ball',
                description='Manually annotate ball positions in key frames',
                command=f'python -m src.modules.annotator --video {self.video_file}',
                optional=True,
                requires=[]
            ),
            PipelineStep(
                name='annotate_hoop',
                description='Manually annotate basketball hoop position',
                command='python scripts/annotate_hoop.py',
                optional=True,
                requires=[]
            ),
            PipelineStep(
                name='mark_backboard',
                description='Mark the 4 corners of the backboard for perspective',
                command='python scripts/mark_backboard_corners.py',
                optional=True,
                requires=[]
            ),
            PipelineStep(
                name='mark_inner_box',
                description='Mark the 4 corners of the inner target box',
                command='python scripts/mark_inner_box.py',
                optional=True,
                requires=[f'{output_dir}/backboard.json']
            ),
            PipelineStep(
                name='generate_trajectory',
                description='Generate ball trajectory with auto-detection',
                command=f'python -m src.modules.trajectory_detector --video {self.video_file}',
                optional=True,
                requires=[f'{output_dir}/annotations.json']
            ),
            PipelineStep(
                name='create_base_video',
                description='Create base video with players and ball trajectory',
                command='python scripts/create_video_optimized.py --output outputs/final_video_clean.mp4',
                optional=False,
                requires=[]
            ),
            PipelineStep(
                name='add_backboard',
                description='Add backboard and hoop to final video',
                command='python scripts/add_hoop_with_marked_backboard.py',
                optional=False,
                requires=[f'{output_dir}/final_video_clean.mp4', f'{output_dir}/backboard.json']
            )
        ]

    def run_step(self, step: PipelineStep) -> str:
        """
        Run a pipeline step.

        Args:
            step: Pipeline step to run

        Returns:
            Result: 'success', 'skip', 'error', or 'quit'
        """
        logger.info("=" * 70)
        logger.info(f"STEP: {step.name}")
        logger.info("=" * 70)
        logger.info(f"Description: {step.description}")
        logger.info(f"Command: {step.command}")

        # Check if step should be skipped
        if step.name in self.skip_steps:
            logger.info("Step marked for skipping")
            return 'skip'

        # Check requirements
        if step.requires:
            if not step.check_requirements():
                logger.warning("Required files not found:")
                for req in step.requires:
                    logger.warning(f"  - {req}")

                if step.optional:
                    logger.info("This step is optional. Skipping...")
                    return 'skip'
                else:
                    logger.error("Cannot proceed without required files")
                    return 'error'

        # Ask user if interactive
        if self.interactive:
            response = input("Run this step? [Y/n/q]: ").strip().lower()

            if response == 'q':
                return 'quit'
            elif response == 'n':
                logger.info("Skipped by user")
                return 'skip'
            elif response not in ['', 'y', 'yes']:
                logger.info("Skipped by user")
                return 'skip'

        # Run command
        logger.info("Running...")
        logger.info("-" * 70)

        try:
            result = subprocess.run(step.command, shell=True)

            if result.returncode == 0:
                logger.info("-" * 70)
                logger.info("Step completed successfully")
                return 'success'
            else:
                logger.warning(f"Step exited with code {result.returncode}")

                if self.interactive:
                    retry = input("Retry this step? [y/N]: ").strip().lower()
                    if retry in ['y', 'yes']:
                        return self.run_step(step)  # Recursive retry
                    else:
                        if step.optional:
                            cont = input("Continue to next step? [Y/n]: ").strip().lower()
                            if cont in ['', 'y', 'yes']:
                                return 'skip'
                            else:
                                return 'quit'
                        else:
                            return 'error'
                else:
                    return 'error' if not step.optional else 'skip'

        except KeyboardInterrupt:
            logger.warning("Interrupted by user")
            if self.interactive:
                cont = input("Continue to next step? [y/N]: ").strip().lower()
                if cont in ['y', 'yes']:
                    return 'skip'
                else:
                    return 'quit'
            else:
                return 'quit'

    def run(self) -> int:
        """
        Run the complete pipeline.

        Returns:
            Exit code (0 for success, 1 for error)
        """
        logger.info("=" * 70)
        logger.info("BASKETBALL TRACKING PIPELINE")
        logger.info("=" * 70)

        # Find video
        if not self.find_video():
            return 1

        # Setup steps
        self.setup_steps()

        if self.interactive:
            logger.info("This pipeline will guide you through all tracking steps.")
            logger.info("You can skip optional steps or quit at any time.")
            input("Press ENTER to start...")

        # Run steps
        for i, step in enumerate(self.steps, 1):
            logger.info(f"\n[{i}/{len(self.steps)}]")

            result = self.run_step(step)

            if result == 'quit':
                logger.info("=" * 70)
                logger.info("Pipeline stopped by user")
                logger.info("=" * 70)
                return 0
            elif result == 'error':
                logger.error("=" * 70)
                logger.error("Pipeline stopped due to error")
                logger.error("=" * 70)
                return 1

        # Pipeline completed
        self.show_results()
        return 0

    def show_results(self):
        """Show pipeline results."""
        logger.info("\n" + "=" * 70)
        logger.info("PIPELINE COMPLETED!")
        logger.info("=" * 70)

        output_dir = self.config.get_output_dir()

        # Check for generated files
        files_to_check = [
            f"{output_dir}/tracked_players_filtered.json",
            f"{output_dir}/tracked_players_named.json",
            f"{output_dir}/tracked_players_named_teams.json",
            f"{output_dir}/annotations.json",
            f"{output_dir}/detections.json",
            f"{output_dir}/annotated_video.mp4",
        ]

        logger.info("\nGenerated files:")
        for file_path in files_to_check:
            if os.path.exists(file_path):
                logger.info(f"  [+] {file_path}")

        logger.info("\nNext steps:")
        logger.info("  - Open outputs/annotated_video.mp4 to view results")
        logger.info("  - Re-run specific steps if needed")


def save_pipeline_state(state: Dict, output_dir: str):
    """Save pipeline state for resume functionality."""
    state_file = Path(output_dir) / "pipeline_state.json"
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)
    logger.debug(f"Pipeline state saved to {state_file}")


def load_pipeline_state(output_dir: str) -> Optional[Dict]:
    """Load pipeline state."""
    state_file = Path(output_dir) / "pipeline_state.json"
    if state_file.exists():
        try:
            with open(state_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load pipeline state: {e}")
    return None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Basketball tracking pipeline - runs all processing steps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Steps:
  1. filter_roi      - Filter tracking data to court area
  2. assign_names    - Assign names to players (optional)
  3. assign_teams    - Assign players to teams (optional)
  4. annotate_ball   - Manually annotate ball positions (optional)
  5. annotate_hoop   - Annotate hoop position (optional)
  6. mark_backboard  - Mark backboard corners (optional)
  7. mark_inner_box  - Mark inner target box (optional)
  8. generate_trajectory - Generate ball trajectory (optional)
  9. create_base_video - Create video with players and ball
 10. add_backboard   - Add backboard and hoop to video

Examples:
  %(prog)s
  %(prog)s --non-interactive
  %(prog)s --skip assign_names assign_teams
  %(prog)s --config custom_config.yaml
        """
    )

    parser.add_argument(
        '--config', '-c',
        help='Path to config.yaml',
        type=str,
        default=None
    )

    parser.add_argument(
        '--non-interactive', '-n',
        help='Run without interactive prompts (auto-yes)',
        action='store_true'
    )

    parser.add_argument(
        '--skip',
        help='Skip specific steps (space-separated step names)',
        nargs='+',
        default=[]
    )

    parser.add_argument(
        '--log-level',
        help='Logging level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO'
    )

    parser.add_argument(
        '--list-steps',
        help='List all pipeline steps and exit',
        action='store_true'
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = get_config()
        if args.config:
            config.load(args.config)
        config.ensure_directories()
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1

    # Setup logger
    logger.setLevel(args.log_level)

    # List steps if requested
    if args.list_steps:
        print("\nAvailable pipeline steps:")
        temp_pipeline = Pipeline(config, interactive=False)
        temp_pipeline.find_video()
        temp_pipeline.setup_steps()

        for i, step in enumerate(temp_pipeline.steps, 1):
            optional_marker = " (optional)" if step.optional else " (required)"
            print(f"  {i}. {step.name}{optional_marker}")
            print(f"     {step.description}")
        print()
        return 0

    # Create and run pipeline
    pipeline = Pipeline(
        config=config,
        interactive=not args.non_interactive,
        skip_steps=args.skip
    )

    return pipeline.run()


if __name__ == '__main__':
    sys.exit(main())
