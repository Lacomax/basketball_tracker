"""
Professional basketball visualization using mplbasketball.

This module provides high-quality visualizations using matplotlib
and mplbasketball library for publication-ready graphics.
"""

import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.collections import LineCollection
    MPL_AVAILABLE = True
    Figure = plt.Figure
except ImportError:
    MPL_AVAILABLE = False
    Figure = Any  # Fallback type

try:
    from mplbasketball import BasketballPlot
    MPLBASKETBALL_AVAILABLE = True
except ImportError:
    MPLBASKETBALL_AVAILABLE = False

from ..config import setup_logging

logger = setup_logging(__name__)

if not MPL_AVAILABLE:
    logger.warning("Matplotlib not available. Install with: pip install matplotlib")

if not MPLBASKETBALL_AVAILABLE:
    logger.warning("mplbasketball not available. Install with: pip install mplbasketball")


class ProfessionalVisualizer:
    """Create professional basketball visualizations."""

    def __init__(self, court_type: str = 'nba', figsize: Tuple[int, int] = (12, 11)):
        """
        Initialize professional visualizer.

        Args:
            court_type: Type of court ('nba', 'fiba', 'ncaa', 'wnba')
            figsize: Figure size (width, height) in inches
        """
        if not MPL_AVAILABLE or not MPLBASKETBALL_AVAILABLE:
            raise ImportError("ProfessionalVisualizer requires matplotlib and mplbasketball")

        self.court_type = court_type
        self.figsize = figsize

        # Team colors
        self.team_colors = {
            'Team_0': '#FF6B6B',  # Red
            'Team_1': '#4ECDC4',  # Teal/Blue
            'Team_Red': '#FF6B6B',
            'Team_Blue': '#4ECDC4',
            'Team_White': '#FFFFFF',
            'Team_Black': '#2C3E50',
            'Unknown': '#95A5A6'
        }

        logger.info(f"ProfessionalVisualizer initialized (court={court_type})")

    def create_shot_chart(self, shot_data: List[Dict],
                         title: str = "Shot Chart",
                         output_path: Optional[str] = None) -> Figure:
        """
        Create a professional shot chart showing made and missed shots.

        Args:
            shot_data: List of shot events with 'position', 'made', 'player_id', 'team'
            title: Chart title
            output_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Draw basketball court
        court = BasketballPlot(ax=ax, court_type=self.court_type)
        court.draw()

        # Separate made and missed shots
        made_shots = []
        missed_shots = []

        for shot in shot_data:
            if 'position' in shot or 'court_position' in shot:
                pos = shot.get('court_position') or shot.get('position')
                is_made = shot.get('made', False)

                if is_made:
                    made_shots.append(pos)
                else:
                    missed_shots.append(pos)

        # Plot made shots (green)
        if made_shots:
            made_x = [p[0] for p in made_shots]
            made_y = [p[1] for p in made_shots]
            ax.scatter(made_x, made_y, c='green', s=100, alpha=0.6,
                      edgecolors='darkgreen', linewidths=2, label='Made')

        # Plot missed shots (red)
        if missed_shots:
            missed_x = [p[0] for p in missed_shots]
            missed_y = [p[1] for p in missed_shots]
            ax.scatter(missed_x, missed_y, c='red', s=100, alpha=0.6,
                      edgecolors='darkred', linewidths=2, marker='x', label='Missed')

        # Add legend and title
        ax.legend(loc='upper right', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Shot chart saved to {output_path}")

        return fig

    def create_player_movement_chart(self, trajectory: List[Tuple[float, float]],
                                    player_id: int,
                                    team: str = 'Unknown',
                                    title: Optional[str] = None,
                                    output_path: Optional[str] = None) -> Figure:
        """
        Create a chart showing player movement patterns.

        Args:
            trajectory: List of (x, y) positions
            player_id: Player ID
            team: Team name
            title: Chart title
            output_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Draw basketball court
        court = BasketballPlot(ax=ax, court_type=self.court_type)
        court.draw()

        if not trajectory:
            logger.warning("Empty trajectory")
            return fig

        # Get team color
        color = self.team_colors.get(team, '#95A5A6')

        # Plot trajectory as lines with gradient
        x_coords = [p[0] for p in trajectory]
        y_coords = [p[1] for p in trajectory]

        # Create line segments
        points = np.array([x_coords, y_coords]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create gradient colors (darker -> lighter)
        n_segments = len(segments)
        colors_gradient = plt.cm.Blues(np.linspace(0.3, 1, n_segments))

        # Plot line collection
        lc = LineCollection(segments, colors=colors_gradient, linewidths=2, alpha=0.7)
        ax.add_collection(lc)

        # Mark start and end points
        ax.plot(x_coords[0], y_coords[0], 'go', markersize=12, label='Start',
               markeredgecolor='darkgreen', markeredgewidth=2)
        ax.plot(x_coords[-1], y_coords[-1], 'ro', markersize=12, label='End',
               markeredgecolor='darkred', markeredgewidth=2)

        # Add legend and title
        ax.legend(loc='upper right', fontsize=12)

        if title is None:
            title = f"Player {player_id} Movement ({team})"

        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Movement chart saved to {output_path}")

        return fig

    def create_heatmap(self, positions: List[Tuple[float, float]],
                      title: str = "Position Heatmap",
                      output_path: Optional[str] = None) -> Figure:
        """
        Create a heatmap showing position density.

        Args:
            positions: List of (x, y) positions
            title: Chart title
            output_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Draw basketball court
        court = BasketballPlot(ax=ax, court_type=self.court_type)
        court.draw()

        if not positions:
            logger.warning("No positions to plot")
            return fig

        # Extract coordinates
        x_coords = [p[0] for p in positions]
        y_coords = [p[1] for p in positions]

        # Create 2D histogram (heatmap)
        heatmap, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=50)

        # Plot heatmap
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im = ax.imshow(heatmap.T, extent=extent, origin='lower',
                      cmap='YlOrRd', alpha=0.6, aspect='auto')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Density', rotation=270, labelpad=20, fontsize=12)

        # Add title
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Heatmap saved to {output_path}")

        return fig

    def create_team_comparison(self, team1_data: Dict, team2_data: Dict,
                              team1_name: str = "Team 1",
                              team2_name: str = "Team 2",
                              output_path: Optional[str] = None) -> Figure:
        """
        Create a comparison chart for two teams.

        Args:
            team1_data: Dictionary with metrics for team 1
            team2_data: Dictionary with metrics for team 2
            team1_name: Name of team 1
            team2_name: Name of team 2
            output_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"{team1_name} vs {team2_name}", fontsize=18, fontweight='bold')

        metrics = ['total_distance_m', 'average_speed_kmh', 'total_sprints', 'max_speed_kmh']
        titles = ['Total Distance (m)', 'Average Speed (km/h)', 'Total Sprints', 'Max Speed (km/h)']
        colors = [self.team_colors.get(team1_name, '#FF6B6B'),
                 self.team_colors.get(team2_name, '#4ECDC4')]

        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]

            values = [team1_data.get(metric, 0), team2_data.get(metric, 0)]
            bars = ax.bar([team1_name, team2_name], values, color=colors, alpha=0.7,
                         edgecolor='black', linewidth=2)

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=12, fontweight='bold')

            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_ylabel('Value', fontsize=12)
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Team comparison saved to {output_path}")

        return fig

    def create_game_timeline(self, events: List[Dict],
                           title: str = "Game Timeline",
                           output_path: Optional[str] = None) -> Figure:
        """
        Create a timeline visualization of game events.

        Args:
            events: List of event dictionaries with 'frame_number', 'event_type', 'team'
            title: Chart title
            output_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(16, 6))

        if not events:
            logger.warning("No events to plot")
            return fig

        # Group events by type
        event_types = defaultdict(list)

        for event in events:
            event_type = event.get('event_type', 'unknown')
            frame = event.get('frame_number', 0)
            team = event.get('team', 'Unknown')

            event_types[event_type].append((frame, team))

        # Plot events
        y_pos = 0
        type_colors = {
            'made_basket': 'green',
            'missed_basket': 'red',
            'shot': 'orange',
            'pass': 'blue',
            'dribble': 'purple',
            'turnover': 'darkred'
        }

        for event_type, occurrences in event_types.items():
            frames = [o[0] for o in occurrences]
            teams = [o[1] for o in occurrences]

            color = type_colors.get(event_type, 'gray')

            ax.scatter(frames, [y_pos] * len(frames), c=color, s=100,
                      alpha=0.7, label=event_type, edgecolors='black', linewidths=1)

            y_pos += 1

        # Configure axes
        ax.set_xlabel('Frame Number', fontsize=14, fontweight='bold')
        ax.set_ylabel('Event Type', fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_yticks(range(len(event_types)))
        ax.set_yticklabels(list(event_types.keys()))
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Timeline saved to {output_path}")

        return fig


def main():
    """Example usage of professional visualizer."""
    import argparse
    parser = argparse.ArgumentParser(description='Professional basketball visualization')
    parser.add_argument('--mode', required=True,
                       choices=['shot_chart', 'movement', 'heatmap', 'timeline'],
                       help='Visualization mode')
    parser.add_argument('--data', required=True, help='Path to data JSON')
    parser.add_argument('--output', required=True, help='Output image path')
    parser.add_argument('--court', default='nba',
                       choices=['nba', 'fiba', 'ncaa', 'wnba'],
                       help='Court type')
    args = parser.parse_args()

    # Load data
    with open(args.data, 'r') as f:
        data = json.load(f)

    # Initialize visualizer
    viz = ProfessionalVisualizer(court_type=args.court)

    # Create visualization based on mode
    if args.mode == 'shot_chart':
        viz.create_shot_chart(data, output_path=args.output)
    elif args.mode == 'heatmap':
        # Extract positions from data
        positions = [d.get('position') or d.get('court_position') for d in data if 'position' in d or 'court_position' in d]
        viz.create_heatmap(positions, output_path=args.output)
    elif args.mode == 'timeline':
        viz.create_game_timeline(data, output_path=args.output)

    print(f"Visualization saved to {args.output}")


if __name__ == '__main__':
    main()
