"""
Zero-shot team classification using Fashion CLIP.

This module uses Fashion CLIP (CLIP fine-tuned on fashion/clothing)
to classify players into teams based on jersey colors without training.
"""

import cv2
import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from PIL import Image

try:
    from transformers import CLIPProcessor, CLIPModel
    import torch
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logging.warning("Fashion CLIP not available. Install with: pip install transformers torch")

from ..config import setup_logging

logger = setup_logging(__name__)


@dataclass
class TeamClassification:
    """Data class for team classification result."""
    player_id: int
    team: str
    confidence: float
    color_description: str
    bbox: List[int]


class FashionCLIPTeamClassifier:
    """Zero-shot team classification using Fashion CLIP."""

    def __init__(self, model_name: str = "patrickjohncyh/fashion-clip",
                 team_prompts: Optional[List[str]] = None,
                 confidence_threshold: float = 0.5):
        """
        Initialize Fashion CLIP team classifier.

        Args:
            model_name: HuggingFace model name
            team_prompts: List of text prompts for teams (e.g., ["red jersey", "blue jersey"])
            confidence_threshold: Minimum confidence for classification
        """
        self.confidence_threshold = confidence_threshold

        if not CLIP_AVAILABLE:
            raise ImportError("Fashion CLIP requires transformers and torch. Install with: pip install transformers torch")

        # Load Fashion CLIP model
        logger.info(f"Loading Fashion CLIP model: {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # Default team prompts (can be customized)
        if team_prompts is None:
            self.team_prompts = [
                "basketball player wearing red jersey",
                "basketball player wearing blue jersey",
                "basketball player wearing white jersey",
                "basketball player wearing black jersey",
                "basketball player wearing yellow jersey",
                "basketball player wearing green jersey"
            ]
        else:
            self.team_prompts = team_prompts

        # Cache for team assignments (player_id -> team)
        self.team_assignments = {}

        logger.info(f"Fashion CLIP initialized with {len(self.team_prompts)} team prompts")

    def classify_player(self, frame: np.ndarray, bbox: List[int],
                       player_id: int) -> TeamClassification:
        """
        Classify a single player's team using Fashion CLIP.

        Args:
            frame: Full frame image (BGR format from OpenCV)
            bbox: Player bounding box [x1, y1, x2, y2]
            player_id: Player ID for tracking

        Returns:
            TeamClassification object
        """
        x1, y1, x2, y2 = bbox

        # Crop player region
        player_crop = frame[y1:y2, x1:x2]

        if player_crop.size == 0:
            return TeamClassification(
                player_id=player_id,
                team="Unknown",
                confidence=0.0,
                color_description="Invalid crop",
                bbox=bbox
            )

        # Focus on upper body (jersey area)
        h, w = player_crop.shape[:2]
        jersey_crop = player_crop[:int(h * 0.5), :]  # Upper 50%

        if jersey_crop.size == 0:
            jersey_crop = player_crop

        # Convert BGR to RGB for CLIP
        jersey_rgb = cv2.cvtColor(jersey_crop, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(jersey_rgb)

        # Prepare inputs for CLIP
        inputs = self.processor(
            text=self.team_prompts,
            images=pil_image,
            return_tensors="pt",
            padding=True
        )

        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image  # Image-text similarity
            probs = logits_per_image.softmax(dim=1)  # Convert to probabilities

        # Get best matching team
        probs_cpu = probs.cpu().numpy()[0]
        best_idx = int(np.argmax(probs_cpu))
        confidence = float(probs_cpu[best_idx])

        # Extract team name from prompt
        team_description = self.team_prompts[best_idx]
        team_name = self._extract_team_name(team_description, best_idx)

        # Cache assignment if confidence is high
        if confidence >= self.confidence_threshold:
            self.team_assignments[player_id] = team_name

        return TeamClassification(
            player_id=player_id,
            team=team_name,
            confidence=confidence,
            color_description=team_description,
            bbox=bbox
        )

    def _extract_team_name(self, prompt: str, idx: int) -> str:
        """
        Extract team name from prompt.

        Args:
            prompt: Text prompt like "basketball player wearing red jersey"
            idx: Prompt index

        Returns:
            Team name like "Team_Red"
        """
        # Extract color from prompt
        colors = ["red", "blue", "white", "black", "yellow", "green", "orange", "purple"]

        for color in colors:
            if color in prompt.lower():
                return f"Team_{color.capitalize()}"

        # Fallback to index
        return f"Team_{idx}"

    def classify_players_in_frame(self, frame: np.ndarray,
                                  player_detections: List[Dict]) -> List[TeamClassification]:
        """
        Classify all players in a frame.

        Args:
            frame: Full frame image
            player_detections: List of player detections with 'bbox' and 'player_id'

        Returns:
            List of TeamClassification objects
        """
        classifications = []

        for detection in player_detections:
            bbox = detection.get('bbox')
            player_id = detection.get('player_id') or detection.get('track_id')

            if bbox is None or player_id is None:
                continue

            # Check cache first
            if player_id in self.team_assignments:
                # Use cached team
                classifications.append(TeamClassification(
                    player_id=player_id,
                    team=self.team_assignments[player_id],
                    confidence=1.0,  # High confidence from cache
                    color_description=f"Cached: {self.team_assignments[player_id]}",
                    bbox=bbox
                ))
            else:
                # Classify with Fashion CLIP
                classification = self.classify_player(frame, bbox, player_id)
                classifications.append(classification)

        return classifications

    def process_video(self, video_path: str, player_detections_path: str,
                     output_path: str, sample_rate: int = 30) -> Dict:
        """
        Process entire video with Fashion CLIP team classification.

        Args:
            video_path: Path to input video
            player_detections_path: Path to player detections JSON
            output_path: Path to save team classifications
            sample_rate: Only classify every N frames (for speed)

        Returns:
            Dictionary with team classifications
        """
        # Load player detections
        with open(player_detections_path, 'r') as f:
            detections = json.load(f)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        classifications_all = {}
        frame_idx = 0

        logger.info(f"Processing video with Fashion CLIP: {video_path}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Only classify every N frames for speed
            if frame_idx % sample_rate == 0:
                frame_detections = detections.get(str(frame_idx), [])

                if frame_detections:
                    classifications = self.classify_players_in_frame(frame, frame_detections)
                    classifications_all[frame_idx] = [asdict(c) for c in classifications]

                    # Log progress
                    if frame_idx % (sample_rate * 10) == 0:
                        logger.info(f"Processed frame {frame_idx}")

            frame_idx += 1

        cap.release()

        # Propagate team assignments to all frames
        classifications_all = self._propagate_team_assignments(detections, classifications_all)

        # Save results
        with open(output_path, 'w') as f:
            json.dump(classifications_all, f, indent=2)

        logger.info(f"Team classifications saved to {output_path}")
        logger.info(f"Total unique teams: {len(set(self.team_assignments.values()))}")

        return classifications_all

    def _propagate_team_assignments(self, all_detections: Dict,
                                   sampled_classifications: Dict) -> Dict:
        """
        Propagate team assignments from sampled frames to all frames.

        Args:
            all_detections: All player detections
            sampled_classifications: Classifications from sampled frames

        Returns:
            Complete classifications for all frames
        """
        complete_classifications = {}

        for frame_idx_str, frame_detections in all_detections.items():
            frame_idx = int(frame_idx_str)
            frame_classifications = []

            for detection in frame_detections:
                player_id = detection.get('player_id') or detection.get('track_id')
                bbox = detection.get('bbox')

                if player_id is None or bbox is None:
                    continue

                # Use cached team assignment
                team = self.team_assignments.get(player_id, "Unknown")

                frame_classifications.append({
                    'player_id': player_id,
                    'team': team,
                    'confidence': 1.0 if team != "Unknown" else 0.0,
                    'color_description': f"Cached: {team}",
                    'bbox': bbox
                })

            complete_classifications[frame_idx] = frame_classifications

        return complete_classifications

    def get_team_statistics(self) -> Dict:
        """Get team assignment statistics."""
        team_counts = {}
        for team in self.team_assignments.values():
            team_counts[team] = team_counts.get(team, 0) + 1

        return {
            'total_players': len(self.team_assignments),
            'team_counts': team_counts,
            'unique_teams': len(team_counts)
        }


def main():
    """Example usage of Fashion CLIP team classifier."""
    import argparse
    parser = argparse.ArgumentParser(description='Zero-shot team classification with Fashion CLIP')
    parser.add_argument('--video', required=True, help='Path to input video')
    parser.add_argument('--detections', required=True, help='Path to player detections JSON')
    parser.add_argument('--output', default='outputs/team_classifications.json',
                       help='Output JSON file')
    parser.add_argument('--sample-rate', type=int, default=30,
                       help='Sample every N frames (default: 30)')
    parser.add_argument('--prompts', nargs='+',
                       help='Custom team prompts (e.g., "red jersey" "blue jersey")')
    args = parser.parse_args()

    # Initialize classifier
    team_prompts = None
    if args.prompts:
        team_prompts = [f"basketball player wearing {prompt}" for prompt in args.prompts]

    classifier = FashionCLIPTeamClassifier(team_prompts=team_prompts)

    # Process video
    classifier.process_video(
        video_path=args.video,
        player_detections_path=args.detections,
        output_path=args.output,
        sample_rate=args.sample_rate
    )

    # Print statistics
    stats = classifier.get_team_statistics()
    print("\nTeam Classification Statistics:")
    print(f"Total players: {stats['total_players']}")
    print(f"Unique teams: {stats['unique_teams']}")
    print(f"Team counts: {stats['team_counts']}")


if __name__ == '__main__':
    main()
