"""
Basketball Tracker orchestrator.

Main pipeline for automated basketball detection:
1. Manual annotation
2. Trajectory detection (Kalman filtering)
3. Verification and correction
4. YOLO model training
5. Inference/prediction
"""

import os
import importlib
import sys
import json
import logging
from pathlib import Path

from config import setup_logging

logger = setup_logging(__name__)


class UltraBasketballTracker:
    """Orchestrator for the complete basketball tracking pipeline."""

    def __init__(self, video_path, model='yolov11x.pt', output_dir='outputs'):
        """
        Initialize the basketball tracker.

        Args:
            video_path: Path to input video file
            model: YOLO model to use (default: yolov11x)
            output_dir: Directory for outputs
        """
        self.video = video_path
        self.model = model
        self.output = Path(output_dir)
        self.output.mkdir(exist_ok=True)
        # Define file paths for each stage
        self.files = {
            'annotations': self.output / 'annotations.json',
            'detections': self.output / 'detections.json',
            'verified': self.output / 'verified.json',
            'dataset': self.output / 'yolo_dataset'
        }

    @staticmethod
    def _safe_load_json(filepath):
        """
        Safely load JSON file with error handling.

        Args:
            filepath: Path to JSON file

        Returns:
            Dictionary from JSON or empty dict if file not found
        """
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"File not found: {filepath}")
            return {}
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON in: {filepath}")
            return {}

    def _import_module(self, module_name):
        """Dynamically import a module."""
        return importlib.import_module(module_name, package=None)

    def annotate(self):
        """Run manual annotation tool. Returns self for method chaining."""
        logger.info("Starting annotation phase")
        annotator_cls = self._import_module('_1_ball_annotator').BallAnnotator
        annotator_cls(self.video, str(self.files['annotations'])).run()
        return self

    def detect(self):
        """Run trajectory detection using Kalman filtering. Returns self for method chaining."""
        logger.info("Starting trajectory detection phase")
        detector_func = self._import_module('_2_trajectory_detector').process_trajectory_video
        detector_func(self.video, str(self.files['annotations']), str(self.files['detections']))
        return self

    def verify(self):
        """Run verification/correction tool if detections exist. Returns self for method chaining."""
        logger.info("Starting verification phase")
        detections_data = self._safe_load_json(self.files['detections'])
        if detections_data:
            verifier_cls = self._import_module('_3_verification_tool').CompactBallVerifier
            verifier_cls(str(self.video), str(self.files['detections']), str(self.files['verified'])).run()
        else:
            logger.warning("No detections found, skipping verification")
        return self

    def train_yolo(self, epochs=50):
        """
        Train YOLO model on verified or annotated data. Returns self for method chaining.

        Args:
            epochs: Number of training epochs
        """
        logger.info("Starting YOLO training phase")
        # Choose verified annotations if available, otherwise use manual annotations
        ann_file = str(self.files['verified']) if os.path.exists(self.files['verified']) and os.path.getsize(self.files['verified']) > 2 \
                   else str(self.files['annotations'])
        trainer_cls = self._import_module('_4_yolo_trainer').UltraYOLOBallTrainer
        yolo_trainer = trainer_cls(str(self.video), ann_file, str(self.files['dataset']), self.model)
        # Train YOLO model (this will internally create the dataset)
        yolo_trainer.train(epochs=epochs)
        return self

    def predict(self, conf=0.3):
        """
        Run inference on video using trained model. Returns self for method chaining.

        Args:
            conf: Confidence threshold for detections
        """
        logger.info("Starting inference phase")
        trainer_cls = self._import_module('_4_yolo_trainer').UltraYOLOBallTrainer
        output_path = str(self.output / 'predicted_video.mp4')
        # Try to find the best trained weights, otherwise use the base model
        model_candidates = [
            str(self.files['dataset'] / 'weights/best.pt'),
            self.model
        ]
        model_path = next((m for m in model_candidates if os.path.exists(m)), None)
        if model_path:
            logger.info(f"Using model: {model_path}")
            trainer_cls.detect(self.video, model_path, output_path, conf)
        else:
            logger.warning("No model found for inference")
        return self

    def full_pipeline(self):
        """
        Run the complete basketball tracking pipeline.

        Executes all stages in sequence: annotate, detect, verify, train, predict.
        """
        logger.info("Starting full basketball tracking pipeline")
        return (self.annotate()
                .detect()
                .verify()
                .train_yolo()
                .predict())


def main():
    """Main entry point."""
    video_path = sys.argv[1] if len(sys.argv) > 1 else 'data/input_video.mp4'
    UltraBasketballTracker(video_path).full_pipeline()

if __name__ == '__main__':
    main()
