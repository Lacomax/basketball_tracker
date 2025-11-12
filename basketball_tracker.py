import os, importlib, sys, json
from pathlib import Path

class UltraBasketballTracker:
    def __init__(self, video_path, model='yolov11x.pt', output_dir='outputs'):
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

    def _safe_load_json(self, filepath):
        try:
            return json.load(open(filepath))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _import_module(self, module_name):
        return importlib.import_module(module_name, package=None)

    def annotate(self):
        # Run manual annotation tool
        annotator_cls = self._import_module('_1_ball_annotator').BallAnnotator
        annotator_cls(self.video, str(self.files['annotations'])).run()
        return self

    def detect(self):
        # Run trajectory detection to produce initial detections JSON
        detector_func = self._import_module('_2_trajectory_detector').process_trajectory_video
        detector_func(self.video, str(self.files['annotations']), str(self.files['detections']))
        return self

    def verify(self):
        # Run verification/correction tool if detections exist
        detections_data = self._safe_load_json(self.files['detections'])
        if detections_data:
            verifier_cls = self._import_module('_3_verification_tool').CompactBallVerifier
            verifier_cls(str(self.video), str(self.files['detections']), str(self.files['verified'])).run()
        return self

    def train_yolo(self, epochs=50):
        # Choose verified annotations if available, otherwise use manual annotations
        ann_file = str(self.files['verified']) if os.path.exists(self.files['verified']) and os.path.getsize(self.files['verified']) > 2 \
                   else str(self.files['annotations'])
        trainer_cls = self._import_module('_4_yolo_trainer').UltraYOLOBallTrainer
        yolo_trainer = trainer_cls(str(self.video), ann_file, str(self.files['dataset']), self.model)
        # Train YOLO model (this will internally create the dataset)
        yolo_trainer.train(epochs=epochs)
        return self

    def predict(self, conf=0.3):
        # Use the trained model to predict on the video (if available) and save output
        trainer_cls = self._import_module('_4_yolo_trainer').UltraYOLOBallTrainer
        output_path = str(self.output / 'predicted_video.mp4')
        # Try to find the best trained weights, otherwise use the base model
        model_candidates = [
            str(self.files['dataset'] / 'weights/best.pt'),
            self.model
        ]
        model_path = next((m for m in model_candidates if os.path.exists(m)), None)
        if model_path:
            trainer_cls.detect(self.video, model_path, output_path, conf)
        return self

    def full_pipeline(self):
        return (self.annotate()
                    .detect()
                    .verify()
                    .train_yolo()
                    .predict())

def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else 'data/input_video.mp4'
    UltraBasketballTracker(video_path).full_pipeline()

if __name__ == '__main__':
    main()
