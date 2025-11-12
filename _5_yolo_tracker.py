import os
import argparse
import torch

# Monkey-patch: forzar torch.load a usar weights_only=False
old_torch_load = torch.load
def new_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return old_torch_load(*args, **kwargs)
torch.load = new_torch_load

from ultralytics import YOLO

def detect_video(video_path, model_path, output_path=None, conf=0.3):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")
    # Cargar el modelo desde el checkpoint (best.pt)
    model = YOLO(model_path)
    # Ejecutar detección en el video
    results = model.predict(source=video_path, conf=conf, save=output_path is not None)
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Detección de balones de baloncesto usando YOLO (best.pt)"
    )
    parser.add_argument("--video", required=True, help="Ruta al video a analizar")
    parser.add_argument(
        "--model",
        default="outputs/yolo_dataset/weights/best.pt",
        help="Ruta al checkpoint best.pt",
    )
    parser.add_argument(
        "--output",
        default="outputs/detected_basketball.mp4",
        help="Archivo de video de salida",
    )
    parser.add_argument(
        "--conf", type=float, default=0.6, help="Confianza mínima para detecciones"
    )
    args = parser.parse_args()

    detect_video(args.video, args.model, output_path=args.output, conf=args.conf)

if __name__ == "__main__":
    main()
