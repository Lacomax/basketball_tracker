#!/usr/bin/env python3
"""
Script para usar modelos pre-entrenados de YOLO sin necesidad de entrenar.

Este script permite probar rápidamente la detección de baloncesto usando:
1. Modelos YOLO pre-entrenados (YOLOv8/v11 con COCO dataset)
2. Modelos específicos de baloncesto descargables desde Roboflow/Ultralytics

Uso:
    # Usar YOLO pre-entrenado (detección general de "sports ball")
    python scripts/use_pretrained_model.py --video input_video.mp4

    # Usar modelo específico de baloncesto (si lo tienes)
    python scripts/use_pretrained_model.py --video input_video.mp4 --model models/basketball_detector.pt

    # Descargar y usar modelo de ejemplo desde Roboflow
    python scripts/use_pretrained_model.py --video input_video.mp4 --download-model
"""

import argparse
import os
import sys
from pathlib import Path
import cv2

try:
    from ultralytics import YOLO
except ImportError:
    print("[X] Error: ultralytics no está instalado")
    print("📦 Instala con: pip install ultralytics")
    sys.exit(1)


def download_pretrained_basketball_model():
    """
    Descarga un modelo pre-entrenado de baloncesto desde Roboflow.

    Returns:
        Path al modelo descargado o None si falla
    """
    print("\n📥 Descargando modelo pre-entrenado de baloncesto...")

    try:
        from roboflow import Roboflow

        # Usar modelo público de ejemplo
        # Nota: Este es un modelo de ejemplo, para mejor precisión entrena tu propio modelo
        api_key = "demo"  # API key pública de demo
        rf = Roboflow(api_key=api_key)

        # Descargar modelo de ejemplo
        workspace = "roboflow-100"
        project = "basketball-detection"
        version = 1

        project_obj = rf.workspace(workspace).project(project)
        version_obj = project_obj.version(version)

        model_path = "models/pretrained"
        os.makedirs(model_path, exist_ok=True)

        dataset = version_obj.download(
            model_format="yolov8",
            location=model_path
        )

        # Buscar el archivo de pesos
        weights_path = os.path.join(model_path, "weights", "best.pt")
        if os.path.exists(weights_path):
            print(f"[+] Modelo descargado: {weights_path}")
            return weights_path
        else:
            print("[!]️  No se encontraron pesos pre-entrenados")
            return None

    except Exception as e:
        print(f"[X] Error descargando modelo: {str(e)}")
        print("\n💡 Usa un modelo YOLO pre-entrenado genérico en su lugar")
        return None


def detect_basketball_generic(video_path, output_path, model_name="yolo11l.pt", conf_threshold=0.3):
    """
    Detecta balones de baloncesto usando modelo YOLO pre-entrenado genérico.
    Usa la clase "sports ball" del dataset COCO.

    Args:
        video_path: Path al video de entrada
        output_path: Path al video de salida
        model_name: Nombre del modelo YOLO a usar
        conf_threshold: Umbral de confianza
    """
    print(f"\n🎾 Usando modelo YOLO pre-entrenado: {model_name}")
    print("   Detectando clase 'sports ball' del dataset COCO")

    # Cargar modelo
    model = YOLO(model_name)

    # Verificar si el archivo existe
    if not os.path.exists(video_path):
        print(f"[X] Error: No se encuentra el video: {video_path}")
        return False

    # Abrir video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[X] Error: No se puede abrir el video")
        return False

    # Obtener propiedades del video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n📹 Video: {width}x{height} @ {fps}fps ({total_frames} frames)")

    # Crear directorio de salida
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # Crear video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    detections_count = 0

    print(f"\n🚀 Procesando video...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Realizar detección
            # Clase 32 = "sports ball" en COCO dataset
            results = model.predict(
                frame,
                conf=conf_threshold,
                classes=[32],  # Solo sports ball
                verbose=False
            )

            # Dibujar detecciones
            annotated_frame = frame.copy()

            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes

                for box in boxes:
                    # Obtener coordenadas
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()

                    # Dibujar rectángulo
                    cv2.rectangle(
                        annotated_frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 255, 0),
                        2
                    )

                    # Dibujar centro
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 0, 255), -1)

                    # Etiqueta
                    label = f"Basketball {conf:.2f}"
                    cv2.putText(
                        annotated_frame,
                        label,
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )

                    detections_count += 1

            # Agregar contador de frames
            cv2.putText(
                annotated_frame,
                f"Frame: {frame_count}/{total_frames}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

            # Escribir frame
            out.write(annotated_frame)

            # Mostrar progreso
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"   Progreso: {progress:.1f}% ({frame_count}/{total_frames} frames)")

    finally:
        cap.release()
        out.release()

    print(f"\n✅ Procesamiento completo!")
    print(f"   Frames procesados: {frame_count}")
    print(f"   Detecciones totales: {detections_count}")
    print(f"   Video guardado: {output_path}")

    return True


def detect_basketball_custom(video_path, output_path, model_path, conf_threshold=0.5):
    """
    Detecta balones usando modelo personalizado.

    Args:
        video_path: Path al video de entrada
        output_path: Path al video de salida
        model_path: Path al modelo personalizado
        conf_threshold: Umbral de confianza
    """
    print(f"\n🏀 Usando modelo personalizado: {model_path}")

    if not os.path.exists(model_path):
        print(f"[X] Error: No se encuentra el modelo: {model_path}")
        return False

    # Cargar modelo
    model = YOLO(model_path)

    # Procesar video
    results = model.predict(
        source=video_path,
        conf=conf_threshold,
        save=True,
        project=os.path.dirname(output_path),
        name=os.path.basename(output_path).replace('.mp4', ''),
        verbose=True
    )

    print(f"\n✅ Detección completa!")
    print(f"   Video guardado: {output_path}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Usa modelos pre-entrenados para detectar baloncesto',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:

  # Usar YOLO pre-entrenado (detección general)
  python scripts/use_pretrained_model.py --video input_video.mp4

  # Usar modelo personalizado
  python scripts/use_pretrained_model.py --video input_video.mp4 --model models/basketball_detector.pt

  # Cambiar umbral de confianza
  python scripts/use_pretrained_model.py --video input_video.mp4 --conf 0.5

  # Usar modelo más grande (mejor precisión, más lento)
  python scripts/use_pretrained_model.py --video input_video.mp4 --yolo-model yolo11x.pt

Notas:
  - Sin --model usa YOLO pre-entrenado genérico (clase "sports ball")
  - Para mejor precisión, entrena tu propio modelo con tus videos
  - Modelos disponibles: yolo11n, yolo11s, yolo11m, yolo11l, yolo11x
    (n=nano, s=small, m=medium, l=large, x=extra large)
        """
    )

    parser.add_argument(
        '--video',
        type=str,
        required=True,
        help='Path al video de entrada'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Path al video de salida (default: outputs/detected_<input>.mp4)'
    )

    parser.add_argument(
        '--model',
        type=str,
        help='Path al modelo personalizado (.pt file)'
    )

    parser.add_argument(
        '--yolo-model',
        type=str,
        default='yolo11l.pt',
        help='Modelo YOLO a usar si no se especifica --model (default: yolo11l.pt)'
    )

    parser.add_argument(
        '--conf',
        type=float,
        default=0.3,
        help='Umbral de confianza para detecciones (default: 0.3)'
    )

    parser.add_argument(
        '--download-model',
        action='store_true',
        help='Descargar modelo pre-entrenado de baloncesto desde Roboflow'
    )

    args = parser.parse_args()

    # Verificar que el video existe
    if not os.path.exists(args.video):
        print(f"[X] Error: No se encuentra el video: {args.video}")
        return 1

    # Generar path de salida
    if not args.output:
        video_name = Path(args.video).stem
        args.output = f"outputs/detected_{video_name}.mp4"

    print("=" * 70)
    print("Detección de Baloncesto con Modelo Pre-entrenado")
    print("=" * 70)

    # Descargar modelo si se solicita
    if args.download_model:
        downloaded_model = download_pretrained_basketball_model()
        if downloaded_model:
            args.model = downloaded_model

    # Usar modelo personalizado o genérico
    if args.model:
        success = detect_basketball_custom(
            args.video,
            args.output,
            args.model,
            args.conf
        )
    else:
        print("\n💡 Usando modelo YOLO pre-entrenado genérico")
        print("   Para mejor precisión, entrena un modelo personalizado:")
        print("   python scripts/train_basketball_detector_simple.py")

        success = detect_basketball_generic(
            args.video,
            args.output,
            args.yolo_model,
            args.conf
        )

    if success:
        print("\n" + "=" * 70)
        print("🎉 ¡Detección completa!")
        print("=" * 70)
        print(f"\n📹 Video procesado: {args.output}")

        # Sugerencias
        print("\n💡 Sugerencias:")
        if not args.model:
            print("   • Entrena un modelo personalizado para mejor precisión")
            print("     python scripts/train_basketball_detector_simple.py")
        print("   • Ajusta --conf si hay muchas/pocas detecciones")
        print("   • Usa --yolo-model yolo11x.pt para mejor precisión (más lento)")

        return 0
    else:
        return 1


if __name__ == '__main__':
    sys.exit(main())
