#!/usr/bin/env python3
"""
Script para verificar detecciones YOLO sin filtros.

Muestra TODAS las detecciones del modelo YOLO entrenado, sin aplicar filtros de distancia
o validación. Esto permite verificar visualmente si el modelo está detectando correctamente.

Uso:
    python scripts/verify_yolo_detections.py
    python scripts/verify_yolo_detections.py --conf 0.15  # Cambiar umbral de confianza
    python scripts/verify_yolo_detections.py --save-video  # Guardar video con detecciones
"""

import cv2
import json
import argparse
from pathlib import Path
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import setup_logging
from src.utils.video_utils import open_video_robust, create_video_writer_robust
from src.utils.ball_detection import get_yolo_model

logger = setup_logging(__name__)


def draw_detection(frame, x1, y1, x2, y2, conf, class_id, class_name, color=(0, 255, 0)):
    """Draw bounding box and label on frame."""
    # Draw bounding box
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    # Calculate center
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    # Draw center point
    cv2.circle(frame, (cx, cy), 5, color, -1)

    # Draw label with confidence
    label = f"{class_name} {conf:.2f}"
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

    # Background for label
    cv2.rectangle(frame,
                  (int(x1), int(y1) - label_size[1] - 10),
                  (int(x1) + label_size[0], int(y1)),
                  color, -1)

    # Label text
    cv2.putText(frame, label, (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    return cx, cy


def verify_yolo_detections(video_path, output_path=None, conf_threshold=0.15,
                          save_video=False, show_window=True, min_size=15, max_size=60):
    """
    Procesa el video mostrando TODAS las detecciones YOLO sin filtros.

    Args:
        video_path: Ruta al video de entrada
        output_path: Ruta para guardar detecciones JSON (opcional)
        conf_threshold: Umbral de confianza mínimo
        save_video: Si True, guarda video con detecciones visualizadas
        show_window: Si True, muestra ventana con preview (solo primeros 50 frames)
        min_size: Tamaño mínimo del balón en píxeles (default: 15)
        max_size: Tamaño máximo del balón en píxeles (default: 60)
    """
    logger.info(f"📹 Verificando detecciones YOLO en: {video_path}")
    logger.info(f"   Umbral de confianza: {conf_threshold}")
    logger.info(f"   Rango de tamaño permitido: {min_size}-{max_size} píxeles")

    # Load YOLO model
    model = get_yolo_model()
    if model is None:
        logger.error("[X] No se pudo cargar el modelo YOLO")
        return

    # Open video
    cap = open_video_robust(video_path)
    if cap is None:
        logger.error(f"[X] No se pudo abrir el video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(f"   Total frames: {total_frames}, FPS: {fps}, Resolución: {width}x{height}")

    # Video writer for saving annotated video
    writer = None
    if save_video:
        output_video_path = 'outputs/yolo_detections_raw.mp4'
        writer = create_video_writer_robust(output_video_path, fps, width, height)
        if writer:
            logger.info(f"   💾 Guardando video anotado en: {output_video_path}")

    # Statistics
    all_detections = {}
    frame_stats = {
        'total_frames': 0,
        'frames_with_detections': 0,
        'total_detections': 0,
        'basketball_detections': 0,
        'other_detections': 0,
        'high_conf_detections': 0  # conf > 0.5
    }

    frame_num = 0

    logger.info("[D] Procesando frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        frame_stats['total_frames'] += 1

        # Run YOLO detection (class 0 = basketball, custom trained model)
        results = model(frame, classes=[0], verbose=False, conf=conf_threshold)

        # Also try detecting all classes to see what else is being detected
        results_all = model(frame, verbose=False, conf=conf_threshold)

        detections_this_frame = []

        # Process basketball detections (class 0)
        if len(results) > 0 and len(results[0].boxes) > 0:
            has_valid_detection = False

            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                class_id = int(box.cls[0])

                # Calculate size
                w = x2 - x1
                h = y2 - y1
                size = max(w, h)

                frame_stats['total_detections'] += 1
                frame_stats['basketball_detections'] += 1

                # Filter by size (basketball should be 15-60 pixels, not a huge window!)
                if size < min_size or size > max_size:
                    # Draw red box for rejected (wrong size)
                    cx, cy = draw_detection(frame, x1, y1, x2, y2, conf, class_id,
                                           f"REJECTED (size={size:.0f}px)", color=(0, 0, 255))
                    continue

                has_valid_detection = True

                if conf > 0.5:
                    frame_stats['high_conf_detections'] += 1

                # Draw green box for basketball
                cx, cy = draw_detection(frame, x1, y1, x2, y2, conf, class_id,
                                       f"basketball ({size:.0f}px)", color=(0, 255, 0))

                detections_this_frame.append({
                    'class': 'basketball',
                    'class_id': class_id,
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'center': [int(cx), int(cy)],
                    'confidence': float(conf),
                    'width': float(w),
                    'height': float(h),
                    'size': float(size)
                })

            if has_valid_detection:
                frame_stats['frames_with_detections'] += 1

        # Check for other detections (all classes)
        if len(results_all) > 0 and len(results_all[0].boxes) > 0:
            for box in results_all[0].boxes:
                class_id = int(box.cls[0])

                # Skip basketball (class 0), we already processed it
                if class_id == 0:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])

                frame_stats['other_detections'] += 1

                # Draw yellow box for other objects
                class_name = f"class_{class_id}"
                cx, cy = draw_detection(frame, x1, y1, x2, y2, conf, class_id,
                                       class_name, color=(0, 255, 255))

                detections_this_frame.append({
                    'class': class_name,
                    'class_id': class_id,
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'center': [int(cx), int(cy)],
                    'confidence': float(conf),
                    'width': float(x2 - x1),
                    'height': float(y2 - y1)
                })

        # Store detections for this frame
        if detections_this_frame:
            all_detections[frame_num] = detections_this_frame

        # Add frame counter
        info_text = f"Frame: {frame_num}/{total_frames} | Detections: {len(detections_this_frame)}"
        cv2.putText(frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Add legend
        cv2.putText(frame, "Verde = Basketball (class 0)", (10, height - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, "Amarillo = Otros objetos", (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Save to video
        if writer:
            writer.write(frame)

        # Show window (only first 50 frames to avoid blocking)
        if show_window and frame_num <= 50:
            cv2.imshow('YOLO Detections (Raw)', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("⏹ Usuario canceló la visualización")
                break

        # Progress update every 50 frames
        if frame_num % 50 == 0:
            logger.info(f"   Procesado {frame_num}/{total_frames} frames...")

    cap.release()
    if writer:
        writer.release()
    if show_window:
        cv2.destroyAllWindows()

    # Save detections to JSON
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(all_detections, f, indent=2)
        logger.info(f"💾 Detecciones guardadas en: {output_path}")

    # Print statistics
    logger.info("\n" + "="*70)
    logger.info("📊 ESTADÍSTICAS DE DETECCIÓN YOLO")
    logger.info("="*70)
    logger.info(f"Total de frames procesados: {frame_stats['total_frames']}")
    logger.info(f"Frames con detecciones: {frame_stats['frames_with_detections']} "
               f"({frame_stats['frames_with_detections']/frame_stats['total_frames']*100:.1f}%)")
    logger.info(f"\nDetecciones totales: {frame_stats['total_detections']}")
    logger.info(f"  - Basketball (class 0): {frame_stats['basketball_detections']}")
    logger.info(f"  - Otros objetos: {frame_stats['other_detections']}")
    logger.info(f"  - Alta confianza (>0.5): {frame_stats['high_conf_detections']} "
               f"({frame_stats['high_conf_detections']/max(1,frame_stats['total_detections'])*100:.1f}%)")

    # Analyze detection distribution
    if all_detections:
        detections_per_frame = [len(dets) for dets in all_detections.values()]
        logger.info(f"\nDetecciones por frame:")
        logger.info(f"  - Promedio: {np.mean(detections_per_frame):.2f}")
        logger.info(f"  - Máximo: {np.max(detections_per_frame)}")
        logger.info(f"  - Mínimo: {np.min(detections_per_frame)}")

        # Find frames with multiple detections
        multi_detection_frames = [f for f, dets in all_detections.items() if len(dets) > 1]
        if multi_detection_frames:
            logger.info(f"\n[!]️ Frames con múltiples detecciones: {len(multi_detection_frames)}")
            logger.info(f"   Ejemplos: {multi_detection_frames[:10]}")

    logger.info("="*70)

    # Recommendations
    logger.info("\n💡 RECOMENDACIONES:")

    if frame_stats['basketball_detections'] == 0:
        logger.warning("[X] NO se detectó ningún basketball!")
        logger.warning("   → Verifica que el modelo esté entrenado correctamente")
        logger.warning("   → Verifica que la ruta del modelo sea correcta")
    elif frame_stats['basketball_detections'] < frame_stats['total_frames'] * 0.3:
        logger.warning(f"[!]️ Solo {frame_stats['basketball_detections']} detecciones de basketball "
                      f"en {frame_stats['total_frames']} frames")
        logger.warning("   → El modelo podría necesitar más entrenamiento")
        logger.warning("   → Considera agregar más datos de entrenamiento")
    else:
        logger.info(f"[+] Buen ratio de detección: {frame_stats['basketball_detections']} detecciones")

    if frame_stats['high_conf_detections'] < frame_stats['total_detections'] * 0.5:
        logger.warning(f"[!]️ Solo {frame_stats['high_conf_detections']/max(1,frame_stats['total_detections'])*100:.1f}% "
                      "de las detecciones tienen alta confianza (>0.5)")
        logger.warning("   → Considera aumentar el umbral de confianza")
        logger.warning("   → El modelo podría necesitar más entrenamiento")

    if len(multi_detection_frames) > frame_stats['frames_with_detections'] * 0.3:
        logger.warning(f"[!]️ Muchos frames con múltiples detecciones: {len(multi_detection_frames)}")
        logger.warning("   → Podrían ser falsos positivos")
        logger.warning("   → Considera usar Non-Maximum Suppression (NMS) más agresivo")

    return all_detections, frame_stats


def main():
    parser = argparse.ArgumentParser(
        description='Verificar detecciones YOLO sin filtros de distancia'
    )
    parser.add_argument('--video', type=str,
                       default='data/input_video.mp4',
                       help='Ruta al video de entrada')
    parser.add_argument('--output', type=str,
                       default='outputs/yolo_detections_raw.json',
                       help='Ruta para guardar detecciones JSON')
    parser.add_argument('--conf', type=float, default=0.15,
                       help='Umbral de confianza mínimo (default: 0.15)')
    parser.add_argument('--min-size', type=int, default=15,
                       help='Tamaño mínimo del balón en píxeles (default: 15)')
    parser.add_argument('--max-size', type=int, default=60,
                       help='Tamaño máximo del balón en píxeles (default: 60)')
    parser.add_argument('--save-video', action='store_true',
                       help='Guardar video con detecciones visualizadas')
    parser.add_argument('--no-window', action='store_true',
                       help='No mostrar ventana de preview')

    args = parser.parse_args()

    # Check if video exists
    video_path = Path(args.video)
    if not video_path.exists():
        # Try data/ folder
        video_path = Path('data') / video_path.name
        if not video_path.exists():
            logger.error(f"[X] Video no encontrado: {args.video}")
            return

    # Create outputs directory
    Path('outputs').mkdir(exist_ok=True)

    # Run verification
    verify_yolo_detections(
        str(video_path),
        output_path=args.output,
        conf_threshold=args.conf,
        save_video=args.save_video,
        show_window=not args.no_window,
        min_size=args.min_size,
        max_size=args.max_size
    )


if __name__ == '__main__':
    main()
