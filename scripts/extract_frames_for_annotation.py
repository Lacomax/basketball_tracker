#!/usr/bin/env python3
"""
Script para extraer frames de un video para anotación en Roboflow u otras herramientas.

Este script extrae frames de un video a intervalos regulares, útil para:
1. Anotar en Roboflow
2. Crear datasets personalizados
3. Revisar contenido del video

Uso:
    # Extraer un frame cada 10
    python scripts/extract_frames_for_annotation.py --video input_video.mp4 --interval 10

    # Extraer solo los primeros 100 frames
    python scripts/extract_frames_for_annotation.py --video input_video.mp4 --max-frames 100

    # Extraer en formato específico
    python scripts/extract_frames_for_annotation.py --video input_video.mp4 --format png
"""

import argparse
import os
import sys
from pathlib import Path
import cv2


def extract_frames(
    video_path,
    output_dir,
    frame_interval=10,
    max_frames=None,
    image_format='jpg',
    quality=95
):
    """
    Extrae frames de un video para anotación.

    Args:
        video_path: Path al video
        output_dir: Directorio de salida
        frame_interval: Extraer un frame cada N frames
        max_frames: Número máximo de frames a extraer (None = todos)
        image_format: Formato de imagen (jpg, png)
        quality: Calidad de compresión JPEG (1-100)

    Returns:
        Número de frames extraídos
    """
    if not os.path.exists(video_path):
        print(f"❌ Error: No se encuentra el video: {video_path}")
        return 0

    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)

    # Abrir video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: No se puede abrir el video")
        return 0

    # Obtener información del video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n📹 Video: {width}x{height} @ {fps}fps ({total_frames} frames)")
    print(f"📁 Salida: {output_dir}")
    print(f"⚙️  Intervalo: 1 frame cada {frame_interval}")

    if max_frames:
        estimated = min(total_frames // frame_interval, max_frames)
    else:
        estimated = total_frames // frame_interval

    print(f"📊 Frames estimados: ~{estimated}\n")

    frame_count = 0
    saved_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Extraer frame si cumple el intervalo
            if frame_count % frame_interval == 0:
                # Nombre del archivo
                if image_format.lower() == 'png':
                    filename = f"frame_{frame_count:06d}.png"
                    params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
                else:  # jpg
                    filename = f"frame_{frame_count:06d}.jpg"
                    params = [cv2.IMWRITE_JPEG_QUALITY, quality]

                output_path = os.path.join(output_dir, filename)

                # Guardar frame
                cv2.imwrite(output_path, frame, params)
                saved_count += 1

                # Mostrar progreso
                if saved_count % 10 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"   Progreso: {progress:.1f}% ({saved_count} frames guardados)")

                # Límite de frames
                if max_frames and saved_count >= max_frames:
                    break

            frame_count += 1

    finally:
        cap.release()

    print(f"\n✅ Extracción completa!")
    print(f"   Total frames procesados: {frame_count}")
    print(f"   Frames guardados: {saved_count}")
    print(f"   Directorio: {output_dir}")

    return saved_count


def main():
    parser = argparse.ArgumentParser(
        description='Extrae frames de un video para anotación',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:

  # Extraer un frame cada 10
  python scripts/extract_frames_for_annotation.py --video input_video.mp4 --interval 10

  # Extraer solo 100 frames
  python scripts/extract_frames_for_annotation.py --video input_video.mp4 --max-frames 100

  # Extraer en PNG (mayor calidad)
  python scripts/extract_frames_for_annotation.py --video input_video.mp4 --format png

  # Extraer con alta compresión
  python scripts/extract_frames_for_annotation.py --video input_video.mp4 --quality 70

Recomendaciones:
  - Para anotación rápida: --interval 30 (1 frame cada segundo @ 30fps)
  - Para anotación detallada: --interval 10
  - Para acción rápida: --interval 5
  - Formato PNG: mejor calidad, archivos más grandes
  - Formato JPG: menor calidad, archivos más pequeños
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
        help='Directorio de salida (default: data/frames_to_annotate/<video_name>)'
    )

    parser.add_argument(
        '--interval',
        type=int,
        default=10,
        help='Extraer un frame cada N frames (default: 10)'
    )

    parser.add_argument(
        '--max-frames',
        type=int,
        help='Número máximo de frames a extraer (default: sin límite)'
    )

    parser.add_argument(
        '--format',
        type=str,
        choices=['jpg', 'png'],
        default='jpg',
        help='Formato de imagen (default: jpg)'
    )

    parser.add_argument(
        '--quality',
        type=int,
        default=95,
        help='Calidad JPEG 1-100 (default: 95)'
    )

    args = parser.parse_args()

    # Generar directorio de salida
    if not args.output:
        video_name = Path(args.video).stem
        args.output = f"data/frames_to_annotate/{video_name}"

    print("=" * 70)
    print("Extracción de Frames para Anotación")
    print("=" * 70)

    # Extraer frames
    saved = extract_frames(
        video_path=args.video,
        output_dir=args.output,
        frame_interval=args.interval,
        max_frames=args.max_frames,
        image_format=args.format,
        quality=args.quality
    )

    if saved > 0:
        print("\n" + "=" * 70)
        print("🎉 ¡Extracción completa!")
        print("=" * 70)
        print(f"\n📁 Frames listos para anotar: {args.output}/")
        print("\n💡 Siguientes pasos:")
        print("   1. Sube los frames a Roboflow para anotación")
        print("   2. O anota manualmente con tu herramienta preferida")
        print("   3. Descarga las anotaciones en formato YOLO")
        print("   4. Entrena el modelo con los datos anotados")

        return 0
    else:
        return 1


if __name__ == '__main__':
    sys.exit(main())
