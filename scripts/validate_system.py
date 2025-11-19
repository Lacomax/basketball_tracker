#!/usr/bin/env python3
"""
Validate basketball tracker system setup.
Checks all components and outputs status.
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config_loader import get_config

def check_file(path, description):
    """Check if file exists and return status."""
    exists = os.path.exists(path)
    status = "[OK]" if exists else "[X]"
    size = ""
    if exists and os.path.isfile(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        size = f" ({size_mb:.1f} MB)"
    print(f"  {status} {description}: {path}{size}")
    return exists

def main():
    print("=" * 70)
    print("BASKETBALL TRACKER - VALIDACION DEL SISTEMA")
    print("=" * 70)
    print()

    config = get_config()
    output_dir = config.get_output_dir()
    all_ok = True

    # Check video input
    print("1. VIDEO DE ENTRADA")
    print("-" * 70)
    video_found = False
    for path in config.get_video_paths():
        if os.path.exists(path):
            check_file(path, "Video encontrado")
            video_found = True
            break
    if not video_found:
        print("  [X] Video no encontrado en:")
        for path in config.get_video_paths():
            print(f"      - {path}")
        all_ok = False
    print()

    # Check model
    print("2. MODELO YOLO CUSTOM")
    print("-" * 70)
    model_path = config.get('ball.model_path', 'models/basketball_detector_custom.pt')
    if not check_file(model_path, "Modelo YOLO"):
        print("      Necesitas entrenar el modelo:")
        print("      python scripts/train_custom_model.py")
        all_ok = False
    print()

    # Check tracking data
    print("3. DATOS DE TRACKING")
    print("-" * 70)
    tracking_files = [
        (f"{output_dir}/tracked_players.json", "Tracking original"),
        (f"{output_dir}/tracked_players_filtered.json", "Tracking filtrado (ROI)"),
        (f"{output_dir}/tracked_players_named.json", "Tracking con nombres"),
    ]
    for path, desc in tracking_files:
        if not check_file(path, desc):
            all_ok = False
    print()

    # Check player data
    print("4. DATOS DE JUGADORES")
    print("-" * 70)
    player_files = [
        (f"{output_dir}/player_names.json", "Nombres de jugadores"),
        (f"{output_dir}/team_assignments.json", "Asignación de equipos"),
    ]
    for path, desc in player_files:
        exists = check_file(path, desc)
        if exists and path.endswith('player_names.json'):
            with open(path) as f:
                names = json.load(f)
                print(f"      -> {len(names)} jugadores nombrados")
    print()

    # Check ball detection
    print("5. DETECCION DEL BALON")
    print("-" * 70)
    ball_files = [
        (f"{output_dir}/detections.json", "Detecciones del balón"),
        (f"{output_dir}/ball_trajectory.json", "Trayectoria suavizada"),
    ]
    for path, desc in ball_files:
        exists = check_file(path, desc)
        if exists and path.endswith('detections.json'):
            with open(path) as f:
                dets = json.load(f)
                print(f"      -> {len(dets)} frames detectados")
    print()

    # Check hoop/backboard
    print("6. CANASTA Y TABLERO")
    print("-" * 70)
    hoop_ok = check_file(f"{output_dir}/hoop.json", "Posición de la canasta")
    backboard_ok = check_file(f"{output_dir}/backboard.json", "Tablero marcado")

    if backboard_ok:
        with open(f"{output_dir}/backboard.json") as f:
            backboard = json.load(f)
            has_outer = 'corners' in backboard
            has_inner = 'inner_box_corners' in backboard
            print(f"      -> Tablero exterior: {'SI' if has_outer else 'NO'}")
            print(f"      -> Cuadro interior: {'SI' if has_inner else 'NO'}")
            if not has_inner:
                print("      [!] Marca el cuadro interior con:")
                print("          python scripts/mark_inner_box.py")
    print()

    # Check ROI
    print("7. ROI (AREA DE LA CANCHA)")
    print("-" * 70)
    check_file(f"{output_dir}/court_roi.json", "ROI definido")
    print()

    # Check final videos
    print("8. VIDEOS GENERADOS")
    print("-" * 70)
    videos = [
        (f"{output_dir}/final_video_COMPLETO.mp4", "Video final COMPLETO"),
        (f"{output_dir}/final_video_clean.mp4", "Video sin canasta"),
        (f"{output_dir}/ball_trajectory_visualization.mp4", "Visualización de trayectoria"),
    ]
    for path, desc in videos:
        check_file(path, desc)
    print()

    # Summary
    print("=" * 70)
    if all_ok:
        print("SISTEMA COMPLETO Y LISTO PARA USAR!")
        print()
        print("Para generar un nuevo video:")
        print("  python scripts/pipeline_improved.py")
        print()
        print("O ejecuta pasos individuales:")
        print("  python scripts/mark_backboard_corners.py")
        print("  python scripts/mark_inner_box.py")
        print("  python scripts/add_hoop_with_marked_backboard.py")
    else:
        print("SISTEMA INCOMPLETO")
        print()
        print("Ejecuta el pipeline completo:")
        print("  python scripts/pipeline_improved.py")
        print()
        print("O sigue los pasos indicados arriba")
    print("=" * 70)

    return 0 if all_ok else 1

if __name__ == '__main__':
    sys.exit(main())
