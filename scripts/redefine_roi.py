#!/usr/bin/env python3
"""
Redefinir ROI de manera más estricta para eliminar jugadores fuera de la cancha.
"""

import sys
import os
import cv2
import json
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.video_utils import open_video_robust
from src.utils.config_loader import get_config

config = get_config()

print("=" * 70)
print("ROI RE-DEFINITION - Filtrado más estricto")
print("=" * 70)
print()
print("Este script te permite redefinir el ROI para eliminar:")
print("  - Jugadores en el banquillo")
print("  - Entrenadores")
print("  - Público en las gradas")
print("  - Árbitros fuera de juego")
print()

# Find video
input_video = None
for path in config.get_video_paths():
    if os.path.exists(path):
        input_video = path
        break

if not input_video:
    print("[X] Video not found")
    sys.exit(1)

# Load original tracking data
output_dir = config.get_output_dir()
tracking_file = f"{output_dir}/tracked_players.json"

if not os.path.exists(tracking_file):
    print(f"[X] Original tracking data not found: {tracking_file}")
    sys.exit(1)

with open(tracking_file, 'r') as f:
    tracking_data = json.load(f)

print(f"[+] Video: {input_video}")
print(f"[+] Tracking data: {tracking_file}")
print()

# Open video
try:
    cap = open_video_robust(input_video)
except IOError as e:
    print(f"[X] {e}")
    sys.exit(1)

ret, first_frame = cap.read()
cap.release()

if not ret:
    print("[X] Cannot read first frame")
    sys.exit(1)

print("=" * 70)
print("INSTRUCCIONES:")
print("=" * 70)
print()
print("1. Define el ROI haciendo click en las ESQUINAS DE LA CANCHA")
print("2. Haz click en al menos 4 puntos (esquinas)")
print("3. Sé MÁS ESTRICTO - solo incluye el área de juego")
print("4. NO incluyas:")
print("   - Banquillos")
print("   - Líneas fuera de la cancha")
print("   - Zonas de entrenadores")
print("5. Presiona ENTER cuando termines")
print()

# Global variables for ROI selection
roi_points = []

def mouse_callback(event, x, y, flags, param):
    global roi_points

    if event == cv2.EVENT_LBUTTONDOWN:
        roi_points.append([x, y])

        # Redraw frame
        frame_copy = param.copy()

        # Draw all points
        for i, pt in enumerate(roi_points):
            cv2.circle(frame_copy, tuple(pt), 8, (0, 255, 0), -1)
            cv2.putText(frame_copy, str(i+1), (pt[0]+15, pt[1]-15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Draw lines between points
        if len(roi_points) > 1:
            for i in range(len(roi_points) - 1):
                cv2.line(frame_copy, tuple(roi_points[i]), tuple(roi_points[i+1]),
                        (0, 255, 0), 3)

        # Close polygon if 4+ points
        if len(roi_points) >= 4:
            cv2.line(frame_copy, tuple(roi_points[-1]), tuple(roi_points[0]),
                    (0, 255, 0), 3)

            # Fill polygon semi-transparent
            overlay = frame_copy.copy()
            pts = np.array(roi_points, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            cv2.addWeighted(overlay, 0.3, frame_copy, 0.7, 0, frame_copy)

        cv2.imshow("Redefinir ROI - SOLO CANCHA", frame_copy)

# Show frame for selection
display_frame = first_frame.copy()
cv2.putText(display_frame, "Click en esquinas de la CANCHA (solo area de juego)",
           (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
cv2.putText(display_frame, "Presiona ENTER cuando termines",
           (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

cv2.imshow("Redefinir ROI - SOLO CANCHA", display_frame)
cv2.setMouseCallback("Redefinir ROI - SOLO CANCHA", mouse_callback, first_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(roi_points) < 4:
    print("[X] Necesitas al menos 4 puntos")
    sys.exit(1)

print(f"[+] ROI definido con {len(roi_points)} puntos")

# Save new ROI
roi_data = {
    'points': roi_points,
    'num_points': len(roi_points)
}

roi_file = f"{output_dir}/court_roi.json"
with open(roi_file, 'w') as f:
    json.dump(roi_data, f, indent=2)

print(f"[+] ROI guardado en {roi_file}")
print()

# Apply strict filtering
print("=" * 70)
print("APLICANDO FILTRADO ESTRICTO")
print("=" * 70)
print()

h, w = first_frame.shape[:2]
roi_mask = np.zeros((h, w), dtype=np.uint8)
pts = np.array(roi_points, dtype=np.int32)
cv2.fillPoly(roi_mask, [pts], 255)

def is_inside_strict_roi(player, mask):
    """
    Verificación ESTRICTA: usa los PIES del jugador (bottom-center).
    Si los pies no están en la cancha, el jugador está fuera.
    """
    bbox = player.get('bbox')
    if not bbox:
        return False

    x1, y1, x2, y2 = bbox

    # Posición de los pies (bottom-center del bbox)
    feet_x = int((x1 + x2) / 2)
    feet_y = int(y2)  # Bottom del bbox

    # Verificar si los pies están dentro del ROI
    if 0 <= feet_x < mask.shape[1] and 0 <= feet_y < mask.shape[0]:
        return mask[feet_y, feet_x] > 0

    return False

# Filter with strict criteria
filtered_data = {}
players_filtered_out = 0
players_kept = 0

# Maximum players per frame
MAX_PLAYERS = config.get('tracking', {}).get('max_players_per_frame', 10)

for frame_idx, players in tracking_data.items():
    # Filter by strict ROI (feet must be inside)
    roi_filtered = []

    for player in players:
        if is_inside_strict_roi(player, roi_mask):
            roi_filtered.append(player)
        else:
            players_filtered_out += 1

    # Limit to max players (keep highest confidence)
    if len(roi_filtered) > MAX_PLAYERS:
        def get_conf(p):
            if 'confidence' in p:
                return p['confidence']
            bbox = p.get('bbox')
            if bbox:
                x1, y1, x2, y2 = bbox
                return (x2-x1) * (y2-y1)
            return 0

        roi_filtered.sort(key=get_conf, reverse=True)
        kept = roi_filtered[:MAX_PLAYERS]
        players_filtered_out += len(roi_filtered) - MAX_PLAYERS
        roi_filtered = kept

    players_kept += len(roi_filtered)

    if roi_filtered:
        filtered_data[frame_idx] = roi_filtered

print(f"[+] Jugadores dentro de la cancha: {players_kept}")
print(f"[+] Jugadores eliminados (banquillo/público): {players_filtered_out}")
print(f"[+] Frames con jugadores: {len(filtered_data)}/{len(tracking_data)}")
print()

# Get unique player IDs
unique_ids = set()
for players in filtered_data.values():
    for player in players:
        track_id = player.get('track_id')
        if track_id is not None:
            unique_ids.add(track_id)

print(f"[+] IDs únicos después del filtrado: {len(unique_ids)}")
print(f"    IDs: {sorted(unique_ids)}")
print()

# Save filtered data
output_file = f"{output_dir}/tracked_players_filtered.json"
with open(output_file, 'w') as f:
    json.dump(filtered_data, f, indent=2)

print(f"[+] Datos filtrados guardados en: {output_file}")
print()

print("=" * 70)
print("COMPLETADO!")
print("=" * 70)
print()
print("Ahora ejecuta:")
print("  python scripts/assign_names.py")
print()
print("Deberías ver SOLO jugadores dentro de la cancha")
print()
