#!/usr/bin/env python3
"""
Mark the 4 corners of the inner target box on the backboard.
"""

# Fix OpenMP error
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import json
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.video_utils import open_video_robust
from src.utils.config_loader import get_config

config = get_config()

print("=" * 70)
print("MARCAR CUADRO INTERIOR DEL TABLERO (TARGET BOX)")
print("=" * 70)
print()

# Get video from config
video_paths = config.get_video_paths()
video_file = None
for path in video_paths:
    if os.path.exists(path):
        video_file = path
        break

if not video_file:
    print("[X] Video no encontrado")
    sys.exit(1)

print(f"[+] Video: {video_file}")

# Load backboard data to show reference
output_dir = config.get_output_dir()
backboard_file = f"{output_dir}/backboard.json"

if os.path.exists(backboard_file):
    with open(backboard_file, 'r') as f:
        backboard_data = json.load(f)
    backboard_corners = np.array(backboard_data['corners'], dtype=np.int32)
    print("[+] Tablero exterior ya marcado")
else:
    print("[!] No se encontro backboard.json")
    backboard_corners = None

print()

# Open video
try:
    cap = open_video_robust(video_file)
except IOError as e:
    print(f"[X] {e}")
    sys.exit(1)

# Read first frame
ret, first_frame = cap.read()
cap.release()

if not ret:
    print("[X] No se puede leer el primer frame")
    sys.exit(1)

print("=" * 70)
print("INSTRUCCIONES:")
print("=" * 70)
print()
print("Marca las 4 ESQUINAS del CUADRO INTERIOR (target) en orden:")
print()
print("  1. ARRIBA IZQUIERDA")
print("  2. ARRIBA DERECHA")
print("  3. ABAJO DERECHA")
print("  4. ABAJO IZQUIERDA")
print()
print("El cuadro interior es el rectangulo rojo/naranja dentro del tablero")
print()
print("Haz click en cada esquina en orden.")
print("Si te equivocas, presiona 'R' para reiniciar.")
print("Presiona ENTER cuando hayas marcado las 4 esquinas.")
print()

# Global variables
inner_box_corners = []
corner_labels = ["ARRIBA IZQ", "ARRIBA DER", "ABAJO DER", "ABAJO IZQ"]

def mouse_callback(event, x, y, flags, param):
    global inner_box_corners

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(inner_box_corners) < 4:
            inner_box_corners.append([x, y])
            print(f"[+] Esquina {len(inner_box_corners)}: {corner_labels[len(inner_box_corners)-1]} = ({x}, {y})")

            # Redraw frame
            frame_copy = param.copy()

            # Draw backboard reference (if available)
            if backboard_corners is not None:
                cv2.polylines(frame_copy, [backboard_corners], True, (100, 100, 255), 2)
                cv2.putText(frame_copy, "TABLERO EXTERIOR (referencia)",
                           (backboard_corners[0][0], backboard_corners[0][1] - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)

            # Draw all inner box points
            for i, pt in enumerate(inner_box_corners):
                color = (0, 255, 0) if i < len(inner_box_corners) - 1 else (0, 255, 255)
                cv2.circle(frame_copy, tuple(pt), 8, color, -1)
                cv2.putText(frame_copy, str(i+1), (pt[0]+15, pt[1]-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(frame_copy, corner_labels[i], (pt[0]+15, pt[1]+5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Draw lines between points
            if len(inner_box_corners) > 1:
                for i in range(len(inner_box_corners) - 1):
                    cv2.line(frame_copy, tuple(inner_box_corners[i]),
                            tuple(inner_box_corners[i+1]), (0, 165, 255), 3)

            # Close polygon if 4 points
            if len(inner_box_corners) == 4:
                cv2.line(frame_copy, tuple(inner_box_corners[3]),
                        tuple(inner_box_corners[0]), (0, 165, 255), 3)

                # Fill polygon semi-transparent
                overlay = frame_copy.copy()
                pts = np.array(inner_box_corners, dtype=np.int32)
                cv2.fillPoly(overlay, [pts], (0, 165, 255))
                cv2.addWeighted(overlay, 0.3, frame_copy, 0.7, 0, frame_copy)

                # Add completion message
                cv2.putText(frame_copy, "COMPLETO! Presiona ENTER",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

            # Show next corner to mark
            if len(inner_box_corners) < 4:
                next_corner = corner_labels[len(inner_box_corners)]
                cv2.putText(frame_copy, f"Siguiente: {next_corner} (CUADRO INTERIOR)",
                           (10, frame_copy.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("Marcar Cuadro Interior", frame_copy)

# Show frame for selection
display_frame = first_frame.copy()

# Draw backboard reference
if backboard_corners is not None:
    cv2.polylines(display_frame, [backboard_corners], True, (100, 100, 255), 2)

cv2.putText(display_frame, "Click en las 4 esquinas del CUADRO INTERIOR (target box)",
           (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.putText(display_frame, "Presiona 'R' para reiniciar, ENTER cuando termines",
           (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

cv2.imshow("Marcar Cuadro Interior", display_frame)
cv2.setMouseCallback("Marcar Cuadro Interior", mouse_callback, first_frame)

while True:
    key = cv2.waitKey(1) & 0xFF

    if key == 13:  # ENTER
        if len(inner_box_corners) == 4:
            break
        else:
            print(f"[!] Necesitas marcar 4 esquinas (tienes {len(inner_box_corners)})")

    elif key == ord('r') or key == ord('R'):  # Reset
        print("[!] Reiniciando...")
        inner_box_corners = []
        display_frame = first_frame.copy()
        if backboard_corners is not None:
            cv2.polylines(display_frame, [backboard_corners], True, (100, 100, 255), 2)
        cv2.putText(display_frame, "Click en las 4 esquinas del CUADRO INTERIOR",
                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Marcar Cuadro Interior", display_frame)

    elif key == 27:  # ESC
        print("[!] Cancelado")
        cv2.destroyAllWindows()
        sys.exit(0)

cv2.destroyAllWindows()

if len(inner_box_corners) != 4:
    print("[X] Necesitas marcar exactamente 4 esquinas")
    sys.exit(1)

print()
print("=" * 70)
print("CUADRO INTERIOR MARCADO!")
print("=" * 70)
print()
print("Esquinas del cuadro interior:")
for i, corner in enumerate(inner_box_corners):
    print(f"  {i+1}. {corner_labels[i]}: {corner}")
print()

# Update backboard data with inner box
if backboard_data:
    backboard_data['inner_box_corners'] = inner_box_corners
    backboard_data['inner_box_labels'] = corner_labels
else:
    print("[!] Creando nuevo archivo de tablero")
    backboard_data = {
        'inner_box_corners': inner_box_corners,
        'inner_box_labels': corner_labels,
        'note': 'Inner target box marked manually'
    }

with open(backboard_file, 'w') as f:
    json.dump(backboard_data, f, indent=2)

print(f"[+] Cuadro interior guardado en: {backboard_file}")
print()
print("=" * 70)
print("SIGUIENTE PASO:")
print("=" * 70)
print()
print("Ahora ejecuta:")
print("  python scripts/add_hoop_with_marked_backboard.py")
print()
print("Esto creara el video con ambos cuadros en perspectiva exacta!")
print()
