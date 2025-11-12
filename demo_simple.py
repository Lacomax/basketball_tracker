"""
Demo simplificado de las nuevas funcionalidades.
"""

import json
import sys
sys.path.insert(0, '/home/user/basketball_tracker')

from src.modules.hoop_detector import HoopDetector

# Cargar datos
print("=" * 60)
print("DEMOSTRACIÓN DE NUEVAS FUNCIONALIDADES - v2.0")
print("=" * 60)
print()

# 1. Cargar canasta
print("1️⃣  DETECCIÓN DE CANASTA")
print("-" * 60)
with open('outputs/demo/hoop.json', 'r') as f:
    hoop_data = json.load(f)

hoop_position = hoop_data['center']
print(f"✅ Canasta detectada en: {hoop_position}")
print(f"   Radio: {hoop_data['radius']} píxeles")
print(f"   Confianza: {hoop_data['confidence']:.1%}")
print()

# 2. Cargar detecciones del balón
print("2️⃣  DETECCIONES DEL BALÓN")
print("-" * 60)
with open('verified.json', 'r') as f:
    ball_detections = json.load(f)

print(f"✅ Total frames con balón: {len(ball_detections)}")
print(f"   Rango: frame {min(int(k) for k in ball_detections.keys())} "
      f"a {max(int(k) for k in ball_detections.keys())}")
print()

# 3. Analizar trayectorias para detectar tiros
print("3️⃣  ANÁLISIS DE TIROS")
print("-" * 60)

# Buscar secuencias que podrían ser tiros (movimiento hacia arriba)
def find_potential_shots(ball_data, window_size=30):
    """Encuentra trayectorias potenciales de tiros."""
    frames = sorted([int(k) for k in ball_data.keys()])
    shots = []

    i = 0
    while i < len(frames) - window_size:
        trajectory = []
        for j in range(window_size):
            if i + j >= len(frames):
                break
            frame_idx = frames[i + j]
            if str(frame_idx) in ball_data:
                center = ball_data[str(frame_idx)]['center']
                trajectory.append(center)

        if len(trajectory) >= 10:
            # Verificar movimiento vertical significativo (potencial tiro)
            y_coords = [p[1] for p in trajectory]
            height_change = max(y_coords) - min(y_coords)

            if height_change > 50:  # Movimiento vertical significativo
                shots.append({
                    'frame_start': frames[i],
                    'frame_end': frames[i + len(trajectory) - 1],
                    'trajectory': trajectory,
                    'height_change': height_change
                })
                i += window_size
                continue

        i += 1

    return shots

# Encontrar tiros
detector = HoopDetector()
detector.cached_hoop_position = hoop_position

shots = find_potential_shots(ball_detections, window_size=25)

print(f"✅ Tiros potenciales detectados: {len(shots)}")
print()

# 4. Clasificar tiros como anotados/fallados
print("4️⃣  CLASIFICACIÓN DE TIROS (ANOTADOS vs FALLADOS)")
print("-" * 60)

made_count = 0
missed_count = 0

for i, shot in enumerate(shots, 1):
    trajectory = shot['trajectory']
    is_made, confidence = detector.is_basket_made(trajectory, hoop_position)

    status = "🎯 ANOTADO" if is_made else "❌ FALLADO"
    shot['made'] = is_made
    shot['confidence'] = confidence

    print(f"Tiro #{i}: {status} (confianza: {confidence:.1%})")
    print(f"   Frames: {shot['frame_start']} - {shot['frame_end']}")
    print(f"   Cambio de altura: {shot['height_change']} px")
    print()

    if is_made:
        made_count += 1
    else:
        missed_count += 1

# 5. Estadísticas finales
print("=" * 60)
print("📊 ESTADÍSTICAS FINALES")
print("=" * 60)
print(f"Total de tiros: {len(shots)}")
print(f"Tiros anotados: {made_count}")
print(f"Tiros fallados: {missed_count}")
if len(shots) > 0:
    print(f"Porcentaje de acierto: {(made_count/len(shots))*100:.1f}%")
print()

print("=" * 60)
print("✨ NUEVAS FUNCIONALIDADES DEMOSTRADAS:")
print("=" * 60)
print("✅ Detección automática de canasta")
print("✅ Análisis de trayectorias de tiros")
print("✅ Clasificación automática (anotado/fallado)")
print("✅ Cálculo de porcentaje de acierto")
print()
print("📝 Nota: Esta es una demo simplificada.")
print("   Con el pipeline completo obtendrías:")
print("   - Tracking de jugadores (DeepSORT)")
print("   - Análisis de posesión")
print("   - Re-identification")
print("   - Video visualizado profesional")
print("=" * 60)
