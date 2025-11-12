import cv2, numpy as np, json, os
from filterpy.kalman import KalmanFilter

def clamp(val, min_val, max_val):
    return max(min(val, max_val), min_val)

def create_kalman_filter(initial_pos):
    # Estado: [x, y, vx, vy]
    kf = KalmanFilter(dim_x=4, dim_z=2)
    x, y = initial_pos
    kf.x = np.array([x, y, 0, 0], dtype=float)
    dt = 1.0  # asumiendo 1 fotograma por unidad de tiempo
    kf.F = np.array([[1, 0, dt, 0],
                     [0, 1, 0, dt],
                     [0, 0, 1, 0 ],
                     [0, 0, 0, 1 ]], dtype=float)
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]], dtype=float)
    kf.P *= 500.
    kf.R = np.eye(2) * 5.
    kf.Q = np.eye(4) * 0.1
    return kf

def process_trajectory_video(video_path: str, annotations_path: str, output_path: str):
    """Genera detecciones del balón en cada fotograma usando filtro de Kalman entre anotaciones manuales."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Cargar anotaciones manuales
    try:
        with open(annotations_path, 'r') as f:
            annotations = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        annotations = {}
    if not annotations:
        with open(output_path, 'w') as f:
            json.dump({}, f, indent=2)
        return {}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detection_points = {}
    # Usar las anotaciones manuales como fotogramas clave
    manual_frames = sorted(int(k) for k in annotations.keys())

    # Umbral para detectar movimiento abrupto (posible oclusión o rebote)
    VELOCITY_THRESHOLD = 50.0

    # Procesar segmentos entre anotaciones manuales
    for idx in range(len(manual_frames) - 1):
        start_f = manual_frames[idx]
        end_f = manual_frames[idx + 1]
        start_ann = annotations[str(start_f)]
        end_ann = annotations[str(end_f)]
        detection_points[start_f] = {'center': start_ann['center'], 'radius': start_ann['radius']}
        
        # Calcular interpolación lineal para el radio
        def interp_radius(f):
            ratio = (f - start_f) / (end_f - start_f)
            return int(start_ann['radius'] + ratio * (end_ann['radius'] - start_ann['radius']))
        
        # Inicializar filtro de Kalman con el fotograma de inicio
        kf = create_kalman_filter(start_ann['center'])
        
        for f in range(start_f + 1, end_f):
            kf.predict()
            pred_center = [kf.x[0], kf.x[1]]
            # Clampeamos la posición a los límites del fotograma
            pred_center[0] = int(clamp(pred_center[0], 0, frame_width - 1))
            pred_center[1] = int(clamp(pred_center[1], 0, frame_height - 1))
            radius = interp_radius(f)
            # Detectar occlusión/rebote por velocidad excesiva
            velocity = np.sqrt(kf.x[2]**2 + kf.x[3]**2)
            detection = {'center': pred_center, 'radius': radius}
            if velocity > VELOCITY_THRESHOLD:
                detection['occluded'] = True
            detection_points[f] = detection

        # En fotograma final, forzar anotación manual y reiniciar el filtro
        detection_points[end_f] = {'center': end_ann['center'], 'radius': end_ann['radius']}
    
    # Procesar fotogramas después de la última anotación
    last_frame = manual_frames[-1]
    last_ann = annotations[str(last_frame)]
    detection_points[last_frame] = {'center': last_ann['center'], 'radius': last_ann['radius']}
    if last_frame < total_frames - 1:
        kf = create_kalman_filter(last_ann['center'])
        for f in range(last_frame + 1, total_frames):
            kf.predict()
            pred_center = [kf.x[0], kf.x[1]]
            pred_center[0] = int(clamp(pred_center[0], 0, frame_width - 1))
            pred_center[1] = int(clamp(pred_center[1], 0, frame_height - 1))
            # Mantener el radio constante en la última sección
            detection_points[f] = {'center': pred_center, 'radius': last_ann['radius']}
    
    cap.release()
    with open(output_path, 'w') as f:
        json.dump(detection_points, f, indent=2)
    print(f"Generadas {len(detection_points)} detecciones -> {output_path}")
    return detection_points

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Detector de trayectoria con filtro de Kalman")
    parser.add_argument("--video", default="data/input_video.mp4", help="Ruta del video de entrada")
    parser.add_argument("--annotations", default="outputs/annotations.json", help="JSON con anotaciones manuales")
    parser.add_argument("--output", default="outputs/detections.json", help="Ruta de salida para las detecciones")
    args = parser.parse_args()
    process_trajectory_video(args.video, args.annotations, args.output)
