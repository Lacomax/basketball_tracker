import cv2, numpy as np, json, os, argparse

class CompactBallVerifier:
    def __init__(self, video_path: str, detection_file: str = None, output_file: str = "outputs/verified.json"):
        self.video_path = video_path
        self.output_file = output_file
        # Cargar verificaciones previas si existen en output_file; sino, cargar detecciones
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r') as f:
                    self.verified = json.load(f)
            except json.JSONDecodeError:
                self.verified = {}
        else:
            try:
                self.verified = json.load(open(detection_file)) if detection_file and os.path.exists(detection_file) else {}
            except (json.JSONDecodeError, FileNotFoundError):
                self.verified = {}
        # Asegurar que las claves sean strings
        self.verified = {str(k): v for k, v in self.verified.items()}
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise IOError(f"No se puede abrir el video: {video_path}")
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_idx = 0
        self.dirty = False
        self.current_frame = None
        self.full_traj = False  # Toggle para previsualizar trayectoria completa
        self.anomaly_threshold = 50  # Umbral (en píxeles) para detectar anomalías

    def _auto_detect_ball(self, frame: np.ndarray, point: tuple) -> dict:
        x, y = int(point[0]), int(point[1])
        h, w = frame.shape[:2]
        roi_x0, roi_y0 = max(0, x - 30), max(0, y - 30)
        roi = frame[roi_y0: min(h, y + 30), roi_x0: min(w, x + 30)]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                   param1=50, param2=30, minRadius=5, maxRadius=50)
        if circles is None:
            circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                       param1=50, param2=15, minRadius=5, maxRadius=50)
        if circles is not None:
            cx, cy, r = circles[0][0]
            return {'center': [int(cx) + roi_x0, int(cy) + roi_y0], 'radius': int(r)}
        return {'center': [x, y], 'radius': 15}

    def _get_anomaly(self):
        # Detecta anomalía comparando con la anotación previa
        current_key = str(self.frame_idx)
        sorted_keys = sorted([int(k) for k in self.verified.keys()])
        prev_frame = None
        for k in sorted_keys:
            if k < self.frame_idx:
                prev_frame = k
            else:
                break
        if prev_frame is not None and current_key in self.verified:
            prev_center = np.array(self.verified[str(prev_frame)]['center'])
            curr_center = np.array(self.verified[current_key]['center'])
            if np.linalg.norm(curr_center - prev_center) > self.anomaly_threshold:
                return True
        return False

    def _compute_anomaly_frames(self):
        # Retorna una lista ordenada de fotogramas con anomalías
        anomaly_frames = []
        sorted_keys = sorted([int(k) for k in self.verified.keys()])
        for i in range(1, len(sorted_keys)):
            prev = sorted_keys[i-1]
            curr = sorted_keys[i]
            prev_center = np.array(self.verified[str(prev)]['center'])
            curr_center = np.array(self.verified[str(curr)]['center'])
            if np.linalg.norm(curr_center - prev_center) > self.anomaly_threshold:
                anomaly_frames.append(curr)
        return anomaly_frames

    def _mouse_handler(self, event: int, x: int, y: int, flags: int, param: any):
        frame_key = str(self.frame_idx)
        if event == cv2.EVENT_LBUTTONDOWN:
            if frame_key in self.verified:
                existing = self.verified[frame_key]
                # Si clic cerca de la anotación existente, mover centro
                if np.hypot(x - existing['center'][0], y - existing['center'][1]) <= existing['radius']:
                    self.verified[frame_key]['center'] = [x, y]
                else:
                    self.verified[frame_key] = self._auto_detect_ball(self.current_frame, (x, y))
            else:
                self.verified[frame_key] = self._auto_detect_ball(self.current_frame, (x, y))
            self.dirty = True
            self._update_display()

    def _update_display(self):
        if self.current_frame is None:
            return
        vis_frame = self.current_frame.copy()
        frame_key = str(self.frame_idx)
        # Seleccionar puntos a mostrar: trayectoria completa o ventana local
        if self.full_traj:
            points = [(int(k), v) for k, v in self.verified.items()]
            points.sort(key=lambda x: x[0])
            line_color = (150, 150, 150)
        else:
            points = [(int(k), v) for k, v in self.verified.items() if abs(int(k) - self.frame_idx) <= 90]
            points.sort(key=lambda x: x[0])
            line_color = (0, 100, 255)
        for (prev_idx, prev_pt), (curr_idx, curr_pt) in zip(points[:-1], points[1:]):
            if abs(curr_idx - prev_idx) <= 30:
                cv2.line(vis_frame,
                         tuple(map(int, prev_pt['center'])),
                         tuple(map(int, curr_pt['center'])),
                         line_color, 1)
        # Dibujar la anotación del fotograma actual
        if frame_key in self.verified:
            center = tuple(map(int, self.verified[frame_key]['center']))
            radius = self.verified[frame_key]['radius']
            if self.verified[frame_key].get('hidden', False):
                cv2.circle(vis_frame, center, radius, (0, 0, 255), 2)
                cv2.putText(vis_frame, "OCULTO", (center[0] - 10, center[1] - radius - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.circle(vis_frame, center, radius, (0, 255, 0), 2)
                cv2.circle(vis_frame, center, 3, (0, 0, 255), -1)
        # Mostrar indicador de anomalía
        if self._get_anomaly():
            cv2.putText(vis_frame, "ANOMALY", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        # Información en pantalla
        cv2.putText(vis_frame, f"Frame: {self.frame_idx}/{self.total_frames}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        info_text = "a/d: Prev/Next   +/-: Radius   S: Save   Q: Quit   t: Toggle Traj   h: Toggle Hidden"
        cv2.putText(vis_frame, info_text, (10, vis_frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(vis_frame, "p: Prev anomaly   n: Next anomaly", (10, vis_frame.shape[0]-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.imshow("Ball Verifier", vis_frame)

    def run(self):
        cv2.namedWindow("Ball Verifier")
        cv2.setMouseCallback("Ball Verifier", self._mouse_handler)
        while self.frame_idx < self.total_frames:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx)
            ret, self.current_frame = self.cap.read()
            if not ret:
                break
            self._update_display()
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                with open(self.output_file, 'w') as f:
                    json.dump(self.verified, f, indent=2)
                self.dirty = False
            elif key in (ord('a'), 81):
                self.frame_idx = max(0, self.frame_idx - 1)
            elif key in (ord('d'), 83):
                self.frame_idx = min(self.total_frames - 1, self.frame_idx + 1)
            elif key in (ord('+'), 82):
                fk = str(self.frame_idx)
                if fk in self.verified:
                    self.verified[fk]['radius'] += 1
                    self.dirty = True
            elif key in (ord('-'), 84):
                fk = str(self.frame_idx)
                if fk in self.verified:
                    self.verified[fk]['radius'] = max(1, self.verified[fk]['radius'] - 1)
                    self.dirty = True
            elif key == ord('t'):
                self.full_traj = not self.full_traj
            elif key == ord('h'):
                fk = str(self.frame_idx)
                if fk in self.verified:
                    self.verified[fk]['hidden'] = not self.verified[fk].get('hidden', False)
                    self.dirty = True
            elif key == ord('p'):
                # Navegar a la anomalía previa
                anomaly_frames = self._compute_anomaly_frames()
                prev_anom = [f for f in anomaly_frames if f < self.frame_idx]
                if prev_anom:
                    self.frame_idx = max(prev_anom)
            elif key == ord('n'):
                # Navegar a la siguiente anomalía
                anomaly_frames = self._compute_anomaly_frames()
                next_anom = [f for f in anomaly_frames if f > self.frame_idx]
                if next_anom:
                    self.frame_idx = min(next_anom)
            # Si no se presiona una tecla de navegación, se permanece en el mismo fotograma para ajustes
        cv2.destroyAllWindows()
        if self.dirty:
            with open(self.output_file, 'w') as f:
                json.dump(self.verified, f, indent=2)
        self.cap.release()

def main():
    parser = argparse.ArgumentParser(description="Herramienta de verificación de detecciones de balones")
    parser.add_argument("--video", default="data/input_video.mp4", help="Ruta del video de entrada")
    parser.add_argument("--detections", default="outputs/detections.json", help="Archivo JSON con detecciones")
    parser.add_argument("--output", default="outputs/verified.json", help="Ruta de salida para las verificaciones")
    args = parser.parse_args()
    CompactBallVerifier(args.video, args.detections, output_file=args.output).run()

if __name__ == '__main__':
    main()
