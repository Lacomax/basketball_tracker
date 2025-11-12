import os, json, yaml, cv2, numpy as np, argparse, random
from ultralytics import YOLO

class UltraYOLOBallTrainer:
    def __init__(self, video_path: str, annotations: str, output_dir: str = 'outputs/yolo_dataset',
                 model: str = 'model/yolov8n.pt', sports_model: str = None):
        self.video = video_path
        with open(annotations, 'r') as f:
            self.annotations = json.load(f)
        self.output = output_dir
        # Usar modelo preentrenado específico de deportes si se proporciona y existe
        if sports_model and os.path.exists(sports_model):
            self.model = os.path.normpath(sports_model)
        else:
            self.model = os.path.normpath(model)
        if not os.path.exists(self.model):
            raise FileNotFoundError(f"Modelo YOLO no encontrado en: {self.model}")
        # Crear directorios necesarios
        for sub in ['images/train', 'images/val', 'labels/train', 'labels/val']:
            os.makedirs(os.path.join(output_dir, sub), exist_ok=True)
    
    def augment_image(self, img, ann, w, h):
        """Genera versiones aumentadas de la imagen y etiqueta correspondiente."""
        aug_imgs = []
        aug_labels = []
        cx, cy = ann['center']
        r = ann['radius']
        box_w = box_h = 2 * r
        
        # Augmentación 1: Rotación
        angle = random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h))
        center_pt = np.array([cx, cy, 1])
        new_center = M.dot(center_pt)
        new_label = [0, new_center[0]/w, new_center[1]/h, box_w/w, box_h/h]
        aug_imgs.append(rotated)
        aug_labels.append(new_label)
        
        # Augmentación 2: Escalado
        scale = random.uniform(0.8, 1.2)
        scaled = cv2.resize(img, None, fx=scale, fy=scale)
        new_w, new_h = int(w*scale), int(h*scale)
        new_cx = cx * scale
        new_cy = cy * scale
        new_box_w = box_w * scale
        new_box_h = box_h * scale
        # Redimensionar la imagen escalada a tamaño original
        scaled_back = cv2.resize(scaled, (w, h))
        new_label2 = [0, new_cx/w, new_cy/h, new_box_w/w, new_box_h/h]
        aug_imgs.append(scaled_back)
        aug_labels.append(new_label2)
        
        # Augmentación 3: Blur (desenfoque)
        ksize = random.choice([3, 5])
        blurred = cv2.GaussianBlur(img, (ksize, ksize), 0)
        new_label3 = [0, cx/w, cy/h, box_w/w, box_h/w]  # Mantener la misma etiqueta
        aug_imgs.append(blurred)
        aug_labels.append(new_label3)
        
        return aug_imgs, aug_labels

    def extract_frames(self, max_frames: int = 500, val_split: float = 0.2, augment: bool = True):
        cap = cv2.VideoCapture(self.video)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_indices = sorted(int(k) for k in self.annotations.keys())
        if not frame_indices:
            cap.release()
            raise ValueError("No hay anotaciones para extraer fotogramas.")
        use_frames = np.random.choice(frame_indices, size=min(max_frames, len(frame_indices)), replace=False)
        # Separar frames para validación
        val_frames = set(np.random.choice(use_frames, size=int(len(use_frames) * val_split), replace=False))
        for idx in use_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            split = 'val' if idx in val_frames else 'train'
            base_name = f'{idx:08d}'
            # Guardar imagen original
            img_path = os.path.join(self.output, 'images', split, base_name + '.jpg')
            cv2.imwrite(img_path, frame)
            ann = self.annotations[str(idx)]
            cx = ann['center'][0] / w
            cy = ann['center'][1] / h
            bw = (ann['radius'] * 2) / w
            bh = (ann['radius'] * 2) / h
            label_txt = f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"
            label_path = os.path.join(self.output, 'labels', split, base_name + '.txt')
            with open(label_path, 'w') as f:
                f.write(label_txt)
            # Generar aumentaciones solo para el conjunto de entrenamiento
            if split == 'train' and augment:
                aug_imgs, aug_labels = self.augment_image(frame, ann, w, h)
                for i, (a_img, a_label) in enumerate(zip(aug_imgs, aug_labels)):
                    aug_name = f'{base_name}_aug{i}.jpg'
                    cv2.imwrite(os.path.join(self.output, 'images', split, aug_name), a_img)
                    with open(os.path.join(self.output, 'labels', split, f'{base_name}_aug{i}.txt'), 'w') as f:
                        f.write(f"{a_label[0]} {a_label[1]:.6f} {a_label[2]:.6f} {a_label[3]:.6f} {a_label[4]:.6f}\n")
        cap.release()
        dataset_yaml = {
            'path': os.path.abspath(self.output).replace("\\", "/"),
            'train': 'images/train',
            'val': 'images/val',
            'names': {0: 'basketball'}
        }
        yaml_path = os.path.join(self.output, 'dataset.yaml')
        with open(yaml_path, 'w') as yf:
            yaml.dump(dataset_yaml, yf, default_flow_style=False, sort_keys=False)
        return yaml_path

    def train(self, epochs: int = 50, batch_size: int = 16, img_size: int = 640, patience: int = 10):
        data_config = self.extract_frames()
        model = YOLO(self.model)
        # Se añade early stopping con el parámetro 'patience' (si el framework lo soporta)
        return model.train(data=data_config, epochs=epochs, imgsz=img_size, batch=batch_size,
                           name='basketball_detector', patience=patience)

    @staticmethod
    def detect(video_path: str, model_path: str, output_path: str = None, conf: float = 0.3):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")
        return YOLO(model_path).predict(source=video_path, conf=conf, save=output_path is not None)

    def optimize_hyperparameters(self, param_grid: dict):
        # Ejemplo mínimo de búsqueda en cuadrícula para optimización de hiperparámetros.
        best_params = None
        best_metric = float('inf')
        for epochs in param_grid.get('epochs', [30]):
            for batch in param_grid.get('batch_size', [16]):
                print(f"Entrenando con epochs={epochs}, batch_size={batch}")
                # Aquí se simula una métrica de validación (para extender según tus datos)
                metric = random.uniform(0, 1)
                if metric < best_metric:
                    best_metric = metric
                    best_params = {'epochs': epochs, 'batch_size': batch}
        print(f"Mejores parámetros encontrados: {best_params} con métrica {best_metric}")
        return best_params

def main():
    parser = argparse.ArgumentParser(description='Pipeline de entrenamiento YOLO para detección de balones')
    parser.add_argument('--video', default='data/input_video.mp4', help='Ruta al video de entrada')
    parser.add_argument('--annotations', default='outputs/verified.json', help='JSON con anotaciones verificadas')
    parser.add_argument('--output', default='outputs/yolo_dataset', help='Directorio para dataset y resultados de YOLO')
    parser.add_argument('--epochs', type=int, default=30, help='Cantidad de épocas de entrenamiento')
    parser.add_argument('--detect', action='store_true', help='Ejecutar detección tras el entrenamiento')
    parser.add_argument('--sports_model', default=None, help='Ruta a modelo preentrenado específico de deportes')
    args = parser.parse_args()

    trainer = UltraYOLOBallTrainer(args.video, args.annotations, args.output,
                                   model='model/yolov8n.pt', sports_model=args.sports_model)
    
    # Ejemplo de optimización de hiperparámetros (comentado, ampliar según necesidad)
    # param_grid = {'epochs': [20, 30], 'batch_size': [8, 16]}
    # best_params = trainer.optimize_hyperparameters(param_grid)
    # if best_params:
    #     args.epochs = best_params['epochs']
    #     batch = best_params['batch_size']
    # else:
    #     batch = 16

    trainer.train(epochs=args.epochs, batch_size=16, img_size=640, patience=10)
    if args.detect:
        best_weights = os.path.join(args.output, 'weights', 'best.pt')
        if os.path.exists(best_weights):
            UltraYOLOBallTrainer.detect(args.video, best_weights,
                                          output_path='outputs/detected_basketball.mp4', conf=0.3)
        else:
            print("No se encontraron pesos entrenados para detección.")

if __name__ == '__main__':
    main()
