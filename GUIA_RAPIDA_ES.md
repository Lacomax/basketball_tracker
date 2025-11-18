# 🚀 Guía Rápida - Basketball Tracker (Español)

Guía rápida para configurar y usar el sistema de detección de baloncesto.

## 📑 Índice

1. [Inicio Rápido](#inicio-rápido)
2. [Usar Modelo Pre-entrenado](#usar-modelo-pre-entrenado)
3. [Descargar Datasets de Roboflow](#descargar-datasets-de-roboflow)
4. [Anotar tus Propios Videos](#anotar-tus-propios-videos)
5. [Entrenar Modelo Personalizado](#entrenar-modelo-personalizado)

---

## Inicio Rápido

### 1. Instalación

```bash
# Clonar repositorio
git clone <repository-url>
cd basketball_tracker

# Instalar dependencias
pip install -r requirements.txt

# Instalar Roboflow (opcional, para descarga automática)
pip install roboflow
```

### 2. Verificar Instalación

```bash
# Probar que todo funciona
python -c "from ultralytics import YOLO; print('✅ YOLO instalado correctamente')"
```

---

## Usar Modelo Pre-entrenado

**¿Quieres probar sin entrenar? ¡Usa un modelo pre-entrenado!**

### Opción 1: YOLO Pre-entrenado Genérico

El modelo detecta "sports ball" (incluye baloncesto):

```bash
python scripts/use_pretrained_model.py --video input_video.mp4
```

**Salida:** `outputs/detected_input_video.mp4`

### Opción 2: Modelo Específico de Baloncesto

Si ya tienes un modelo entrenado:

```bash
python scripts/use_pretrained_model.py \
    --video input_video.mp4 \
    --model models/basketball_detector.pt
```

### Ajustar Sensibilidad

```bash
# Más detecciones (menos confianza)
python scripts/use_pretrained_model.py --video input_video.mp4 --conf 0.2

# Menos detecciones (más confianza)
python scripts/use_pretrained_model.py --video input_video.mp4 --conf 0.7
```

### Usar Modelo Más Preciso

```bash
# Modelo más grande = mejor precisión (más lento)
python scripts/use_pretrained_model.py \
    --video input_video.mp4 \
    --yolo-model yolo11x.pt
```

**Modelos disponibles:**
- `yolo11n.pt` - Nano (más rápido, menos preciso)
- `yolo11s.pt` - Small
- `yolo11m.pt` - Medium
- `yolo11l.pt` - Large (recomendado)
- `yolo11x.pt` - Extra Large (más preciso, más lento)

---

## Descargar Datasets de Roboflow

### 1. Obtener API Key de Roboflow

1. Crea cuenta gratuita en [https://roboflow.com/](https://roboflow.com/)
2. Ve a **Settings → API Keys**
3. Copia tu **Private API Key**

### 2. Listar Datasets Recomendados

```bash
python scripts/download_roboflow_dataset.py --list
```

Esto muestra datasets populares de baloncesto.

### 3. Descargar Todos los Datasets Recomendados

```bash
python scripts/download_roboflow_dataset.py \
    --api-key TU_API_KEY \
    --download-all
```

**Esto descarga:**
- Basketball Detection (Roboflow 100) - Dataset oficial verificado

**Nota:** El script incluye solo datasets públicos verificados. Para agregar más datasets, consulta `docs/COMO_AGREGAR_DATASETS_ROBOFLOW.md`

**Los guarda en:** `data/basketball_training/`

### 4. Descargar Dataset Específico

```bash
python scripts/download_roboflow_dataset.py \
    --api-key TU_API_KEY \
    --workspace roboflow-100 \
    --project basketball-detection \
    --version 1
```

### 5. Buscar Más Datasets

Ve a [Roboflow Universe](https://universe.roboflow.com/search?q=basketball) y busca "basketball".

---

## Anotar tus Propios Videos

**¿Quieres entrenar con tus propios videos? ¡Debes anotarlos primero!**

### Método 1: Anotación Rápida (Herramienta Incluida)

#### Paso 1: Preparar Video

```bash
# Copiar video a la carpeta correcta
cp mi_video.mp4 data/raw/
```

#### Paso 2: Anotar

```python
# Crear script: scripts/annotate_my_video.py
from src.modules.annotator import BallAnnotator

annotator = BallAnnotator(
    video="data/raw/mi_video.mp4",
    output="data/annotations/mi_video.json"
)
annotator.run()
```

```bash
# Ejecutar
python scripts/annotate_my_video.py
```

**Controles:**
- **Click:** Marcar posición del balón
- **A/D:** Frame anterior/siguiente
- **S:** Guardar
- **Q:** Salir

**Consejo:** No anotes cada frame. Anota cada 10-30 frames (el sistema interpola el resto).

#### Paso 3: Interpolar con Kalman

```python
from src.modules.trajectory_detector import process_trajectory_video

process_trajectory_video(
    video_path="data/raw/mi_video.mp4",
    annotations_path="data/annotations/mi_video.json",
    output_path="data/detections/mi_video.json"
)
```

#### Paso 4: Verificar y Corregir

```python
from src.modules.verifier import CompactBallVerifier

verifier = CompactBallVerifier(
    video_path="data/raw/mi_video.mp4",
    detection_file="data/detections/mi_video.json",
    output_file="data/verified/mi_video.json"
)
verifier.run()
```

### Método 2: Anotación Profesional (Roboflow)

#### Paso 1: Extraer Frames

```bash
python scripts/extract_frames_for_annotation.py \
    --video data/raw/mi_video.mp4 \
    --interval 10
```

**Salida:** `data/frames_to_annotate/mi_video/`

#### Paso 2: Subir a Roboflow

1. Crea proyecto en Roboflow
2. Sube las imágenes
3. Anota el balón con bounding boxes
4. Genera dataset en formato YOLOv8

#### Paso 3: Descargar Dataset

```bash
python scripts/download_roboflow_dataset.py \
    --api-key TU_API_KEY \
    --workspace tu-workspace \
    --project tu-proyecto \
    --version 1
```

### 📖 Guía Completa

Lee la [Guía Completa de Anotación](docs/GUIA_ANOTACION_VIDEOS.md) para más detalles.

---

## Entrenar Modelo Personalizado

### Con Datasets de Roboflow

```bash
# Después de descargar datasets
python scripts/train_basketball_detector_simple.py
```

Esto:
1. Encuentra todos los datasets en `data/basketball_training/`
2. Los combina en un solo dataset
3. Entrena YOLO11-L por 50 épocas
4. Guarda el modelo en `models/basketball_detector_yolo11l.pt`

### Con tus Propias Anotaciones

```python
from src.modules.yolo_trainer import UltraYOLOBallTrainer

trainer = UltraYOLOBallTrainer(
    video_path="data/raw/mi_video.mp4",
    annotations="data/verified/mi_video.json",
    output_dir="models/trained/mi_detector",
    model="yolo11l.pt"
)

trainer.train(epochs=50, batch_size=16, img_size=640)
```

### Configuración Avanzada

```python
# Entrenar más tiempo
trainer.train(epochs=100, batch_size=16)

# Modelo más grande (mejor precisión)
trainer = UltraYOLOBallTrainer(
    video_path="data/raw/mi_video.mp4",
    annotations="data/verified/mi_video.json",
    model="yolo11x.pt"  # Extra Large
)

# GPU específica
trainer.train(epochs=50, device="cuda:0")
```

---

## Flujo de Trabajo Completo

### Escenario 1: Prueba Rápida (sin entrenar)

```bash
# Probar con modelo pre-entrenado
python scripts/use_pretrained_model.py --video input_video.mp4

# Ver resultado
vlc outputs/detected_input_video.mp4
```

**Tiempo:** 2-5 minutos

### Escenario 2: Entrenar con Roboflow (mejor precisión)

```bash
# 1. Descargar datasets
python scripts/download_roboflow_dataset.py \
    --api-key TU_API_KEY \
    --download-all

# 2. Entrenar modelo
python scripts/train_basketball_detector_simple.py

# 3. Probar modelo entrenado
python scripts/use_pretrained_model.py \
    --video input_video.mp4 \
    --model models/basketball_detector_yolo11l.pt
```

**Tiempo:** 1-3 horas (dependiendo de GPU)

### Escenario 3: Entrenar con tus Videos (máxima precisión)

```bash
# 1. Preparar video
cp mi_video.mp4 data/raw/

# 2. Extraer frames para anotar
python scripts/extract_frames_for_annotation.py \
    --video data/raw/mi_video.mp4 \
    --interval 10

# 3. Anotar en Roboflow (o manual)
# ... (subir, anotar, descargar)

# 4. Descargar anotaciones
python scripts/download_roboflow_dataset.py \
    --api-key TU_API_KEY \
    --workspace tu-workspace \
    --project mi-video \
    --version 1

# 5. Entrenar
python scripts/train_basketball_detector_simple.py

# 6. Probar
python scripts/use_pretrained_model.py \
    --video nuevo_video.mp4 \
    --model models/basketball_detector_yolo11l.pt
```

**Tiempo:** Medio día (incluye anotación manual)

---

## Solución de Problemas

### ❌ "ModuleNotFoundError: No module named 'roboflow'"

```bash
pip install roboflow
```

### ❌ "ModuleNotFoundError: No module named 'ultralytics'"

```bash
pip install ultralytics
```

### ❌ "Cannot open video"

```bash
# Verificar que el archivo existe
ls -lh data/raw/mi_video.mp4

# Convertir a MP4 si es necesario
ffmpeg -i video_original.avi -c:v libx264 data/raw/mi_video.mp4
```

### ❌ "CUDA out of memory"

```python
# Reducir batch size
trainer.train(epochs=50, batch_size=8)  # En vez de 16
```

### ❌ "Model accuracy is low"

1. **Más datos:** Anota más videos (objetivo: 1000+ frames)
2. **Mejor calidad:** Revisa y corrige anotaciones
3. **Más épocas:** Entrena por más tiempo (100-200 épocas)
4. **Modelo más grande:** Usa `yolo11x.pt` en vez de `yolo11l.pt`

---

## Recursos Adicionales

### Documentación Completa

- [README.md](README.md) - Documentación completa en inglés
- [GUIA_ANOTACION_VIDEOS.md](docs/GUIA_ANOTACION_VIDEOS.md) - Guía detallada de anotación
- [QUICKSTART.md](QUICKSTART.md) - Guía rápida en inglés

### Herramientas

- [Roboflow](https://roboflow.com/) - Anotación profesional
- [Roboflow Universe](https://universe.roboflow.com/) - Datasets públicos
- [Ultralytics YOLO](https://docs.ultralytics.com/) - Documentación de YOLO

### Tutoriales

- [Roboflow Basketball Tutorial](https://blog.roboflow.com/basketball-detection/)
- [YOLO Training Guide](https://docs.ultralytics.com/modes/train/)

---

## Comandos Útiles

```bash
# Ver información de video
ffprobe input_video.mp4

# Convertir video a MP4
ffmpeg -i video.avi -c:v libx264 -c:a aac output.mp4

# Extraer frames
python scripts/extract_frames_for_annotation.py --video input.mp4

# Listar datasets de Roboflow
python scripts/download_roboflow_dataset.py --list

# Usar modelo pre-entrenado
python scripts/use_pretrained_model.py --video input.mp4

# Entrenar modelo
python scripts/train_basketball_detector_simple.py

# Probar modelo entrenado
python scripts/test_basketball_model.py
```

---

## Ayuda

¿Problemas o preguntas?

1. Lee la documentación completa: [README.md](README.md)
2. Revisa la guía de anotación: [GUIA_ANOTACION_VIDEOS.md](docs/GUIA_ANOTACION_VIDEOS.md)
3. Abre un issue en GitHub

---

**¡Buena suerte con tu proyecto de detección de baloncesto!** 🏀🚀

**Última actualización:** Noviembre 2024
