# 🎯 Configuración Roboflow y Modelos Pre-entrenados

Este documento describe las nuevas funcionalidades agregadas al proyecto para facilitar el uso de Roboflow y modelos pre-entrenados.

## 📦 Nuevos Scripts Agregados

### 1. `scripts/download_roboflow_dataset.py`

**Propósito:** Descarga automática de datasets de baloncesto desde Roboflow.

**Características:**
- ✅ Descarga datasets públicos de baloncesto
- ✅ Lista de datasets recomendados pre-configurados
- ✅ Descarga múltiple automática
- ✅ Organización automática de archivos
- ✅ Estadísticas de descarga

**Uso básico:**
```bash
# Listar datasets recomendados
python scripts/download_roboflow_dataset.py --list

# Descargar todos los datasets recomendados
python scripts/download_roboflow_dataset.py --api-key YOUR_API_KEY --download-all

# Descargar dataset específico
python scripts/download_roboflow_dataset.py \
    --api-key YOUR_API_KEY \
    --workspace roboflow-100 \
    --project basketball-detection \
    --version 1
```

**Datasets recomendados incluidos:**
1. Basketball Detection (Roboflow 100) - Dataset oficial
2. Basketball Object Detection - Múltiples ángulos
3. Basketball Ball Detection - Enfocado en el balón

### 2. `scripts/use_pretrained_model.py`

**Propósito:** Usar modelos pre-entrenados para probar sin necesidad de entrenar.

**Características:**
- ✅ Usa modelos YOLO pre-entrenados (detecta "sports ball")
- ✅ Soporta modelos personalizados entrenados
- ✅ Configuración de umbral de confianza
- ✅ Múltiples tamaños de modelo (nano a extra-large)
- ✅ Visualización de detecciones en video

**Uso básico:**
```bash
# Usar YOLO pre-entrenado genérico
python scripts/use_pretrained_model.py --video input_video.mp4

# Usar modelo personalizado
python scripts/use_pretrained_model.py \
    --video input_video.mp4 \
    --model models/basketball_detector.pt

# Ajustar confianza
python scripts/use_pretrained_model.py \
    --video input_video.mp4 \
    --conf 0.5

# Usar modelo más grande (mejor precisión)
python scripts/use_pretrained_model.py \
    --video input_video.mp4 \
    --yolo-model yolo11x.pt
```

**Modelos YOLO disponibles:**
- `yolo11n.pt` - Nano (rápido, menos preciso)
- `yolo11s.pt` - Small
- `yolo11m.pt` - Medium
- `yolo11l.pt` - Large (recomendado, por defecto)
- `yolo11x.pt` - Extra Large (más preciso, más lento)

### 3. `scripts/extract_frames_for_annotation.py`

**Propósito:** Extraer frames de videos para anotación profesional en Roboflow.

**Características:**
- ✅ Extracción a intervalos configurables
- ✅ Límite de frames opcional
- ✅ Soporte para JPG y PNG
- ✅ Control de calidad de compresión
- ✅ Barra de progreso

**Uso básico:**
```bash
# Extraer un frame cada 10
python scripts/extract_frames_for_annotation.py \
    --video input_video.mp4 \
    --interval 10

# Extraer solo 100 frames
python scripts/extract_frames_for_annotation.py \
    --video input_video.mp4 \
    --max-frames 100

# Extraer en PNG (mejor calidad)
python scripts/extract_frames_for_annotation.py \
    --video input_video.mp4 \
    --format png

# Personalizar salida
python scripts/extract_frames_for_annotation.py \
    --video input_video.mp4 \
    --output mis_frames/ \
    --interval 5
```

## 📚 Documentación Agregada

### 1. `docs/GUIA_ANOTACION_VIDEOS.md`

**Guía completa en español sobre cómo anotar videos.**

**Contenido:**
- ✅ Introducción a la anotación
- ✅ Preparación del video
- ✅ Métodos de anotación (manual y Roboflow)
- ✅ Uso de la herramienta incluida
- ✅ Uso de Roboflow profesional
- ✅ Mejores prácticas
- ✅ Solución de problemas
- ✅ Flujo de trabajo completo
- ✅ Scripts de ejemplo

**Secciones destacadas:**
- Controles de las herramientas de anotación
- Guía paso a paso con Roboflow
- Estrategias de anotación eficiente
- Checklist de calidad
- Troubleshooting común

### 2. `GUIA_RAPIDA_ES.md`

**Guía rápida de inicio en español.**

**Contenido:**
- ✅ Instalación rápida
- ✅ Uso de modelos pre-entrenados
- ✅ Descarga de datasets de Roboflow
- ✅ Anotación de videos propios
- ✅ Entrenamiento de modelos
- ✅ Flujos de trabajo completos
- ✅ Solución de problemas comunes
- ✅ Comandos útiles

## 🚀 Flujos de Trabajo

### Flujo 1: Prueba Rápida (Sin Entrenar)

**Tiempo:** 5 minutos

```bash
# 1. Usar modelo pre-entrenado
python scripts/use_pretrained_model.py --video input_video.mp4

# 2. Ver resultado
# Abre: outputs/detected_input_video.mp4
```

**Ventajas:**
- ⚡ Muy rápido
- 💻 No requiere GPU potente
- 🎯 Bueno para pruebas iniciales

**Limitaciones:**
- Precisión moderada (modelo genérico)
- Puede no detectar balones en condiciones difíciles

---

### Flujo 2: Entrenar con Datasets de Roboflow

**Tiempo:** 2-4 horas (dependiendo de GPU)

```bash
# 1. Instalar dependencias
pip install roboflow

# 2. Obtener API key de Roboflow.com
# (Crear cuenta gratis y copiar API key)

# 3. Descargar datasets
python scripts/download_roboflow_dataset.py \
    --api-key YOUR_API_KEY \
    --download-all

# 4. Entrenar modelo
python scripts/train_basketball_detector_simple.py

# 5. Probar modelo entrenado
python scripts/use_pretrained_model.py \
    --video input_video.mp4 \
    --model models/basketball_detector_yolo11l.pt
```

**Ventajas:**
- 🎯 Alta precisión
- 📊 Datasets profesionales
- 🔄 Reproducible

**Requisitos:**
- GPU recomendada (CUDA)
- 2-4 GB espacio en disco
- 2-4 horas de tiempo

---

### Flujo 3: Entrenar con tus Propios Videos

**Tiempo:** Medio día (incluye anotación)

#### Opción A: Anotación Rápida (Herramienta Incluida)

```bash
# 1. Preparar video
cp mi_video.mp4 data/raw/

# 2. Anotar manualmente
python -c "from src.modules.annotator import BallAnnotator; \
           BallAnnotator('data/raw/mi_video.mp4', 'data/annotations/mi_video.json').run()"

# 3. Interpolar
python -c "from src.modules.trajectory_detector import process_trajectory_video; \
           process_trajectory_video('data/raw/mi_video.mp4', \
                                    'data/annotations/mi_video.json', \
                                    'data/detections/mi_video.json')"

# 4. Verificar
python -c "from src.modules.verifier import CompactBallVerifier; \
           CompactBallVerifier('data/raw/mi_video.mp4', \
                              'data/detections/mi_video.json', \
                              'data/verified/mi_video.json').run()"

# 5. Entrenar
python -c "from src.modules.yolo_trainer import UltraYOLOBallTrainer; \
           trainer = UltraYOLOBallTrainer('data/raw/mi_video.mp4', \
                                          'data/verified/mi_video.json', \
                                          'models/trained/mi_detector', \
                                          'yolo11l.pt'); \
           trainer.train(epochs=50, batch_size=16, img_size=640)"
```

#### Opción B: Anotación Profesional (Roboflow)

```bash
# 1. Extraer frames
python scripts/extract_frames_for_annotation.py \
    --video data/raw/mi_video.mp4 \
    --interval 10

# 2. Anotar en Roboflow
# - Subir frames a roboflow.com
# - Anotar balón con bounding boxes
# - Generar dataset en formato YOLOv8

# 3. Descargar dataset anotado
python scripts/download_roboflow_dataset.py \
    --api-key YOUR_API_KEY \
    --workspace tu-workspace \
    --project mi-video \
    --version 1

# 4. Entrenar
python scripts/train_basketball_detector_simple.py

# 5. Probar
python scripts/use_pretrained_model.py \
    --video nuevo_video.mp4 \
    --model models/basketball_detector_yolo11l.pt
```

**Ventajas:**
- 🎯 Máxima precisión para tus videos
- 🔧 Control total sobre datos
- 📈 Mejora continua

**Requisitos:**
- Tiempo para anotar (200-500 frames por video)
- GPU recomendada para entrenamiento
- Paciencia para iteración

---

## 🔧 Configuración Inicial

### 1. Instalar Dependencias

```bash
# Dependencias base (si no están instaladas)
pip install -r requirements.txt

# Roboflow (nuevo)
pip install roboflow
```

### 2. Obtener API Key de Roboflow

1. Ve a [https://roboflow.com/](https://roboflow.com/)
2. Crea una cuenta gratuita
3. Ve a **Settings** → **API Keys**
4. Copia tu **Private API Key**

**Límites de cuenta gratuita:**
- 1000 imágenes de entrenamiento
- 3 proyectos
- Suficiente para empezar

### 3. Verificar Instalación

```bash
# Verificar YOLO
python -c "from ultralytics import YOLO; print('✅ YOLO OK')"

# Verificar Roboflow
python -c "from roboflow import Roboflow; print('✅ Roboflow OK')"

# Verificar OpenCV
python -c "import cv2; print('✅ OpenCV OK')"
```

---

## 📊 Comparación de Métodos

| Método | Tiempo | Precisión | Complejidad | Costo |
|--------|--------|-----------|-------------|-------|
| **Pre-entrenado** | 5 min | Media | Muy baja | Gratis |
| **Roboflow Datasets** | 2-4h | Alta | Baja | Gratis (cuenta básica) |
| **Tus Videos (Manual)** | 1 día | Muy alta | Media | Gratis |
| **Tus Videos (Roboflow)** | 4-8h | Muy alta | Baja-Media | Gratis (cuenta básica) |

**Recomendaciones:**
- 🚀 **Principiantes:** Empezar con pre-entrenado
- 🎯 **Proyectos serios:** Roboflow Datasets
- 💪 **Máxima precisión:** Tus propios videos con Roboflow
- 💡 **Presupuesto cero:** Herramientas incluidas + Roboflow gratis

---

## 🎯 Mejores Prácticas

### Para Descarga de Datasets

1. **Empieza con datasets recomendados**
   - Usa `--download-all` para obtener todos
   - Combina múltiples datasets para mejor generalización

2. **Verifica la descarga**
   ```bash
   # Ver estadísticas
   ls -lh data/basketball_training/*/train/images/ | wc -l
   ```

3. **Organiza tus datos**
   ```
   data/
   ├── basketball_training/     # Datasets de Roboflow
   ├── raw/                      # Tus videos originales
   ├── annotations/              # Anotaciones manuales
   └── verified/                 # Listo para entrenar
   ```

### Para Uso de Modelos Pre-entrenados

1. **Ajusta el umbral según necesidad**
   - Más detecciones: `--conf 0.2`
   - Menos falsos positivos: `--conf 0.6`
   - Balance: `--conf 0.4` (recomendado)

2. **Elige el modelo correcto**
   - Pruebas rápidas: `yolo11n.pt`
   - Producción: `yolo11l.pt` o `yolo11x.pt`

3. **Evalúa resultados**
   - Mira el video completo
   - Cuenta falsos positivos/negativos
   - Decide si necesitas entrenar

### Para Anotación

1. **Cantidad mínima recomendada**
   - Prueba: 500 frames
   - Producción: 2000+ frames
   - Múltiples videos: 5-10 videos variados

2. **Diversidad es clave**
   - Diferentes ángulos
   - Diferentes iluminaciones
   - Diferentes velocidades de balón
   - Diferentes distancias de cámara

3. **Calidad sobre cantidad**
   - Mejor 500 anotaciones precisas
   - Que 2000 anotaciones imprecisas

---

## 🐛 Solución de Problemas Comunes

### Error: "roboflow module not found"

```bash
pip install roboflow
```

### Error: "Invalid API key"

1. Verifica que copiaste la API key correcta
2. Usa la **Private API Key**, no la pública
3. Crea una cuenta nueva si es necesario

### Error: "Dataset not found"

1. Verifica que el workspace/project existen
2. Ve a la URL: `https://universe.roboflow.com/WORKSPACE/PROJECT`
3. Usa los nombres exactos (case-sensitive)

### Modelo no detecta bien

1. **Usa modelo más grande:**
   ```bash
   python scripts/use_pretrained_model.py --video input.mp4 --yolo-model yolo11x.pt
   ```

2. **Entrena con más datos:**
   - Descarga todos los datasets de Roboflow
   - Añade tus propios videos anotados

3. **Ajusta umbral:**
   ```bash
   python scripts/use_pretrained_model.py --video input.mp4 --conf 0.3
   ```

### Entrenamiento lento

1. **Verifica GPU:**
   ```python
   import torch
   print(torch.cuda.is_available())  # Debe ser True
   ```

2. **Reduce batch size:**
   - Cambiar de `batch_size=16` a `batch_size=8`
   - O incluso `batch_size=4` para GPUs pequeñas

3. **Usa modelo más pequeño:**
   - `yolo11m.pt` en vez de `yolo11l.pt`

---

## 📈 Próximos Pasos

Después de configurar Roboflow y probar los modelos:

1. **Experimenta con diferentes datasets**
   - Prueba cada dataset individual
   - Compara resultados

2. **Anota tus propios videos**
   - Empieza con 1-2 videos cortos
   - Itera y mejora

3. **Optimiza el pipeline**
   - Ajusta parámetros de entrenamiento
   - Experimenta con augmentations
   - Fine-tune modelos pre-entrenados

4. **Comparte tus resultados**
   - Publica tu modelo en Roboflow Universe
   - Contribuye al proyecto

---

## 📚 Referencias

- [Roboflow Documentation](https://docs.roboflow.com/)
- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [Roboflow Universe](https://universe.roboflow.com/)
- [Basketball Detection Tutorial](https://blog.roboflow.com/basketball-detection/)

---

## 🤝 Contribuir

Si mejoras estos scripts o documentación:

1. Abre un Pull Request
2. Describe tus cambios
3. Comparte tus resultados

---

**¡Disfruta entrenando tu modelo de detección de baloncesto!** 🏀🚀

**Creado:** Noviembre 2024
**Última actualización:** Noviembre 2024
