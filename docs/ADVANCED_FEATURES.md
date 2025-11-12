# Advanced Basketball Tracker - Guía Completa

## 🎯 Visión General

El sistema avanzado de análisis de baloncesto proporciona análisis completo de partidos, incluyendo:

- 🏀 **Tracking del balón** con detección mejorada de oclusiones
- 👥 **Detección de jugadores** con asignación automática de equipos
- 📊 **Análisis de eventos** (canastas, pases, botes, rebotes)
- 📈 **Estadísticas por jugador** (tiros, asistencias, distancia recorrida)
- 💾 **Base de datos SQLite** para persistencia de datos

---

## 🚀 Inicio Rápido

### Análisis Completo de un Partido

```bash
python -m src.advanced_tracker \
    --video data/raw/partido.mp4 \
    --annotations data/annotations/partido_annotations.json \
    --output outputs/mi_analisis \
    --db data/stats.db
```

Esto ejecutará el pipeline completo:
1. Detección del balón con manejo de oclusiones
2. Detección y tracking de jugadores
3. Análisis de eventos (canastas, pases, botes)
4. Generación de estadísticas
5. Persistencia en base de datos

---

## 📦 Módulos Principales

### 1. Detección Mejorada de Oclusiones

**¿Qué es?** Es muy común que el balón quede tapado por jugadores durante el juego. El sistema ahora detecta estas oclusiones.

**Características:**
- Detección de alta velocidad (indica movimiento rápido o oclusión)
- Detección de aceleración repentina (rebotes o cambios bruscos)
- Sistema de confianza que decae durante predicciones largas
- Marcado automático de frames con baja confianza

**Archivo:** `src/modules/trajectory_detector.py`

**Ejemplo de uso:**
```python
from src.modules.trajectory_detector import process_trajectory_video

detections = process_trajectory_video(
    video_path="partido.mp4",
    annotations_path="anotaciones.json",
    output_path="detecciones.json"
)

# Cada detección incluye:
# - center: [x, y]
# - radius: int
# - confidence: float (0-1)
# - velocity: float
# - occluded: bool (si está tapado)
# - occlusion_reason: "high_velocity" o "high_acceleration"
```

---

### 2. Detección de Jugadores

**¿Qué hace?** Detecta jugadores usando YOLOv8 y opcionalmente asigna equipos según el color de la camiseta.

**Características:**
- Detección de personas con YOLO
- Pose estimation opcional (para detectar acciones)
- Asignación automática de equipos por color de camiseta
- Tracking de jugadores entre frames

**Archivo:** `src/modules/player_detector.py`

**Ejemplo de uso:**
```python
from src.modules.player_detector import PlayerDetector

detector = PlayerDetector()

# Procesar video completo
detections = detector.process_video(
    video_path="partido.mp4",
    output_path="jugadores.json",
    use_pose=True,          # Activar pose estimation
    detect_teams=True       # Detectar equipos
)

# Para un frame individual
players = detector.detect_players(frame, conf_threshold=0.5)
for player in players:
    print(f"Jugador {player.player_id} en {player.center}, Equipo: {player.team}")
```

---

### 3. Análisis de Eventos

**¿Qué detecta?**
- 🏀 **Tiros** (shots) - Detecta trayectorias parabólicas del balón
- 🤝 **Pases** - Detecta movimiento horizontal rápido entre jugadores
- ⛹️ **Botes** (dribbles) - Detecta patrón de rebote del balón
- 📥 **Rebotes** (próximamente)
- 🛡️ **Robos** (próximamente)

**Archivo:** `src/modules/event_analyzer.py`

**Ejemplo de uso:**
```python
from src.modules.event_analyzer import EventAnalyzer

# Cargar detecciones
with open('balones.json') as f:
    ball_detections = json.load(f)
with open('jugadores.json') as f:
    player_detections = json.load(f)

# Analizar eventos
analyzer = EventAnalyzer(ball_detections, player_detections)
events = analyzer.analyze_all_events()

# Filtrar eventos específicos
shots = analyzer.detect_shots(window_size=60)
passes = analyzer.detect_passes()
dribbles = analyzer.detect_dribbles()

# Guardar eventos
analyzer.save_events('eventos.json')
```

**Formato de eventos:**
```json
{
  "event_type": "shot",
  "frame_start": 1234,
  "frame_end": 1294,
  "player_id": 5,
  "ball_trajectory": [[x1, y1], [x2, y2], ...],
  "confidence": 0.85,
  "metadata": {"height_change": 150}
}
```

---

### 4. Generación de Estadísticas

**¿Qué genera?** Estadísticas completas por jugador basadas en los eventos detectados.

**Métricas incluidas:**
- Tiros: intentados, anotados, porcentaje
- Pases y asistencias
- Botes y rebotes totales
- Tiempo en cancha (frames visibles)
- Distancia recorrida (en píxeles)
- Robos y tapones

**Archivo:** `src/modules/statistics_generator.py`

**Ejemplo de uso:**
```python
from src.modules.statistics_generator import StatisticsGenerator

generator = StatisticsGenerator(
    events_file='eventos.json',
    players_file='jugadores.json'
)

# Generar todas las estadísticas
stats = generator.generate_all_statistics()

# Guardar estadísticas
generator.save_statistics('estadisticas.json')

# Generar reporte legible
generator.save_summary_report('reporte.txt')
```

**Ejemplo de estadísticas:**
```
============================================================
BASKETBALL GAME STATISTICS SUMMARY
============================================================

Team_0
------------------------------------------------------------

Player #1:
  Shots: 5/12 (41.7%)
  Passes: 23
  Dribbles: 15 (45 bounces)
  Time on court: 2340 frames
  Distance traveled: 15420.5 pixels

Player #2:
  Shots: 8/15 (53.3%)
  Passes: 18
  Dribbles: 22 (68 bounces)
  Time on court: 2890 frames
  Distance traveled: 18950.2 pixels
```

---

### 5. Base de Datos SQLite

**¿Para qué?** Almacena todos los datos de forma persistente para análisis histórico.

**Tablas:**
- `games` - Información de partidos
- `players` - Datos de jugadores
- `player_statistics` - Estadísticas por jugador
- `events` - Eventos del juego
- `ball_detections` - Detecciones del balón (caché)

**Archivo:** `src/utils/database.py`

**Ejemplo de uso:**
```python
from src.utils.database import BasketballDatabase

# Usar como context manager
with BasketballDatabase('stats.db') as db:
    # Insertar partido
    game_id = db.insert_game(
        video_path='partido.mp4',
        total_frames=5000,
        duration_seconds=180.0
    )

    # Insertar jugador
    db.insert_player(game_id, player_id=10, team='Lakers', name='Tu hijo')

    # Consultar estadísticas
    stats = db.get_player_statistics(game_id, player_id=10)
    print(stats)

    # Obtener eventos
    shots = db.get_events(game_id, event_type='shot')
    print(f"Total de tiros: {len(shots)}")
```

---

## 🎮 Casos de Uso

### Caso 1: Analizar un Partido Completo

```bash
# 1. Primero anotar manualmente algunos frames clave del balón
python -m src.modules.annotator --video partido.mp4 --output anotaciones.json

# 2. Ejecutar análisis completo
python -m src.advanced_tracker \
    --video partido.mp4 \
    --annotations anotaciones.json \
    --output outputs/partido1 \
    --pose \
    --db stats.db

# 3. Ver resultados
cat outputs/partido1/statistics_report.txt
```

### Caso 2: Solo Detectar Jugadores

```python
from src.modules.player_detector import PlayerDetector

detector = PlayerDetector()
detections = detector.process_video(
    video_path='partido.mp4',
    output_path='solo_jugadores.json',
    use_pose=False,
    detect_teams=True
)
```

### Caso 3: Analizar Solo Tiros

```python
from src.modules.event_analyzer import EventAnalyzer
import json

with open('balones.json') as f:
    ball = json.load(f)
with open('jugadores.json') as f:
    players = json.load(f)

analyzer = EventAnalyzer(ball, players)
shots = analyzer.detect_shots(window_size=60)

print(f"Se detectaron {len(shots)} tiros")
for shot in shots:
    print(f"Tiro en frame {shot.frame_start} por jugador {shot.player_id}")
```

### Caso 4: Comparar Estadísticas de Múltiples Partidos

```python
from src.utils.database import BasketballDatabase

with BasketballDatabase('stats.db') as db:
    # Obtener todos los partidos
    games = db.get_all_games()

    for game in games:
        print(f"\nPartido: {game['video_path']}")

        # Estadísticas del partido
        stats = db.get_player_statistics(game['game_id'])

        for stat in stats:
            player_id = stat['player_id']
            shooting_pct = stat['shooting_percentage']
            print(f"  Jugador {player_id}: {shooting_pct:.1f}% tiros")
```

---

## ❓ Preguntas Frecuentes

### ¿Hay una base de datos de balones de baloncesto?

**Respuesta:** YOLO viene preentrenado con datasets como COCO que incluyen "sports ball", pero **no es específico para balones de baloncesto naranjas**. Por eso este sistema:

1. Requiere **anotación manual inicial** de algunos frames
2. **Entrena un modelo YOLO específico** con tus videos
3. Usa **detección por color y forma** (círculos) para mejorar precisión

Para mejores resultados, puedes usar modelos preentrenados en deportes si los tienes.

### ¿Es normal que el balón esté tapado por jugadores?

**¡Absolutamente sí!** Esto se llama **"oclusión"** y es uno de los mayores desafíos. El sistema lo maneja:

- ✅ Detección de oclusiones por alta velocidad
- ✅ Detección de oclusiones por aceleración repentina
- ✅ Filtro de Kalman para predecir posición durante oclusiones
- ✅ Sistema de confianza que marca detecciones de baja calidad
- ✅ Flags específicos: `occluded: true` y `occlusion_reason`

### ¿Cómo mejoro la precisión de detección?

1. **Anotar más frames clave** manualmente (más datos = mejor modelo)
2. **Usar pose estimation** para detectar mejor las acciones
3. **Ajustar umbrales** en `src/config.py`
4. **Entrenar más épocas** el modelo YOLO
5. **Usar un modelo YOLO más grande** (yolov8m o yolov8l en vez de yolov8n)

### ¿Puedo procesar múltiples videos en lote?

Sí, con un script simple:

```python
from src.advanced_tracker import AdvancedBasketballTracker
import os

videos = ['partido1.mp4', 'partido2.mp4', 'partido3.mp4']

for video in videos:
    annotations = video.replace('.mp4', '_annotations.json')
    output_dir = f'outputs/{video[:-4]}'

    tracker = AdvancedBasketballTracker(video, output_dir)
    tracker.run_full_analysis(annotations)
```

---

## 🔧 Optimizaciones de Rendimiento

El sistema incluye varias optimizaciones:

### 1. Procesamiento en Lotes
```python
# En yolo_trainer.py - procesa frames en lotes
trainer.extract_frames(batch_size=32)  # Procesa 32 frames a la vez
```

### 2. Caché de Frames Preprocesados
```python
# En ball_detection.py - caché automático
from src.utils.ball_detection import clear_cache

# Liberar memoria si es necesario
clear_cache()
```

### 3. Bulk Insert en Base de Datos
```python
# Inserciones masivas son más rápidas
db.bulk_insert_ball_detections(game_id, all_detections)
```

### 4. Context Managers
```python
# Cierre automático de recursos
with BasketballDatabase('stats.db') as db:
    # Trabajo con la base de datos
    pass  # Se cierra automáticamente
```

---

## 📊 Formatos de Datos

### Ball Detections JSON
```json
{
  "0": {
    "center": [640, 360],
    "radius": 12,
    "confidence": 1.0,
    "velocity": 0.0
  },
  "1": {
    "center": [642, 358],
    "radius": 12,
    "confidence": 0.95,
    "velocity": 2.83,
    "occluded": false
  }
}
```

### Player Detections JSON
```json
{
  "0": [
    {
      "player_id": 1,
      "bbox": [100, 200, 150, 350],
      "confidence": 0.92,
      "center": [125, 275],
      "team": "Team_0",
      "keypoints": [[x1, y1], [x2, y2], ...]
    }
  ]
}
```

### Events JSON
```json
[
  {
    "event_type": "shot",
    "frame_start": 100,
    "frame_end": 160,
    "player_id": 5,
    "confidence": 0.85,
    "ball_trajectory": [[x1, y1], [x2, y2], ...],
    "metadata": {"height_change": 150}
  }
]
```

---

## 🎓 Próximos Pasos

Para extender el sistema:

1. **Detectar la canasta** - Añadir detección de la canasta para determinar tiros anotados vs fallados
2. **Mejorar tracking** - Implementar algoritmos como DeepSORT para tracking más robusto
3. **Análisis táctico** - Detectar formaciones, jugadas, zonas defensivas
4. **Visualización** - Crear videos con overlay de estadísticas en tiempo real
5. **Machine Learning avanzado** - Predecir próximas jugadas o resultado de tiros

---

## 📞 Soporte

Para problemas o preguntas:
1. Revisa los logs en la consola
2. Verifica que los archivos JSON de entrada existan
3. Asegúrate de tener suficiente espacio en disco
4. Consulta la documentación principal en `README.md`

---

**¡Disfruta analizando los partidos de tu hijo!** 🏀
