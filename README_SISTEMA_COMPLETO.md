# Basketball Tracker - Sistema Completo

Sistema profesional de tracking y análisis de basketball con detección personalizada, seguimiento de jugadores, y visualización con perspectiva exacta.

## 🎯 Características Principales

### ✅ Detección de Jugadores
- Tracking automático con YOLO
- Filtrado por ROI (solo jugadores en cancha)
- Auto-merge de IDs duplicados
- Asignación de nombres y equipos
- 10 jugadores únicos rastreados

### ✅ Detección del Balón
- Modelo YOLO custom entrenado (500 imágenes)
- Detección automática (83.9%)
- Anotación manual para frames difíciles (10.5%)
- Interpolación inteligente (5.6%)
- Suavizado con Kalman + Savitzky-Golay

### ✅ Visualización de Canasta
- Tablero con perspectiva exacta (marcado manual)
- Cuadro interior/target box
- Aro con efecto 3D
- Red con malla detallada
- Todo siguiendo la perspectiva de la cámara

---

## 🚀 Uso Rápido

### Pipeline Completo (Recomendado)

```bash
python scripts/pipeline_improved.py
```

El pipeline te guiará paso a paso:

1. **Filtrar ROI** - Marca el área de la cancha
2. **Asignar nombres** - Nombra a los jugadores
3. **Asignar equipos** - Clasifica por equipo
4. **Anotar balón** - Marca posiciones clave del balón (opcional)
5. **Anotar canasta** - Marca el centro del aro
6. **Marcar tablero** - Define las 4 esquinas del tablero
7. **Marcar cuadro interior** - Define el target box
8. **Generar trayectoria** - Detecta automáticamente el balón
9. **Crear video final** - Genera el video completo

### Comandos Individuales

#### 1. Entrenar Modelo Custom
```bash
python scripts/train_custom_model.py
```

#### 2. Detectar Trayectoria del Balón
```bash
python -m src.modules.trajectory_detector --video data/input_video.mp4
```

#### 3. Visualizar Trayectoria
```bash
python scripts/visualize_ball_trajectory.py
```

#### 4. Auto-Merge de Jugadores
```bash
python scripts/auto_merge_players.py
```

#### 5. Marcar Tablero y Cuadro Interior
```bash
# Tablero exterior
python scripts/mark_backboard_corners.py

# Cuadro interior
python scripts/mark_inner_box.py
```

#### 6. Crear Video Final
```bash
python scripts/add_hoop_with_marked_backboard.py
```

---

## 📁 Estructura del Proyecto

```
basketball_tracker/
├── config.yaml                    # Configuración central
├── data/
│   └── input_video.mp4           # Video de entrada
├── outputs/                       # Resultados generados
│   ├── final_video_COMPLETO.mp4  # ⭐ VIDEO FINAL
│   ├── backboard.json            # Coordenadas del tablero
│   ├── detections.json           # Detecciones del balón
│   ├── tracked_players_named.json # Jugadores con nombres
│   └── ...
├── models/
│   └── basketball_detector_custom.pt # Modelo YOLO entrenado
├── scripts/                       # Scripts de pipeline
│   ├── pipeline_improved.py      # Pipeline principal
│   ├── train_custom_model.py     # Entrenar modelo
│   ├── mark_backboard_corners.py # Marcar tablero
│   ├── mark_inner_box.py         # Marcar cuadro interior
│   ├── auto_merge_players.py     # Consolidar jugadores
│   └── ...
└── src/                          # Código fuente
    ├── modules/                  # Módulos principales
    ├── utils/                    # Utilidades
    └── ...
```

---

## ⚙️ Configuración (config.yaml)

### Detección del Balón
```yaml
ball:
  model_path: "models/basketball_detector_custom.pt"
  confidence_threshold: 0.3
  min_size: 15  # Tamaño mínimo en pixels
  max_size: 60  # Tamaño máximo en pixels
  yolo_conf_primary: 0.15
  yolo_conf_fallback: 0.05
```

### Suavizado de Trayectoria
```yaml
smoothing:
  kalman:
    process_variance: 0.25      # Más bajo = más suave
    measurement_variance: 10.0  # Más alto = más suave
  savgol:
    window_length: 17           # Ventana para suavizado
    polyorder: 3                # Orden del polinomio
```

### Tracking de Jugadores
```yaml
tracking:
  max_players_per_frame: 10  # 4 red + 4 yellow + 2 referees
  confidence_threshold: 0.7
```

---

## 📊 Archivos de Salida

### Videos Generados

| Archivo | Descripción |
|---------|-------------|
| `final_video_COMPLETO.mp4` | ⭐ Video final con todo (jugadores, balón, tablero con perspectiva exacta) |
| `final_video_clean.mp4` | Video sin marcador de canasta |
| `ball_trajectory_visualization.mp4` | Solo trayectoria del balón con métodos de detección |

### Datos JSON

| Archivo | Contenido |
|---------|-----------|
| `backboard.json` | Esquinas del tablero y cuadro interior marcadas manualmente |
| `hoop.json` | Centro y radio de la canasta |
| `detections.json` | Todas las detecciones del balón (323 frames) |
| `tracked_players_named.json` | Jugadores con nombres (10 únicos) |
| `player_names.json` | Mapeo de IDs a nombres |
| `team_assignments.json` | Equipos asignados |
| `court_roi.json` | Área de la cancha |

---

## 🎨 Personalización

### Cambiar Colores del Equipo
Edita `config.yaml`:
```yaml
team_colors:
  default:
    - [255, 0, 0]     # Azul
    - [0, 255, 0]     # Verde
    - [0, 0, 255]     # Rojo
    # ... más colores
```

### Ajustar Suavizado de Trayectoria
Más suave:
```yaml
smoothing:
  kalman:
    process_variance: 0.2     # Más bajo
    measurement_variance: 12.0  # Más alto
  savgol:
    window_length: 21         # Más grande (debe ser impar)
```

Menos suave (más preciso):
```yaml
smoothing:
  kalman:
    process_variance: 0.3
    measurement_variance: 8.0
  savgol:
    window_length: 13
```

### Cambiar Tamaño de Detección del Balón
```yaml
ball:
  min_size: 10  # Más pequeño
  max_size: 80  # Más grande
```

---

## 🏀 Resultados del Sistema

### Jugadores Rastreados (10 únicos)

**EQUIPO ROJO (4 jugadores):**
- ID 1: RED 5
- ID 2: RED 10
- ID 3: RED 11
- ID 4: RED 9 Mateo

**EQUIPO AMARILLO (4 jugadores):**
- ID 5: YEL 11
- ID 7: YEL 4
- ID 9: YEL 7
- ID 12: YEL 8

**ÁRBITROS (1):**
- ID 20: referee 1

**SIN NOMBRE (1):**
- ID 34: Player 34

### Detección del Balón (323 frames)

- ✅ **83.9% Auto-detectado** - Modelo YOLO custom
- ✅ **10.5% Manual** - Anotaciones en frames difíciles
- ✅ **5.6% Interpolado** - Relleno inteligente

### Modelo YOLO Custom

- **Dataset:** 500 imágenes (34 manuales + 466 filtradas)
- **Arquitectura:** YOLOv11 nano
- **Entrenado:** 100 epochs con early stopping
- **Rendimiento:** Detecta balones de 15-60px

---

## 🛠️ Solución de Problemas

### Error: OpenMP library already initialized
**Solución:** Ya configurado automáticamente
```bash
setx KMP_DUPLICATE_LIB_OK TRUE
```

### Error: Video no encontrado
**Solución:** Coloca el video en `data/input_video.mp4` o actualiza `config.yaml`

### Jugadores del público aparecen
**Solución:** Redefine el ROI más estricto
```bash
python scripts/redefine_roi.py
```

### Demasiados IDs duplicados
**Solución:** Usa auto-merge
```bash
python scripts/auto_merge_players.py
```

### Trayectoria del balón muy ruidosa
**Solución:** Ajusta parámetros de suavizado en `config.yaml`

---

## 📝 Notas Técnicas

### Requisitos de Hardware
- **GPU:** NVIDIA RTX (recomendado) - Usado para entrenamiento YOLO
- **RAM:** 8GB mínimo, 16GB recomendado
- **Espacio:** ~2GB para modelo + outputs

### Dependencias Principales
- Python 3.12
- PyTorch 2.5.1 con CUDA
- Ultralytics YOLOv11
- OpenCV (cv2)
- NumPy, SciPy

### Rendimiento
- **Procesamiento de video:** ~70 frames/seg
- **Entrenamiento YOLO:** ~41 epochs en RTX 4060 Laptop
- **Detección en tiempo real:** Posible con GPU

---

## 🎯 Próximos Pasos

### Para Producción
1. Entrenar con más datos (1000+ imágenes)
2. Implementar tracking multi-cámara
3. Agregar análisis de estadísticas (tiros, pases, etc.)
4. Exportar a formatos estándar (JSON, CSV, XML)

### Para Mejorar Precisión
1. Usar Re-ID (Re-Identification) para jugadores
2. Implementar filtro de partículas para tracking
3. Agregar detección de eventos (tiros, rebotes, etc.)
4. Calibración de cámara para métricas reales

---

## 📄 Licencia

Este proyecto usa:
- YOLOv11 (Ultralytics) - AGPL-3.0
- OpenCV - Apache 2.0

---

## 🙏 Créditos

Desarrollado con:
- **Custom YOLO training** - Transfer learning de YOLOv11 nano
- **Kalman filtering** - Suavizado de trayectorias
- **Manual perspective marking** - Precisión visual exacta
- **Auto-merge algorithm** - Consolidación de jugadores

---

## 📞 Soporte

Para problemas o preguntas:
1. Verifica este README
2. Revisa los logs en `logs/basketball_tracker.log`
3. Ejecuta con `--log-level DEBUG` para más información

---

**¡Sistema listo para usar! 🏀🎬**

Video final: `outputs/final_video_COMPLETO.mp4`
