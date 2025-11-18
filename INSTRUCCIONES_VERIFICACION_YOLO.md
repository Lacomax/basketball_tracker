# 🔍 Instrucciones para Verificar Detecciones YOLO

## ¿Por qué usar este script?

Este script te permite ver **TODAS las detecciones de YOLO** sin ningún filtro de distancia o validación. Esto es útil para:

1. ✅ Verificar si el modelo entrenado está detectando correctamente el balón
2. ✅ Identificar falsos positivos (detecta cosas que no son balones)
3. ✅ Ver dónde YOLO está funcionando bien y dónde falla
4. ✅ Decidir si necesitas más entrenamiento o más anotaciones manuales

## 📋 Cómo Ejecutar

### 1. Activa tu entorno conda (si usas conda):

```powershell
conda activate basketball_tracker
# O si usaste otro nombre:
conda activate rtx_env
```

### 2. Ejecuta el script de verificación:

**Opción A: Generar video + JSON con detecciones**

```powershell
python scripts/verify_yolo_detections.py --save-video
```

**Opción B: Cambiar umbral de confianza**

```powershell
# Usar confianza más alta (menos detecciones, más seguras)
python scripts/verify_yolo_detections.py --save-video --conf 0.25

# Usar confianza más baja (más detecciones, incluye menos seguras)
python scripts/verify_yolo_detections.py --save-video --conf 0.10
```

**Opción C: Usar otro video**

```powershell
python scripts/verify_yolo_detections.py --video data/otro_video.mp4 --save-video
```

## 📊 Qué Esperar

### Archivos Generados:

1. **`outputs/yolo_detections_raw.mp4`** - Video anotado con:
   - Bounding boxes VERDES = Basketball (class 0, tu modelo entrenado)
   - Bounding boxes AMARILLOS = Otros objetos detectados
   - Centro del balón marcado con círculo
   - Confianza de cada detección

2. **`outputs/yolo_detections_raw.json`** - Detecciones en formato JSON:
   ```json
   {
     "1": [
       {
         "class": "basketball",
         "class_id": 0,
         "bbox": [x1, y1, x2, y2],
         "center": [cx, cy],
         "confidence": 0.85,
         "width": 30,
         "height": 32
       }
     ]
   }
   ```

### Estadísticas en Consola:

```
======================================================================
📊 ESTADÍSTICAS DE DETECCIÓN YOLO
======================================================================
Total de frames procesados: 323
Frames con detecciones: 250 (77.4%)

Detecciones totales: 280
  - Basketball (class 0): 280
  - Otros objetos: 0
  - Alta confianza (>0.5): 150 (53.6%)

Detecciones por frame:
  - Promedio: 1.12
  - Máximo: 3
  - Mínimo: 1

⚠️ Frames con múltiples detecciones: 25
   Ejemplos: [10, 25, 42, 58, 91, ...]
======================================================================

💡 RECOMENDACIONES:
✓ Buen ratio de detección: 280 detecciones
⚠️ Solo 53.6% de las detecciones tienen alta confianza (>0.5)
   → Considera aumentar el umbral de confianza
   → El modelo podría necesitar más entrenamiento
```

## 🔍 Cómo Interpretar los Resultados

### ✅ Señales de que el modelo está funcionando BIEN:

- **70-90%** de los frames tienen detecciones
- **>50%** de las detecciones tienen confianza >0.5
- Pocas detecciones múltiples por frame (<10%)
- Las detecciones están centradas en el balón (verifica en el video)

### ❌ Señales de PROBLEMAS:

#### Problema 1: "NO se detectó ningún basketball"
```
❌ NO se detectó ningún basketball!
   → Verifica que el modelo esté entrenado correctamente
```

**Solución:**
- Verifica que `models/best.pt` existe y es tu modelo entrenado
- Re-entrena el modelo con más datos
- Verifica que las anotaciones de entrenamiento sean correctas

#### Problema 2: "Pocas detecciones" (<30% de frames)
```
⚠️ Solo 80 detecciones de basketball en 323 frames
   → El modelo podría necesitar más entrenamiento
```

**Solución:**
- Descarga más datasets desde Roboflow
- Agrega tus propias anotaciones al dataset de entrenamiento
- Entrena por más epochs (aumenta `epochs` en `train_basketball_detector_simple.py`)

#### Problema 3: "Baja confianza" (<50% alta confianza)
```
⚠️ Solo 30.0% de las detecciones tienen alta confianza (>0.5)
   → El modelo podría necesitar más entrenamiento
```

**Solución:**
- Usa umbral de confianza más alto: `--conf 0.25` o `--conf 0.30`
- Re-entrena el modelo con más datos
- Usa más augmentations durante entrenamiento

#### Problema 4: "Muchas detecciones múltiples" (>30%)
```
⚠️ Muchos frames con múltiples detecciones: 100
   → Podrían ser falsos positivos
```

**Solución:**
- Usa NMS (Non-Maximum Suppression) más agresivo
- Aumenta el umbral de confianza
- Filtra detecciones por tamaño (balón debe ser 20-50px)

## 🎥 Revisar el Video Generado

### 1. Abre el video:

```powershell
# Windows
start outputs/yolo_detections_raw.mp4

# O usando VLC/Media Player
```

### 2. Qué buscar en el video:

**✅ BUENO:**
- Bounding box verde sigue al balón en la mayoría de frames
- El centro (círculo verde) está dentro del balón
- Confianza >0.3 en la mayoría de detecciones
- Pocas detecciones cuando el balón NO está visible

**❌ MALO:**
- Bounding box verde aparece en objetos que NO son balones
- Múltiples cajas verdes en el mismo frame
- Confianza muy baja (<0.2) en la mayoría de detecciones
- El balón está visible pero NO hay detección

### 3. Capturas de pantalla:

Si ves algo extraño, toma capturas de pantalla de esos frames y compártelas.

## 📝 Comparar con Detecciones Actuales

### Script actual (con filtros):

```powershell
# Esto es lo que usa el pipeline normal
python scripts/pipeline.py
```

- ❌ Solo 2.8% detecciones YOLO
- ❌ Muchas rechazadas por estar lejos de la predicción

### Script de verificación (SIN filtros):

```powershell
# Esto muestra TODO lo que YOLO detecta
python scripts/verify_yolo_detections.py --save-video
```

- ✅ Muestra el 100% de lo que YOLO detecta
- ✅ Sin filtros de distancia

**Si el script de verificación muestra 70-90% de detecciones:**
→ El modelo está bien, el problema es la interpolación/anotaciones manuales
→ Solución: Agregar más anotaciones manuales (especialmente frames 229-323)

**Si el script de verificación muestra <30% de detecciones:**
→ El modelo necesita más entrenamiento
→ Solución: Más datos, más epochs, mejor dataset

## 🚀 Próximos Pasos

1. **Ejecuta el script:**
   ```powershell
   python scripts/verify_yolo_detections.py --save-video
   ```

2. **Revisa las estadísticas en la consola**

3. **Revisa el video generado:** `outputs/yolo_detections_raw.mp4`

4. **Decide:**
   - Si YOLO detecta bien (>70%): Agregar más anotaciones manuales
   - Si YOLO detecta mal (<30%): Re-entrenar el modelo

5. **Comparte los resultados:**
   - Copia las estadísticas de la consola
   - Comparte el archivo JSON si es necesario
   - Describe qué ves en el video

## ❓ Preguntas Frecuentes

**P: ¿Por qué hay cajas amarillas?**
R: Son otros objetos que YOLO detecta (no basketball). Si hay muchos, podría ser ruido.

**P: ¿Qué es una buena tasa de detección?**
R: 70-90% de frames con detecciones. Menos de 50% indica problemas.

**P: ¿Qué umbral de confianza usar?**
R: Empieza con 0.15 (default). Si hay muchos falsos positivos, sube a 0.25-0.30.

**P: El video muestra detecciones pero el pipeline no las usa. ¿Por qué?**
R: Porque el pipeline filtra detecciones que están lejos de la predicción interpolada. Esto significa que necesitas más anotaciones manuales para que la interpolación sea más precisa.

**P: ¿Cuánto tarda en procesar?**
R: ~5-10 segundos por frame en GPU RTX. Un video de 323 frames tarda ~30-50 minutos.
