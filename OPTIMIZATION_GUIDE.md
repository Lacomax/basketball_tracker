# Basketball Tracker - Optimización para RTX GPU

Guía de optimización para Windows + Conda + AMD Ryzen 9 + NVIDIA RTX GPU

## 🚀 Instalación Optimizada

### 1. Crear Entorno Conda con CUDA

```bash
# Crear entorno optimizado para RTX
conda env create -f environment_rtx.yml

# Activar entorno
conda activate basketball_tracker_rtx
```

### 2. Verificar Instalación de CUDA

```bash
# Verificar que PyTorch detecta la GPU
python -c "import torch; print('CUDA disponible:', torch.cuda.is_available())"
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
python -c "import torch; print('CUDA version:', torch.version.cuda)"
```

Deberías ver algo como:
```
CUDA disponible: True
GPU: NVIDIA GeForce RTX 4090
CUDA version: 12.1
```

### 3. Configurar Ultralytics (YOLO) para GPU

```bash
# Verificar que YOLO detecta la GPU
python -c "from ultralytics import YOLO; import torch; print('Devices:', torch.cuda.device_count())"
```

## ⚙️ Configuración de Rendimiento

### Archivo de Configuración: `config_performance.yaml`

Este archivo contiene todas las optimizaciones para tu hardware. Las configuraciones principales son:

#### Para RTX 4090 / 4080 (24GB VRAM):
```yaml
active_preset: 'max_performance'
training:
  batch_size: 48
  workers: 8
  cache: true
```

#### Para RTX 4070 / 3080 / 3090 (10-12GB VRAM):
```yaml
active_preset: 'balanced'
training:
  batch_size: 24
  workers: 6
  cache: true
```

#### Para RTX 3060 / 3070 (8GB VRAM):
```yaml
active_preset: 'memory_efficient'
training:
  batch_size: 16
  workers: 4
  cache: 'disk'
```

### Características de Optimización

1. **Mixed Precision (FP16)**: Usa los Tensor Cores de RTX para entrenamiento 2-3x más rápido
2. **Batch Processing**: Procesa múltiples frames simultáneamente para mejor utilización de GPU
3. **Persistent Workers**: Reduce overhead de procesos en Windows
4. **Smart Caching**: Cachea datasets en RAM para acceso más rápido
5. **cuDNN Auto-tuner**: Encuentra los mejores algoritmos para tu GPU

## 📊 Benchmark de Rendimiento

Antes de entrenar, ejecuta el benchmark para encontrar la configuración óptima:

```bash
# Benchmark completo (15-20 minutos)
python scripts/benchmark_gpu.py

# Benchmark rápido (5 minutos)
python scripts/benchmark_gpu.py --quick

# Solo inferencia (sin entrenamiento)
python scripts/benchmark_gpu.py --skip-training
```

Esto te mostrará:
- FPS de inferencia con diferentes batch sizes
- Uso de memoria
- Configuración óptima para tu GPU
- Comparación FP16 vs FP32

## 🎯 Entrenamiento Optimizado

### Script Optimizado

Usa el nuevo script optimizado en lugar del original:

```bash
# Con configuración balanceada (recomendado)
python scripts/train_basketball_detector_optimized.py --preset balanced

# Máximo rendimiento (requiere GPU potente)
python scripts/train_basketball_detector_optimized.py --preset max_performance

# Bajo uso de memoria
python scripts/train_basketball_detector_optimized.py --preset memory_efficient

# Más épocas para mejor precisión
python scripts/train_basketball_detector_optimized.py --preset balanced --epochs 150
```

### Ventajas del Script Optimizado:

- ✅ Detecta automáticamente las capacidades de tu GPU
- ✅ Usa FP16 (Mixed Precision) en GPUs RTX
- ✅ Batch size optimizado para tu VRAM
- ✅ Workers configurados para Windows
- ✅ Monitoreo de memoria GPU en tiempo real
- ✅ Estimación de VRAM antes de iniciar
- ✅ Opción de exportar a TensorRT al finalizar

### Tiempo Estimado de Entrenamiento

Con RTX GPU y optimizaciones:
- **RTX 4090**: ~15-20 min (100 épocas, batch 48)
- **RTX 4080**: ~20-25 min (100 épocas, batch 32)
- **RTX 3080**: ~25-35 min (100 épocas, batch 24)
- **RTX 3060**: ~40-50 min (100 épocas, batch 16)

Sin optimizaciones (CPU o FP32): 3-5 horas

## 🔥 Inferencia Ultra-Rápida

### Opción 1: FP16 (Recomendado)

```python
from ultralytics import YOLO

model = YOLO('models/basketball_detector_yolo11l.pt')

# Inferencia con FP16 (2x más rápido)
results = model(frame, half=True, device=0)
```

### Opción 2: TensorRT (Máximo Rendimiento)

TensorRT puede dar hasta 5x más velocidad, pero requiere instalación adicional.

#### Instalar TensorRT:

```bash
# Opción 1: Con pip (más fácil)
pip install tensorrt

# Opción 2: Desde NVIDIA (más control)
# Descargar de: https://developer.nvidia.com/tensorrt
# Elegir versión compatible con CUDA 12.1
```

#### Exportar Modelo a TensorRT:

```python
from ultralytics import YOLO

# Cargar modelo entrenado
model = YOLO('models/basketball_detector_yolo11l.pt')

# Exportar a TensorRT con FP16
model.export(format='engine', half=True, device=0)

# Esto crea: basketball_detector_yolo11l.engine
```

#### Usar TensorRT:

```python
# Cargar engine de TensorRT
model = YOLO('models/basketball_detector_yolo11l.engine')

# ¡Inferencia ultra-rápida!
results = model(frame)
```

### Opción 3: Batch Processing

Para procesar videos, usa batch processing:

```python
from src.utils.ball_detection_optimized import BatchBallDetector

detector = BatchBallDetector(batch_size=16)

# Procesar múltiples frames a la vez
frames = [frame1, frame2, frame3, ...]  # lista de frames
results = detector.detect_batch(frames)
```

## 💡 Optimizaciones Adicionales para Windows

### 1. Prioridad de Proceso

Ejecuta Python con alta prioridad (abre cmd como administrador):

```bash
start /high python scripts/train_basketball_detector_optimized.py
```

O configura en `config_performance.yaml`:
```yaml
windows:
  process_priority: 'high'
```

### 2. Desactivar Windows Defender (Opcional)

Para máximo rendimiento, excluye la carpeta del proyecto de Windows Defender:

1. Configuración de Windows → Actualización y seguridad → Seguridad de Windows
2. Protección contra virus y amenazas → Configuración
3. Exclusiones → Agregar exclusión → Carpeta
4. Seleccionar carpeta del proyecto

### 3. Modo de Alto Rendimiento

1. Panel de Control → Opciones de energía
2. Seleccionar "Alto rendimiento"
3. O crear plan personalizado con CPU al 100%

### 4. Temperatura GPU

Monitorea la temperatura con MSI Afterburner o HWiNFO:
- Temperatura ideal: 60-75°C
- Máximo seguro: 85°C
- Si llega a 85°C+: mejora ventilación o reduce batch size

## 📈 Comparación de Rendimiento

### Entrenamiento (100 épocas, YOLO11-L)

| Configuración | Tiempo | Speedup |
|--------------|--------|---------|
| CPU (Ryzen 9) | ~4-5 horas | 1x |
| RTX GPU (FP32) | ~60 min | 4-5x |
| RTX GPU (FP16) | ~20-30 min | 8-12x |

### Inferencia (640x640 frames)

| Configuración | FPS | Latency |
|--------------|-----|---------|
| CPU | ~5 FPS | 200ms |
| RTX (FP32, batch=1) | ~45 FPS | 22ms |
| RTX (FP16, batch=1) | ~90 FPS | 11ms |
| RTX (FP16, batch=8) | ~180 FPS | 5.5ms |
| RTX + TensorRT | ~300+ FPS | 3ms |

## 🛠️ Troubleshooting

### Error: "CUDA out of memory"

**Solución**: Reduce el batch size en `config_performance.yaml`

```yaml
training:
  batch_size: 16  # Prueba valores más bajos: 12, 8
```

### Error: "CUDA not available"

**Solución**: Reinstala PyTorch con CUDA

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### Entrenamiento Lento en Windows

**Solución**: Reduce workers

```yaml
training:
  workers: 4  # Windows tiene overhead con multiprocessing
```

### GPU no se usa al 100%

**Posibles causas**:
1. Batch size muy pequeño → aumentar
2. Workers muy bajos → aumentar (pero no mucho en Windows)
3. CPU es cuello de botella → usar cache: true
4. Disco lento → usar SSD o cache: 'disk'

### Verificar Uso de GPU

```bash
# En otra terminal, monitorea la GPU
nvidia-smi -l 1
```

Deberías ver:
- GPU-Util cerca de 100%
- Memory-Usage alto pero no 100%
- Power cerca del máximo (TDP)

## 📝 Checklist de Optimización

- [ ] Conda environment con CUDA instalado
- [ ] CUDA funciona (verificado con python)
- [ ] config_performance.yaml configurado para tu GPU
- [ ] Benchmark ejecutado para encontrar batch size óptimo
- [ ] Windows Defender excluye la carpeta (opcional)
- [ ] Modo de alto rendimiento activado
- [ ] nvidia-smi muestra utilización alta durante entrenamiento
- [ ] Temperatura GPU bajo control (<85°C)

## 🎓 Próximos Pasos

1. **Ejecuta el benchmark**:
   ```bash
   python scripts/benchmark_gpu.py
   ```

2. **Ajusta config_performance.yaml** con los resultados

3. **Entrena el modelo**:
   ```bash
   python scripts/train_basketball_detector_optimized.py
   ```

4. **Exporta a TensorRT** (opcional, máximo rendimiento):
   ```python
   from ultralytics import YOLO
   model = YOLO('models/basketball_detector_yolo11l.pt')
   model.export(format='engine', half=True)
   ```

5. **Usa el modelo optimizado** en tus scripts

## 📚 Recursos Adicionales

- [Ultralytics Docs - Performance](https://docs.ultralytics.com/guides/model-optimization/)
- [PyTorch Mixed Precision](https://pytorch.org/docs/stable/amp.html)
- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt)
- [cuDNN Documentation](https://docs.nvidia.com/deeplearning/cudnn/)

## ❓ Preguntas Frecuentes

**P: ¿Cuánta VRAM necesito?**
R: Mínimo 6GB para entrenamiento básico, 8GB+ recomendado, 12GB+ para batch sizes grandes.

**P: ¿FP16 afecta la precisión?**
R: En YOLO, FP16 tiene impacto mínimo (<0.5% mAP) pero 2-3x más rápido.

**P: ¿Vale la pena TensorRT?**
R: Sí para producción, no necesario para experimentación. Da ~3-5x speedup extra.

**P: ¿Puedo usar múltiples GPUs?**
R: Sí, configura en config_performance.yaml: `devices: [0, 1]`

**P: Mi GPU es GTX (no RTX), ¿funcionan estas optimizaciones?**
R: Sí, pero sin Tensor Cores. FP16 puede no dar speedup. Usa FP32.

---

¿Problemas o preguntas? Abre un issue en GitHub.
