# 🔍 Guía Rápida: Solucionar "No datasets found"

## Situación Actual

Has descargado exitosamente el dataset:
```
✓ Dataset descargado: data/basketball_training\basketball-detection_v1
```

Pero el script de entrenamiento no lo encuentra:
```
❌ No datasets found!
```

## 🚀 Solución Paso a Paso

### Paso 1: Diagnosticar el Problema

Ejecuta el nuevo script de diagnóstico:

```powershell
python scripts/diagnose_datasets.py
```

Este script te dirá:
- ✅ Qué encuentra en `data/basketball_training/`
- ✅ Si la estructura es correcta
- ✅ Exactamente qué archivos tiene cada directorio
- ✅ Cuántas imágenes y labels hay

**Ejemplo de salida esperada:**
```
📂 basketball-detection_v1/
------------------------------------------------------------------
  ✓ data.yaml
  ✓ train/
  ✓ train/images/
      → 450 imágenes
  ✓ train/labels/
      → 450 archivos de etiquetas
  ✓ valid/
  ✓ valid/images/
      → 150 imágenes
  ✓ valid/labels/
      → 150 archivos de etiquetas

  ✅ VÁLIDO - Dataset YOLO detectado
```

### Paso 2: Soluciones Según el Problema

#### Problema A: Estructura Incorrecta

Si el diagnóstico muestra que falta `train/images/` o `data.yaml`:

**Solución: Verificar estructura de Roboflow**

Roboflow a veces crea estructuras anidadas. Verifica si tu estructura es:

```
data/basketball_training/
└── basketball-detection_v1/
    └── basketball-detection_v1/   ← Estructura anidada
        ├── data.yaml
        ├── train/
        └── valid/
```

Si es así, mueve el contenido un nivel arriba:

```powershell
# PowerShell
Move-Item -Path "data\basketball_training\basketball-detection_v1\basketball-detection_v1\*" -Destination "data\basketball_training\basketball-detection_v1\"
Remove-Item -Path "data\basketball_training\basketball-detection_v1\basketball-detection_v1" -Recurse
```

#### Problema B: Dataset en Subdirectorio Incorrecto

Si descargaste a una ubicación diferente:

```powershell
# Verificar dónde está realmente
Get-ChildItem -Path . -Filter "data.yaml" -Recurse

# Mover al lugar correcto si está en otro lado
Move-Item -Path "ruta\actual\dataset" -Destination "data\basketball_training\dataset"
```

#### Problema C: Directorio Vacío

Si el directorio `data/basketball_training/` está vacío o no existe:

```powershell
# Crear directorio
New-Item -ItemType Directory -Force -Path data\basketball_training

# Descargar dataset de nuevo
python scripts/download_roboflow_dataset.py --api-key TU_API_KEY --download-all
```

### Paso 3: Volver a Intentar Entrenamiento

Después de solucionar, prueba de nuevo:

```powershell
# 1. Solucionar error OpenMP primero
.\fix_windows_openmp.ps1

# 2. Ejecutar script de entrenamiento (ahora con más diagnóstico)
python scripts/train_basketball_detector_simple.py
```

El script mejorado ahora te dirá:
```
[Step 1/3] Looking for datasets in data/basketball_training/...
   Buscando datasets en: C:\Users\...\basketball_tracker\data\basketball_training
   Encontrados 1 item(s) en el directorio

   Verificando: basketball-detection_v1/
      - data.yaml: ✓
      - train/: ✓
      - train/images/: ✓
      ✅ Dataset válido encontrado

   ✓ Found 1 dataset(s):
     - basketball-detection_v1
```

## 🔧 Solución Rápida Alternativa

Si nada funciona, puedes entrenar directamente con el dataset sin combinarlo:

### Opción A: Entrenar Directamente con YOLO

```powershell
# En PowerShell
python -c "from ultralytics import YOLO; model = YOLO('yolo11l.pt'); model.train(data='data/basketball_training/basketball-detection_v1/data.yaml', epochs=50, batch=16, imgsz=640, name='basketball_detector', device=0)"
```

### Opción B: Crear Script Personalizado

Crea `train_direct.py`:

```python
from ultralytics import YOLO
import os

# Configurar variable OpenMP
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Entrenar directamente
model = YOLO('yolo11l.pt')

results = model.train(
    data='data/basketball_training/basketball-detection_v1/data.yaml',
    epochs=50,
    batch=16,
    imgsz=640,
    name='basketball_detector',
    patience=10,
    device=0,  # 0 para GPU, 'cpu' para CPU
    project='runs/basketball'
)

print("\n✅ Entrenamiento completo!")
print(f"Modelo guardado en: runs/basketball/basketball_detector/weights/best.pt")
```

Ejecuta:

```powershell
python train_direct.py
```

## 📊 Verificación Final

Después de cualquier solución, verifica:

```powershell
# 1. Diagnosticar
python scripts/diagnose_datasets.py

# 2. Si sale ✅, entrenar
python scripts/train_basketball_detector_simple.py
```

## 💡 Consejos

1. **Siempre ejecuta primero `diagnose_datasets.py`** - Te ahorra tiempo
2. **Verifica la estructura** - Debe ser `data/basketball_training/dataset_name/train/images/`
3. **Revisa el error OpenMP** - Ejecuta `.\fix_windows_openmp.ps1` antes de entrenar
4. **Si estás en CPU** - Cambia `device=0` a `device='cpu'` en los scripts

## 🆘 Si Nada Funciona

1. **Borra todo y empieza de nuevo:**

```powershell
# Limpiar
Remove-Item -Path data\basketball_training\* -Recurse -Force

# Descargar de nuevo
python scripts/download_roboflow_dataset.py --api-key TU_API_KEY --download-all

# Diagnosticar
python scripts/diagnose_datasets.py

# Entrenar
python scripts/train_basketball_detector_simple.py
```

2. **Comparte el output del diagnóstico:**

```powershell
python scripts/diagnose_datasets.py > diagnostico.txt
# Envía diagnostico.txt para ayuda
```

3. **Prueba entrenamiento directo con YOLO** (Opción A arriba)

---

**Siguiente paso:** Ejecuta `python scripts/diagnose_datasets.py` y comparte el output si necesitas más ayuda.
