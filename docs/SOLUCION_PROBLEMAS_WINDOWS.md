# 🔧 Solución de Problemas en Windows

Esta guía resuelve problemas comunes al ejecutar el proyecto en Windows.

## ❌ Error: "Initializing libiomp5md.dll already initialized"

### Descripción del Error

```
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program.
```

### Causa

Este error ocurre cuando múltiples librerías de Python (como NumPy, PyTorch, scikit-learn) incluyen sus propias copias del runtime de OpenMP de Intel. Es común en Windows cuando se usa Anaconda o múltiples paquetes científicos.

### ✅ Soluciones

#### Solución 1: Usar Script de Configuración (Recomendado)

##### Para PowerShell:

```powershell
# Ejecutar una vez
.\fix_windows_openmp.ps1

# Luego ejecutar tus scripts normalmente
python scripts/use_pretrained_model.py --video input_video.mp4
```

##### Para CMD:

```cmd
# Ejecutar una vez
fix_windows_openmp.bat

# Luego ejecutar tus scripts normalmente
python scripts\use_pretrained_model.py --video input_video.mp4
```

#### Solución 2: Configurar Variable Manualmente

##### En PowerShell:

```powershell
# Configurar para esta sesión
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# Ejecutar tu script
python scripts/use_pretrained_model.py --video input_video.mp4
```

##### En CMD:

```cmd
# Configurar para esta sesión
set KMP_DUPLICATE_LIB_OK=TRUE

# Ejecutar tu script
python scripts\use_pretrained_model.py --video input_video.mp4
```

#### Solución 3: Configurar Permanentemente

##### Opción A: Variables de Entorno del Sistema

1. **Abrir Configuración:**
   - Presiona `Win + R`
   - Escribe `sysdm.cpl` y presiona Enter
   - Ve a la pestaña "Avanzado"
   - Click en "Variables de entorno"

2. **Agregar Variable:**
   - En "Variables del sistema" click "Nueva"
   - Nombre: `KMP_DUPLICATE_LIB_OK`
   - Valor: `TRUE`
   - Click "Aceptar"

3. **Reiniciar PowerShell/CMD**

##### Opción B: Agregar a Perfil de PowerShell

```powershell
# Editar perfil de PowerShell
notepad $PROFILE

# Agregar esta línea al archivo:
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# Guardar y cerrar
# Reiniciar PowerShell
```

##### Opción C: Crear Script de Inicio

Crea un archivo `setup_env.ps1`:

```powershell
# setup_env.ps1
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
Write-Host "Entorno configurado para basketball_tracker" -ForegroundColor Green
```

Luego ejecuta antes de trabajar:
```powershell
.\setup_env.ps1
```

#### Solución 4: Modificar Scripts Directamente (No Recomendado)

Si las soluciones anteriores no funcionan, puedes agregar al inicio de cada script Python:

```python
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ... resto del código
```

**Nota:** Esta solución no es ideal porque modifica el código.

---

## 🚀 Guía Rápida de Uso en Windows

### 1. Configurar Entorno (una sola vez)

```powershell
# PowerShell
.\fix_windows_openmp.ps1
```

O permanentemente siguiendo "Solución 3".

### 2. Usar Modelo Pre-entrenado

```powershell
python scripts/use_pretrained_model.py --video input_video.mp4
```

### 3. Entrenar Modelo

```powershell
python scripts/train_basketball_detector_simple.py
```

### 4. Descargar Datasets

```powershell
python scripts/download_roboflow_dataset.py --list
python scripts/download_roboflow_dataset.py --api-key TU_API_KEY --download-all
```

---

## 🐛 Otros Problemas Comunes en Windows

### Error: "ModuleNotFoundError: No module named 'cv2'"

**Solución:**
```powershell
pip install opencv-python
```

### Error: "ModuleNotFoundError: No module named 'roboflow'"

**Solución:**
```powershell
pip install roboflow
```

### Error: "No module named 'torch'"

**Solución:**
```powershell
# Para CPU
pip install torch torchvision

# Para GPU (NVIDIA CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Error: "CUDA out of memory"

**Causa:** GPU sin suficiente memoria.

**Solución:**
```python
# Reducir batch size
# En scripts/train_basketball_detector_simple.py línea ~220
# Cambiar de:
batch_size=16
# A:
batch_size=8  # o incluso 4
```

### Error: Rutas con espacios no funcionan

**Problema:**
```powershell
python scripts/use_pretrained_model.py --video C:\Users\Mi Usuario\video.mp4
```

**Solución:**
```powershell
# Usar comillas
python scripts/use_pretrained_model.py --video "C:\Users\Mi Usuario\video.mp4"
```

### Error: Permisos para crear directorios

**Solución:**
```powershell
# Ejecutar PowerShell como Administrador
# O crear manualmente los directorios:
New-Item -ItemType Directory -Force -Path data\basketball_training
New-Item -ItemType Directory -Force -Path outputs
New-Item -ItemType Directory -Force -Path models\trained
```

---

## 💻 Configuración Recomendada para Windows

### 1. Usar Anaconda/Miniconda

```powershell
# Crear entorno
conda create -n basketball python=3.10
conda activate basketball

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Configurar Variable de Entorno Permanentemente

```powershell
# En PowerShell (Administrador)
[System.Environment]::SetEnvironmentVariable('KMP_DUPLICATE_LIB_OK', 'TRUE', 'User')
```

### 3. Verificar Instalación

```powershell
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "from ultralytics import YOLO; print('YOLO: OK')"
```

---

## 📊 Comparación GPU vs CPU en Windows

### Con GPU NVIDIA (CUDA):

```powershell
# Verificar CUDA
python -c "import torch; print('CUDA disponible:', torch.cuda.is_available())"

# Si devuelve True, puedes usar GPU
# Entrenamiento: 30-60 minutos
# Inferencia: Tiempo real (30+ FPS)
```

### Solo CPU:

```powershell
# Entrenamiento: 2-4 horas
# Inferencia: 5-10 FPS
```

**Recomendación:** Si tienes NVIDIA GPU, instala versión CUDA de PyTorch:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 🔍 Debugging en Windows

### Ver variables de entorno activas:

```powershell
# PowerShell
Get-ChildItem Env: | Where-Object {$_.Name -like "*KMP*"}

# CMD
set | findstr KMP
```

### Ver versiones instaladas:

```powershell
pip list | Select-String "torch|opencv|ultralytics|roboflow"
```

### Limpiar cache de pip:

```powershell
pip cache purge
```

### Reinstalar dependencias:

```powershell
pip uninstall -y torch torchvision opencv-python ultralytics
pip install -r requirements.txt
```

---

## 📝 Checklist de Configuración Inicial

Para evitar problemas en Windows:

```
□ Python 3.8+ instalado
□ pip actualizado (python -m pip install --upgrade pip)
□ Variable KMP_DUPLICATE_LIB_OK configurada
□ requirements.txt instalado
□ Directorios creados (data/, outputs/, models/)
□ Git instalado (opcional)
□ CUDA instalado (si tienes NVIDIA GPU)
```

Verificar todo:

```powershell
# Verificar Python
python --version

# Verificar pip
pip --version

# Verificar variable de entorno
$env:KMP_DUPLICATE_LIB_OK

# Verificar librerías
python -c "import torch, cv2, ultralytics; print('OK')"

# Verificar CUDA (si tienes GPU)
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 🚀 Script de Configuración Automática

Crea `setup_windows.ps1`:

```powershell
# setup_windows.ps1 - Configuración automática para Windows

Write-Host "Configurando entorno para Basketball Tracker..." -ForegroundColor Cyan

# 1. Configurar variable de entorno
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
Write-Host "[OK] Variable KMP_DUPLICATE_LIB_OK configurada" -ForegroundColor Green

# 2. Crear directorios necesarios
$directories = @(
    "data\raw",
    "data\basketball_training",
    "data\annotations",
    "data\detections",
    "data\verified",
    "data\frames_to_annotate",
    "outputs",
    "models\trained",
    "models\pretrained"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
        Write-Host "[OK] Creado: $dir" -ForegroundColor Green
    }
}

# 3. Verificar instalación
Write-Host "`nVerificando instalación..." -ForegroundColor Yellow

try {
    python -c "import torch; print('PyTorch:', torch.__version__)" 2>$null
    Write-Host "[OK] PyTorch instalado" -ForegroundColor Green
} catch {
    Write-Host "[!] PyTorch no instalado" -ForegroundColor Red
}

try {
    python -c "import cv2; print('OpenCV:', cv2.__version__)" 2>$null
    Write-Host "[OK] OpenCV instalado" -ForegroundColor Green
} catch {
    Write-Host "[!] OpenCV no instalado" -ForegroundColor Red
}

try {
    python -c "from ultralytics import YOLO" 2>$null
    Write-Host "[OK] Ultralytics instalado" -ForegroundColor Green
} catch {
    Write-Host "[!] Ultralytics no instalado" -ForegroundColor Red
}

Write-Host "`n[OK] Configuración completa!" -ForegroundColor Green
Write-Host "`nPuedes empezar a usar los scripts:" -ForegroundColor Cyan
Write-Host "  python scripts\use_pretrained_model.py --video input_video.mp4" -ForegroundColor White
```

**Uso:**
```powershell
.\setup_windows.ps1
```

---

## 📞 Soporte

Si sigues teniendo problemas:

1. Revisa que la variable `KMP_DUPLICATE_LIB_OK` esté configurada
2. Verifica que todas las dependencias estén instaladas
3. Intenta en un entorno virtual limpio (conda o venv)
4. Abre un issue en GitHub con:
   - Versión de Windows
   - Versión de Python
   - Output de `pip list`
   - Mensaje de error completo

---

**Última actualización:** Noviembre 2024
