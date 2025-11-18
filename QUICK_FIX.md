# 🔧 Quick Fix - Instalación Fallida

## Problema
El comando `conda env create -f environment_rtx.yml` falló con error de `faiss-gpu`.

## ✅ Solución Rápida (2 minutos)

Tu environment **ya está creado** y **CUDA funciona correctamente** (verificado ✓).
Solo faltan algunos paquetes Python. Instálalos así:

### Opción 1: Script Automático (Recomendado)

```powershell
# Asegúrate de estar en el environment
conda activate basketball_tracker_rtx

# Ejecuta el script de instalación
.\install_packages.ps1
```

### Opción 2: Manual

```powershell
conda activate basketball_tracker_rtx

pip install ultralytics>=8.3.0
pip install faiss-cpu>=1.8.0
pip install filterpy>=1.4.5
pip install deep-sort-realtime>=1.3.2
pip install boxmot>=10.0.0
pip install transformers>=4.40.0
pip install mplbasketball>=1.0.0
pip install py3nvml>=0.2.7
pip install gputil>=1.4.0
```

## ✓ Verificar Instalación

```powershell
# Debe imprimir "YOLO OK"
python -c "from ultralytics import YOLO; print('YOLO OK')"

# Debe imprimir "CUDA: True"
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Debe imprimir tu GPU
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
```

## 🚀 Siguiente Paso: Benchmark

```powershell
# Copia el archivo de configuración
copy config_performance.yaml.example config_performance.yaml

# Ejecuta benchmark rápido para encontrar tu batch size óptimo
python scripts/benchmark_gpu.py --quick
```

## 📊 Configuración para tu RTX 4060 Laptop

Edita `config_performance.yaml`:

```yaml
# Para RTX 4060 Laptop (8GB VRAM)
active_preset: 'balanced'

training:
  batch_size: 16  # Empieza con 16, el benchmark te dirá el óptimo
  workers: 4
  cache: true

inference:
  batch_size: 6
  half: true
```

## ❓ ¿Por qué falló faiss-gpu?

`faiss-gpu` tiene compatibilidad limitada con Python 3.11/3.12 en Windows. Usamos `faiss-cpu` que es:
- ✅ Compatible con todas las versiones de Python
- ✅ Suficientemente rápido para basketball tracking
- ✅ La diferencia con GPU es mínima para este caso de uso

## 📚 Más Ayuda

Revisa `OPTIMIZATION_GUIDE.md` para guía completa.
