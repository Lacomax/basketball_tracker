# setup_windows.ps1 - Configuración automática para Windows
# Basketball Tracker - Windows Setup Script

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Basketball Tracker - Configuración Windows" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Configurar variable de entorno para OpenMP
Write-Host "[1/4] Configurando variables de entorno..." -ForegroundColor Yellow
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
Write-Host "  [OK] KMP_DUPLICATE_LIB_OK = TRUE" -ForegroundColor Green

# 2. Crear directorios necesarios
Write-Host "`n[2/4] Creando estructura de directorios..." -ForegroundColor Yellow

$directories = @(
    "data\raw",
    "data\basketball_training",
    "data\annotations",
    "data\detections",
    "data\verified",
    "data\frames_to_annotate",
    "outputs",
    "models\trained",
    "models\pretrained",
    "runs"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
        Write-Host "  [OK] $dir" -ForegroundColor Green
    } else {
        Write-Host "  [OK] $dir (ya existe)" -ForegroundColor Gray
    }
}

# 3. Verificar Python y dependencias
Write-Host "`n[3/4] Verificando instalación de Python..." -ForegroundColor Yellow

try {
    $pythonVersion = python --version 2>&1
    Write-Host "  [OK] $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "  [ERROR] Python no encontrado" -ForegroundColor Red
    Write-Host "  Instala Python 3.8+ desde: https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}

# 4. Verificar librerías críticas
Write-Host "`n[4/4] Verificando librerías..." -ForegroundColor Yellow

$libraries = @(
    @{Name="torch"; Import="import torch; print(torch.__version__)"; DisplayName="PyTorch"},
    @{Name="cv2"; Import="import cv2; print(cv2.__version__)"; DisplayName="OpenCV"},
    @{Name="ultralytics"; Import="from ultralytics import YOLO; print('OK')"; DisplayName="Ultralytics YOLO"},
    @{Name="roboflow"; Import="from roboflow import Roboflow; print('OK')"; DisplayName="Roboflow"}
)

$allInstalled = $true

foreach ($lib in $libraries) {
    try {
        $result = python -c $lib.Import 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  [OK] $($lib.DisplayName)" -ForegroundColor Green
        } else {
            Write-Host "  [!] $($lib.DisplayName) - No instalado" -ForegroundColor Yellow
            $allInstalled = $false
        }
    } catch {
        Write-Host "  [!] $($lib.DisplayName) - No instalado" -ForegroundColor Yellow
        $allInstalled = $false
    }
}

# 5. Verificar CUDA (opcional)
Write-Host "`n[Opcional] Verificando soporte CUDA..." -ForegroundColor Yellow
try {
    $cudaAvailable = python -c "import torch; print(torch.cuda.is_available())" 2>&1
    if ($cudaAvailable -eq "True") {
        Write-Host "  [OK] CUDA disponible - Puedes usar GPU" -ForegroundColor Green
        $cudaVersion = python -c "import torch; print(torch.version.cuda)" 2>&1
        Write-Host "  [OK] CUDA Version: $cudaVersion" -ForegroundColor Green
    } else {
        Write-Host "  [INFO] CUDA no disponible - Usarás CPU" -ForegroundColor Gray
        Write-Host "  Para GPU, instala: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118" -ForegroundColor Gray
    }
} catch {
    Write-Host "  [INFO] No se pudo verificar CUDA" -ForegroundColor Gray
}

# Resumen
Write-Host "`n================================================" -ForegroundColor Cyan
Write-Host "  Resumen de Configuración" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

if ($allInstalled) {
    Write-Host "`n[OK] ¡Todo está configurado correctamente!" -ForegroundColor Green
    Write-Host "`nPuedes empezar a usar el proyecto:" -ForegroundColor White
    Write-Host ""
    Write-Host "  1. Probar modelo pre-entrenado:" -ForegroundColor Yellow
    Write-Host "     python scripts\use_pretrained_model.py --video input_video.mp4" -ForegroundColor White
    Write-Host ""
    Write-Host "  2. Descargar datasets de Roboflow:" -ForegroundColor Yellow
    Write-Host "     python scripts\download_roboflow_dataset.py --list" -ForegroundColor White
    Write-Host ""
    Write-Host "  3. Entrenar modelo:" -ForegroundColor Yellow
    Write-Host "     python scripts\train_basketball_detector_simple.py" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host "`n[!] Algunas librerías faltan" -ForegroundColor Yellow
    Write-Host "`nInstala las dependencias faltantes:" -ForegroundColor White
    Write-Host "  pip install -r requirements.txt" -ForegroundColor White
    Write-Host ""
    Write-Host "Luego ejecuta este script de nuevo:" -ForegroundColor White
    Write-Host "  .\setup_windows.ps1" -ForegroundColor White
    Write-Host ""
}

Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Documentación útil:" -ForegroundColor Cyan
Write-Host "  - Guía rápida: GUIA_RAPIDA_ES.md" -ForegroundColor White
Write-Host "  - Problemas Windows: docs\SOLUCION_PROBLEMAS_WINDOWS.md" -ForegroundColor White
Write-Host "  - README completo: README.md" -ForegroundColor White
Write-Host ""

# Nota sobre la variable de entorno
Write-Host "IMPORTANTE:" -ForegroundColor Yellow
Write-Host "La variable KMP_DUPLICATE_LIB_OK solo está configurada para esta sesión." -ForegroundColor Yellow
Write-Host "Si abres una nueva ventana de PowerShell, ejecuta:" -ForegroundColor Yellow
Write-Host "  .\fix_windows_openmp.ps1" -ForegroundColor White
Write-Host ""
Write-Host "O configúrala permanentemente siguiendo:" -ForegroundColor Yellow
Write-Host "  docs\SOLUCION_PROBLEMAS_WINDOWS.md" -ForegroundColor White
Write-Host ""
