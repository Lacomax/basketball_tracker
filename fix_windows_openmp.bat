@echo off
REM Script para solucionar el error de OpenMP en Windows
REM Error: "Initializing libiomp5md.dll, but found libiomp5md.dll already initialized"

echo ================================================
echo Solucionando error de OpenMP en Windows
echo ================================================
echo.

REM Configurar variable de entorno para esta sesión
set KMP_DUPLICATE_LIB_OK=TRUE

echo [OK] Variable KMP_DUPLICATE_LIB_OK configurada
echo.
echo Ahora puedes ejecutar los scripts normalmente:
echo.
echo   python scripts/train_basketball_detector_simple.py
echo   python scripts/use_pretrained_model.py --video input_video.mp4
echo.

REM Mantener la ventana abierta
cmd /k
