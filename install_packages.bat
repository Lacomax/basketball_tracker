@echo off
REM Quick fix script to install missing packages
REM Run this if environment creation failed

echo ================================================
echo Installing missing packages for Basketball Tracker
echo ================================================
echo.

echo Installing Ultralytics (YOLO)...
pip install ultralytics>=8.3.0

echo Installing tracking libraries...
pip install filterpy>=1.4.5
pip install deep-sort-realtime>=1.3.2
pip install boxmot>=10.0.0

echo Installing transformers...
pip install transformers>=4.40.0

echo Installing faiss (CPU version - compatible with all Python versions)...
pip install faiss-cpu>=1.8.0

echo Installing visualization libraries...
pip install mplbasketball>=1.0.0

echo Installing performance monitoring...
pip install py3nvml>=0.2.7
pip install gputil>=1.4.0

echo.
echo ================================================
echo Installation complete!
echo ================================================
echo.
echo Verify installation:
echo   python -c "from ultralytics import YOLO; print('YOLO OK')"
echo   python -c "import torch; print('CUDA:', torch.cuda.is_available())"
echo.
