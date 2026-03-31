@echo off
cd /d "%~dp0"
echo Testing Python...
.\.venv\Scripts\python.exe --version
echo.
echo Testing imports...
.\.venv\Scripts\python.exe -c "import torch; print(' 'PyTorch OK')" 
.\.venv\Scripts\python.exe -c "import pandas; print('Pandas OK')"
echo.
echo Running main.py...
.\.venv\Scripts\python.exe main.py > batch_output.txt 2>&1
echo Pipeline execution complete
