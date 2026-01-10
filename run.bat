@echo off
setlocal
cd /d %~dp0

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] .venv not found. Create venv first.
  pause
  exit /b 1
)

set PYTHONPATH=src
".venv\Scripts\python.exe" "src\app\run_camera.py"
pause
