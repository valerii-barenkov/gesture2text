#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "[ERROR] .venv not found. Create venv first."
  exit 1
fi

source .venv/bin/activate
PYTHONPATH=src python src/app/run_camera.py
