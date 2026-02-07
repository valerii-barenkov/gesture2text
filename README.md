# Gesture2Text

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey)
![License](https://img.shields.io/badge/License-MIT-green)

**Gesture2Text** is a research-oriented prototype exploring accessible, camera-based communication through classical machine learning.

**Gesture2Text** is an experimental hand gesture recognition system based on machine learning, focused on inclusive human–computer interaction and assistive technologies. The project explores how feature-based machine learning models can enable basic communication through predefined hand gestures captured via a standard webcam, without the need for speech, touch input, or specialized hardware.

The system is designed for people with speech or motor impairments, as well as for environments where voice control is unreliable (noise, privacy constraints). All processing is performed locally and offline.

---

## Demo
> A short demo (real-time webcam recognition) will be added in a future update.

---

## Key Features

* Classical machine learning–based gesture classification
* Real-time hand gesture recognition using a webcam
* Offline inference (no cloud services required)
* Lightweight ML pipeline (classical ML, no deep learning inference at runtime)
* Built on **MediaPipe Hands** and **scikit-learn**
* Gesture stabilization and confidence filtering
* Optional text-to-speech output
* Modular structure suitable for academic and research use  

---

## Supported Gestures

The current version supports a fixed vocabulary of intentional gestures:

* HELP
* STOP
* WATER
* PAIN
* YES
* NO
* CALL
* OK  

In addition, the system includes a dedicated fallback class:

* **UNKNOWN** — used to group all gestures outside the trained set and reduce false positives  

The `UNKNOWN` class is intentionally included to improve safety and robustness. It allows the system to ignore untrained or ambiguous hand poses. The gesture set can be extended in future versions by collecting additional data and retraining the model.

---

## Technical Overview

The model is trained on a custom-collected dataset and evaluated using held-out samples to assess generalization across gestures and users.

The processing pipeline consists of the following stages:

### 1. Hand detection and tracking
MediaPipe Hands detects a single hand and outputs 21 three-dimensional landmarks per frame.

### 2. Feature extraction
Each frame is converted into a 63-dimensional feature vector:

* Coordinates are centered at the wrist
* Hand scale is normalized using the middle finger length
* Left and right hands are aligned into a unified coordinate system
* Features are flattened into a fixed-length vector  

### 3. Classification
A scikit-learn pipeline is used:

* `StandardScaler` for normalization
* Multinomial Logistic Regression for classification  

### 4. Post-processing

* Temporal smoothing
* Confidence thresholds
* UNKNOWN class filtering to reduce false positives

---

## Project Structure

```
gesture2text/
├── src/
│   ├── app/
│   │   ├── main.py              # Entry point (optional wrapper)
│   │   └── run_camera.py        # Real-time webcam application
│   ├── ml/
│   │   ├── features.py          # Feature extraction logic
│   │   ├── train.py             # Model training script
│   │   └── predict_one.py       # Offline prediction / evaluation tool
│   ├── vision/
│   │   └── hand_tracker.py      # MediaPipe Hands wrapper
│   └── data/
│       ├── analyze_dataset.py   # Dataset inspection and sanity checks
│       ├── collector.py         # Interactive data collection tool
│       ├── raw/                 # Raw datasets (ignored by Git)
│       └── models/              # Trained models (ignored by Git)
├── requirements.txt
├── .gitignore
├── run.sh
├── run.command
└── run.bat
```

---

## Installation

### Clone the repository

`git clone https://github.com/valerii-barenkov/gesture2text.git`
`cd gesture2text`

### Create and activate virtual environment

`python -m venv .venv`

**macOS / Linux:**
`source .venv/bin/activate`

**Windows (PowerShell):**
`.venv\Scripts\activate`

### Install dependencies

`pip install -r requirements.txt`

---

## Running the Application

**macOS / Linux:**
`./run.sh`
**or**
`./run.command`

**Windows:**
`run.bat`

**Alternatively, you can run directly:**
`PYTHONPATH=src python src/app/run_camera.py`

The application starts the webcam feed and displays recognized gestures in real time.

---

## Data Collection

**To collect new gesture samples:**
`PYTHONPATH=src python src/data/collector.py`

This tool allows:
* Switching users
* Assigning gesture labels via keyboard
* Saving collected samples to CSV files

Collected data is stored locally and is not tracked by Git.

---

## Model Training

**To train a new model from collected data:**
`PYTHONPATH=src python src/ml/train.py --dataset combined`

The training script:
* Loads and validates the dataset
* Extracts features
* Trains a classifier
* Saves a bundled model (pipeline and metadata)

---

## Offline Evaluation

**To evaluate a trained model on stored samples:**
`PYTHONPATH=src python src/ml/predict_one.py --n 500`

**To inspect a single sample:**
`PYTHONPATH=src python src/ml/predict_one.py --row 0`

---

## Results

* Trained on: custom multi-user gesture dataset
* Model: classical ML baseline with UNKNOWN class
* Status: working research prototype
* Evaluation: metrics will be added in future experiments

---

## License

This project is licensed under the MIT License.
