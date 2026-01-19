# BIRD: Vision-Based UAV Landing System

Autonomous landing module for UAVs using monocular vision. The system detects a marked landing pad in the camera stream and outputs stable 2D/3D landing cues suitable for integration with a flight controller.

---

## Features

- Real-time landing pad detection using OpenCV and deep learning–based detectors (YOLO-style).
- Robust tracking pipeline for stable target localization across frames.
- Designed for deployment on a companion computer with constrained compute.
- Modular code structure for experimenting with different detectors and trackers.

---

## Tech Stack

- Python
- OpenCV
- PyTorch
- NumPy / Pandas (for data handling)
- (Optional) ROS2 or MAVLink integration hooks if you add them later

---

## Repository Structure

- `data/` – Sample images / videos and annotations (if public).
- `models/` – Trained weights or model configs.
- `src/`
  - `detection/` – Detector models and inference code.
  - `tracking/` – Tracking and temporal smoothing.
  - `utils/` – Helper functions (I/O, visualization, metrics).
- `notebooks/` – Experiments and prototyping.
- `scripts/` – Training, evaluation, and demo scripts.

folders.
---

## Getting Started

```bash
git clone https://github.com/gauravs1303/BIRD.git
cd BIRD
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt