# SafeVision — Full Runnable Major Project

Pipeline: YOLO person detection -> ByteTrack -> MediaPipe Pose -> pose history -> movement features -> explainable heuristic risk score -> evidence -> SQLite -> Flask dashboard.

This is an academic prototype, not a validated crime/violence detector. It reports potential high-risk activity and should not automatically accuse people.

## Install
Recommended: Python 3.11.

```bash
python3.11 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python download_models.py
python app.py
```

Open http://127.0.0.1:5000

## Configure
Copy `.env.example` to `.env`. Default camera is 0. Set `CAMERA_SOURCE=/absolute/path/video.mp4` for a video.

## Troubleshooting
```bash
python -c "import mediapipe as mp; print(mp.__file__); print(getattr(mp,'__version__','unknown')); print(hasattr(mp,'tasks'))"
python -c "from ultralytics import YOLO; print('YOLO OK')"
```
Do not keep a local `mediapipe.py` file or `mediapipe/` folder in this project.

## Next upgrade
Replace the heuristic risk engine with a trained temporal behavior model and report precision, recall, F1 and confusion matrix. Do not claim accuracy without evaluation.
