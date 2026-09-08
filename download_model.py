from pathlib import Path
from urllib.request import urlretrieve
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

# -----------------------------
# YOLO model
# -----------------------------
yolo_path = MODELS / "yolov8n.pt"

if not yolo_path.exists():
    print("Downloading YOLOv8n...")
    YOLO(str(yolo_path))
else:
    print("YOLO model already exists.")

# -----------------------------
# MediaPipe Pose model
# -----------------------------
pose_url = (
    "https://storage.googleapis.com/"
    "mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/"
    "pose_landmarker_full.task"
)

pose_path = MODELS / "pose_landmarker_full.task"

if not pose_path.exists():
    print("Downloading MediaPipe Pose Landmarker...")
    urlretrieve(pose_url, pose_path)
else:
    print("MediaPipe pose model already exists.")

print("\nModels ready:")
print("YOLO:", yolo_path)
print("MediaPipe:", pose_path)
