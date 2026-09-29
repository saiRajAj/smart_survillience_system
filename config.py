import os


def get_float(name, default):
    return float(os.getenv(name, default))


def get_int(name, default):
    return int(os.getenv(name, default))


# Camera
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "0")
CAMERA_WIDTH = get_int("CAMERA_WIDTH", 960)
CAMERA_HEIGHT = get_int("CAMERA_HEIGHT", 540)

# Models
MODEL_PATH = os.getenv("MODEL_PATH", "models/yolov8n.pt")
POSE_MODEL_PATH = os.getenv(
    "POSE_MODEL_PATH",
    "models/yolov8n-pose.pt"
)

# Detection
PERSON_CONF = get_float("PERSON_CONF", 0.35)
TRACK_IOU = get_float("TRACK_IOU", 0.50)
POSE_CONF = get_float("POSE_CONF", 0.30)

# Performance
INFER_WIDTH = get_int("INFER_WIDTH", 640)
POSE_INTERVAL = get_int("POSE_INTERVAL", 2)

# History
HISTORY_SECONDS = get_float("HISTORY_SECONDS", 2.5)

# Risk confirmation
HIGH_CONFIRM_FRAMES = get_int("HIGH_CONFIRM_FRAMES", 8)
ALERT_COOLDOWN_SECONDS = get_float(
    "ALERT_COOLDOWN_SECONDS",
    8
)

# Behavior thresholds
SUDDEN_SPEED = get_float("SUDDEN_SPEED", 0.045)
SUDDEN_ACCEL = get_float("SUDDEN_ACCEL", 0.025)

DIRECTION_CHANGE_DEG = get_float(
    "DIRECTION_CHANGE_DEG",
    65
)

VIGOROUS_MOTION = get_float(
    "VIGOROUS_MOTION",
    0.055
)

POSE_MOTION = get_float(
    "POSE_MOTION",
    0.080
)

FALL_DROP = get_float(
    "FALL_DROP",
    0.12
)

# Interaction is only contextual.
# It must NEVER independently create HIGH risk.
PROXIMITY_NORM = get_float(
    "PROXIMITY_NORM",
    1.20
)

DB_PATH = os.getenv(
    "DB_PATH",
    "safevision.db"
)

HOST = os.getenv(
    "HOST",
    "127.0.0.1"
)

PORT = get_int(
    "PORT",
    5000
)
