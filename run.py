import subprocess,sys
from pathlib import Path
if not Path('models/yolov8n.pt').exists() or not Path('models/pose_landmarker_full.task').exists():print('Run: python download_models.py');sys.exit(1)
subprocess.run([sys.executable,'app.py'])
