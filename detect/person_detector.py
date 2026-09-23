from ultralytics import YOLO
class PersonDetector:
    def __init__(self,model_path,conf=0.5,iou=0.5):
        self.model=YOLO(model_path); self.conf=conf; self.iou=iou
    def track(self,frame,tracker_config):
        return self.model.track(source=frame,persist=True,tracker=tracker_config,conf=self.conf,iou=self.iou,classes=[0],verbose=False)
