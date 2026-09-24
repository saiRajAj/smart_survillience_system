from pathlib import Path
import json,cv2
class EvidenceRecorder:
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
    def save(self,frame,metadata):
        folder=self.root/metadata['incident_id'];folder.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(folder/'frame.jpg'),frame);(folder/'metadata.json').write_text(json.dumps(metadata,indent=2,default=str));return str(folder)
