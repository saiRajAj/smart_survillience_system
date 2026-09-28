class PersonTracker:
    def extract(self,result):
        people=[]
        if result is None or result.boxes is None or result.boxes.id is None: return people
        for i in range(len(result.boxes)):
            people.append({'track_id':int(result.boxes.id[i]),'bbox':[float(v) for v in result.boxes.xyxy[i].tolist()],'confidence':float(result.boxes.conf[i])})
        return people
