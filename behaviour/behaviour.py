from collections import deque
from math import hypot
class FeatureExtractor:
    def __init__(self,size=30): self.history={}; self.size=size
    def update(self,tid,landmarks,t):
        if not landmarks:return {'speed':0.0,'acceleration':0.0,'wrist_speed':0.0}
        vis=[x for x in landmarks if x['visibility']>=.5]
        if not vis:return {'speed':0.0,'acceleration':0.0,'wrist_speed':0.0}
        cx=sum(x['x'] for x in vis)/len(vis); cy=sum(x['y'] for x in vis)/len(vis); q=self.history.setdefault(tid,deque(maxlen=self.size)); cur={'t':t,'cx':cx,'cy':cy,'landmarks':landmarks}; q.append(cur)
        speed=accel=wrist=0.0
        if len(q)>=2:
            p=q[-2]; dt=max(t-p['t'],.001); speed=hypot(cx-p['cx'],cy-p['cy'])/dt; pw={x['id']:x for x in p['landmarks']}; cw={x['id']:x for x in landmarks}; vals=[]
            for wid in (9,10):
                if wid in pw and wid in cw: vals.append(hypot(cw[wid]['x']-pw[wid]['x'],cw[wid]['y']-pw[wid]['y'])/dt)
            if vals:wrist=sum(vals)/len(vals)
        if len(q)>=3:
            p=q[-2]; p2=q[-3]; old=hypot(p['cx']-p2['cx'],p['cy']-p2['cy'])/max(p['t']-p2['t'],.001); accel=abs(speed-old)/max(t-p['t'],.001)
        return {'speed':speed,'acceleration':accel,'wrist_speed':wrist}
    def remove_missing(self,active):
        for tid in list(self.history):
            if tid not in active:del self.history[tid]
