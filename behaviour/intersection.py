from math import hypot
def pairwise_distances(people):
    out={}
    for i,a in enumerate(people):
        ax=(a['bbox'][0]+a['bbox'][2])/2; ay=(a['bbox'][1]+a['bbox'][3])/2
        for b in people[i+1:]:
            bx=(b['bbox'][0]+b['bbox'][2])/2; by=(b['bbox'][1]+b['bbox'][3])/2; out[(a['track_id'],b['track_id'])]=hypot(ax-bx,ay-by)
    return out
