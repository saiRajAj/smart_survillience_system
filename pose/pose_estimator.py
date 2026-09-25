from ultralytics import YOLO


class PoseEstimator:

    def __init__(self, model_path, confidence=0.5, iou=0.5):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.iou = iou

    def estimate(self, frame, bbox):
        """
        Estimate pose only inside the person's bounding box.

        Returns:
            List of 17 pose landmarks.
        """

        x1, y1, x2, y2 = map(int, bbox)

        h, w = frame.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return []

        crop = frame[y1:y2, x1:x2]

        results = self.model.predict(
            source=crop,
            conf=self.confidence,
            iou=self.iou,
            verbose=False
        )

        if not results:
            return []

        result = results[0]

        if result.keypoints is None:
            return []

        if len(result.keypoints) == 0:
            return []

        keypoints = result.keypoints

        # xy coordinates relative to cropped image
        xy = keypoints.xy[0].cpu().numpy()

        # confidence of each keypoint
        if keypoints.conf is not None:
            conf = keypoints.conf[0].cpu().numpy()
        else:
            conf = [1.0] * len(xy)

        landmarks = []

        for idx, point in enumerate(xy):

            px = float(point[0]) + x1
            py = float(point[1]) + y1

            visibility = float(conf[idx])

            landmarks.append({
                "id": idx,
                "x": px,
                "y": py,
                "z": 0.0,
                "visibility": visibility
            })

        return landmarks

    def close(self):
        pass
