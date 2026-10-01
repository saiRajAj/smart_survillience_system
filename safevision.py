import cv2
import threading
import time

from ultralytics import YOLO

import config

from behavior_analyzer import BehaviorAnalyzer
from risk_engine import RiskEngine
from database import Database


class SafeVision:

    def __init__(self):

        print("=" * 60)

        print(
            "             SAFE VISION - INITIALIZING"
        )

        print("=" * 60)

        # --------------------------------
        # YOLO PERSON
        # --------------------------------

        print(
            "[1/7] Loading YOLO person detector..."
        )

        self.detector = YOLO(
            config.MODEL_PATH
        )

        print(
            "      ✓ YOLO detector loaded"
        )

        # --------------------------------
        # BYTE TRACK
        # --------------------------------

        print(
            "[2/7] Initializing ByteTrack..."
        )

        print(
            "      ✓ ByteTrack enabled"
        )

        # --------------------------------
        # YOLO POSE
        # --------------------------------

        print(
            "[3/7] Loading YOLO pose model..."
        )

        self.pose_model = YOLO(
            config.POSE_MODEL_PATH
        )

        print(
            "      ✓ YOLO pose model loaded"
        )

        # --------------------------------
        # BEHAVIOR
        # --------------------------------

        print(
            "[4/7] Initializing behaviour analysis..."
        )

        self.behavior = BehaviorAnalyzer(
            config.HISTORY_SECONDS
        )

        print(
            "      ✓ Behaviour analyzer ready"
        )

        # --------------------------------
        # RISK
        # --------------------------------

        print(
            "[5/7] Initializing risk engine..."
        )

        self.risk_engine = RiskEngine(
            config
        )

        print(
            "      ✓ Risk engine ready"
        )

        # --------------------------------
        # DATABASE
        # --------------------------------

        print(
            "[6/7] Initializing database..."
        )

        self.db = Database(
            config.DB_PATH
        )

        print(
            "      ✓ SQLite database ready"
        )

        # --------------------------------
        # CAMERA
        # --------------------------------

        print(
            "[7/7] Initializing camera..."
        )

        source = (
            int(config.CAMERA_SOURCE)
            if str(
                config.CAMERA_SOURCE
            ).isdigit()
            else config.CAMERA_SOURCE
        )

        self.cap = cv2.VideoCapture(
            source,
            cv2.CAP_AVFOUNDATION
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            config.CAMERA_WIDTH
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            config.CAMERA_HEIGHT
        )

        if not self.cap.isOpened():

            raise RuntimeError(
                "Camera could not be opened. "
                "Grant camera permission to "
                "Terminal/VS Code."
            )

        print(
            f"      ✓ Camera opened: {source}"
        )

        print("=" * 60)

        print(
            "             SAFE VISION READY"
        )

        print("=" * 60)

        self.running = True

        self.frame_count = 0

        self.latest_frame = None

        self.lock = threading.Lock()

        self.fps = 0

        self.fps_counter = 0

        self.fps_time = time.monotonic()

        self.last_alert = 0

    # ==================================================
    # POSE
    # ==================================================

    def get_pose(self, frame):

        result = self.pose_model.predict(

            frame,

            conf=config.POSE_CONF,

            imgsz=config.INFER_WIDTH,

            verbose=False
        )[0]

        detections = []

        if (
            result.boxes is None
            or
            result.keypoints is None
        ):

            return detections

        boxes = (
            result.boxes
            .xyxy
            .cpu()
            .numpy()
        )

        keypoints = (
            result.keypoints
            .xy
            .cpu()
            .numpy()
        )

        count = min(
            len(boxes),
            len(keypoints)
        )

        for i in range(count):

            detections.append({

                "bbox":
                    boxes[i].tolist(),

                "keypoints":
                    keypoints[i].tolist()
            })

        return detections

    # ==================================================
    # IOU
    # ==================================================

    @staticmethod
    def iou(box_a, box_b):

        ax1, ay1, ax2, ay2 = box_a

        bx1, by1, bx2, by2 = box_b

        ix1 = max(ax1, bx1)

        iy1 = max(ay1, by1)

        ix2 = min(ax2, bx2)

        iy2 = min(ay2, by2)

        width = max(
            0,
            ix2 - ix1
        )

        height = max(
            0,
            iy2 - iy1
        )

        intersection = (
            width *
            height
        )

        area_a = (
            max(0, ax2-ax1) *
            max(0, ay2-ay1)
        )

        area_b = (
            max(0, bx2-bx1) *
            max(0, by2-by1)
        )

        union = (
            area_a +
            area_b -
            intersection
        )

        if union <= 0:

            return 0

        return (
            intersection /
            union
        )

    # ==================================================
    # POSE ↔ TRACK ID
    # ==================================================

    def associate_pose(
        self,
        tracks,
        pose_detections
    ):

        associations = {}

        used = set()

        for track in tracks:

            best_index = None

            best_iou = 0

            for i, pose in enumerate(
                pose_detections
            ):

                if i in used:

                    continue

                current_iou = self.iou(
                    track["bbox"],
                    pose["bbox"]
                )

                if (
                    current_iou >
                    best_iou
                ):

                    best_iou = current_iou

                    best_index = i

            if (
                best_index is not None
                and
                best_iou >= 0.20
            ):

                associations[
                    track["id"]
                ] = pose_detections[
                    best_index
                ]["keypoints"]

                used.add(
                    best_index
                )

        return associations

    # ==================================================
    # MAIN LOOP
    # ==================================================

    def run(self):

        print(
            "SafeVision camera loop started."
        )

        while self.running:

            success, frame = (
                self.cap.read()
            )

            if not success:

                time.sleep(0.03)

                continue

            self.frame_count += 1

            frame = cv2.resize(
                frame,
                (
                    config.CAMERA_WIDTH,
                    config.CAMERA_HEIGHT
                )
            )

            # --------------------------------
            # PERSON + BYTE TRACK
            # --------------------------------

            result = self.detector.track(

                frame,

                persist=True,

                tracker="bytetrack.yaml",

                conf=config.PERSON_CONF,

                iou=config.TRACK_IOU,

                classes=[0],

                imgsz=config.INFER_WIDTH,

                verbose=False
            )[0]

            tracks = []

            if (
                result.boxes is not None
                and
                result.boxes.id is not None
            ):

                boxes = (
                    result.boxes
                    .xyxy
                    .cpu()
                    .numpy()
                )

                ids = (
                    result.boxes
                    .id
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                for bbox, track_id in zip(
                    boxes,
                    ids
                ):

                    x1, y1, x2, y2 = (
                        bbox.tolist()
                    )

                    tracks.append({

                        "id":
                            int(track_id),

                        "bbox":
                            bbox.tolist(),

                        "center": (
                            (x1+x2)/2,
                            (y1+y2)/2
                        )
                    })

            # --------------------------------
            # POSE
            # --------------------------------

            pose_map = {}

            if (
                tracks
                and
                self.frame_count %
                max(
                    1,
                    config.POSE_INTERVAL
                ) == 0
            ):

                pose_detections = (
                    self.get_pose(frame)
                )

                pose_map = (
                    self.associate_pose(
                        tracks,
                        pose_detections
                    )
                )

            # --------------------------------
            # BEHAVIOR
            # --------------------------------

            for track in tracks:

                track["behavior"] = (
                    self.behavior.update(

                        track["id"],

                        track["center"],

                        track["bbox"],

                        pose_map.get(
                            track["id"]
                        )
                    )
                )

            # --------------------------------
            # RISK
            # --------------------------------

            overall_risk, severity, details = (
                self.risk_engine.evaluate(
                    tracks
                )
            )

            # --------------------------------
            # CONFIRM HIGH RISK
            # --------------------------------

            confirmed = []

            for track in tracks:

                track_id = track["id"]

                person_risk = details[
                    track_id
                ]["score"]

                high_frames = (
                    self.behavior
                    .update_confirmation(
                        track_id,
                        person_risk
                    )
                )

                if (
                    high_frames >=
                    config.HIGH_CONFIRM_FRAMES
                ):

                    confirmed.append(
                        track_id
                    )

            # --------------------------------
            # ALERT
            # --------------------------------

            now = time.monotonic()

            if (
                confirmed
                and
                now - self.last_alert
                >=
                config.ALERT_COOLDOWN_SECONDS
            ):

                reasons = []

                for track_id in confirmed:

                    reasons.append(
                        details[
                            track_id
                        ]["reason"]
                    )

                reason = "; ".join(
                    set(reasons)
                )

                self.db.add(

                    "HIGH",

                    overall_risk,

                    reason,

                    confirmed
                )

                self.last_alert = now

            # --------------------------------
            # DRAW
            # --------------------------------

            for track in tracks:

                x1, y1, x2, y2 = map(
                    int,
                    track["bbox"]
                )

                track_id = track["id"]

                score = details[
                    track_id
                ]["score"]

                cv2.rectangle(

                    frame,

                    (x1, y1),

                    (x2, y2),

                    (0, 255, 0),

                    2
                )

                cv2.putText(

                    frame,

                    f"ID {track_id}  R:{score:.0f}",

                    (
                        x1,
                        max(
                            25,
                            y1-8
                        )
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.6,

                    (0, 255, 0),

                    2
                )

            cv2.putText(

                frame,

                (
                    f"SAFEVISION | "
                    f"People:{len(tracks)} | "
                    f"Risk:{overall_risk:.0f} | "
                    f"{severity}"
                ),

                (15, 30),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.65,

                (255, 255, 255),

                2
            )

            # --------------------------------
            # FPS
            # --------------------------------

            self.fps_counter += 1

            elapsed = (
                now -
                self.fps_time
            )

            if elapsed >= 1:

                self.fps = (
                    self.fps_counter /
                    elapsed
                )

                self.fps_counter = 0

                self.fps_time = now

            # --------------------------------
            # STATUS
            # --------------------------------

            public_tracks = []

            for track in tracks:

                track_id = track["id"]

                public_tracks.append({

                    "id":
                        track_id,

                    "risk":
                        round(
                            details[
                                track_id
                            ]["score"],
                            1
                        ),

                    "severity":
                        details[
                            track_id
                        ]["severity"],

                    "reason":
                        details[
                            track_id
                        ]["reason"],

                    "behavior":
                        track[
                            "behavior"
                        ]
                })

            with self.lock:

                self.latest_frame = (
                    frame.copy()
                )

                self.status = {

                    "camera":
                        "ACTIVE",

                    "people":
                        len(tracks),

                    "risk":
                        round(
                            overall_risk,
                            1
                        ),

                    "severity":
                        severity,

                    "fps":
                        round(
                            self.fps,
                            1
                        ),

                    "tracks":
                        public_tracks
                }

    # ==================================================
    # MJPEG
    # ==================================================

    def mjpeg(self):

        while True:

            with self.lock:

                frame = (
                    None
                    if self.latest_frame
                    is None
                    else
                    self.latest_frame.copy()
                )

            if frame is None:

                time.sleep(0.03)

                continue

            success, jpeg = (
                cv2.imencode(
                    ".jpg",
                    frame,
                    [
                        int(
                            cv2.IMWRITE_JPEG_QUALITY
                        ),
                        80
                    ]
                )
            )

            if success:

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    +
                    jpeg.tobytes()
                    +
                    b"\r\n"
                )

            time.sleep(0.01)

    def stop(self):

        self.running = False

        if self.cap:

            self.cap.release()
