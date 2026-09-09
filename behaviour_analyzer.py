from collections import deque
from dataclasses import dataclass, field
import math
import time

import numpy as np


@dataclass
class TrackHistory:

    samples: deque = field(
        default_factory=lambda: deque(maxlen=100)
    )

    high_frames: int = 0

    last_seen: float = 0.0


class BehaviorAnalyzer:

    def __init__(self, history_seconds=2.5):

        self.history_seconds = history_seconds

        self.histories = {}

    def get_history(self, track_id):

        if track_id not in self.histories:

            self.histories[track_id] = TrackHistory()

        return self.histories[track_id]

    @staticmethod
    def calculate_pose_center(keypoints):

        if keypoints is None:

            return None

        points = np.asarray(
            keypoints,
            dtype=float
        )

        if points.ndim != 2 or points.shape[1] < 2:

            return None

        valid = points[
            np.isfinite(points).all(axis=1)
        ]

        if len(valid) == 0:

            return None

        return (
            float(valid[:, 0].mean()),
            float(valid[:, 1].mean())
        )

    def update(
        self,
        track_id,
        center,
        bbox,
        keypoints=None
    ):

        now = time.monotonic()

        history = self.get_history(track_id)

        x1, y1, x2, y2 = bbox

        width = max(
            1.0,
            x2 - x1
        )

        height = max(
            1.0,
            y2 - y1
        )

        pose_center = self.calculate_pose_center(
            keypoints
        )

        pose_motion = 0.0

        if history.samples:

            previous_pose = (
                history.samples[-1]
                .get("pose_center")
            )

            if (
                previous_pose is not None
                and pose_center is not None
            ):

                distance = math.hypot(
                    pose_center[0] -
                    previous_pose[0],

                    pose_center[1] -
                    previous_pose[1]
                )

                pose_motion = (
                    distance /
                    height
                )

        history.samples.append({

            "time": now,

            "center": center,

            "width": width,

            "height": height,

            "pose_center": pose_center,

            "pose_motion": pose_motion
        })

        history.last_seen = now

        # Remove old samples.

        while history.samples:

            age = (
                now -
                history.samples[0]["time"]
            )

            if age <= self.history_seconds:

                break

            history.samples.popleft()

        if len(history.samples) < 3:

            return {

                "speed": 0.0,

                "acceleration": 0.0,

                "direction_change": 0.0,

                "fall": 0.0,

                "vigorous": 0.0,

                "pose_motion": pose_motion,

                "motion": 0.0
            }

        a = history.samples[-3]

        b = history.samples[-2]

        c = history.samples[-1]

        dt1 = max(
            0.001,
            b["time"] - a["time"]
        )

        dt2 = max(
            0.001,
            c["time"] - b["time"]
        )

        width = max(
            1.0,
            c["width"]
        )

        height = max(
            1.0,
            c["height"]
        )

        # Previous velocity

        vx1 = (
            b["center"][0] -
            a["center"][0]
        ) / width / dt1

        vy1 = (
            b["center"][1] -
            a["center"][1]
        ) / height / dt1

        # Current velocity

        vx2 = (
            c["center"][0] -
            b["center"][0]
        ) / width / dt2

        vy2 = (
            c["center"][1] -
            b["center"][1]
        ) / height / dt2

        previous_speed = math.hypot(
            vx1,
            vy1
        )

        current_speed = math.hypot(
            vx2,
            vy2
        )

        acceleration = (
            abs(
                current_speed -
                previous_speed
            ) / dt2
        )

        # Direction change

        denominator = (
            previous_speed *
            current_speed
        )

        if denominator > 1e-6:

            cosine = (
                vx1 * vx2 +
                vy1 * vy2
            ) / denominator

            cosine = max(
                -1.0,
                min(1.0, cosine)
            )

            direction_change = math.degrees(
                math.acos(cosine)
            )

        else:

            direction_change = 0.0

        # Fall detection.

        vertical_change = (
            c["center"][1] -
            a["center"][1]
        ) / height

        old_ratio = (
            a["width"] /
            max(1.0, a["height"])
        )

        new_ratio = (
            c["width"] /
            max(1.0, c["height"])
        )

        fall = 0.0

        if (
            vertical_change >= 0.12
            and
            new_ratio >= old_ratio * 1.35
        ):

            fall = 1.0

        motion = min(
            1.0,
            current_speed / 0.08
        )

        vigorous = 0.0

        if (
            current_speed >= 0.055
            or acceleration >= 0.025
            or direction_change >= 65
        ):

            vigorous = 1.0

        return {

            "speed": round(
                current_speed,
                4
            ),

            "acceleration": round(
                acceleration,
                4
            ),

            "direction_change": round(
                direction_change,
                1
            ),

            "fall": fall,

            "vigorous": vigorous,

            "pose_motion": round(
                pose_motion,
                4
            ),

            "motion": round(
                motion,
                3
            )
        }

    def update_confirmation(
        self,
        track_id,
        risk
    ):

        history = self.get_history(
            track_id
        )

        if risk >= 60:

            history.high_frames += 1

        else:

            history.high_frames = max(
                0,
                history.high_frames - 1
            )

        return history.high_frames

    def cleanup(
        self,
        active_ids,
        max_age=5
    ):

        now = time.monotonic()

        for track_id in list(
            self.histories.keys()
        ):

            history = self.histories[
                track_id
            ]

            if (
                track_id not in active_ids
                and
                now - history.last_seen >
                max_age
            ):

                del self.histories[
                    track_id
                ]
