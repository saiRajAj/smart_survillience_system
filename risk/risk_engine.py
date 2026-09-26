import math


class RiskEngine:

    def __init__(self, config):

        self.config = config

    def evaluate(self, tracks):

        results = {}

        for person in tracks:

            behavior = person["behavior"]

            score = 0

            reasons = []

            # -------------------------
            # FALL
            # -------------------------

            if behavior["fall"]:

                score += 48

                reasons.append(
                    "fall-like movement"
                )

            # -------------------------
            # SUDDEN ACCELERATION
            # -------------------------

            if (
                behavior["acceleration"]
                >= self.config.SUDDEN_ACCEL
            ):

                score += 22

                reasons.append(
                    "sudden acceleration"
                )

            # -------------------------
            # RAPID MOVEMENT
            # -------------------------

            if (
                behavior["speed"]
                >= self.config.SUDDEN_SPEED
            ):

                score += 12

                reasons.append(
                    "rapid movement"
                )

            # -------------------------
            # DIRECTION CHANGE
            # -------------------------

            if (
                behavior["direction_change"]
                >=
                self.config.DIRECTION_CHANGE_DEG
            ):

                score += 14

                reasons.append(
                    "rapid direction change"
                )

            # -------------------------
            # VIGOROUS MOVEMENT
            # -------------------------

            if behavior["vigorous"]:

                score += 10

                reasons.append(
                    "vigorous movement"
                )

            # -------------------------
            # POSE MOVEMENT
            # -------------------------

            if (
                behavior["pose_motion"]
                >= self.config.POSE_MOTION
            ):

                score += 8

                reasons.append(
                    "rapid body movement"
                )

            # -------------------------
            # PROXIMITY
            #
            # IMPORTANT:
            # ONLY +4.
            # NEVER enough for HIGH.
            # -------------------------

            close_to_person = False

            for other in tracks:

                if (
                    other["id"]
                    ==
                    person["id"]
                ):

                    continue

                dx = (
                    person["center"][0] -
                    other["center"][0]
                )

                dy = (
                    person["center"][1] -
                    other["center"][1]
                )

                distance = math.hypot(
                    dx,
                    dy
                )

                person_width = max(
                    1,
                    person["bbox"][2] -
                    person["bbox"][0]
                )

                normalized_distance = (
                    distance /
                    person_width
                )

                if (
                    normalized_distance
                    <
                    self.config.PROXIMITY_NORM
                ):

                    close_to_person = True

                    break

            if close_to_person:

                score += 4

                reasons.append(
                    "close interaction context"
                )

            score = min(
                100,
                score
            )

            if score < 30:

                severity = "LOW"

            elif score < 60:

                severity = "MEDIUM"

            else:

                severity = "HIGH"

            results[
                person["id"]
            ] = {

                "score": score,

                "severity": severity,

                "reason":
                    ", ".join(
                        dict.fromkeys(
                            reasons
                        )
                    )
                    or
                    "normal movement"
            }

        overall = max(
            [
                x["score"]
                for x in results.values()
            ],
            default=0
        )

        if overall < 30:

            overall_severity = "LOW"

        elif overall < 60:

            overall_severity = "MEDIUM"

        else:

            overall_severity = "HIGH"

        return (
            overall,
            overall_severity,
            results
        )
