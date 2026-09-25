import cv2


CONNECTIONS = [
    (5, 6),

    (5, 7),
    (7, 9),

    (6, 8),
    (8, 10),

    (5, 11),
    (6, 12),

    (11, 12),

    (11, 13),
    (13, 15),

    (12, 14),
    (14, 16),
]


def draw_landmarks(frame, landmarks, min_visibility=0.5):

    points = {}

    for landmark in landmarks:

        if landmark["visibility"] < min_visibility:
            continue

        x = int(landmark["x"])
        y = int(landmark["y"])

        if (
            0 <= x < frame.shape[1]
            and 0 <= y < frame.shape[0]
        ):
            points[landmark["id"]] = (x, y)

            cv2.circle(
                frame,
                (x, y),
                4,
                (0, 255, 0),
                -1
            )

    for a, b in CONNECTIONS:

        if a in points and b in points:

            cv2.line(
                frame,
                points[a],
                points[b],
                (0, 255, 0),
                2
            )
