import cv2
import numpy as np

VIDEO_PATH = "input/volleyball.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

success, frame = cap.read()

if not success:
    print("Could not open video.")
    exit()

points = []


def mouse_callback(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:

        if len(points) < 4:

            points.append((x, y))

            print(f"Point {len(points)}: ({x}, {y})")


cv2.namedWindow("Calibration")
cv2.setMouseCallback("Calibration", mouse_callback)


while True:

    display = frame.copy()

    # Draw selected points
    for i, point in enumerate(points):

        cv2.circle(
            display,
            point,
            8,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            display,
            str(i + 1),
            (point[0] + 10, point[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    cv2.putText(
        display,
        "Click: TOP-LEFT -> TOP-RIGHT -> BOTTOM-RIGHT -> BOTTOM-LEFT",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        display,
        "Press S to save | R to reset | Q to quit",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.imshow("Calibration", display)

    key = cv2.waitKey(1) & 0xFF

    # Reset
    if key == ord("r"):

        points = []

        print("Points reset.")

    # Save
    elif key == ord("s"):

        if len(points) == 4:

            np.save(
                "court_points.npy",
                np.array(points, dtype=np.float32)
            )

            print()
            print("================================")
            print("Court points saved!")
            print("================================")
            print(points)

            break

        else:

            print(
                f"Need 4 points. Currently selected: {len(points)}"
            )

    # Quit
    elif key == ord("q"):

        break


cap.release()

cv2.destroyAllWindows()