import cv2
import numpy as np
from ultralytics import YOLO
import os


# ============================================================
# VOLLEYVISION - BALL ONLY
# ============================================================

VIDEO = "input/volleyball.mp4"
OUTPUT = "output/volleyvision_ball_only.mp4"

# YOLO COCO:
# 32 = sports ball
model = YOLO("yolo26n.pt")


# ============================================================
# 1. LOAD COURT CALIBRATION
# ============================================================

source_points = np.load(
    "court_points.npy"
).astype(np.float32)

print("Court points:")
print(source_points)


# ============================================================
# 2. OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    print("ERROR: Cannot open video.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30


# ============================================================
# 3. COURT REGION
# ============================================================
#
# We already calibrated the court.
# Instead of running YOLO over the whole image, crop around
# the calibrated court and enlarge it before detecting the ball.
# This makes the small volleyball easier for YOLO to see.
# ============================================================

court_x1 = int(max(0, np.min(source_points[:, 0]) - 20))
court_y1 = int(max(0, np.min(source_points[:, 1]) - 20))

court_x2 = int(min(width, np.max(source_points[:, 0]) + 20))
court_y2 = int(min(height, np.max(source_points[:, 1]) + 20))


# ============================================================
# 4. TACTICAL COURT
# ============================================================

court_width = 400

left = 20
right = 380
top = 90
bottom = 270

net_y = 180


# ============================================================
# 5. HOMOGRAPHY
# ============================================================

target_points = np.array([
    [left, bottom],
    [left, top],
    [right, top],
    [right, bottom]
], dtype=np.float32)

H = cv2.getPerspectiveTransform(
    source_points,
    target_points
)


# ============================================================
# 6. OUTPUT VIDEO
# ============================================================

output_width = width + court_width

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT,
    fourcc,
    fps,
    (output_width, height)
)


# ============================================================
# 7. DRAW TACTICAL COURT
# ============================================================

def draw_court():

    court = np.zeros(
        (height, court_width, 3),
        dtype=np.uint8
    )

    court[:] = (40, 120, 40)

    # Boundary
    cv2.rectangle(
        court,
        (left, top),
        (right, bottom),
        (255, 255, 255),
        3
    )

    # Net
    cv2.line(
        court,
        (left, net_y),
        (right, net_y),
        (255, 255, 255),
        4
    )

    # Attack lines
    cv2.line(
        court,
        (left, net_y - 35),
        (right, net_y - 35),
        (255, 255, 255),
        2
    )

    cv2.line(
        court,
        (left, net_y + 35),
        (right, net_y + 35),
        (255, 255, 255),
        2
    )

    # Title
    cv2.putText(
        court,
        "BALL TRACKING",
        (120, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        court,
        "NET",
        (180, net_y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    # Legend
    cv2.circle(
        court,
        (80, height - 35),
        8,
        (0, 140, 255),
        -1
    )

    cv2.putText(
        court,
        "Ball",
        (95, height - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )

    return court


# ============================================================
# 8. BALL DETECTION
# ============================================================

def detect_ball(frame):

    # --------------------------------------------------------
    # Crop the calibrated court
    # --------------------------------------------------------

    crop = frame[
        court_y1:court_y2,
        court_x1:court_x2
    ]

    if crop.size == 0:
        return None

    # --------------------------------------------------------
    # Enlarge crop
    # --------------------------------------------------------

    scale = 2.0

    enlarged = cv2.resize(
        crop,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )

    # --------------------------------------------------------
    # YOLO sports-ball detection
    # --------------------------------------------------------

    results = model.predict(
        enlarged,
        classes=[32],
        conf=0.08,
        imgsz=1280,
        verbose=False
    )

    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return None

    xyxy = boxes.xyxy.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()

    # Select highest-confidence ball
    best = int(np.argmax(confidences))

    x1, y1, x2, y2 = xyxy[best]
    confidence = float(confidences[best])

    # Center in enlarged crop
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # Convert back to original frame coordinates
    ball_x = (
        center_x / scale
        + court_x1
    )

    ball_y = (
        center_y / scale
        + court_y1
    )

    return (
        float(ball_x),
        float(ball_y),
        confidence
    )


# ============================================================
# 9. PROCESS VIDEO
# ============================================================

frame_number = 0
last_ball = None

while True:

    success, frame = cap.read()

    if not success:
        break

    frame_number += 1

    annotated = frame.copy()
    court = draw_court()

    # Detect ball
    detection = detect_ball(frame)

    if detection is not None:

        ball_x, ball_y, confidence = detection

        # ----------------------------------------------------
        # DRAW BALL ON ORIGINAL VIDEO
        # ----------------------------------------------------

        cv2.circle(
            annotated,
            (int(ball_x), int(ball_y)),
            8,
            (0, 140, 255),
            -1
        )

        cv2.circle(
            annotated,
            (int(ball_x), int(ball_y)),
            14,
            (255, 255, 255),
            2
        )

        cv2.putText(
            annotated,
            f"BALL {confidence:.2f}",
            (
                int(ball_x) + 12,
                int(ball_y) - 10
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 140, 255),
            2
        )

        # Remember last detection
        last_ball = (
            ball_x,
            ball_y
        )

        # ----------------------------------------------------
        # HOMOGRAPHY
        # ----------------------------------------------------

        point = np.array(
            [[[ball_x, ball_y]]],
            dtype=np.float32
        )

        mapped = cv2.perspectiveTransform(
            point,
            H
        )

        bx = float(
            mapped[0][0][0]
        )

        by = float(
            mapped[0][0][1]
        )

        # ----------------------------------------------------
        # DRAW ON TACTICAL COURT
        # ----------------------------------------------------

        if (
            left <= bx <= right
            and
            top <= by <= bottom
        ):

            bx = int(bx)
            by = int(by)

            cv2.circle(
                court,
                (bx, by),
                10,
                (0, 140, 255),
                -1
            )

            cv2.circle(
                court,
                (bx, by),
                14,
                (255, 255, 255),
                2
            )

            cv2.putText(
                court,
                "BALL",
                (bx + 15, by),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

    else:

        # Tell us when YOLO cannot see the ball
        cv2.putText(
            annotated,
            "BALL LOST",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    # ========================================================
    # COMBINE
    # ========================================================

    combined = np.hstack(
        (annotated, court)
    )

    out.write(combined)

    # Live preview
    cv2.imshow(
        "VolleyVision - Ball Only",
        combined
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# 10. CLEANUP
# ============================================================

cap.release()
out.release()
cv2.destroyAllWindows()


print()
print("==========================================")
print("VolleyVision - BALL DETECTION COMPLETE")
print("==========================================")
print(f"Frames processed: {frame_number}")
print(f"Output saved to: {OUTPUT}")


if os.path.exists(OUTPUT):

    os.startfile(
        os.path.abspath(OUTPUT)
    )
