import cv2
import numpy as np
from ultralytics import YOLO


# -----------------------------
# 1. LOAD YOLO MODEL
# -----------------------------

model = YOLO("yolo26n.pt")


# -----------------------------
# 2. INPUT / OUTPUT
# -----------------------------

video_path = "input/volleyball.mp4"
output_path = "output/volleyvision.mp4"


# -----------------------------
# 3. OPEN VIDEO
# -----------------------------

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()


width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0:
    fps = 30


# -----------------------------
# 4. OUTPUT VIDEO
# -----------------------------

output_width = width + 400

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (output_width, height)
)


# -----------------------------
# 5. CREATE TACTICAL COURT
# -----------------------------

def create_court(width, height):

    court_width = 400

    court = np.zeros(
        (height, court_width, 3),
        dtype=np.uint8
    )

    # Court background
    court[:] = (40, 120, 40)

    margin_x = 40
    margin_y = 100

    left = margin_x
    right = court_width - margin_x

    top = margin_y
    bottom = height - margin_y

    # Outer court
    cv2.rectangle(
        court,
        (left, top),
        (right, bottom),
        (255, 255, 255),
        3
    )

    # Net
    center_y = height // 2

    cv2.line(
        court,
        (left, center_y),
        (right, center_y),
        (255, 255, 255),
        4
    )

    # Attack lines
    attack_offset = 100

    cv2.line(
        court,
        (left, center_y - attack_offset),
        (right, center_y - attack_offset),
        (255, 255, 255),
        2
    )

    cv2.line(
        court,
        (left, center_y + attack_offset),
        (right, center_y + attack_offset),
        (255, 255, 255),
        2
    )

    # Title
    cv2.putText(
        court,
        "VOLLEYBALL TACTICAL VIEW",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        court,
        "NET",
        (175, center_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    return court


# -----------------------------
# 6. PROCESS VIDEO
# -----------------------------

while True:

    success, frame = cap.read()

    if not success:
        break


    # -------------------------
    # YOLO + BYTETRACK
    # -------------------------

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],
        conf=0.4,
        verbose=False
    )


    # -------------------------
    # DRAW DETECTIONS
    # -------------------------

    annotated_frame = results[0].plot()


    # -------------------------
    # CREATE COURT
    # -------------------------

    court = create_court(width, height)


    # -------------------------
    # GET PLAYER POSITIONS
    # -------------------------

    boxes = results[0].boxes


    if boxes.id is not None:

        ids = boxes.id.cpu().numpy()

        coordinates = boxes.xyxy.cpu().numpy()


        for player_id, box in zip(ids, coordinates):

            x1, y1, x2, y2 = box

            # Bottom-center of bounding box
            player_x = int((x1 + x2) / 2)
            player_y = int(y2)


            # --------------------------------
            # SIMPLE POSITION MAPPING
            # --------------------------------

            court_x = int(
                (player_x / width) * 320
            ) + 40

            court_y = int(
                (player_y / height) * (height - 200)
            ) + 100


            # Keep point inside court
            court_x = max(45, min(355, court_x))
            court_y = max(105, min(height - 105, court_y))


            # Draw player
            cv2.circle(
                court,
                (court_x, court_y),
                8,
                (0, 255, 255),
                -1
            )


            # Player ID
            cv2.putText(
                court,
                f"P{int(player_id)}",
                (court_x + 10, court_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )


    # -------------------------
    # COMBINE BOTH VIEWS
    # -------------------------

    combined = np.hstack(
        (annotated_frame, court)
    )


    # -------------------------
    # SAVE
    # -------------------------

    out.write(combined)


    # -------------------------
    # SHOW
    # -------------------------

    cv2.imshow(
        "VolleyVision",
        combined
    )


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -----------------------------
# CLEANUP
# -----------------------------

cap.release()

out.release()

cv2.destroyAllWindows()

print()
print("================================")
print("VolleyVision processing complete!")
print("Output:", output_path)
print("================================")