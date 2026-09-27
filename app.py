import streamlit as st
import tempfile
import os
import cv2
from ultralytics import YOLO


# -----------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------

st.set_page_config(
    page_title="VolleyVision",
    page_icon="🏐",
    layout="wide"
)


# -----------------------------------------
# TITLE
# -----------------------------------------

st.title("🏐 VolleyVision")

st.subheader(
    "AI-Powered Volleyball Player Detection & Tracking"
)

st.write(
    """
    VolleyVision uses Computer Vision to analyze volleyball
    match footage using YOLO object detection, ByteTrack
    multi-object tracking, and OpenCV.
    """
)


# -----------------------------------------
# SIDEBAR
# -----------------------------------------

st.sidebar.title("⚙️ System")

st.sidebar.write("Computer Vision Pipeline")

st.sidebar.write("✅ YOLO Player Detection")
st.sidebar.write("✅ ByteTrack Tracking")
st.sidebar.write("✅ OpenCV Processing")
st.sidebar.write("🚧 Ball Tracking")
st.sidebar.write("🚧 Tactical Analytics")


# -----------------------------------------
# VIDEO UPLOAD
# -----------------------------------------

st.header("🎥 Upload Match Footage")

uploaded_file = st.file_uploader(
    "Upload a volleyball video",
    type=["mp4", "mov", "avi"]
)


# -----------------------------------------
# PROCESS VIDEO
# -----------------------------------------

if uploaded_file is not None:

    st.success("Video uploaded successfully!")

    st.video(uploaded_file)

    if st.button("🚀 Analyze Video"):

        with st.spinner("Running YOLO + ByteTrack..."):

            # Temporary input file
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            ) as temp_input:

                temp_input.write(
                    uploaded_file.read()
                )

                input_path = temp_input.name


            # Load YOLO model
            model = YOLO("yolo26n.pt")


            # Output file
            output_path = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            ).name


            # Open video
            cap = cv2.VideoCapture(input_path)

            width = int(
                cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            )

            height = int(
                cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            )

            fps = cap.get(
                cv2.CAP_PROP_FPS
            )

            fourcc = cv2.VideoWriter_fourcc(
                *"mp4v"
            )

            out = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (width, height)
            )


            # ---------------------------------
            # PROCESS FRAMES
            # ---------------------------------

            while cap.isOpened():

                success, frame = cap.read()

                if not success:
                    break


                results = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=[0],
                    conf=0.4,
                    verbose=False
                )


                annotated_frame = results[0].plot()


                out.write(
                    annotated_frame
                )


            cap.release()
            out.release()


        st.success(
            "Analysis completed successfully! 🎉"
        )


        # ---------------------------------
        # SHOW RESULT
        # ---------------------------------

        st.header("🧠 AI Analysis Result")

        st.video(
            output_path
        )


        st.info(
            """
            YOLO detects the players while ByteTrack
            maintains player identities across frames.
            """
        )