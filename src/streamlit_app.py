import streamlit as st
from ultralytics import YOLO
import tempfile, os, subprocess
from pathlib import Path

st.title("🦺 PPE Detection App")
st.write("App loaded")

uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "avi"])

if uploaded_file:
    # save temporary upload
    temp_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    st.write("Running detection...")
    model = YOLO("runs/train/ppe_yolov8/weights/best.pt")
    results = model.predict(source=temp_path, conf=0.25, save=True)
    result_path = results[0].save_dir

    st.success("✅ Detection complete!")

    if uploaded_file.name.lower().endswith((".jpg", ".jpeg", ".png")):
        st.image(f"{result_path}/{uploaded_file.name}")
    else:
        avi_files = list(Path(result_path).rglob("*.avi"))
        if avi_files:
            avi_file = str(avi_files[0])
            mp4_file = avi_file.replace(".avi", ".mp4")

            # show note while converting
            st.info("Processing video — this may take a moment depending on length and resolution.")

            # convert AVI → MP4 using system ffmpeg
            conversion = subprocess.run(
                ["/usr/bin/ffmpeg", "-y", "-i", avi_file, "-vcodec", "h264", "-crf", "18", mp4_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            if conversion.returncode == 0 and os.path.exists(mp4_file):
                st.video(mp4_file)
            else:
                st.warning("Video conversion failed — check if /usr/bin/ffmpeg exists.")
        else:
            st.warning("No .avi file found.")

    # display detections as JSON
    st.json([
        {"label": results[0].names[int(b.cls)], "confidence": float(b.conf)}
        for b in results[0].boxes
    ])
