import streamlit as st
from ultralytics import YOLO, settings
settings.update({"sync": False})  # disable analytics

import tempfile, os, subprocess
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client
import warnings, logging

import os, tempfile
os.environ["TMPDIR"] = "/tmp"
tempfile.tempdir = "/tmp"

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------
# Get the directory of this script (works on any system)
BASE_DIR = Path(__file__).resolve().parent.parent  # go up from src/ to project root
ENV_PATH = BASE_DIR / ".env"

# Load .env file dynamically
load_dotenv(ENV_PATH)

warnings.filterwarnings("ignore")
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("requests").setLevel(logging.CRITICAL)

# Connect to Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ------------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------------
st.title("🦺 PPE Detection App")
st.write("App loaded")

uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "avi"])

# ------------------------------------------------------------------
# Detection logic
# ------------------------------------------------------------------
if uploaded_file:
    # Save temporary upload
    temp_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    st.write("Running detection...")

    # Load YOLO model
    model = YOLO("runs/train/ppe_yolov8/weights/best.pt")

    # Run prediction and save results
    results = model.predict(source=temp_path, conf=0.25, save=True, project="/tmp", name="detections")
    result_path = Path(results[0].save_dir)

    st.success("✅ Detection complete!")

    # Display image or video results
    output_file = None
    if uploaded_file.name.lower().endswith((".jpg", ".jpeg", ".png")):
        image_files = list(result_path.rglob("*.jpg")) + list(result_path.rglob("*.jpeg")) + list(result_path.rglob("*.png"))
        if image_files:
            output_file = str(image_files[0])
            st.image(output_file)
        else:
            st.warning("No output image found.")
    else:
        avi_files = list(result_path.rglob("*.avi"))
        if avi_files:
            avi_file = str(avi_files[0])
            mp4_file = avi_file.replace(".avi", ".mp4")

            st.info("Processing video — please wait...")

            conversion = subprocess.run(
                ["ffmpeg", "-y", "-i", avi_file, "-vcodec", "h264", "-crf", "18", mp4_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            if conversion.returncode == 0 and os.path.exists(mp4_file):
                output_file = mp4_file
                st.video(mp4_file)
            else:
                st.warning("Video conversion failed.")
        else:
            st.warning("No .avi file found.")

    # ------------------------------------------------------------------
    # Upload detection results to Supabase
    # ------------------------------------------------------------------
    if output_file:
        file_name = os.path.basename(output_file)
        with open(output_file, "rb") as f:
            # upload to storage bucket named 'ppe_results'
            res = supabase.storage.from_("ppe_results").upload(file_name, f)

        # build public URL
        public_url = f"{SUPABASE_URL}/storage/v1/object/public/ppe_results/{file_name}"

        # insert record in detections table
        for box in results[0].boxes:
            supabase.table("detections").insert({
                "filename": file_name,
                "file_url": public_url,
                "confidence": float(box.conf),
                "label": results[0].names[int(box.cls)]
            }).execute()

        st.success(f"✅ Results uploaded to Supabase!\n{public_url}")

    # ------------------------------------------------------------------
    # Display detections as JSON
    # ------------------------------------------------------------------
    st.json([
        {"label": results[0].names[int(b.cls)], "confidence": float(b.conf)}
        for b in results[0].boxes
    ])
