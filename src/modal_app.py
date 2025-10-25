import shlex
import subprocess
from pathlib import Path
import os
from dotenv import load_dotenv
import modal

# -----------------------------
# RUN WITH:
#   modal deploy src/modal_app.py
# -----------------------------

# Load environment variables
load_dotenv()

# Point to Streamlit script
streamlit_script_local_path = Path(__file__).parent / "streamlit_app.py"
streamlit_script_remote_path = "/root/streamlit_app.py"

# Build Modal image
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .uv_pip_install(
        "streamlit", "ultralytics", "opencv-python",
        "python-dotenv", "pandas", "plotly", "pydeck",
        "folium", "streamlit-folium", "supabase"
    )

    .add_local_file(streamlit_script_local_path, streamlit_script_remote_path)
    .add_local_file(".env", "/root/.env")
    .add_local_dir(".streamlit", "/root/.streamlit") 
)


# Create Modal app
app = modal.App(name="ppe-detection-fixed", image=image)

if not streamlit_script_local_path.exists():
    raise RuntimeError("Streamlit app not found at expected path")

@app.function()
@modal.web_server(8000)
def run():
    """Run the Streamlit PPE Detection app on Modal"""
    target = shlex.quote(streamlit_script_remote_path)
    cmd = (
        f"streamlit run {target} "
        "--server.port 8000 "
        "--server.enableCORS=false "
        "--server.enableXsrfProtection=false"
    )

    env_vars = dict(os.environ)
    # Ensure Streamlit can find your weights and dependencies
    env_vars["PYTHONUNBUFFERED"] = "1"

    subprocess.Popen(cmd, shell=True, env=env_vars)
