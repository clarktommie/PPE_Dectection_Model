from ultralytics import YOLO
import json, os, datetime
from supabase import create_client, Client

from dotenv import load_dotenv
load_dotenv()


# initialize supabase connection
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)

def run_etl(model_path="runs/train/ppe_yolov8/weights/best.pt", source=None, conf=0.25):
    if source is None or not os.path.exists(source):
        raise FileNotFoundError("Provide a valid image or video path for 'source'.")

    model = YOLO(model_path)
    results = model.predict(source=source, conf=conf, save=False)
    detections = []

    for box in results[0].boxes:
        detections.append({
            "label": results[0].names[int(box.cls)],
            "confidence": float(box.conf),
            "timestamp": datetime.datetime.now().isoformat()
        })

    data = {"source": source, "detections": detections}
    supabase.table("ppe_detections").insert(data).execute()
    print(f"✅ {len(detections)} detections uploaded to Supabase.")

if __name__ == "__main__":
    path = input("Enter path to image or video: ").strip()
    run_etl(source=path)
