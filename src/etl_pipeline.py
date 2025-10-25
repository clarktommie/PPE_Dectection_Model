from ultralytics import YOLO
import json, os, datetime

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

    print(f"✅ {len(detections)} detections found.")
    print(json.dumps(detections, indent=2))

if __name__ == "__main__":
    path = input("Enter path to image or video: ").strip()
    run_etl(source=path)
