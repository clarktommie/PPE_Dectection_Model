# src/predict_violations.py
from ultralytics import YOLO
import cv2, os

def detect_violations(model_path, source, conf=0.25):
    model = YOLO(model_path)
    results = model.predict(source=source, conf=conf, save=False)
    frame = results[0].plot()  # annotated frame

    violations = []
    for box in results[0].boxes:
        cls = results[0].names[int(box.cls)]
        if cls.lower() in ["no_hardhat", "no_vest"]:
            violations.append({
                "label": cls,
                "confidence": float(box.conf)
            })

    print("⚠️ Violations detected:", violations if violations else "None")

    # optional: show frame
    cv2.imshow("PPE Violations", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

detect_violations(
    model_path="runs/train/ppe_yolov8/weights/best.pt",
    source="/home/tclark/Data Science/multimodal_PPE_dectection/test_videos/hardhat_nohardhat_test.mp4"
)
