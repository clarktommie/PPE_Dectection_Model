# src/predict_image.py
from ultralytics import YOLO
import os

def run_inference(model_path="runs/train/ppe_yolov8/weights/best.pt", source="data/valid/images"):
    """
    Run inference using the trained YOLOv8 PPE model.
    Args:
        model_path (str): Path to trained weights.
        source (str): Path to image or folder for inference.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = YOLO(model_path)
    results = model.predict(source=source, conf=0.4, save=True)

    print(f"\n✅ Inference complete. Results saved to: {results[0].save_dir}")

if __name__ == "__main__":
    run_inference()
