# src/train_model.py
from ultralytics import YOLO
import os

def train_yolov8(data_yaml="data/data.yaml", model_size="yolov8n.pt", epochs=50, imgsz=640):
    """
    Train a YOLOv8 model on the PPE dataset. 
    
    Args:
        data_yaml (str): Path to the dataset configuration file.
        model_size (str): Size of the YOLOv8 model to train.
        epochs (int): Number of training epochs.
        imgsz (int): Size of the input images.
    """
    
    # confirm data file exists
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"data.yaml not found a {data_yaml}")
    
    # initialize model
    model = YOLO(model_size)

     # train
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        device=0,   # GPU; change to 'cpu' if needed
        project="runs/train",
        name="ppe_yolov8"
    )

    print("Training complete. Check the 'runs/train/ppe_yolov8' folder for results.")

if __name__ == "__main__":
    train_yolov8()