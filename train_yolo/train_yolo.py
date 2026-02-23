from ultralytics import YOLO
import torch

def train_psico_yolo_96px():
    # Carregar modelo Nano (o mais adequado para resoluções baixas)
    model = YOLO("yolov8n.pt") 

    # Treinamento focado no dataset (96x96)
    model.train(
        data="../data/YOLO_format/data.yaml", 
        epochs=100,
        imgsz=96,             
        batch=64,             
        device=0,
        project="psico_ai",
        name="affectnet_96px",
        save=True,
        augment=True,         # Para evitar que o modelo decore as fotos
        optimizer="AdamW",    # AdamW costuma convergir mais rápido em resoluções baixas
        lr0=0.001             # LR um pouco menor para precisão cirúrgica
    )

if __name__ == "__main__":
    train_psico_yolo_96px()