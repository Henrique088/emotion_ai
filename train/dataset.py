# train/dataset.py

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2

class FERDataset(Dataset):
    def __init__(self, csv_file, usage="Training"):
        self.data = pd.read_csv(csv_file)
        self.data = self.data[self.data["Usage"] == usage]

        # Transformações de TREINO: Aumento de dados para evitar overfitting
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2) # Simula oclusões no rosto
        ])

        # Transformações de VALIDAÇÃO: Limpas, apenas redimensionamento
        self.val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.usage = usage

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        pixels = np.array(row["pixels"].split(), dtype="uint8")
        image = pixels.reshape(48, 48)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if self.usage == "Training":
            image = self.train_transform(image)
        else:
            image = self.val_transform(image)

        label = torch.tensor(int(row["emotion"]), dtype=torch.long)
        return image, label