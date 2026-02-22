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

        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            # Menos agressivo para manter a estrutura do rosto
            transforms.RandomResizedCrop(224, scale=(0.9, 1.0)), 
            transforms.RandomGrayscale(p=0.2), # Ajuda com imagens PB do FER
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.2))
        ])

        self.val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.usage = usage

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        pixels = np.array(row["pixels"].split(), dtype="uint8")
        image = pixels.reshape(48, 48)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = self.train_transform(image) if self.usage == "Training" else self.val_transform(image)
        return image, torch.tensor(int(row["emotion"]), dtype=torch.long)