import torch.nn as nn
from torchvision import models

def get_model():
    # Usando a versão V2 dos pesos (mais precisa)
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # Adicionando Dropout antes da camada final
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 7)
    )
    return model