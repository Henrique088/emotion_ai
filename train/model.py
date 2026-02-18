import torch.nn as nn
from torchvision import models

def get_model():

    model = models.resnet50(weights="IMAGENET1K_V1")

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 7)

    return model
