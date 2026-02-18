import torch
import torch.nn as nn
from torchvision import models


EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

def load_model(model_path="models/resnet_emotion.pt"):

    # carrega arquitetura base
    model = models.resnet18(weights=None)

    # substitui última camada
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(EMOTIONS))

    # carrega pesos treinados
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    model.eval()

    return model
