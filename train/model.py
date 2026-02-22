import torch.nn as nn
from torchvision import models

def get_model():
    # Usando EfficientNet_V2_S (Superior à ResNet50 para este caso)
    try:
        # Tenta carregar com o novo sistema de pesos
        model = models.efficientnet_b0(weights='DEFAULT')
    except:
        # Tenta carregar no estilo antigo caso o torchvision seja bem antigo
        model = models.efficientnet_b0(pretrained=True)

    # Ajustando a camada final (Classifier)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(num_features, 7)
    )
    return model