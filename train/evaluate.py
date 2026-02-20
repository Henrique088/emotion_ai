# train/evaluate.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from dataset import FERDataset
from model import get_model



# Configurações
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATH_MODEL = "../models/SOTA_FER2013.pt"
CLASSES = ['Raiva', 'Nojo', 'Medo', 'Feliz', 'Triste', 'Surpresa', 'Neutro']

# 1. Carregar Dados e Modelo
val_dataset = FERDataset("../data/fer2013.csv", usage="PublicTest")
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

model = get_model().to(device)
model.load_state_dict(torch.load(PATH_MODEL))
model.eval()

all_preds = []
all_labels = []

# 2. Coletar Predições
print("Avaliando o modelo...")
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# 3. Gerar Relatório de Texto
print("\nRelatório de Classificação:")
print(classification_report(all_labels, all_preds, target_names=CLASSES))

# 4. Criar Matriz de Confusão Visual
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
plt.xlabel('Predito')
plt.ylabel('Real')
plt.title('Matriz de Confusão - FER2013')
plt.savefig('matriz_confusao.png')
print("Imagem salva como matriz_confusao.png")
plt.show()