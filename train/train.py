# train/train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import FERDataset
from model import get_model
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DATASETS
train_dataset = FERDataset("../data/fer2013.csv", usage="Training")
val_dataset   = FERDataset("../data/fer2013.csv", usage="PublicTest")

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

# MODELO E PESOS DE CLASSE
model = get_model().to(device)
labels = train_dataset.data["emotion"].values
class_counts = Counter(labels)
total_samples = sum(class_counts.values())
weights = torch.tensor([total_samples / (7 * class_counts[i]) for i in range(7)], dtype=torch.float).to(device)

# PERDA COM LABEL SMOOTHING (Crítico para FER2013)
criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

# OTIMIZADOR SGD (Melhor generalização que Adam em modelos profundos)
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
                            lr=0.01, momentum=0.9, weight_decay=1e-4)

# SCHEDULER COSINE (Evita estagnação)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)

scaler = torch.cuda.amp.GradScaler()

def run_epoch(epoch, loader, is_train):
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0, 0, 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        with torch.set_grad_enabled(is_train):
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            if is_train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        
        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    return total_loss / len(loader), correct / total

# TREINAMENTO PRINCIPAL
best_acc = 0
epochs = 30 # Aumentado pois o SGD e Cosine demoram mais a convergir, mas chegam mais longe

for epoch in range(epochs):
    train_loss, train_acc = run_epoch(epoch, train_loader, is_train=True)
    val_loss, val_acc = run_epoch(epoch, val_loader, is_train=False)
    
    scheduler.step() # Atualiza LR baseado no ciclo cosseno

    print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "../models/SOTA_FER2013.pt")
        print(">> Novo Recorde de Acurácia!")

print(f"\nTreino Finalizado. Melhor Acurácia: {best_acc:.4f}")