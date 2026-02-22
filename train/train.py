import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import FERDataset
from model import get_model
import numpy as np  
import os

# Configuração de dispositivo e escalonador de precisão
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = torch.cuda.amp.GradScaler() 

# --- CARREGAMENTO (Batch reduzido ao mínimo) ---
train_dataset = FERDataset("../data/fer2013.csv", usage="Training")
val_dataset = FERDataset("../data/fer2013.csv", usage="PublicTest")

# Batch size 4 para caber em qualquer GPU, num_workers 0 para não travar Windows
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

model = get_model().to(device)
# criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

labels = train_dataset.data["emotion"].values
counts = np.bincount(labels)
weights = torch.FloatTensor(len(counts) / (len(counts) * counts)).to(device)
criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

def train_loop(epochs, model, loader, optimizer, scheduler=None):
    best_acc = 0
    # Acumular gradientes por 8 passos (Batch virtual = 4 * 8 = 32)
    accumulation_steps = 8 

    for epoch in range(epochs):
        model.train()
        acc_loss, correct, total = 0, 0, 0
        optimizer.zero_grad()

        for i, (imgs, lbls) in enumerate(loader):
            imgs, lbls = imgs.to(device), lbls.to(device)

            # Precisão Mista (Economiza muita VRAM)
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
                loss = loss / accumulation_steps 

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            acc_loss += loss.item() * accumulation_steps
            _, p = torch.max(outputs, 1)
            correct += (p == lbls).sum().item()
            total += lbls.size(0)

        # Validação
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for v_imgs, v_lbls in val_loader:
                v_imgs, v_lbls = v_imgs.to(device), v_lbls.to(device)
                with torch.cuda.amp.autocast():
                    v_out = model(v_imgs)
                _, vp = torch.max(v_out, 1)
                v_correct += (vp == v_lbls).sum().item()
                v_total += v_lbls.size(0)
        
        if scheduler:
            scheduler.step()

        v_acc = v_correct / v_total
        print(f"Epoch {epoch+1} | Loss: {acc_loss/len(loader):.3f} | Val Acc: {v_acc:.3f}")
        
        if v_acc > best_acc:
            best_acc = v_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }
            torch.save(checkpoint, "../models/checkpoint_fer.pth")
            # Salva o arquivo .pt puro para a webcam também
            torch.save(model.state_dict(), "../models/best_emotion_v2.pt")
            print(">> Checkpoint Salvo!")
        
        # Limpeza forçada de memória ao fim de cada época
        torch.cuda.empty_cache()

if __name__ == '__main__':
    # ETAPA 1: CABEÇA
    print("Etapa 1: Treinando Classifier...")
    for param in model.parameters(): param.requires_grad = False
    for param in model.classifier.parameters(): param.requires_grad = True
    
    opt1 = torch.optim.Adam(model.classifier.parameters(), lr=0.001)
    train_loop(5, model, train_loader, opt1)

    # ETAPA 2: FINE-TUNING TOTAL
    print("\nEtapa 2: Fine-tuning Total (SGD + LR Progressivo)...")
    for param in model.parameters(): param.requires_grad = True

    # LR aumentado para 0.01 para forçar a descida do Loss
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    # Ciclo mais longo (50 épocas) para dar tempo de convergir
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    train_loop(50, model, train_loader, optimizer, scheduler=scheduler)