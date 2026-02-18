import torch
from torch.utils.data import DataLoader
from dataset import FERDataset
from model import get_model
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# DATASETS
# =========================

train_dataset = FERDataset("../data/fer2013.csv", usage="Training")
val_dataset   = FERDataset("../data/fer2013.csv", usage="PublicTest")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)

# =========================
# MODEL
# =========================

model = get_model().to(device)

# =========================
# FREEZE BACKBONE (FASE 1)
# =========================

for param in model.parameters():
    param.requires_grad = False

for param in model.fc.parameters():
    param.requires_grad = True

# =========================
# CLASS WEIGHTING
# =========================

labels = train_dataset.data["emotion"].values
class_counts = Counter(labels)

total_samples = sum(class_counts.values())
num_classes = 7

weights = [total_samples / class_counts[i] for i in range(num_classes)]
weights = torch.tensor(weights, dtype=torch.float).to(device)

criterion = torch.nn.CrossEntropyLoss(weight=weights)

# =========================
# OPTIMIZER (FASE 1)
# =========================

optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-4)

# =========================
# MIXED PRECISION
# =========================

scaler = torch.cuda.amp.GradScaler()

# =========================
# TREINO FASE 1
# =========================

epochs_phase1 = 5
best_acc = 0

print("\n===== FASE 1: TREINANDO SOMENTE FC =====")

for epoch in range(epochs_phase1):

    # -------- TREINO --------
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = total_loss / len(train_loader)
    train_acc = correct / total

    # -------- VALIDAÇÃO --------
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0

    with torch.no_grad():
        for images, labels in val_loader:

            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss = val_loss / len(val_loader)
    val_acc = val_correct / val_total

    print(f"\nEpoch {epoch+1}")
    print(f"Train loss: {train_loss:.3f} | Train acc: {train_acc:.3f}")
    print(f"Val   loss: {val_loss:.3f} | Val   acc: {val_acc:.3f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "../models/best_emotion.pt")
        print(">> Melhor modelo salvo!")

# =========================
# FASE 2 — DESTRAVAR LAYER4
# =========================

print("\n===== FASE 2: FINE-TUNING PROFUNDO =====")

for param in model.layer4.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=3e-5
)

epochs_phase2 = 7

for epoch in range(epochs_phase2):

    # -------- TREINO --------
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = total_loss / len(train_loader)
    train_acc = correct / total

    # -------- VALIDAÇÃO --------
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0

    with torch.no_grad():
        for images, labels in val_loader:

            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss = val_loss / len(val_loader)
    val_acc = val_correct / val_total

    print(f"\nEpoch {epoch+1}")
    print(f"Train loss: {train_loss:.3f} | Train acc: {train_acc:.3f}")
    print(f"Val   loss: {val_loss:.3f} | Val   acc: {val_acc:.3f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "../models/best_emotion.pt")
        print(">> Melhor modelo salvo!")

print("\nTreinamento finalizado.")
print("Melhor Val Acc:", best_acc)
