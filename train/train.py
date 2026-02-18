import torch
from torch.utils.data import DataLoader
from dataset import FERDataset
from model import get_model

device = torch.device("cuda")

train_dataset = FERDataset("../data/fer2013.csv", usage="Training")
val_dataset   = FERDataset("../data/fer2013.csv", usage="PublicTest")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = get_model().to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

epochs = 10
best_acc = 0

for epoch in range(epochs):

    # ===================== TREINO =====================
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = total_loss / len(train_loader)
    train_acc = correct / total

    # ===================== VALIDAÇÃO =====================
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

    # ===================== SALVAR MELHOR MODELO =====================
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "../models/best_emotion.pt")
        print(">> Melhor modelo salvo!")



torch.save(model.state_dict(), "../models/resnet_emotion.pt")
