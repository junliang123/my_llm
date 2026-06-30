import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import HerbDataset
from model import ResNet

BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
NUM_CLASSES = 27
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    #transforms.RandomHorizontalFlip(),
    #transforms.RandomRotation(10),
    #transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

train_dataset = HerbDataset(
    txt_path="data/split/train.txt",
    image_root="data/raw/train",
    transform=train_transform,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

val_dataset = HerbDataset(
    txt_path="data/split/val.txt",
    image_root="data/raw/val",
    transform=val_transform,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

model = timm.create_model(
    "convnext_tiny",
    pretrained=True,
    num_classes=27,
).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        out = model(images)
        loss = criterion(out, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, pred = torch.max(out, 1)
        running_loss += loss.item()*images.size(0)
        running_correct += torch.sum(pred == labels).item()
        total += labels.size(0)
    epoch_loss = running_loss / total
    epoch_acc = running_correct / total
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            out = model(images)
            loss = criterion(out, labels)

            _, pred = torch.max(out, 1)
            running_loss += loss.item()*images.size(0)
            running_correct += torch.sum(pred == labels).item()
            total += labels.size(0)
    epoch_loss = running_loss / total
    epoch_acc = running_correct / total
    return epoch_loss, epoch_acc

for epoch in range(NUM_EPOCHS):
    train_loss, train_acc = train_one_epoch(
        model=model,
        loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE,
    )
    val_loss, val_acc = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=DEVICE,
    )
    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
        f"train_loss: {train_loss:.4f} "
        f"train_acc: {train_acc:.4f} "
        f"val_loss: {val_loss:.4f} "
        f"val_acc: {val_acc:.4f}"
    )