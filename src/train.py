import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataset import get_dataloaders
from model import BoneFractureModel


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        print(f"\tEarlyStopping: {self.counter}/{self.patience}")
        return self.counter >= self.patience


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in tqdm(loader, desc="Training", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in tqdm(loader, desc="Validating", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

def train(
    data_dir="data/BoneFractureDataset",
    batch_size=32,
    head_epochs=10,
    finetune_epochs=20,
    lr_head=1e-3,
    lr_backbone=1e-5,
    patience=7,
    save_dir="checkpoints",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader, test_loader = get_dataloaders(data_dir, batch_size)

    # Model
    model = BoneFractureModel().to(device)

    criterion = nn.CrossEntropyLoss()
    os.makedirs(save_dir, exist_ok=True)
    best_val_acc = 0.0

    #phase pne train head only
    print("\n" + "=" * 50)
    print("PHASE 1: Training classification head")
    print("=" * 50)

    model.freeze_backbone()
    optimizer = AdamW(model.backbone.fc.parameters(), lr=lr_head, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=head_epochs)
    early_stop = EarlyStopping(patience=patience)

    for epoch in range(head_epochs):
        start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - start
        print(f"Epoch {epoch+1}/{head_epochs} ({elapsed:.0f}s) | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print(f"\tSaved best model (val_acc: {val_acc:.4f})")

        if early_stop(val_loss):
            print("\tEarly stopping triggered")
            break

    # ===== Phase 2: Fine-tune backbone =====
    print("\n" + "=" * 50)
    print("PHASE 2: Fine-tuning backbone from layer3")
    print("=" * 50)

    model.unfreeze_backbone("layer3")
    optimizer = AdamW(model.get_param_groups(lr_head=lr_head * 0.1, lr_backbone=lr_backbone), weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=finetune_epochs)
    early_stop = EarlyStopping(patience=patience)

    for epoch in range(finetune_epochs):
        start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - start
        print(f"Epoch {epoch+1}/{finetune_epochs} ({elapsed:.0f}s) | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print(f"\tSaved best model (val_acc: {val_acc:.4f})")

        if early_stop(val_loss):
            print("\tEarly stopping triggered")
            break

    #final test evaluation
    print("\n" + "=" * 50)
    print("FINAL TEST EVAL")
    print("=" * 50)

    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pth")))
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    print(f"Best Val Acc: {best_val_acc:.4f}")

    return model


if __name__ == "__main__":
    train()