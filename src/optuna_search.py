"""
optuna hyperparameter search for bone fracture classifier 
run on VU servers L4 GPU
python src/optuna_search.py
"""


import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm
import json
import optuna

from dataset import (
    ImageListDataset,
    train_transform,
    val_transform,
)




def collect_all_images(data_dir="data/BoneFractureDataset"):
    """collect all image paths and labels"""
    all_images = []
    all_labels = []
    classes = ["not_fractured", "fractured"]

    for split in ["training", "testing"]:
        for idx, cls in enumerate(classes):
            cls_path = os.path.join(data_dir, split, cls)
            for fname in sorted(os.listdir(cls_path)):
                all_images.append(os.path.join(cls_path, fname))
                all_labels.append(idx)

    return all_images, all_labels




def build_model(num_classes=2, hidden_size=256, dropout1=0.5, dropout2=0.3):
    """build ResNet-50 with custom head"""
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)

    #freeze backbone
    for param in model.parameters():
        param.requires_grad = False

    # cstom head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout1),
        nn.Linear(in_features, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout2),
        nn.Linear(hidden_size, num_classes),
    )

    return model


def train_and_evaluate(model, train_loader, val_loader, device,
                       lr_head, lr_backbone, head_epochs, finetune_epochs,
                       trial=None):
    """train model + return best val acc"""
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    
    #PHASE 1: head only
    optimizer = AdamW(model.fc.parameters(), lr=lr_head, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=head_epochs)

    for epoch in range(head_epochs):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        #validation
        val_acc = evaluate(model, val_loader, device)
        best_val_acc = max(best_val_acc, val_acc)

        #optuna pruningstop bad trials early
        if trial:
            trial.report(val_acc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

                

    #PHASE 2: unfreeze from layer3
    layers_to_unfreeze = ["layer3", "layer4", "fc"]
    for name, param in model.named_parameters():
        for layer_name in layers_to_unfreeze:
            if layer_name in name:
                param.requires_grad = True


        

    #differential learning rates
    head_params = list(model.fc.parameters())
    head_ids = set(id(p) for p in head_params)
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids and p.requires_grad]

    
    optimizer = AdamW([
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_params, "lr": lr_head * 0.1},
    ], weight_decay=1e-4)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=finetune_epochs)

    for epoch in range(finetune_epochs):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        val_acc = evaluate(model, val_loader, device)
        best_val_acc = max(best_val_acc, val_acc)

        if trial:
            trial.report(val_acc, head_epochs + epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return best_val_acc






@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total



    
def objective(trial):
    """optuna objective function — one trial = one training run"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    #hyperparameters to search holy mole
    hidden_size = trial.suggest_categorical("hidden_size", [128, 256, 512])
    dropout1 = trial.suggest_float("dropout1", 0.2, 0.6, step=0.1)
    dropout2 = trial.suggest_float("dropout2", 0.1, 0.4, step=0.1)
    lr_head = trial.suggest_float("lr_head", 1e-4, 3e-3, log=True)
    lr_backbone = trial.suggest_float("lr_backbone", 1e-6, 1e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    head_epochs = trial.suggest_int("head_epochs", 3, 8)
    finetune_epochs = trial.suggest_int("finetune_epochs", 5, 15)



    #le data
    all_images, all_labels = collect_all_images()
    train_imgs, val_imgs, train_labels, val_labels = train_test_split(
        all_images, all_labels, test_size=0.2, stratify=all_labels, random_state=42
    )

    train_loader = DataLoader(
        ImageListDataset(train_imgs, train_labels, train_transform),
        batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        ImageListDataset(val_imgs, val_labels, val_transform),
        batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True,
    )


    #modl
    model = build_model(
        hidden_size=hidden_size,
        dropout1=dropout1,
        dropout2=dropout2,
    ).to(device)
    

    #train
    val_acc = train_and_evaluate(
        model, train_loader, val_loader, device,
        lr_head=lr_head,
        lr_backbone=lr_backbone,
        head_epochs=head_epochs,
        finetune_epochs=finetune_epochs,
        trial=trial,
    )

    return val_acc


    

if __name__ == "__main__":
    #create study with pruning
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        study_name="bone_fracture_search",
    )

    
    print("☝️🤓 Starting Optuna hyperparameter search!")
    print("=" * 50)

    study.optimize(objective, n_trials=20, show_progress_bar=True)

    #results 
    print("\n" + "=" * 50)
    print("SEARCH COMPLETE ☝️🤓")
    print("=" * 50)
    print(f"Best trial accuracy: {study.best_trial.value:.4f}")
    print(f"Best hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")

    
    #save em
    results = {
        "best_accuracy": study.best_trial.value,
        "best_params": study.best_trial.params,
        "all_trials": [
            {"number": t.number, "value": t.value, "params": t.params}
            for t in study.trials if t.value is not None
        ],
    }

    os.makedirs("results", exist_ok=True)
    with open("results/optuna_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nresults saved to results/optuna_results.json")

    #giggle beast clan out! ☝️🤓
