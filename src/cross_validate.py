"""
kfold cross validation with best hyperparameters from optuna!
run after optuna
"""




import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from dataset import ImageListDataset, train_transform, val_transform
from optuna_search import collect_all_images, build_model, evaluate




def cross_validate(params, n_folds=5, data_dir="data/BoneFractureDataset"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Running {n_folds}-fold cross-validation")
    print(f"Params: {json.dumps(params, indent=2)}")
    print("=" * 50)

    

    all_images, all_labels = collect_all_images(data_dir)
    all_images = np.array(all_images)
    all_labels = np.array(all_labels)
    

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_results = []
    all_preds = []
    all_true = []

    
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_images, all_labels)):
        print(f"\n--- fold {fold + 1}/{n_folds} ---")

        train_imgs = all_images[train_idx].tolist()
        train_labels = all_labels[train_idx].tolist()
        val_imgs = all_images[val_idx].tolist()
        val_labels = all_labels[val_idx].tolist()

        train_loader = DataLoader(
            ImageListDataset(train_imgs, train_labels, train_transform),
            batch_size=params.get("batch_size", 32),
            shuffle=True, num_workers=0, pin_memory=True,
        )
        val_loader = DataLoader(
            ImageListDataset(val_imgs, val_labels, val_transform),
            batch_size=params.get("batch_size", 32),
            shuffle=False, num_workers=0, pin_memory=True,
        )

        #build model
        model = build_model(
            hidden_size=params.get("hidden_size", 256),
            dropout1=params.get("dropout1", 0.5),
            dropout2=params.get("dropout2", 0.3),
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        best_val_acc = 0.0
        best_state = None


        
        #PHASE 1: head only
        optimizer = AdamW(
            model.fc.parameters(),
            lr=params.get("lr_head", 3e-4),
            weight_decay=1e-4,
        )
        head_epochs = params.get("head_epochs", 5)
        scheduler = CosineAnnealingLR(optimizer, T_max=head_epochs)
        

        for epoch in range(head_epochs):
            model.train()
            running_loss = 0.0
            for imgs, labels in tqdm(train_loader, desc=f"F{fold+1} Head E{epoch+1}", leave=False):
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(imgs), labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            scheduler.step()

            val_acc = evaluate(model, val_loader, device)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  Head Epoch {epoch+1}/{head_epochs} | Val Acc: {val_acc:.4f}")
            

        #PHASE 2: fine-tune
        layers_to_unfreeze = ["layer3", "layer4", "fc"]
        for name, param in model.named_parameters():
            for layer_name in layers_to_unfreeze:
                if layer_name in name:
                    param.requires_grad = True

        head_params = list(model.fc.parameters())
        head_ids = set(id(p) for p in head_params)
        backbone_params = [p for p in model.parameters() if id(p) not in head_ids and p.requires_grad]
        

        optimizer = AdamW([
            {"params": backbone_params, "lr": params.get("lr_backbone", 1e-5)},
            {"params": head_params, "lr": params.get("lr_head", 3e-4) * 0.1},
        ], weight_decay=1e-4)

        

        finetune_epochs = params.get("finetune_epochs", 15)
        scheduler = CosineAnnealingLR(optimizer, T_max=finetune_epochs)
        

        for epoch in range(finetune_epochs):
            model.train()
            for imgs, labels in tqdm(train_loader, desc=f"F{fold+1} FT E{epoch+1}", leave=False):
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(imgs), labels)
                loss.backward()
                optimizer.step()
            scheduler.step()
            

            val_acc = evaluate(model, val_loader, device)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  FT Epoch {epoch+1}/{finetune_epochs} | Val Acc: {val_acc:.4f}")
            

        #evaluate best model on this fold
        model.load_state_dict(best_state)
        model.to(device)

        fold_preds = []
        fold_true = []
        model.eval()
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                preds = model(imgs).argmax(1).cpu().numpy()
                fold_preds.extend(preds)
                fold_true.extend(labels.numpy())

        all_preds.extend(fold_preds)
        all_true.extend(fold_true)

        fold_acc = np.mean(np.array(fold_preds) == np.array(fold_true))
        fold_results.append(fold_acc)
        print(f"\n  Fold {fold + 1} Best Accuracy: {fold_acc:.4f}")

        

    #summary
    print("\n" + "=" * 50)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 50)
    for i, acc in enumerate(fold_results):
        print(f"  Fold {i + 1}: {acc:.4f}")
    print(f"\n  Mean Accuracy: {np.mean(fold_results):.4f} +/- {np.std(fold_results):.4f}")
    print(f"\nClassification Report (all folds combined):")
    print(classification_report(
        all_true, all_preds,
        target_names=["not_fractured", "fractured"]
    ))
    print("Confusion Matrix (all folds combined):")
    print(confusion_matrix(all_true, all_preds))

    # Save results
    os.makedirs("results", exist_ok=True)
    cv_results = {
        "fold_accuracies": fold_results,
        "mean_accuracy": float(np.mean(fold_results)),
        "std_accuracy": float(np.std(fold_results)),
        "params": params,
    }
    with open("results/cv_results.json", "w") as f:
        json.dump(cv_results, f, indent=2)

    print("\nresults saved to results/cv_results.json")




if __name__ == "__main__":
    #load best params from Optuna or use defaults if not available
    params_path = "results/optuna_results.json"

    if os.path.exists(params_path):
        with open(params_path) as f:
            optuna_results = json.load(f)
        params = optuna_results["best_params"]
        print(f"Loaded best params from Optuna (accuracy: {optuna_results['best_accuracy']:.4f})")
    else:
        print("No Optuna results found, using defaults")
        params = {
            "hidden_size": 256,
            "dropout1": 0.5,
            "dropout2": 0.3,
            "lr_head": 3e-4,
            "lr_backbone": 1e-5,
            "batch_size": 32,
            "head_epochs": 5,
            "finetune_epochs": 15,
        }

    cross_validate(params)

    
