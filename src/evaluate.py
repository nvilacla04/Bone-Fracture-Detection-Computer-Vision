import torch
import matplotlib.pyplot as plt
from dataset import get_dataloaders, val_transform, BoneFractureDataset
from model import BoneFractureModel
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#load best model
model = BoneFractureModel().to(device)
model.load_state_dict(torch.load("checkpoints/best_model.pth"))
model.eval()

#get test data
_, _, test_loader = get_dataloaders()

all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(1).cpu()
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())


print(classification_report(all_labels, all_preds, target_names=["not_fractured", "fractured"]))
print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

