import os
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

class BoneFractureDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ["not_fractured", "fractured"] #0 not fractured 1 fractured 
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = []
        self.labels = []

        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            for fname in sorted(os.listdir(cls_path)):
                fpath = os.path.join(cls_path, fname)
                
                try:
                    Image.open(fpath).verify()
                    self.images.append(fpath)
                    self.labels.append(self.class_to_idx[cls])

                except Exception:
                    print(f"smth went wrong bro {fpath}")

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            img = self.transofrm(img)

        return img, label
    
    

        
