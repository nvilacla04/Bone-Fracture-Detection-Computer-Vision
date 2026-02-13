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
            img = self.transform(img)

        return img, label

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])



def get_dataloaders(data_dir="data/BoneFractureDataset", batch_size=32, val_split=0.15):
    """
    creates train, val, and test dataloaders
    splits training data into train/val since dataset only has training/testing folders.
    """
    #full training set (will split val later)
    full_train = BoneFractureDataset(
        root_dir=os.path.join(data_dir, "training"),
        transform=None  # applied later per split
    )

    #split into train and val
    val_size = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size
    train_subset, val_subset = random_split(full_train, [train_size, val_size])

    #wrap subsets with transforms
    train_loader = DataLoader(
        TransformSubset(train_subset, train_transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        TransformSubset(val_subset, val_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    #test
    test_set = BoneFractureDataset(
        root_dir=os.path.join(data_dir, "testing"),
        transform=val_transform,
    )


    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )


    print(f"Train: {train_size}\t Val: {val_size}\t Test: {len(test_set)}")
    return train_loader, val_loader, test_loader


class TransformSubset(Dataset):
    """applies transform to a subset since reandom_split loses transforms"""


    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform



    def __len__(self):
        return len(self.subset)
    


    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label



#quick test
if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()

    batch_imgs, batch_labels = next(iter(train_loader))
    print(f"Batch shape: {batch_imgs.shape}")
    print(f"Labels: {batch_labels[:10]}")
    print(f"Label distribution in batch: {batch_labels.sum().item()}/{len(batch_labels)} fractured")


            
