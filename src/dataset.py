import os
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split


class BoneFractureDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ["not_fractured", "fractured"]  # 0 not fractured 1 fractured
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
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(30),
    transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])



def get_dataloaders(data_dir="data/BoneFractureDataset", batch_size=32):
    """merge train+test, then re-split properly holy mole"""
    all_images = []
    all_labels = []

    classes = ["not_fractured", "fractured"]

    
    for split in ["training", "testing"]:
        for idx, cls in enumerate(classes):
            cls_path = os.path.join(data_dir, split, cls)
            for fname in sorted(os.listdir(cls_path)):
                all_images.append(os.path.join(cls_path, fname))
                all_labels.append(idx)

    
    #stratified split 70% train, 15% val, 15% test
    train_imgs, temp_imgs, train_labels, temp_labels = train_test_split(
        all_images, all_labels, test_size=0.3, stratify=all_labels, random_state=42
    )
    val_imgs, test_imgs, val_labels, test_labels = train_test_split(
        temp_imgs, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
    )

    train_loader = DataLoader(
        ImageListDataset(train_imgs, train_labels, train_transform),
        batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        ImageListDataset(val_imgs, val_labels, val_transform),
        batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True,
    )
    test_loader = DataLoader(
        ImageListDataset(test_imgs, test_labels, val_transform),
        batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True,
    )

    print(f"teain: {len(train_imgs)}\t val: {len(val_imgs)}\t test: {len(test_imgs)}")
    return train_loader, val_loader, test_loader



class ImageListDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


#quick test
if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()
    

    batch_imgs, batch_labels = next(iter(train_loader))
    print(f"batch shape: {batch_imgs.shape}")
    print(f"labels: {batch_labels[:10]}")
    print(f"label distribution in batch: {batch_labels.sum().item()}/{len(batch_labels)} fractured")

    
