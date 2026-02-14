import matplotlib.pyplot as plt
from PIL import Image
import os
import random

data_dir = "data/BoneFractureDataset"

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for col, cls in enumerate(["fractured", "not_fractured"]):
    for row, split in enumerate(["training", "testing"]):
        path = os.path.join(data_dir, split, cls)
        imgs = os.listdir(path)
        samples = random.sample(imgs, 2)
        
        for i in range(2):
            img = Image.open(os.path.join(path, samples[i]))
            ax = axes[row][col * 2 + i]
            ax.imshow(img)
            ax.set_title(f"{split}/{cls}\n{img.size}")
            ax.axis("off")

plt.tight_layout()
plt.savefig("data_comparison.png", dpi=150)
plt.show()
print("saved to data_comparison.png")