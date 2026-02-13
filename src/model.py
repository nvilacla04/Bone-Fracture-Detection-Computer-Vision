import torch
import torch.nn as nn
from torchvision import models


class BoneFractureModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()

        #resnet50 for now
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)

        #freeze backbone initially
        self.freeze_backbone()

        #replace classification head
        in_features = self.backbone.fc.in_features  # 2048
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        #unfreeze fc head if exists
        if hasattr(self.backbone, "fc"):
            for param in self.backbone.fc.parameters():
                param.requires_grad = True

    def unfreeze_backbone(self, from_layer="layer3"):
        """
        unfreeze from a specific layer onwards
        ResNet-50 layers: conv1, bn1, layer1, layer2, layer3, layer4, fc
        Default unfreezes layer3 + layer4 + fc (gradual unfreezing)
        """
        layers = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"]
        unfreeze = False

        for name in layers:
            if name == from_layer:
                unfreeze = True
            if unfreeze:
                layer = getattr(self.backbone, name)
                for param in layer.parameters():
                    param.requires_grad = True

    def get_param_groups(self, lr_head=1e-3, lr_backbone=1e-5):
        """
        returns parameter groups with different learning rates
        higher LR for the head lower for pretrained backbone layers
        """
        head_params = list(self.backbone.fc.parameters())
        head_ids = set(id(p) for p in head_params)
        backbone_params = [p for p in self.parameters() if id(p) not in head_ids and p.requires_grad]

        return [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr_head},
        ]

    def count_params(self):
        """print trainable vs total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")
        return trainable, total


#quick test
if __name__ == "__main__":
    model = BoneFractureModel()

    print("=" * 30)
    print("Frozen backbone (head only)")
    print("=" * 30)
    model.count_params()

    print("\n")
    print("=" * 30)
    print("Unfrozen from layer3")
    model.unfreeze_backbone("layer3")
    model.count_params()

    #test forward pass
    dummy = torch.randn(4, 3, 224, 224)
    out = model(dummy)
    print(f"\noutput shape: {out.shape}")  #[4, 2]
    print(f"predictions: {out.argmax(dim=1)}")
