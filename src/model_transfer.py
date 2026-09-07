import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class MaskResNet18(nn.Module):
    """ResNet-18 model fine-tuned for binary face mask classification."""

    def __init__(self, freeze_backbone: bool = False, dropout_rate: float = 0.3) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        self.backbone = resnet18(weights=weights)

        # Optional: freeze initial convolutional layers for feature extraction
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace default 1000-class fc head with a custom binary classification head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return  self.backbone(x))