import torch
import torch.nn as nn
from torchvision import models

class ResNetEncoder(nn.Module):
    def __init__(self, embed_dim=256, dropout=0.3, freeze_base=False):
        super().__init__()
        base = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        if freeze_base:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.projection = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(2048, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.embed_dim = embed_dim

    def forward(self, x):
        # [B, 3, 224, 224] → [B, 2048, 1, 1] → [B, 2048] → [B, 256]
        features = self.backbone(x)
        features = features.flatten(start_dim=1)
        return self.projection(features)
