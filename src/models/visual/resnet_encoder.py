import torch
import torch.nn as nn
from torchvision import models


class ResNetEncoder(nn.Module):
    """
    ResNet-101 visual encoder for static Instagram posts.

    Phase 3  → embed_dim=256  (projection head active, for quick demo)
    Post-eval → embed_dim=2048 (projection head is identity, raw features
                                 passed to TrimodalFusion's own proj layer)

    Args:
        embed_dim   : output dimension. Use 2048 to skip internal projection.
        dropout     : dropout before projection.
        freeze_base : freeze ResNet backbone weights (recommended for fine-tune).
    """

    def __init__(self, embed_dim: int = 2048, dropout: float = 0.3,
                 freeze_base: bool = False):
        super().__init__()

        base = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)

        # Strip final FC layer → outputs [B, 2048, 1, 1]
        self.backbone = nn.Sequential(*list(base.children())[:-1])

        if freeze_base:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.embed_dim = embed_dim

        # If embed_dim == 2048, projection is identity (no-op)
        # If embed_dim != 2048 (e.g. 256 for Phase 3), project down
        if embed_dim == 2048:
            self.projection = nn.Identity()
        else:
            self.projection = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(2048, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, 3, 224, 224]
        Returns:
            [B, embed_dim]  →  [B, 2048] post-eval  |  [B, 256] Phase 3
        """
        features = self.backbone(x)           # [B, 2048, 1, 1]
        features = features.flatten(start_dim=1)  # [B, 2048]
        return self.projection(features)          # [B, embed_dim]
