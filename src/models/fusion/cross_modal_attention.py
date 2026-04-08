import torch
import torch.nn as nn
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score

class CrossModalAttention(nn.Module):
    """
    Bimodal fusion — Image (ResNet-101) + Text (BERT)
    Uses genuine cross-modal attention, not simple concatenation.
    """

    def __init__(self, embed_dim=256, num_heads=4, dropout=0.3):
        super().__init__()

        # Image attends to Text
        self.image_to_text_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )

        # Text attends to Image
        self.text_to_image_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )

        self.norm_image = nn.LayerNorm(embed_dim)
        self.norm_text  = nn.LayerNorm(embed_dim)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embed_dim),
        )

        self.regression_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, image_emb, text_emb):
        """
        Args:
            image_emb : [B, 256] from ResNetEncoder
            text_emb  : [B, 256] from BertEncoder
        Returns:
            virality_score : [B, 1]
        """
        img = image_emb.unsqueeze(1)   # [B, 1, 256]
        txt = text_emb.unsqueeze(1)    # [B, 1, 256]

        img_attended, _ = self.image_to_text_attn(query=img, key=txt, value=txt)
        img_attended = self.norm_image(img_attended.squeeze(1))

        txt_attended, _ = self.text_to_image_attn(query=txt, key=img, value=img)
        txt_attended = self.norm_text(txt_attended.squeeze(1))

        fused = torch.cat([img_attended, txt_attended], dim=1)
        fused = self.fusion_mlp(fused)
        score = self.regression_head(fused)

        return score


def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            image_emb = batch["image_emb"].to(device)
            text_emb  = batch["text_emb"].to(device)
            labels    = batch["likes"].to(device)
            preds     = model(image_emb, text_emb).squeeze(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    mae  = mean_absolute_error(all_labels, all_preds)
    r2   = r2_score(all_labels, all_preds)
    srcc = spearmanr(all_labels, all_preds).correlation

    return mae, r2, srcc
