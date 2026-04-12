import torch
import torch.nn as nn
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score


class TrimodalFusion(nn.Module):
    PROJ_DIM = 512

    def __init__(self, image_dim=2048, video_dim=768, text_dim=768,
                 num_heads=8, dropout=0.3):
        super().__init__()
        d = self.PROJ_DIM
        self.image_proj = nn.Sequential(nn.Linear(image_dim, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(dropout))
        self.video_proj = nn.Sequential(nn.Linear(video_dim, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(dropout))
        self.text_proj  = nn.Sequential(nn.Linear(text_dim,  d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(dropout))
        self.visual_self_attn    = nn.MultiheadAttention(d, num_heads, dropout=dropout, batch_first=True)
        self.text_self_attn      = nn.MultiheadAttention(d, num_heads, dropout=dropout, batch_first=True)
        self.visual_to_text_attn = nn.MultiheadAttention(d, num_heads, dropout=dropout, batch_first=True)
        self.text_to_visual_attn = nn.MultiheadAttention(d, num_heads, dropout=dropout, batch_first=True)
        self.norm_visual = nn.LayerNorm(d)
        self.norm_text   = nn.LayerNorm(d)
        self.fusion_mlp  = nn.Sequential(nn.Linear(d*2, d), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(d))
        self.regression_head = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 1))

    def forward(self, text_emb, image_emb=None, video_emb=None, return_embedding=False):
        if image_emb is None and video_emb is None:
            raise ValueError("Provide either image_emb or video_emb.")
        if image_emb is not None and video_emb is not None:
            raise ValueError("Provide either image_emb or video_emb — not both.")

        visual = self.image_proj(image_emb) if image_emb is not None else self.video_proj(video_emb)
        text   = self.text_proj(text_emb)

        vis = visual.unsqueeze(1)
        txt = text.unsqueeze(1)

        vis, _ = self.visual_self_attn(vis, vis, vis)
        txt, _ = self.text_self_attn(txt, txt, txt)

        vis_cross, _ = self.visual_to_text_attn(query=vis, key=txt, value=txt)
        txt_cross, _ = self.text_to_visual_attn(query=txt, key=vis, value=vis)

        vis_out = self.norm_visual(vis_cross.squeeze(1))
        txt_out = self.norm_text(txt_cross.squeeze(1))

        fused = self.fusion_mlp(torch.cat([vis_out, txt_out], dim=1))
        score = self.regression_head(fused)

        if return_embedding:
            return score, fused
        return score


CrossModalAttention = TrimodalFusion


def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            text_emb  = batch['text_emb'].to(device)
            image_emb = batch.get('image_emb')
            video_emb = batch.get('video_emb')
            labels    = batch['likes'].to(device)
            if image_emb is not None: image_emb = image_emb.to(device)
            if video_emb is not None: video_emb = video_emb.to(device)
            preds = model(text_emb=text_emb, image_emb=image_emb, video_emb=video_emb).squeeze(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    return mean_absolute_error(all_labels, all_preds), r2_score(all_labels, all_preds), spearmanr(all_labels, all_preds).correlation
