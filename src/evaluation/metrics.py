"""
Evaluation Metrics
- MAE (Mean Absolute Error)
- R2 Score
- SRCC (Spearman Rank Correlation Coefficient)
"""
"""
metrics.py — ViralVisions AI Evaluation Suite

Evaluates and compares:
    1. XGBoost baseline   (ViralityXGBoost)
    2. Deep fusion model  (TrimodalFusion)

Metrics reported for both:
    - MAE   (Mean Absolute Error)       → lower is better
    - R²    (Coefficient of Determination) → higher is better, 1.0 = perfect
    - SRCC  (Spearman Rank Correlation) → higher is better, measures ranking ability

Usage:
    from src.evaluation.metrics import evaluate_all, print_comparison_table
"""

import torch
import numpy as np
import pandas as pd
from typing import Optional
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


# ── Core metric functions ──────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute all regression metrics for a single model's predictions.

    Args:
        y_true : ground truth virality scores/likes
        y_pred : model predictions

    Returns:
        dict with mae, rmse, r2, spearman, pearson
    """
    mae      = mean_absolute_error(y_true, y_pred)
    rmse     = np.sqrt(mean_squared_error(y_true, y_pred))
    r2       = r2_score(y_true, y_pred)
    spearman = spearmanr(y_true, y_pred).correlation
    pearson  = pearsonr(y_true, y_pred)[0]

    return {
        "MAE":      round(mae, 4),
        "RMSE":     round(rmse, 4),
        "R²":       round(r2, 4),
        "Spearman": round(spearman, 4),
        "Pearson":  round(pearson, 4),
    }


# ── Deep fusion model evaluation ──────────────────────────────────────────────

def evaluate_deep_model(model: torch.nn.Module,
                        dataloader,
                        device: str) -> dict:
    """
    Run TrimodalFusion on a dataloader and compute metrics.

    Dataloader batches must contain:
        'text_emb'              : [B, 768]
        'image_emb' OR 'video_emb' : [B, 2048] or [B, 768]
        'likes'                 : [B]  ground truth

    Returns:
        dict of metrics
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            text_emb  = batch['text_emb'].to(device)
            image_emb = batch.get('image_emb')
            video_emb = batch.get('video_emb')
            labels    = batch['likes'].to(device)

            if image_emb is not None:
                image_emb = image_emb.to(device)
            if video_emb is not None:
                video_emb = video_emb.to(device)

            preds = model(
                text_emb=text_emb,
                image_emb=image_emb,
                video_emb=video_emb,
            ).squeeze(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return compute_metrics(np.array(all_labels), np.array(all_preds))


# ── XGBoost baseline evaluation ───────────────────────────────────────────────

def evaluate_xgboost(model,
                     X_test: pd.DataFrame,
                     y_test: np.ndarray) -> dict:
    """
    Evaluate ViralityXGBoost on held-out test set.

    Args:
        model  : trained ViralityXGBoost instance
        X_test : feature DataFrame (preprocessed)
        y_test : ground truth labels

    Returns:
        dict of metrics
    """
    preds = model.predict(X_test)
    return compute_metrics(np.array(y_test), np.array(preds))


# ── Side-by-side comparison ────────────────────────────────────────────────────

def evaluate_all(deep_model:    torch.nn.Module,
                 deep_loader,
                 device:        str,
                 xgb_model      = None,
                 X_test:        Optional[pd.DataFrame] = None,
                 y_test:        Optional[np.ndarray]   = None) -> dict:
    """
    Evaluate both models and return results dict.

    Args:
        deep_model  : trained TrimodalFusion
        deep_loader : DataLoader with text/image/video/likes batches
        device      : 'cuda' or 'cpu'
        xgb_model   : trained ViralityXGBoost (optional)
        X_test      : XGBoost feature DataFrame (required if xgb_model given)
        y_test      : ground truth labels       (required if xgb_model given)

    Returns:
        {
            'TrimodalFusion': { metric: value, ... },
            'XGBoost':        { metric: value, ... },   # only if xgb_model given
        }
    """
    results = {}

    print("⚙️  Evaluating TrimodalFusion...")
    results['TrimodalFusion'] = evaluate_deep_model(deep_model, deep_loader, device)

    if xgb_model is not None:
        assert X_test is not None and y_test is not None, \
            "X_test and y_test required when xgb_model is provided"
        print("⚙️  Evaluating XGBoost baseline...")
        results['XGBoost'] = evaluate_xgboost(xgb_model, X_test, y_test)

    return results


# ── Pretty print ───────────────────────────────────────────────────────────────

def print_comparison_table(results: dict) -> None:
    """
    Print a clean side-by-side comparison table.

    Args:
        results : output of evaluate_all()
    """
    metrics = ["MAE", "RMSE", "R²", "Spearman", "Pearson"]

    # Header
    models = list(results.keys())
    col_w  = 16
    header = f"{'Metric':<12}" + "".join(f"{m:>{col_w}}" for m in models)

    print("\n" + "─" * len(header))
    print("  ViralVisions AI — Model Comparison")
    print("─" * len(header))
    print(header)
    print("─" * len(header))

    for metric in metrics:
        row = f"{metric:<12}"
        values = [results[m].get(metric, "N/A") for m in models]

        # Bold the better value (lower MAE/RMSE, higher rest)
        if len(values) == 2 and all(isinstance(v, float) for v in values):
            if metric in ("MAE", "RMSE"):
                better_idx = int(values[1] < values[0])
            else:
                better_idx = int(values[1] > values[0])
            for i, v in enumerate(values):
                marker = " ✓" if i == better_idx else "  "
                row += f"{str(v) + marker:>{col_w}}"
        else:
            for v in values:
                row += f"{str(v):>{col_w}}"

        print(row)

    print("─" * len(header))
    print("  ✓ = better value\n")


# ── Colab-ready runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Quick test with dummy data — run directly in Colab to verify the script works.

    Replace dummy_loader and dummy_xgb with real objects for actual evaluation.
    """
    import sys
    sys.path.append('/content/viralvisions-ai')

    from src.models.fusion.cross_modal_attention import TrimodalFusion
    from src.models.baseline.xgboost_model import ViralityXGBoost
    from torch.utils.data import DataLoader, TensorDataset

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B = 64  # dummy batch size

    # ── Dummy deep model test ──
    model = TrimodalFusion().to(device).eval()

    dummy_text  = torch.randn(B, 768)
    dummy_image = torch.randn(B, 2048)
    dummy_likes = torch.rand(B)

    dataset = TensorDataset(dummy_text, dummy_image, dummy_likes)

    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self): pass
        def __len__(self): return B
        def __getitem__(self, i):
            return {
                'text_emb':  dummy_text[i],
                'image_emb': dummy_image[i],
                'likes':     dummy_likes[i],
            }

    loader = DataLoader(DummyDataset(), batch_size=16)

    # ── Dummy XGBoost test ──
    xgb = ViralityXGBoost()
    X_dummy = pd.DataFrame(np.random.randn(B, 10),
                           columns=[f'feat_{i}' for i in range(10)])
    y_dummy = np.random.rand(B)
    xgb.model.fit(X_dummy, y_dummy)  # fit on dummy so predict works

    results = evaluate_all(model, loader, device,
                           xgb_model=xgb, X_test=X_dummy, y_test=y_dummy)

    print_comparison_table(results)
