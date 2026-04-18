import os, sys, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

sys.path.insert(0, '/kaggle/working/viralvisions-ai')

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_CSV    = '/kaggle/working/data/flickr_with_sharerate.csv'
FLICKR_IMGS = '/kaggle/working/data/flickr30k/flickr30k_images/flickr30k_images'
CKPT_DIR    = '/kaggle/working/checkpoints'
os.makedirs(CKPT_DIR, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
EMBED_DIM  = 2048
BATCH_SIZE = 32
EPOCHS     = 3
LR         = 3e-4
MAX_LEN    = 128
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

# ── Dataset ───────────────────────────────────────────────────────────────────
class FlickrShareDataset(Dataset):
    def __init__(self, df, img_dir, augment=False):
        self.df        = df.reset_index(drop=True)
        self.img_dir   = img_dir
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        t = [transforms.Resize((224, 224))]
        if augment:
            t += [transforms.RandomHorizontalFlip(),
                  transforms.ColorJitter(0.2, 0.2)]
        t += [transforms.ToTensor(),
              transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]
        self.transform = transforms.Compose(t)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row     = self.df.iloc[idx]
        caption = str(row['caption']).strip()
        label   = float(row['share_rate']) / 100.0  # normalise to 0-1

        img_path = os.path.join(self.img_dir, row['image_name'].strip())
        try:
            tensor = self.transform(Image.open(img_path).convert('RGB'))
        except Exception:
            tensor = torch.zeros(3, 224, 224)

        enc = self.tokenizer(caption, max_length=MAX_LEN,
                             padding='max_length', truncation=True,
                             return_tensors='pt')
        return {
            'image':          tensor,
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label':          torch.tensor(label, dtype=torch.float32),
        }

# ── Models ────────────────────────────────────────────────────────────────────
def load_models(device):
    from src.models.visual.resnet_encoder import ResNetEncoder
    from src.models.text.bert_encoder import ViralBertEncoder
    from src.models.fusion.cross_modal_attention import TrimodalFusion

    visual = ResNetEncoder(embed_dim=EMBED_DIM).to(device)
    bert   = ViralBertEncoder().to(device)
    fusion = TrimodalFusion(image_dim=EMBED_DIM, text_dim=768).to(device)

    # Freeze BERT backbone
    for p in bert.bert.parameters():
        p.requires_grad = False

    trainable = sum(p.numel() for p in fusion.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")
    return visual, bert, fusion

# ── Train / Eval ──────────────────────────────────────────────────────────────
def run_epoch(visual, bert, fusion, loader, optimizer, criterion, device, train=True):
    if train:
        visual.train(); fusion.train(); bert.eval()
    else:
        visual.eval(); fusion.eval(); bert.eval()

    total_loss = 0
    all_preds, all_labels = [], []

    for batch in loader:
        images    = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attn_mask = batch['attention_mask'].to(device)
        labels    = batch['label'].to(device)

        with torch.no_grad() if not train else torch.enable_grad():
            text_emb = bert(input_ids, attn_mask)
            img_emb  = visual(images)
            score    = fusion(text_emb=text_emb,
                              image_emb=img_emb).squeeze(1)
            loss     = criterion(score, labels)

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(visual.parameters()) + list(fusion.parameters()), 1.0)
            optimizer.step()

        total_loss += loss.item()
        all_preds.extend(score.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    p = np.array(all_preds)
    l = np.array(all_labels)
    return (total_loss / len(loader),
            mean_absolute_error(l, p),
            r2_score(l, p),
            spearmanr(l, p).correlation)

# ── XGBoost baseline ──────────────────────────────────────────────────────────
def xgboost_baseline(train_df, test_df):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    import xgboost as xgb

    print("\nXGBoost baseline...")
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
        ('xgb',   xgb.XGBRegressor(n_estimators=300, learning_rate=0.05,
                                    max_depth=6, verbosity=0))
    ])
    pipe.fit(train_df['caption'].fillna(''), train_df['share_rate']/100)
    preds = pipe.predict(test_df['caption'].fillna(''))
    truth = test_df['share_rate'].values / 100

    mae  = mean_absolute_error(truth, preds)
    r2   = r2_score(truth, preds)
    srcc = spearmanr(truth, preds).correlation
    print(f"  XGBoost  MAE={mae:.4f}  R²={r2:.4f}  Spearman={srcc:.4f}")
    return {'MAE': mae, 'R2': r2, 'Spearman': srcc}

# ── Comparison table ──────────────────────────────────────────────────────────
def print_comparison(deep, xgb):
    print("\n" + "═"*52)
    print("  ViralVisions — Share Rate Prediction")
    print("═"*52)
    print(f"{'Metric':<12}{'TrimodalFusion':>18}{'XGBoost':>16}")
    print("─"*52)
    for m in ['MAE','R2','Spearman']:
        dv, xv = deep[m], xgb[m]
        better_d = ' ✓' if (dv < xv if m=='MAE' else dv > xv) else '  '
        better_x = ' ✓' if (xv < dv if m=='MAE' else xv > dv) else '  '
        print(f"{m:<12}{str(round(dv,4))+better_d:>18}{str(round(xv,4))+better_x:>16}")
    print("═"*52)

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    df = pd.read_csv(DATA_CSV)
    print(f"Loaded {len(df)} rows")

    train_val, test_df = train_test_split(df, test_size=0.10, random_state=42)
    train_df, val_df   = train_test_split(train_val, test_size=0.15, random_state=42)
    print(f"Train:{len(train_df)}  Val:{len(val_df)}  Test:{len(test_df)}")

    # XGBoost first
    xgb_metrics = xgboost_baseline(train_df, test_df)

    # Data loaders
    train_loader = DataLoader(FlickrShareDataset(train_df, FLICKR_IMGS, augment=True),
                              batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(FlickrShareDataset(val_df, FLICKR_IMGS),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(FlickrShareDataset(test_df, FLICKR_IMGS),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    visual, bert, fusion = load_models(DEVICE)
    optimizer = torch.optim.AdamW(
        list(visual.parameters()) + list(fusion.parameters()),
        lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()

    print(f"\nTraining {EPOCHS} epochs on {DEVICE}")
    print("─"*70)

    best_val = float('inf')
    history  = []

    for epoch in range(1, EPOCHS+1):
        tr_loss, tr_mae, tr_r2, tr_sr = run_epoch(
            visual, bert, fusion, train_loader, optimizer, criterion, DEVICE, train=True)
        vl_loss, vl_mae, vl_r2, vl_sr = run_epoch(
            visual, bert, fusion, val_loader,   optimizer, criterion, DEVICE, train=False)
        scheduler.step()

        print(f"Ep {epoch:02d}/{EPOCHS} | "
              f"TrLoss:{tr_loss:.4f} VlLoss:{vl_loss:.4f} | "
              f"MAE:{vl_mae:.4f} R²:{vl_r2:.4f} Spearman:{vl_sr:.4f}")

        history.append(dict(epoch=epoch, train_loss=tr_loss, val_loss=vl_loss,
                            val_mae=vl_mae, val_r2=vl_r2, val_srcc=vl_sr))

        if vl_loss < best_val:
            best_val = vl_loss
            torch.save({'epoch':epoch, 'visual':visual.state_dict(),
                        'fusion':fusion.state_dict(), 'val_loss':vl_loss,
                        'val_mae':vl_mae, 'val_r2':vl_r2},
                       f'{CKPT_DIR}/best_model.pt')
            print(f"  ✓ Saved best (val_loss={vl_loss:.4f})")

    # Test evaluation
    print("\nTest set evaluation...")
    ckpt = torch.load(f'{CKPT_DIR}/best_model.pt')
    visual.load_state_dict(ckpt['visual'])
    fusion.load_state_dict(ckpt['fusion'])
    _, te_mae, te_r2, te_sr = run_epoch(
        visual, bert, fusion, test_loader, optimizer, criterion, DEVICE, train=False)
    deep_metrics = {'MAE': te_mae, 'R2': te_r2, 'Spearman': te_sr}
    print(f"  TrimodalFusion  MAE={te_mae:.4f}  R²={te_r2:.4f}  Spearman={te_sr:.4f}")

    print_comparison(deep_metrics, xgb_metrics)

    with open(f'{CKPT_DIR}/history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nDone. Best model → {CKPT_DIR}/best_model.pt")
