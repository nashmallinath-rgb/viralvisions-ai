import os
import sys

# ── Kaggle working-directory + import fix ─────────────────────────────────────
os.chdir('/kaggle/working/viralvisions-ai')
sys.path.insert(0, '/kaggle/working/viralvisions-ai')
sys.path.insert(0, '/kaggle/working')

import json
import time
import pickle
import random
import warnings
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from PIL import Image

import scipy.stats as stats_lib
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer

import xgboost as xgb
from huggingface_hub import HfApi

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('viralvisions.train')


# ══════════════════════════════════════════════════════════════════════════════
# PATH CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

SMPD_IMG_ROOT     = '/kaggle/input/datasets/nishamallinath/smpd-image'
SMPD_VID_ROOT     = '/kaggle/input/datasets/nishamallinath/smpd-video'

# Images are downloaded from Google Drive to here before training starts.
# Structure: {IMG_BASE_DIR}/train/{uid}/{pid}.jpg
IMG_BASE_DIR      = '/kaggle/input/datasets/nishamallinath/smpd-train'

# SMPD-Image raw files
LABEL_PATH        = os.path.join(SMPD_IMG_ROOT, 'train_label.txt')
TEXT_PATH         = os.path.join(SMPD_IMG_ROOT, 'train_text.json')
IMG_FILEPATH_PATH = os.path.join(SMPD_IMG_ROOT,
                                  'train_allmetadata_json',
                                  'train_allmetadata_json',
                                  'train_img_filepath.txt')

# SMPD-Video raw files
VID_POSTS_PATH    = os.path.join(SMPD_VID_ROOT,
                                  'SMP-Video_anonymized_posts_train.jsonl')
VID_POP_PATH      = os.path.join(SMPD_VID_ROOT,
                                  'SMP-Video_anonymized_popularity_train.jsonl')

# Outputs
CHECKPOINT_DIR    = '/kaggle/working/checkpoints'
BEST_MODEL_PATH   = os.path.join(CHECKPOINT_DIR, 'best_model_smpd.pt')
HISTORY_PATH      = os.path.join(CHECKPOINT_DIR, 'history.json')

# HuggingFace
HF_REPO_ID        = 'NishaMallinath/viralvisions-model'
HF_TOKEN          = os.environ.get('HF_TOKEN', '')


# ══════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

EPOCHS       = 10
BATCH_SIZE   = 32
LR           = 3e-4
MAX_LEN      = 128
WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 1.0
VAL_SPLIT    = 0.15
TEST_SPLIT   = 0.10
SEED         = 42
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE == 'cuda':
    torch.cuda.manual_seed_all(SEED)

logger.info(f"Device: {DEVICE}  |  GPUs: {torch.cuda.device_count()}")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL IMPORTS
# ══════════════════════════════════════════════════════════════════════════════

from src.models.visual.resnet_encoder import ResNetEncoder
from src.models.text.bert_encoder import ViralBertEncoder
from src.models.fusion.cross_modal_attention import TrimodalFusion
from src.preprocessing.engine import ViralPreprocessor


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE TRANSFORMS
# ══════════════════════════════════════════════════════════════════════════════

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

EVAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

ZERO_IMAGE = torch.zeros(3, 224, 224)   # fallback when image file is missing


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_smpd_image_records() -> Tuple[List[Dict], float, float]:
    """
    Read the three SMPD-Image files and return a list of record dicts.

    The three files are aligned by line/index order (all 305,613 rows).
    Returns (records, global_label_min, global_label_max) where min/max
    are the raw values across this split — the caller normalises.
    """
    logger.info("Loading SMPD-Image records...")

    with open(LABEL_PATH, 'r') as f:
        raw_labels = [float(line.strip()) for line in f if line.strip()]

    with open(TEXT_PATH, 'r') as f:
        text_meta = json.load(f)          # list of dicts

    with open(IMG_FILEPATH_PATH, 'r') as f:
        img_paths = [line.strip() for line in f if line.strip()]

    n = len(raw_labels)
    assert len(text_meta) == n, "Label / text size mismatch"
    assert len(img_paths) == n, "Label / filepath size mismatch"

    records = []
    for i in range(n):
        meta = text_meta[i]
        caption = f"{meta.get('Title', '')} {meta.get('Alltags', '')}".strip()
        # img_path looks like  train/59@N75/775.jpg  — prepend IMG_BASE_DIR
        abs_img_path = os.path.join(IMG_BASE_DIR, img_paths[i])
        records.append({
            'post_id'   : str(meta.get('Pid', i)),
            'caption'   : caption,
            'raw_label' : raw_labels[i],
            'img_path'  : abs_img_path,
            'post_type' : 'image',
        })

    label_min = min(r['raw_label'] for r in records)
    label_max = max(r['raw_label'] for r in records)
    logger.info(f"  SMPD-Image: {n:,} posts | "
                f"label range [{label_min:.2f}, {label_max:.2f}]")
    return records, label_min, label_max


def load_smpd_video_records() -> Tuple[List[Dict], float, float]:
    """
    Read SMPD-Video posts + popularity files, join on 'pid'.
    Actual .mp4 files are NOT present — video embeddings are synthetic.
    """
    logger.info("Loading SMPD-Video records...")

    # Posts: {pid -> post_content}
    posts: Dict[str, str] = {}
    with open(VID_POSTS_PATH, 'r') as f:
        for line in f:
            obj = json.loads(line)
            posts[obj['pid']] = obj.get('post_content', '')

    # Popularity: {pid -> popularity}
    pops: Dict[str, float] = {}
    with open(VID_POP_PATH, 'r') as f:
        for line in f:
            obj = json.loads(line)
            pops[obj['pid']] = float(obj['popularity'])

    # Join on pid
    records = []
    for pid, caption in posts.items():
        if pid not in pops:
            continue
        records.append({
            'post_id'   : pid,
            'caption'   : caption,
            'raw_label' : pops[pid],
            'img_path'  : None,          # no real video
            'post_type' : 'video',
        })

    label_min = min(r['raw_label'] for r in records)
    label_max = max(r['raw_label'] for r in records)
    logger.info(f"  SMPD-Video: {len(records):,} posts | "
                f"label range [{label_min:.2f}, {label_max:.2f}]")
    return records, label_min, label_max


def normalise_labels(records: List[Dict], global_min: float, global_max: float) -> None:
    """Min-max normalise 'raw_label' → 'label' in place (0-1 range)."""
    span = global_max - global_min
    for r in records:
        r['label'] = (r['raw_label'] - global_min) / (span + 1e-8)


# ══════════════════════════════════════════════════════════════════════════════
# COMBINED DATASET
# ══════════════════════════════════════════════════════════════════════════════

class SMPDCombinedDataset(Dataset):
    """
    Single Dataset wrapping both SMPD-Image and SMPD-Video records.

    Each sample returns:
        {
            'image'         : FloatTensor[3,224,224] | None,
            'video_emb'     : FloatTensor[768]       | None,
            'input_ids'     : LongTensor[MAX_LEN],
            'attention_mask': LongTensor[MAX_LEN],
            'label'         : FloatTensor scalar,
            'post_type'     : str ('image' | 'video'),
        }

    Image posts  → image is a real tensor (or zero fallback), video_emb=None.
    Video posts  → image=None, video_emb is a SYNTHETIC random tensor [768].
                   Real .mp4 files are not yet available. Remove this comment
                   and replace with actual VideoMAE extraction once .mp4s land.
    """

    def __init__(self,
                 records: List[Dict],
                 preprocessor: ViralPreprocessor,
                 image_transform,
                 is_train: bool = True):
        self.records      = records
        self.preprocessor = preprocessor
        self.transform    = image_transform
        self.is_train     = is_train

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]

        # ── Text tokenisation (BERT) ──────────────────────────────────────────
        text_dict = self.preprocessor.process_text(rec['caption'])
        input_ids      = text_dict['input_ids']       # [MAX_LEN]
        attention_mask = text_dict['attention_mask']  # [MAX_LEN]

        label = torch.tensor(rec['label'], dtype=torch.float32)

        # ── Modality-specific ─────────────────────────────────────────────────
        if rec['post_type'] == 'image':
            image     = self._load_image(rec['img_path'])
            video_emb = None

        else:
            # VIDEO POST — synthetic placeholder until actual .mp4 files arrive.
            # Replace torch.randn with VideoMAE extraction when ready.
            image     = None
            video_emb = torch.randn(768)  # ← SYNTHETIC: remove when .mp4s available

        return {
            'image'         : image,
            'video_emb'     : video_emb,
            'input_ids'     : input_ids,
            'attention_mask': attention_mask,
            'label'         : label,
            'post_type'     : rec['post_type'],
        }

    def _load_image(self, path: Optional[str]) -> torch.Tensor:
        """
        Load an image from disk and apply transforms.
        Falls back to a zero tensor [3,224,224] if file is missing.
        This prevents crashes when images haven't finished downloading yet.
        """
        if path is None:
            return ZERO_IMAGE.clone()
        try:
            img = Image.open(path).convert('RGB')
            return self.transform(img)
        except (FileNotFoundError, OSError):
            logger.warning(f"[FALLBACK] Image not found: {path} — using zero tensor")
            return ZERO_IMAGE.clone()


# ══════════════════════════════════════════════════════════════════════════════
# COLLATE FUNCTION (mixed image / video batches)
# ══════════════════════════════════════════════════════════════════════════════

def mixed_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    Custom collate for batches that may contain a mix of image and video posts.

    image posts  → 'images' stacked, 'video_embs' zero-filled for that row
    video posts  → 'video_embs' stacked, 'images' zero-filled for that row

    Returns:
        {
            'images'        : FloatTensor[B,3,224,224],
            'video_embs'    : FloatTensor[B,768],
            'image_mask'    : BoolTensor[B]  — True where post_type=='image',
            'video_mask'    : BoolTensor[B]  — True where post_type=='video',
            'input_ids'     : LongTensor[B,MAX_LEN],
            'attention_mask': LongTensor[B,MAX_LEN],
            'labels'        : FloatTensor[B],
            'post_types'    : List[str],
        }
    """
    image_mask = torch.tensor([s['post_type'] == 'image' for s in batch])
    video_mask = ~image_mask

    input_ids      = torch.stack([s['input_ids']      for s in batch])
    attention_mask = torch.stack([s['attention_mask'] for s in batch])
    labels         = torch.stack([s['label']          for s in batch])
    post_types     = [s['post_type'] for s in batch]

    # Stack images only for image posts; zero-fill for video posts so tensors
    # can still be stacked (they won't be forwarded for video rows).
    image_tensors = []
    for s in batch:
        if s['image'] is not None:
            image_tensors.append(s['image'])
        else:
            image_tensors.append(ZERO_IMAGE.clone())
    images = torch.stack(image_tensors)  # [B,3,224,224]

    # Stack video_embs only for video posts; zero-fill for image posts.
    video_tensors = []
    for s in batch:
        if s['video_emb'] is not None:
            video_tensors.append(s['video_emb'])
        else:
            video_tensors.append(torch.zeros(768))
    video_embs = torch.stack(video_tensors)  # [B,768]

    return {
        'images'        : images,
        'video_embs'    : video_embs,
        'image_mask'    : image_mask,
        'video_mask'    : video_mask,
        'input_ids'     : input_ids,
        'attention_mask': attention_mask,
        'labels'        : labels,
        'post_types'    : post_types,
    }


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN / VAL / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════

def make_splits(records: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Deterministic train / val / test split on the combined record list.
    Proportions: 75% / 15% / 10%
    """
    n     = len(records)
    idxs  = list(range(n))
    rng   = random.Random(SEED)
    rng.shuffle(idxs)

    n_test = int(n * TEST_SPLIT)
    n_val  = int(n * VAL_SPLIT)

    test_idx  = idxs[:n_test]
    val_idx   = idxs[n_test : n_test + n_val]
    train_idx = idxs[n_test + n_val:]

    train = [records[i] for i in train_idx]
    val   = [records[i] for i in val_idx]
    test  = [records[i] for i in test_idx]

    logger.info(f"Split → train: {len(train):,}  val: {len(val):,}  test: {len(test):,}")
    return train, val, test


# ══════════════════════════════════════════════════════════════════════════════
# FORWARD PASS HELPER
# ══════════════════════════════════════════════════════════════════════════════

def forward_batch(
    batch: Dict[str, Any],
    visual_encoder: ResNetEncoder,
    text_encoder: ViralBertEncoder,
    fusion_model: TrimodalFusion,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run one batch through the trimodal pipeline.

    For image posts : ResNet(image) → image_emb [B_img, 2048]
                      BERT(tokens)  → text_emb  [B, 768]
                      fusion(text, image_emb=...)

    For video posts : SYNTHETIC video_emb [B_vid, 768]   ← placeholder
                      BERT(tokens)  → text_emb  [B, 768]
                      fusion(text, video_emb=...)

    Returns (predictions[B], labels[B]).
    """
    images         = batch['images'].to(device)          # [B,3,224,224]
    video_embs     = batch['video_embs'].to(device)      # [B,768]
    input_ids      = batch['input_ids'].to(device)       # [B,128]
    attention_mask = batch['attention_mask'].to(device)  # [B,128]
    labels         = batch['labels'].to(device)          # [B]
    image_mask     = batch['image_mask']                 # [B] bool, CPU is fine
    video_mask     = batch['video_mask']

    B = labels.shape[0]

    # ── BERT text embeddings (all posts) ─────────────────────────────────────
    text_emb = text_encoder(input_ids, attention_mask)  # [B,768]

    # ── Allocate output tensor ────────────────────────────────────────────────
    preds = torch.zeros(B, device=device)

    # ── IMAGE posts ───────────────────────────────────────────────────────────
    img_indices = image_mask.nonzero(as_tuple=True)[0]
    if len(img_indices) > 0:
        img_t   = images[img_indices]       # [B_img, 3, 224, 224]
        txt_t   = text_emb[img_indices]     # [B_img, 768]
        vis_emb = visual_encoder(img_t)     # [B_img, 2048]

        score = fusion_model(
            text_emb=txt_t,
            image_emb=vis_emb,
        )  # [B_img, 1]
        preds[img_indices] = score.squeeze(1)

    # ── VIDEO posts ───────────────────────────────────────────────────────────
    vid_indices = video_mask.nonzero(as_tuple=True)[0]
    if len(vid_indices) > 0:
        vid_t = video_embs[vid_indices]     # [B_vid, 768] — synthetic placeholder
        txt_t = text_emb[vid_indices]       # [B_vid, 768]

        score = fusion_model(
            text_emb=txt_t,
            video_emb=vid_t,
        )  # [B_vid, 1]
        preds[vid_indices] = score.squeeze(1)

    return preds, labels


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(targets, preds)
    r2  = r2_score(targets, preds)
    rho, _ = stats_lib.spearmanr(preds, targets)
    return {'mae': mae, 'r2': r2, 'spearman': rho}


def print_comparison_table(nn_metrics: Dict, xgb_metrics: Dict) -> None:
    header = f"{'Metric':<14} {'TrimodalFusion':>16} {'XGBoost-TF-IDF':>16}"
    sep    = '-' * len(header)
    logger.info('\n' + sep)
    logger.info(header)
    logger.info(sep)
    for key in ('mae', 'r2', 'spearman'):
        nn_val  = f"{nn_metrics[key]:.4f}"
        xgb_val = f"{xgb_metrics[key]:.4f}"
        logger.info(f"{key.upper():<14} {nn_val:>16} {xgb_val:>16}")
    logger.info(sep + '\n')


# ══════════════════════════════════════════════════════════════════════════════
# XGBOOST BASELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_xgboost_baseline(
    train_records: List[Dict],
    test_records: List[Dict],
) -> Dict[str, float]:
    """
    TF-IDF on captions → XGBoost regression.
    Used as a quick sanity-check baseline alongside the neural model.
    """
    logger.info("Running XGBoost baseline (TF-IDF on captions)...")

    train_captions = [r['caption'] for r in train_records]
    test_captions  = [r['caption'] for r in test_records]
    train_labels   = np.array([r['label'] for r in train_records])
    test_labels    = np.array([r['label'] for r in test_records])

    tfidf = TfidfVectorizer(max_features=10_000, ngram_range=(1, 2))
    X_train = tfidf.fit_transform(train_captions)
    X_test  = tfidf.transform(test_captions)

    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='gpu_hist',   # uses T4 GPU
        random_state=SEED,
        verbosity=0,
    )
    model.fit(X_train, train_labels, eval_set=[(X_test, test_labels)],
              verbose=False)

    preds = model.predict(X_test)
    metrics = compute_metrics(preds, test_labels)
    logger.info(f"XGBoost baseline → MAE={metrics['mae']:.4f}  "
                f"R²={metrics['r2']:.4f}  Spearman={metrics['spearman']:.4f}")
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION LOOP
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(
    loader: DataLoader,
    visual_encoder: ResNetEncoder,
    text_encoder: ViralBertEncoder,
    fusion_model: TrimodalFusion,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, Dict[str, float]]:
    """Run one evaluation pass. Returns (avg_loss, metrics_dict)."""
    visual_encoder.eval()
    fusion_model.eval()

    all_preds, all_targets = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            preds, labels = forward_batch(
                batch, visual_encoder, text_encoder, fusion_model, device
            )
            loss = criterion(preds, labels)
            total_loss += loss.item() * labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    metrics  = compute_metrics(np.array(all_preds), np.array(all_targets))
    return avg_loss, metrics


# ══════════════════════════════════════════════════════════════════════════════
# UPLOAD TO HUGGINGFACE
# ══════════════════════════════════════════════════════════════════════════════

def upload_to_huggingface(local_path: str) -> None:
    """
    Upload best_model_smpd.pt to HuggingFace Hub.
    Requires HF_TOKEN env variable with Write permission.
    """
    if not HF_TOKEN:
        logger.warning("HF_TOKEN not set — skipping HuggingFace upload.")
        return

    logger.info(f"Uploading {local_path} → HuggingFace:{HF_REPO_ID} ...")
    api = HfApi()
    try:
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo='best_model_smpd.pt',
            repo_id=HF_REPO_ID,
            token=HF_TOKEN,
        )
        logger.info("✓ Upload complete.")
    except Exception as e:
        logger.error(f"HuggingFace upload failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING SCRIPT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── 1. Load data ──────────────────────────────────────────────────────────
    img_records, img_min, img_max = load_smpd_image_records()
    vid_records, vid_min, vid_max = load_smpd_video_records()

    # Global min/max across both splits for consistent normalisation
    global_min = min(img_min, vid_min)
    global_max = max(img_max, vid_max)
    logger.info(f"Global label range: [{global_min:.2f}, {global_max:.2f}]")

    normalise_labels(img_records, global_min, global_max)
    normalise_labels(vid_records, global_min, global_max)

    all_records = img_records + vid_records
    logger.info(f"Combined dataset: {len(all_records):,} posts")

    # ── 2. Split ──────────────────────────────────────────────────────────────
    train_rec, val_rec, test_rec = make_splits(all_records)

    # ── 3. Preprocessor ───────────────────────────────────────────────────────
    preprocessor = ViralPreprocessor(bert_model='bert-base-uncased', max_len=MAX_LEN)

    # ── 4. Datasets & Loaders ─────────────────────────────────────────────────
    train_ds = SMPDCombinedDataset(train_rec, preprocessor, TRAIN_TRANSFORMS, is_train=True)
    val_ds   = SMPDCombinedDataset(val_rec,   preprocessor, EVAL_TRANSFORMS,  is_train=False)
    test_ds  = SMPDCombinedDataset(test_rec,  preprocessor, EVAL_TRANSFORMS,  is_train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=mixed_collate_fn, num_workers=4,
                              pin_memory=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=mixed_collate_fn, num_workers=2,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=mixed_collate_fn, num_workers=2,
                              pin_memory=True)

    logger.info(f"Train batches: {len(train_loader):,}  "
                f"Val: {len(val_loader):,}  Test: {len(test_loader):,}")

    # ── 5. Models ─────────────────────────────────────────────────────────────
    # ResNet-101: raw 2048-dim output. TrimodalFusion projects internally.
    visual_encoder = ResNetEncoder(embed_dim=2048).to(DEVICE)

    # BERT frozen throughout (trainable=False is already the default in ViralBertEncoder)
    text_encoder = ViralBertEncoder(model_name='bert-base-uncased', trainable=False).to(DEVICE)

    # TrimodalFusion: handles both image (2048-dim) and video (768-dim) paths
    fusion_model = TrimodalFusion(
        image_dim=2048,
        video_dim=768,
        text_dim=768,
        num_heads=8,
        dropout=0.3,
    ).to(DEVICE)

    # ── 6. Optimiser & scheduler ──────────────────────────────────────────────
    # Only ResNet and Fusion are trained; BERT is frozen.
    trainable_params = (
        list(visual_encoder.parameters()) +
        list(fusion_model.parameters())
    )

    optimizer = AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = nn.MSELoss()

    # ── 7. Training loop ──────────────────────────────────────────────────────
    history: Dict[str, List] = {
        'train_loss': [], 'val_loss': [],
        'val_mae': [], 'val_r2': [], 'val_spearman': [],
    }
    best_val_loss = float('inf')

    logger.info(f"\n{'='*60}")
    logger.info(f"Starting training: {EPOCHS} epochs | bs={BATCH_SIZE} | lr={LR}")
    logger.info(f"{'='*60}")

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        # — Train —
        visual_encoder.train()
        fusion_model.train()
        train_loss_accum = 0.0

        for step, batch in enumerate(train_loader, 1):
            optimizer.zero_grad()

            preds, labels = forward_batch(
                batch, visual_encoder, text_encoder, fusion_model, DEVICE
            )
            loss = criterion(preds, labels)
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(trainable_params, GRAD_CLIP)
            optimizer.step()

            train_loss_accum += loss.item() * labels.size(0)

            if step % 500 == 0:
                running_loss = train_loss_accum / (step * BATCH_SIZE)
                logger.info(f"  Epoch {epoch} | step {step}/{len(train_loader)} "
                            f"| loss={running_loss:.4f}")

        scheduler.step()

        avg_train_loss = train_loss_accum / len(train_loader.dataset)

        # — Validate —
        val_loss, val_metrics = evaluate(
            val_loader, visual_encoder, text_encoder, fusion_model, criterion, DEVICE
        )

        elapsed = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"MAE={val_metrics['mae']:.4f} | "
            f"R²={val_metrics['r2']:.4f} | "
            f"ρ={val_metrics['spearman']:.4f} | "
            f"{elapsed:.1f}s"
        )

        # History
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_metrics['mae'])
        history['val_r2'].append(val_metrics['r2'])
        history['val_spearman'].append(val_metrics['spearman'])

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch'           : epoch,
                'visual'          : visual_encoder.state_dict(),
                'fusion'          : fusion_model.state_dict(),
                'val_loss'        : val_loss,
                'val_metrics'     : val_metrics,
                'global_label_min': global_min,
                'global_label_max': global_max,
            }, BEST_MODEL_PATH)
            logger.info(f"  ✓ Best model saved (val_loss={val_loss:.4f})")

    # ── 8. Save history ───────────────────────────────────────────────────────
    with open(HISTORY_PATH, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved to {HISTORY_PATH}")

    # ── 9. Test set evaluation ────────────────────────────────────────────────
    logger.info("\nLoading best checkpoint for test evaluation...")
    ckpt = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
    visual_encoder.load_state_dict(ckpt['visual'])
    fusion_model.load_state_dict(ckpt['fusion'])

    test_loss, test_metrics = evaluate(
        test_loader, visual_encoder, text_encoder, fusion_model, criterion, DEVICE
    )
    logger.info(f"Test set → loss={test_loss:.4f} | "
                f"MAE={test_metrics['mae']:.4f} | "
                f"R²={test_metrics['r2']:.4f} | "
                f"ρ={test_metrics['spearman']:.4f}")

    # ── 10. XGBoost baseline ──────────────────────────────────────────────────
    xgb_metrics = run_xgboost_baseline(train_rec, test_rec)

    # ── 11. Comparison table ──────────────────────────────────────────────────
    print_comparison_table(test_metrics, xgb_metrics)

    # Append test metrics to history file
    history['test_metrics']    = test_metrics
    history['xgboost_metrics'] = xgb_metrics
    with open(HISTORY_PATH, 'w') as f:
        json.dump(history, f, indent=2)

    # ── 12. Upload to HuggingFace ─────────────────────────────────────────────
    upload_to_huggingface(BEST_MODEL_PATH)

    logger.info("\n✓ Training pipeline complete.")
    logger.info(f"  Best model : {BEST_MODEL_PATH}")
    logger.info(f"  History    : {HISTORY_PATH}")
    logger.info(f"  HuggingFace: {HF_REPO_ID}/best_model_smpd.pt")


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    main()
