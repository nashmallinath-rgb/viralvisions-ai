# ViralVisions AI 🚀
### Multimodal Social Media Virality Prediction & Content Optimization System

> VI Semester Mini Project | Dept. of CSE (AI&ML) & CSE (Cyber Security)
> MSRIT | Guide: Mrs. Shobha K

---

## 📌 Overview
ViralVisions AI is a pre-publication advisor that predicts the virality of social media content (images + text) and suggests data-driven modifications to maximize engagement.

### Core Components
| Module | Description |
|--------|-------------|
| Visual Module | ResNet-101 for image feature extraction |
| Text Module | BERT for caption/hashtag sentiment |
| Fusion Layer | Cross-modal attention for multimodal fusion |
| RAG Knowledge Bank | Retrieval-augmented benchmarking against past viral posts |
| Advisory Agent | Suggests content modifications pre-publication |
| Baseline | XGBoost for benchmark comparison |

---

## Repository Structure

```
viralvisions-ai/
├── src/
│   ├── models/
│   │   ├── visual/          # ResNet-101 visual feature extractor
│   │   ├── text/            # BERT text encoder
│   │   ├── fusion/          # Cross-modal attention & fusion
│   │   ├── rag/             # RAG Knowledge Bank
│   │   └── baseline/        # XGBoost baseline model
│   ├── preprocessing/       # Data cleaning, tokenization, image transforms
│   ├── training/            # Training loops, schedulers, loss functions
│   ├── evaluation/          # MAE, R2, SRCC metrics
│   └── agent/               # Advisory AI agent logic
├── app/
│   ├── frontend/            # Streamlit / React UI
│   └── backend/             # FastAPI backend
├── notebooks/               # Exploratory analysis & experiments
├── configs/                 # Model hyperparameters & config files
├── scripts/                 # Utility scripts
├── tests/                   # Unit tests
├── docs/
│   ├── phase1/
│   └── phase2/
└── results/
    ├── checkpoints/
    ├── logs/
    └── plots/
```

> Datasets are maintained on a separate `datasets` branch.

---

## Branch Structure

| Branch | Purpose |
|--------|---------|
| `main` | Stable, production-ready code |
| `dev` | Active development & integration |
| `datasets` | Dataset storage, scripts & documentation |
| `feature/visual-module` | ResNet-101 development |
| `feature/text-module` | BERT development |
| `feature/fusion` | Fusion layer development |
| `feature/rag` | RAG Knowledge Bank development |
| `feature/ui` | Frontend UI development |

---

## Datasets
Stored on the `datasets` branch. See that branch's README for download instructions.

**Primary:** SMPD (Social Media Prediction Dataset) — Flickr
**Secondary:** TPIC17 (Flickr temporal dataset)

---

## Team
- **Nisha Mallinath** (1MS23CI077)
- **Shrusti Mathapati** (1MS23CI118)
- **Guide:** Mrs. Shobha K

---

## Project Timeline
| Phase | Dates | Status |
|-------|-------|--------|
| Phase 1 - Idea Finalization | Completed | Done |
| Phase 2 - System Design | Mar 3 | Done |
| Phase 3 - Initial Implementation | Mar 7 - Apr 3 | In Progress |
| Phase 4 - Core Development | Apr 11 - Apr 22 | Pending |
| Phase 5 - Testing | Apr 25 - May 6 | Pending |
| Final Demo | May 14-15 | Pending |
