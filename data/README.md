# Datasets - ViralVisions AI

Raw data files are NOT stored in this repo (too large for GitHub).
Use the scripts below to set up folder structure and get download links.

---

## Primary: SMPD
- Run: `python scripts/download_smpd.py`
- Size: ~Several GBs
- Use: Main training & evaluation

## Secondary: TPIC17
- Run: `python scripts/download_tpic17.py`  
- Size: ~Several GBs
- Use: Temporal feature experiments only

---

## Recommended: Download inside Google Colab
Colab has faster internet and more storage than local systems.

In Colab:
!python scripts/download_smpd.py

---

## Folder Structure
data/
├── smpd/
│   ├── raw/
│   ├── images/
│   └── annotations/
├── tpic17/
│   ├── raw/
│   ├── images/
│   ├── metadata/
│   └── processed/
└── processed/
    ├── train/
    ├── val/
    └── test/
