"""
TPIC17 - Temporal Prediction of Image Popularity
Download script for ViralVisions AI
"""

import os

def create_folder_structure():
    folders = [
        "data/tpic17/raw",
        "data/tpic17/images",
        "data/tpic17/metadata",
        "data/tpic17/processed"
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Created: {folder}")

def print_instructions():
    print("""
=================================================
TPIC17 - Temporal Prediction of Image Popularity
=================================================

STEP 1 - Download:
    Website : https://social-media-prediction.github.io/TPIC2017
    Paper   : https://arxiv.org/abs/1705.01674

STEP 2 - What you will get:
    - 680K Flickr photos
    - Temporal metadata spanning 36 months
    - User profiles + photo sharing records
    - Popularity scores over time

STEP 3 - After downloading:
    - Place raw files in   → data/tpic17/raw/
    - Place images in      → data/tpic17/images/
    - Place metadata in    → data/tpic17/metadata/

NOTE: Use TPIC17 for temporal feature experiments only.
      SMPD is your primary dataset!
=================================================
    """)

if __name__ == "__main__":
    create_folder_structure()
    print_instructions()