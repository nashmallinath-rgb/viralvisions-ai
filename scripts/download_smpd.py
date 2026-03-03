"""
SMPD - Social Media Prediction Dataset
Download script for ViralVisions AI
"""

import os

def create_folder_structure():
    folders = [
        "data/smpd/raw",
        "data/smpd/images",
        "data/smpd/annotations",
        "data/processed/train",
        "data/processed/val",
        "data/processed/test"
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Created: {folder}")

def print_instructions():
    print("""
=================================================
SMPD - Social Media Prediction Dataset
=================================================

STEP 1 - Register & Download:
    Website  : https://smp-challenge.com/dataset.html
    GitHub   : https://github.com/social-media-prediction/SMPChallenge

STEP 2 - What you will get:
    - 486K Flickr social image posts
    - Images + captions + hashtags + metadata
    - Popularity scores (view counts, likes)

STEP 3 - After downloading:
    - Place raw files in      → data/smpd/raw/
    - Place images in         → data/smpd/images/
    - Place annotations in    → data/smpd/annotations/

STEP 4 - Then run preprocessing:
    python src/preprocessing/preprocess_smpd.py
=================================================
    """)

if __name__ == "__main__":
    create_folder_structure()
    print_instructions()