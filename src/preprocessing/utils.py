import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def prepare_smpd_data(csv_path, img_dir, output_dir='data/processed'):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    
    # 1. Validation: Remove rows with missing images
    print("🔍 Validating images...")
    exists = [os.path.isfile(os.path.join(img_dir, f"{pid}.jpg")) for pid in tqdm(df['post_id'])]
    df = df[exists]
    
    # 2. Splitting: 80/10/10
    train, temp = train_test_split(df, test_size=0.2, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)
    
    # 3. Save
    train.to_csv(f"{output_dir}/train.csv", index=False)
    val.to_csv(f"{output_dir}/val.csv", index=False)
    test.to_csv(f"{output_dir}/test.csv", index=False)
    print(f"✅ Data processed. Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
