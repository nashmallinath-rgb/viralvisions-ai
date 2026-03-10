import pandas as pd
from torch.utils.data import Dataset, DataLoader
from .engine import ViralPreprocessor

class ViralDataset(Dataset):
    def __init__(self, csv_file, img_dir, max_len=128):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.preprocessor = ViralPreprocessor(max_len=max_len)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load Image and Caption from CSV
        img_path = f"{self.img_dir}/{row['post_id']}.jpg"
        caption = str(row['caption'])
        label = torch.tensor(row['virality_score'], dtype=torch.float)

        # Preprocess
        processed = self.preprocessor(img_path, caption)
        
        return {
            'image': processed['image_tensor'],
            'input_ids': processed['text_tensors']['input_ids'],
            'attention_mask': processed['text_tensors']['attention_mask'],
            'label': label
        }

def get_loader(csv_file, img_dir, batch_size=32):
    ds = ViralDataset(csv_file, img_dir)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)
