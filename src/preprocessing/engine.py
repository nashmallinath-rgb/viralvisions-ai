import torch
import re
import emoji
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer

class ViralPreprocessor:
    def __init__(self, bert_model='bert-base-uncased', max_len=128):
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.max_len = max_len
        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _clean_text(self, text):
        # Handle potential NaN/float values in CSV
        text = str(text) if text else ""
        text = emoji.demojize(text, delimiters=(" ", " "))
        text = re.sub(r"[^a-zA-Z0-9#@\s:]", "", text)
        return " ".join(text.split()).lower()

    def process_text(self, text):
        cleaned = self._clean_text(text)
        encoding = self.tokenizer(
            cleaned, add_special_tokens=True, max_length=self.max_len,
            padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt'
        )
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten()}

    def process_image(self, image_path):
        try:
            # Check if file exists before opening
            with Image.open(image_path) as img:
                return self.img_transform(img.convert('RGB'))
        except Exception as e:
            # Return a zero tensor if image is missing/corrupt
            return torch.zeros(3, 224, 224)

    def __call__(self, image_path, caption):
        return {
            "image_tensor": self.process_image(image_path),
            "text_tensors": self.process_text(caption)
        }
