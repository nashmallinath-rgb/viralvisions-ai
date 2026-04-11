import torch
import numpy as np
import cv2
# Use ImageProcessor instead of FeatureExtractor
from transformers import VideoMAEModel, VideoMAEImageProcessor
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

MODEL_NAME = "MCG-NJU/videomae-base"
NUM_FRAMES = 16
FRAME_SIZE = (224, 224)

def extract_frames(video_path, num_frames=NUM_FRAMES):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb).resize(FRAME_SIZE)
            frames.append(pil_frame)
    cap.release()
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else Image.new('RGB', FRAME_SIZE))
    return frames[:num_frames]

class VideoFeatureExtractor:
    def __init__(self, model_name=MODEL_NAME, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        # Updated to ImageProcessor
        self.image_processor = VideoMAEImageProcessor.from_pretrained(model_name)
        self.model = VideoMAEModel.from_pretrained(model_name)
        self.model.to(self.device).eval()

    def extract_features(self, video_path):
        frames = extract_frames(video_path, NUM_FRAMES)
        # Updated to image_processor
        inputs = self.image_processor(frames, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.device)
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)
        return outputs.last_hidden_state.mean(dim=1).squeeze()

if __name__ == "__main__":
    print("Testing VideoMAE with updated ImageProcessor...")
    dummy_frames = [Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)) for _ in range(NUM_FRAMES)]
    extractor = VideoFeatureExtractor()
    
    # Process dummy frames
    inputs = extractor.image_processor(dummy_frames, return_tensors="pt")
    pixel_values = inputs['pixel_values'].to(extractor.device)
    with torch.no_grad():
        features = extractor.model(pixel_values=pixel_values).last_hidden_state.mean(dim=1).squeeze()
    
    print(f"✅ Success! Feature shape: {features.shape}")
