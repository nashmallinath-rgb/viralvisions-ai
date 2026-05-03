from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch, torch.nn as nn, io, sys, os
import numpy as np
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download
import shutil
import os
sys.path.append('/content/viralvisions-ai')

from src.models.fusion.cross_modal_attention import TrimodalFusion
from src.models.visual.resnet_encoder        import ResNetEncoder
from src.models.text.bert_encoder            import ViralBertEncoder
from src.preprocessing.engine               import ViralPreprocessor
from src.models.rag.knowledge_bank           import ViralKnowledgeBank

HF_REPO   = 'NishaMallinath/viralvisions-model'
EMBED_DIM = 512
# In main.py, replace line 21:
GROQ_KEY = os.environ.get('GROQ_KEY', '')

app = FastAPI(title="ViralVisions API", version="4.0-smpd")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

visual_encoder = ResNetEncoder(embed_dim=2048).to(device).eval()
bert_encoder   = ViralBertEncoder().to(device).eval()
fusion_model   = TrimodalFusion(image_dim=2048, text_dim=768).to(device).eval()
preprocessor   = ViralPreprocessor()

try:
    ckpt_path = hf_hub_download(repo_id=HF_REPO, filename='best_model_smpd.pt')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    visual_encoder.load_state_dict(ckpt['visual'])
    fusion_model.load_state_dict(ckpt['fusion'])
    print("✅ Weights loaded")
except Exception as e:
    print(f"⚠️  Weights not loaded: {e}")

bank = ViralKnowledgeBank(embed_dim=EMBED_DIM, groq_key=GROQ_KEY)
try:
    idx_path  = hf_hub_download(repo_id=HF_REPO, filename='knowledge_bank.index')
    meta_path = hf_hub_download(repo_id=HF_REPO, filename='knowledge_bank_metadata.pkl')
    shutil.copy(idx_path,  '/tmp/knowledge_bank.index')
    shutil.copy(meta_path, '/tmp/knowledge_bank_metadata.pkl')
    bank.load('/tmp/knowledge_bank')
    print(f"✅ Knowledge bank loaded — {bank.index.ntotal:,} entries")

    manual_posts = [
        {"caption": "18 years of wait. Worth every second. Champions. #Cricket #IPL #Virat", "score": 0.97},
        {"caption": "Champions again. Keeping the flag flying high. #RCB #WPL #Cricket", "score": 0.96},
        {"caption": "This team made the dream possible. For every fan who never gave up. #RCB #Champions", "score": 0.98},
        {"caption": "We did it. World Cup winners. This one's for everyone who believed. #Football #WorldCup", "score": 0.97},
        {"caption": "Quit my job and bought a one way ticket. Best decision of my life. #Travel #Adventure", "score": 0.96},
        {"caption": "6 months transformation. The only thing that changed was my mindset. #Fitness #Transformation", "score": 0.96},
        {"caption": "Broke at 22. Millionaire at 27. Consistency is everything. #Entrepreneur #Success", "score": 0.95},
        {"caption": "AI is not going to take your job. Someone using AI will. Start learning. #Tech #AI", "score": 0.95},
        {"caption": "This place doesn't look real. Earth is insane. #Travel #Nature #Paradise", "score": 0.95},
        {"caption": "India won and the whole country felt it at the same time #India #Cricket #ProudIndian", "score": 0.97},
        {"caption": "Stop waiting for the perfect moment. Take the moment and make it perfect. #Motivation", "score": 0.94},
        {"caption": "He looked at me like this and I gave him all my food obviously #Dogs #Pets #Cute", "score": 0.94},
        {"caption": "This scene lives rent free in my head forever #Movies #Nostalgia #PopCulture", "score": 0.94},
        {"caption": "Outfit of the day but make it effortless #OOTD #Fashion #Style", "score": 0.91},
        {"caption": "Made this from scratch and I'm never ordering out again #Foodie #Homecooking", "score": 0.92},
        {"caption": "Desi wedding energy is unmatched anywhere in the world #IndianWedding #Desi", "score": 0.92},
        {"caption": "Sunday mood. Coffee, sunlight, no plans. #Aesthetic #Lifestyle #Cozy", "score": 0.89},
        {"caption": "My cat has no idea she's the CEO of this household #Cats #Pets #Funny", "score": 0.93},
        {"caption": "POV: you just discovered the best show on Netflix #Netflix #Binge #Trending", "score": 0.93},
        {"caption": "5am club. While the world sleeps, we grind. #Fitness #Discipline #Gym", "score": 0.91},
    ]
    manual_embeddings = []
    for p in manual_posts:
        td = preprocessor.process_text(p["caption"])
        ids_ = td['input_ids'].unsqueeze(0).to(device)
        mask_ = td['attention_mask'].unsqueeze(0).to(device)
        with torch.no_grad():
            t_emb = bert_encoder(ids_, mask_)
            proj = fusion_model.text_proj(t_emb).squeeze(0).cpu().numpy()
        manual_embeddings.append(proj.astype(np.float32))
    manual_meta = [{"post_id": f"manual_{i}", "caption": p["caption"],
                    "score": p["score"], "post_type": "image", "source": "manual"}
                   for i, p in enumerate(manual_posts)]
    bank._add(np.stack(manual_embeddings), manual_meta)
    print(f"✅ Manual posts added — {bank.index.ntotal:,} total entries")

except Exception as e:
    print(f"⚠️  Knowledge bank not loaded: {e}")

img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

@app.get("/health")
def health():
    return {
        "status"      : "ok",
        "device"      : str(device),
        "bank_entries": bank.index.ntotal,
        "groq_ready"  : bool(GROQ_KEY),
    }

@app.post("/predict")
async def predict(caption: str = Form(...), image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        img    = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = img_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            image_emb = visual_encoder(tensor)

        text_dict = preprocessor.process_text(caption)
        input_ids = text_dict['input_ids'].unsqueeze(0).to(device)
        attn_mask = text_dict['attention_mask'].unsqueeze(0).to(device)
        with torch.no_grad():
            text_emb = bert_encoder(input_ids, attn_mask)

        with torch.no_grad():
            score_tensor, fused_emb = fusion_model(
                text_emb=text_emb, image_emb=image_emb, return_embedding=True)

        # Use text projection for RAG retrieval (same space as manual posts)
        fused_np     = fused_emb.squeeze(0).cpu().numpy()
        text_proj_np = fusion_model.text_proj(text_emb).squeeze(0).cpu().numpy()
        similar      = bank.retrieve(text_proj_np, k=10)

        if similar:
            top_sim   = similar[0]['similarity']
            avg_score = float(np.mean([s['score'] for s in similar[:3]]))
        else:
            top_sim   = 0.5
            avg_score = 0.5

        # Caption quality gate
        words    = len(caption.split())
        hashtags = caption.count('#')
        emojis   = sum(1 for c in caption if ord(c) > 127)
        has_strong_emotion = any(w in caption.lower() for w in [
            'dream', 'champion', 'win', 'never forget', 'worth', 'history',
            'legend', 'glory', 'proud', 'incredible', 'journey', '18 years'
        ])
        has_mild_emotion = any(w in caption.lower() for w in [
            'love', 'amazing', 'beautiful', 'happy', 'blessed', 'grateful',
            'moment', 'heart', 'feel', 'life', 'good', 'great', 'best'
        ])

        word_score    = min(words / 40, 0.4)
        hashtag_score = min(hashtags * 0.05, 0.2)
        emoji_score   = min(emojis * 0.02, 0.1)
        emotion_score = 0.3 if has_strong_emotion else 0.15 if has_mild_emotion else 0.0

        caption_quality = min(word_score + hashtag_score + emoji_score + emotion_score, 1.0)
        final_score     = (top_sim * avg_score * caption_quality) * 100

        label = "HIGH" if final_score >= 70 else "MEDIUM" if final_score >= 40 else "LOW"

        advisory = bank.generate_advisory(
            query_emb=fused_np,
            caption=caption,
            predicted_score=final_score / 100.0,
            post_type="image",
            k=5,
        )

        return {"score": round(final_score, 2), "label": label,
                "advisory": advisory, "embed_dim": EMBED_DIM}

    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
