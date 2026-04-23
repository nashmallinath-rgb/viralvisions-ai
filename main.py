from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch, torch.nn as nn, io, sys, os, numpy as np
from PIL import Image
from torchvision import transforms

sys.path.append('/content/viralvisions-ai')

from src.models.fusion.cross_modal_attention import TrimodalFusion
from src.models.visual.resnet_encoder        import ResNetEncoder
from src.models.text.bert_encoder            import ViralBertEncoder
from src.preprocessing.engine               import ViralPreprocessor
from src.models.rag.knowledge_bank                 import ViralKnowledgeBank

EMBED_DIM        = 512
FUSED_DIM        = 512
KNOWLEDGE_BANK_PATH = "/content/viralvisions-ai/knowledge_bank"

app = FastAPI(title="ViralVisions API", version="3.0-phase3")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

visual_encoder  = ResNetEncoder(embed_dim=2048).to(device).eval()
bert_encoder    = ViralBertEncoder().to(device).eval()
text_projection = nn.Linear(768, EMBED_DIM).to(device).eval()
fusion_model    = TrimodalFusion(image_dim=2048, text_dim=768).to(device).eval()
preprocessor    = ViralPreprocessor()
print(f"All models loaded — embed_dim={EMBED_DIM}")

img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

GEMINI_KEY = os.environ.get("GEMINI_KEY", None)
bank = ViralKnowledgeBank(embed_dim=FUSED_DIM, gemini_key=GEMINI_KEY)
try:
    bank.load(KNOWLEDGE_BANK_PATH)
    print(f"Knowledge bank loaded — {bank.index.ntotal:,} entries")
except Exception as e:
    print(f"Knowledge bank not found — using empty bank (fallback advisory). Error: {e}")

@app.get("/health")
def health():
    return {
        "status"      : "ok",
        "embed_dim"   : EMBED_DIM,
        "device"      : str(device),
        "bank_entries": bank.index.ntotal,
        "gemini_ready": GEMINI_KEY is not None,
    }

@app.post("/predict")
async def predict(caption: str = Form(...), image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        img         = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor      = img_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            image_emb = visual_encoder(tensor)

        text_dict    = preprocessor.process_text(caption)
        input_ids    = text_dict['input_ids'].unsqueeze(0).to(device)
        attn_mask    = text_dict['attention_mask'].unsqueeze(0).to(device)
        with torch.no_grad():
            text_emb_768 = bert_encoder(input_ids, attn_mask)
            text_emb     = text_emb_768

        with torch.no_grad():
            score_tensor, fused_emb = fusion_model(
                text_emb        = text_emb,
                image_emb       = image_emb,
                return_embedding= True,
            )
        score = float(score_tensor.squeeze()) * 100
        label = "HIGH" if score >= 70 else "MEDIUM" if score >= 40 else "LOW"

        query_np = fused_emb.squeeze(0).cpu().numpy()
        advisory = bank.generate_advisory(
            query_emb       = query_np,
            caption         = caption,
            predicted_score = score / 100.0,
            post_type       = "image",
            k               = 5,
        )

        return {
            "score"    : round(score, 2),
            "label"    : label,
            "advisory" : advisory,
            "embed_dim": EMBED_DIM,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
