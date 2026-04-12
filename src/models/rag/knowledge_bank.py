import os
import pickle
import numpy as np
import torch
import faiss
import google.genai as genai

from pathlib import Path
from datetime import datetime, timezone
from PIL import Image
from torchvision import transforms

GEMINI_MODEL   = "gemini-1.5-flash"
PROJ_DIM       = 512
TOP_K_RETRIEVE = 5

IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class ViralKnowledgeBank:
    def __init__(self, embed_dim=PROJ_DIM, gemini_key=None):
        self.embed_dim = embed_dim
        self.metadata  = []
        self.index     = faiss.IndexFlatIP(embed_dim)
        if gemini_key:
            self.client = genai.Client(api_key=gemini_key)
        else:
            self.client = None
            print("No Gemini key — advisory generation disabled")

    def _add(self, embeddings, metadata):
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        embeddings = (embeddings / norms).astype(np.float32)
        self.index.add(embeddings)
        self.metadata.extend(metadata)
        print(f"Added {len(metadata)} entries — total: {self.index.ntotal}")

    def build_from_smpd(self, csv_path, img_dir, text_encoder,
                        visual_encoder, device="cpu", top_n=None, min_score=0.5):
        import pandas as pd
        from src.preprocessing.engine import ViralPreprocessor

        df = pd.read_csv(csv_path)
        df = df[df["virality_score"] >= min_score].copy()
        df = df.sort_values("virality_score", ascending=False)
        if top_n:
            df = df.head(top_n)

        print(f"Building from {len(df)} viral posts (score >= {min_score})...")

        img_proj  = torch.nn.Linear(2048, PROJ_DIM).to(device).eval()
        text_proj = torch.nn.Linear(768,  PROJ_DIM).to(device).eval()
        prep = ViralPreprocessor()
        text_encoder.eval()
        visual_encoder.eval()

        embeddings, metadata, skipped = [], [], 0

        for i, row in df.iterrows():
            try:
                post_id  = str(row["post_id"])
                caption  = str(row.get("caption", ""))
                score    = float(row["virality_score"])
                img_path = os.path.join(img_dir, f"{post_id}.jpg")

                text_dict = prep.process_text(caption)
                input_ids = text_dict["input_ids"].unsqueeze(0).to(device)
                attn_mask = text_dict["attention_mask"].unsqueeze(0).to(device)

                with torch.no_grad():
                    text_emb      = text_encoder(input_ids, attn_mask)
                    text_proj_emb = text_proj(text_emb)

                if os.path.exists(img_path):
                    img    = Image.open(img_path).convert("RGB")
                    tensor = IMG_TRANSFORM(img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        img_emb      = visual_encoder(tensor)
                        img_proj_emb = img_proj(img_emb)
                    fused = (img_proj_emb + text_proj_emb) / 2
                else:
                    fused = text_proj_emb

                embeddings.append(fused.squeeze(0).cpu().numpy())
                metadata.append({
                    "post_id":    post_id,
                    "caption":    caption,
                    "score":      score,
                    "has_image":  os.path.exists(img_path),
                    "post_type":  "image",
                    "source":     "smpd",
                    "indexed_at": datetime.now(timezone.utc).isoformat(),
                })

            except Exception as e:
                skipped += 1
                if skipped <= 5:
                    print(f"  Skipping {row.get('post_id', i)}: {e}")
                continue

            if len(embeddings) % 500 == 0 and len(embeddings) > 0:
                print(f"  Indexed {len(embeddings)}/{len(df)}...")

        if embeddings:
            self._add(np.array(embeddings, dtype=np.float32), metadata)
        print(f"Done — {self.index.ntotal} entries, {skipped} skipped")

    def retrieve(self, query_embedding, k=TOP_K_RETRIEVE):
        if self.index.ntotal == 0:
            return []
        q = query_embedding.reshape(1, -1).astype(np.float32)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm
        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(q, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            entry = dict(self.metadata[idx])
            entry["similarity"] = round(float(score), 4)
            results.append(entry)
        return results

    def generate_advisory(self, query_embedding, caption,
                           predicted_score, post_type="image", k=TOP_K_RETRIEVE):
        similar = self.retrieve(query_embedding, k=k)
        if similar:
            context = "\n".join([
                f"{i+1}. Score:{p['score']:.2f} | "
                f"Sim:{p['similarity']:.2f} | {p['caption'][:100]}"
                for i, p in enumerate(similar)
            ])
        else:
            context = "No similar posts found."

        prompt = f"""You are a social media virality expert for Instagram.
Post caption: "{caption}"
Predicted virality score: {predicted_score:.1f}/100
Similar viral posts for context:
{context}
Give 3-4 specific actionable bullet points to improve virality before posting.
{"Also suggest 2-3 trending audio styles for this reel." if post_type == "video" else ""}"""

        if self.client is None:
            return "Advisory unavailable — Gemini key not configured."
        try:
            response = self.client.models.generate_content(
                model=GEMINI_MODEL, contents=prompt
            )
            return response.text
        except Exception as e:
            return f"Advisory generation failed: {e}"

    def save(self, path):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(p / "faiss.index"))
        with open(p / "metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"Saved to {path} ({self.index.ntotal} entries)")

    def load(self, path):
        p = Path(path)
        self.index = faiss.read_index(str(p / "faiss.index"))
        with open(p / "metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)
        print(f"Loaded from {path} ({self.index.ntotal} entries)")

    def stats(self):
        if not self.metadata:
            return {"total_entries": 0}
        scores = [m["score"] for m in self.metadata]
        return {
            "total_entries": self.index.ntotal,
            "embed_dim":     self.embed_dim,
            "avg_score":     round(float(np.mean(scores)), 4),
            "max_score":     round(float(np.max(scores)), 4),
            "min_score":     round(float(np.min(scores)), 4),
        }
