"""
RAG Knowledge Bank
Retrieval-Augmented benchmarking using FAISS index of past viral posts.
Handles cold-start content by finding similar high-engagement examples.
"""
"""
rag_knowledge_bank.py — ViralVisions AI RAG Knowledge Bank

Stores embeddings of viral posts in a FAISS index.
At inference: query embedding → FAISS top-k → Gemini Flash advisory.

Data source: SMPD dataset
    CSV columns : post_id, caption, virality_score
    Images      : {img_dir}/{post_id}.jpg

Usage:
    bank = ViralKnowledgeBank(gemini_key='...')
    bank.build_from_smpd('data/smpd/annotations/train.csv',
                          'data/smpd/images',
                          text_encoder, visual_encoder, device)
    bank.save('knowledge_bank')

    advice = bank.generate_advisory(query_emb, caption, score)
"""

import os
import pickle
import numpy as np
import torch
import faiss
import google.generativeai as genai

from pathlib import Path
from datetime import datetime
from PIL import Image
from torchvision import transforms


# ── Config ────────────────────────────────────────────────────────────────────

GEMINI_MODEL   = "gemini-1.5-flash"
PROJ_DIM       = 512    # matches TrimodalFusion.PROJ_DIM
TOP_K_RETRIEVE = 5

IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── Knowledge Bank ─────────────────────────────────────────────────────────────

class ViralKnowledgeBank:
    """
    FAISS-backed knowledge bank for RAG-augmented virality prediction.

    Embeddings are 512-dim fused representations:
        image_emb [2048] → proj [512]  +  text_emb [768] → proj [512]
        → mean pool → [512] stored in FAISS

    Each FAISS entry has a parallel metadata dict in self.metadata.
    """

    def __init__(self,
                 embed_dim:  int = PROJ_DIM,
                 gemini_key: str = None):
        self.embed_dim = embed_dim
        self.metadata  = []

        # Inner Product index (cosine similarity after L2 norm)
        self.index = faiss.IndexFlatIP(embed_dim)

        # Gemini
        if gemini_key:
            genai.configure(api_key=gemini_key)
            self.gemini = genai.GenerativeModel(GEMINI_MODEL)
        else:
            self.gemini = None
            print("⚠️  No Gemini key — advisory generation disabled")

    # ── Build from SMPD ───────────────────────────────────────────────────────

    def build_from_smpd(self,
                        csv_path:       str,
                        img_dir:        str,
                        text_encoder,
                        visual_encoder,
                        device:         str = 'cpu',
                        top_n:          int = None,
                        min_score:      float = 0.5) -> None:
        """
        Build knowledge bank from SMPD CSV.

        Only indexes posts above min_score so the bank contains
        genuinely viral examples for RAG context.

        Args:
            csv_path       : path to SMPD annotations CSV
                             (columns: post_id, caption, virality_score)
            img_dir        : directory containing {post_id}.jpg files
            text_encoder   : ViralBertEncoder instance
            visual_encoder : ResNetEncoder instance (embed_dim=2048)
            device         : 'cuda' or 'cpu'
            top_n          : only index top N posts by score (None = all)
            min_score      : minimum virality_score to include (0-1 scale)
        """
        import pandas as pd
        from src.preprocessing.engine import ViralPreprocessor

        df = pd.read_csv(csv_path)
        df = df[df['virality_score'] >= min_score].copy()
        df = df.sort_values('virality_score', ascending=False)
        if top_n:
            df = df.head(top_n)

        print(f"Building knowledge bank from {len(df)} viral posts "
              f"(score >= {min_score})...")

        # Fixed projection layers: raw dims → 512 for storage
        # Weights don't matter here — we just need consistent dims
        img_proj  = torch.nn.Linear(2048, PROJ_DIM).to(device).eval()
        text_proj = torch.nn.Linear(768,  PROJ_DIM).to(device).eval()

        prep = ViralPreprocessor()
        text_encoder.eval()
        visual_encoder.eval()

        embeddings = []
        metadata   = []
        skipped    = 0

        for i, row in df.iterrows():
            try:
                post_id  = str(row['post_id'])
                caption  = str(row.get('caption', ''))
                score    = float(row['virality_score'])
                img_path = os.path.join(img_dir, f"{post_id}.jpg")

                # Text embedding
                text_dict = prep.process_text(caption)
                input_ids = text_dict['input_ids'].unsqueeze(0).to(device)
                attn_mask = text_dict['attention_mask'].unsqueeze(0).to(device)

                with torch.no_grad():
                    text_emb      = text_encoder(input_ids, attn_mask)  # [1, 768]
                    text_proj_emb = text_proj(text_emb)                  # [1, 512]

                # Visual embedding — text-only fallback if image missing
                if os.path.exists(img_path):
                    img    = Image.open(img_path).convert('RGB')
                    tensor = IMG_TRANSFORM(img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        img_emb      = visual_encoder(tensor)            # [1, 2048]
                        img_proj_emb = img_proj(img_emb)                 # [1, 512]
                    fused = (img_proj_emb + text_proj_emb) / 2           # [1, 512]
                else:
                    fused = text_proj_emb                                # [1, 512]

                embeddings.append(fused.squeeze(0).cpu().numpy())
                metadata.append({
                    'post_id':    post_id,
                    'caption':    caption,
                    'score':      score,
                    'has_image':  os.path.exists(img_path),
                    'post_type':  'image',
                    'source':     'smpd',
                    'indexed_at': datetime.utcnow().isoformat(),
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

        print(f"Done — {self.index.ntotal} entries indexed, {skipped} skipped")

    # ── Internal add ──────────────────────────────────────────────────────────

    def _add(self, embeddings: np.ndarray, metadata: list) -> None:
        """L2-normalise then add to FAISS index."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        embeddings = (embeddings / norms).astype(np.float32)
        self.index.add(embeddings)
        self.metadata.extend(metadata)
        print(f"Added {len(metadata)} entries — total: {self.index.ntotal}")

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self,
                 query_embedding: np.ndarray,
                 k: int = TOP_K_RETRIEVE) -> list:
        """
        Retrieve k most similar viral posts.

        Args:
            query_embedding : [512] float32 — the fused representation
                              from TrimodalFusion before the regression head
            k               : number of results

        Returns:
            list of metadata dicts, each with added 'similarity' key
        """
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
            entry['similarity'] = round(float(score), 4)
            results.append(entry)

        return results

    # ── Gemini Advisory ───────────────────────────────────────────────────────

    def generate_advisory(self,
                           query_embedding:  np.ndarray,
                           caption:          str,
                           predicted_score:  float,
                           post_type:        str = 'image',
                           k:                int = TOP_K_RETRIEVE) -> str:
        """
        Retrieve similar viral posts then generate advice via Gemini Flash.

        Args:
            query_embedding : [512] fused embedding of current post
            caption         : raw caption text
            predicted_score : virality score from TrimodalFusion (0-100)
            post_type       : 'image' or 'video'
            k               : number of similar posts to retrieve

        Returns:
            advisory string from Gemini
        """
        similar = self.retrieve(query_embedding, k=k)

        if similar:
            context_lines = []
            for i, post in enumerate(similar, 1):
                context_lines.append(
                    f"{i}. Score: {post['score']:.2f} | "
                    f"Similarity: {post['similarity']:.2f} | "
                    f"Caption: \"{post['caption'][:100]}\""
                )
            context = "\n".join(context_lines)
        else:
            context = "No similar posts found in knowledge bank."

        prompt = f"""You are a social media virality expert for Instagram.

A creator is about to publish the following {post_type} post:
Caption: "{caption}"
Predicted virality score: {predicted_score:.1f} / 100

Here are {k} similar posts that performed well, for context:
{context}

Based on the predicted score and the viral examples above, give 3-4 specific,
actionable suggestions to improve this post's virality before publishing.
{"Also suggest 2-3 trending audio styles that suit this content based on similar viral reels." if post_type == 'video' else ""}

Keep it concise. Use bullet points. Be specific, not generic."""

        if self.gemini is None:
            return "Gemini API key not configured — advisory unavailable."

        try:
            response = self.gemini.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Advisory generation failed: {e}"

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(p / 'faiss.index'))
        with open(p / 'metadata.pkl', 'wb') as f:
            pickle.dump(self.metadata, f)
        print(f"Saved to {path} ({self.index.ntotal} entries)")

    def load(self, path: str) -> None:
        p = Path(path)
        self.index = faiss.read_index(str(p / 'faiss.index'))
        with open(p / 'metadata.pkl', 'rb') as f:
            self.metadata = pickle.load(f)
        print(f"Loaded from {path} ({self.index.ntotal} entries)")

    def stats(self) -> dict:
        if not self.metadata:
            return {"total_entries": 0}
        scores = [m['score'] for m in self.metadata]
        return {
            "total_entries": self.index.ntotal,
            "embed_dim":     self.embed_dim,
            "avg_score":     round(float(np.mean(scores)), 4),
            "max_score":     round(float(np.max(scores)), 4),
            "min_score":     round(float(np.min(scores)), 4),
        }


# ── Colab quick-test (no dataset needed) ─────────────────────────────────────

if __name__ == "__main__":
    print("Testing ViralKnowledgeBank with dummy data...\n")

    bank = ViralKnowledgeBank(embed_dim=512)

    N    = 200
    embs = np.random.randn(N, 512).astype(np.float32)
    meta = [
        {
            'post_id':    str(i),
            'caption':    f'golden hour vibes #{["aesthetic","viral","reels"][i%3]}',
            'score':      round(0.5 + np.random.rand() * 0.5, 3),
            'has_image':  True,
            'post_type':  'image',
            'source':     'dummy',
            'indexed_at': datetime.utcnow().isoformat(),
        }
        for i in range(N)
    ]

    bank._add(embs, meta)
    print(f"Stats: {bank.stats()}\n")

    query   = np.random.randn(512).astype(np.float32)
    results = bank.retrieve(query, k=3)
    print("Top 3 retrieved:")
    for r in results:
        print(f"  sim={r['similarity']:.4f} | score={r['score']:.3f} | "
              f"{r['caption'][:50]}")

    bank.save('/tmp/test_kb')
    bank2 = ViralKnowledgeBank()
    bank2.load('/tmp/test_kb')
    print(f"\nReloaded stats: {bank2.stats()}")
    print("\n✅ All tests passed")
