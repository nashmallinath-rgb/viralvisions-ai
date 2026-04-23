"""
knowledge_bank.py — ViralVisions AI: RAG Knowledge Bank
"""
import os, pickle, logging, json
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
import numpy as np
import faiss
import torch
import torch.nn as nn

try:
    from groq import Groq as GroqClient
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False

logger = logging.getLogger('viralvisions.rag')

class ViralKnowledgeBank:
    def __init__(self, embed_dim: int = 512, gemini_key: Optional[str] = None,
                 groq_key: Optional[str] = None):
        self.embed_dim = embed_dim
        self.index     = faiss.IndexFlatIP(embed_dim)
        self.metadata: List[Dict[str, Any]] = []
        self._groq_client = None
        key = groq_key or os.environ.get('GROQ_KEY') or os.environ.get('GROK_KEY')
        if key and _GROQ_AVAILABLE:
            try:
                self._groq_client = GroqClient(api_key=key)
                logger.info("Groq client initialised.")
            except Exception as e:
                logger.warning(f"Groq init failed: {e}")

    def _add(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        emb = embeddings.astype(np.float32)
        faiss.normalize_L2(emb)
        self.index.add(emb)
        self.metadata.extend(metadata)

    def retrieve(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            return []
        q = query_embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(q)
        k_actual = min(k, self.index.ntotal)
        distances, indices = self.index.search(q, k_actual)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            item = dict(self.metadata[idx])
            item['similarity'] = float(dist)
            results.append(item)
        return results

    def generate_advisory(self, query_emb: np.ndarray, caption: str,
                          predicted_score: float, post_type: str,
                          k: int = 5) -> str:
        similar = self.retrieve(query_emb, k=k)
        context_lines = []
        for i, item in enumerate(similar, 1):
            context_lines.append(
                f"{i}. [{item.get('post_type','?')}] "
                f"score={item['score']*100:.1f}% | "
                f"sim={item['similarity']*100:.1f}% | "
                f"caption: {str(item.get('caption',''))[:120]}"
            )
        context_str = '\n'.join(context_lines) if context_lines else "No similar posts found."

        prompt = (
            f"You are ViralVisions AI, an expert Instagram growth advisor.\n\n"
            f"A user wants feedback on their upcoming post:\n"
            f"  Type    : {post_type}\n"
            f"  Caption : {caption[:300]}\n"
            f"  Predicted virality score: {predicted_score * 100:.1f}%\n\n"
            f"Here are the {len(similar)} most similar high-performing posts:\n"
            f"{context_str}\n\n"
            f"Provide:\n"
            f"1. A one-sentence virality verdict.\n"
            f"2. Three specific, actionable improvements to the caption or content.\n"
            f"3. Best time to post for maximum Instagram reach.\n"
            f"Keep response under 200 words. Be direct and practical."
        )

        if self._groq_client:
            try:
                response = self._groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"Groq call failed: {e}. Using fallback.")

        label = ("HIGH" if predicted_score >= 0.7
                 else "MEDIUM" if predicted_score >= 0.4
                 else "LOW")
        return (
            f"Virality outlook: {label} ({predicted_score * 100:.1f}%). "
            f"Found {len(similar)} similar posts in knowledge bank. "
            "Tip: use specific niche hashtags, post between 6-9pm local time, "
            "and open with a strong hook in the first line of your caption."
        )

    def save(self, path: str) -> None:
        faiss.write_index(self.index, f"{path}.index")
        with open(f"{path}_metadata.pkl", 'wb') as f:
            pickle.dump(self.metadata, f)
        logger.info(f"Knowledge bank saved — {self.index.ntotal} entries")

    def load(self, path: str) -> None:
        self.index = faiss.read_index(f"{path}.index")
        with open(f"{path}_metadata.pkl", 'rb') as f:
            self.metadata = pickle.load(f)
        logger.info(f"Knowledge bank loaded — {self.index.ntotal} entries")

    def stats(self) -> Dict[str, Any]:
        n = self.index.ntotal
        if n == 0:
            return {'total': 0}
        scores = [m.get('score', 0.0) for m in self.metadata]
        return {'total': n, 'mean_score': float(np.mean(scores)),
                'max_score': float(np.max(scores))}

    def build_from_smpd(self, label_path, text_path, img_filepath_path,
                        img_base_dir, text_encoder, visual_encoder,
                        device='cpu', top_n=None, min_score=0.5):
        from torchvision import transforms
        from PIL import Image as PILImage
        from src.preprocessing.engine import ViralPreprocessor

        with open(label_path) as f:
            raw_labels = [float(l.strip()) for l in f if l.strip()]
        with open(text_path) as f:
            text_meta = json.load(f)
        with open(img_filepath_path) as f:
            img_rel_paths = [l.strip() for l in f if l.strip()]

        label_arr   = np.array(raw_labels, dtype=np.float32)
        norm_labels = (label_arr - label_arr.min()) / (label_arr.max() - label_arr.min() + 1e-8)
        keep_indices = [i for i, s in enumerate(norm_labels) if s >= min_score]
        if top_n:
            keep_indices = keep_indices[:top_n]

        text_proj = nn.Linear(768, 512, bias=False).to(device)
        img_proj  = nn.Linear(2048, 512, bias=False).to(device)
        text_proj.eval(); img_proj.eval()

        img_transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        preprocessor = ViralPreprocessor(bert_model='bert-base-uncased', max_len=128)
        text_encoder.eval().to(device)
        visual_encoder.eval().to(device)

        all_embeddings, all_metadata = [], []
        with torch.no_grad():
            for count, i in enumerate(keep_indices):
                meta    = text_meta[i]
                caption = f"{meta.get('Title','')} {meta.get('Alltags','')}".strip()
                pid     = str(meta.get('Pid', i))
                td      = preprocessor.process_text(caption)
                ids     = td['input_ids'].unsqueeze(0).to(device)
                mask    = td['attention_mask'].unsqueeze(0).to(device)
                t_emb   = text_proj(text_encoder(ids, mask))
                fused   = t_emb.clone()
                abs_path = os.path.join(img_base_dir, img_rel_paths[i])
                if os.path.exists(abs_path):
                    try:
                        img   = PILImage.open(abs_path).convert('RGB')
                        v_emb = img_proj(visual_encoder(
                            img_transform(img).unsqueeze(0).to(device)))
                        fused = (t_emb + v_emb) / 2.0
                    except:
                        pass
                all_embeddings.append(fused.squeeze(0).cpu().numpy())
                all_metadata.append({
                    'post_id': pid, 'caption': caption,
                    'score': float(norm_labels[i]),
                    'post_type': 'image', 'source': 'smpd',
                    'indexed_at': datetime.now(timezone.utc).isoformat()
                })
                if (count+1) % 1000 == 0:
                    logger.info(f"Indexed {count+1}/{len(keep_indices)}")

        self._add(np.stack(all_embeddings, 0), all_metadata)
        logger.info(f"Done. {self.index.ntotal} entries.")
