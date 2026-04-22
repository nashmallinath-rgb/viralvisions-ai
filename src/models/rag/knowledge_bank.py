"""
knowledge_bank.py — ViralVisions AI: RAG Knowledge Bank
========================================================
FAISS-backed vector store of high-virality posts.
Provides retrieval-augmented advisory generation via Gemini.

Authors : Nisha Mallinath (1MS23CI077) & Shrusti Mathapati (1MS23CI118)
Institute: MSRIT Bangalore — VI Sem CSE AI&ML
"""

import os
import pickle
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

import numpy as np
import faiss
import torch
import torch.nn as nn

# ── Instaloader (for refresh_from_instagram) ──────────────────────────────────
import instaloader
from instaloader import QueryReturnedNotFoundException

# ── Gemini — use google.genai (NOT google.generativeai which is deprecated) ───
try:
    from google import genai as google_genai
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False

logger = logging.getLogger('viralvisions.rag')


class ViralKnowledgeBank:
    """
    FAISS IndexFlatIP knowledge bank of social-media posts.

    Each entry stores:
        - A 512-dim L2-normalised embedding (cosine similarity via inner product)
        - A metadata dict with post details

    Typical usage
    -------------
    bank = ViralKnowledgeBank(embed_dim=512, gemini_key=os.environ['GEMINI_KEY'])
    bank.build_from_smpd(label_path, text_path, img_fp_path, img_base, text_enc, vis_enc)
    advisory = bank.generate_advisory(query_emb, caption, score, 'image')
    bank.save('knowledge_bank')
    """

    def __init__(self, embed_dim: int = 512, gemini_key: Optional[str] = None):
        self.embed_dim    = embed_dim
        self.index        = faiss.IndexFlatIP(embed_dim)  # cosine via L2-norm + IP
        self.metadata: List[Dict[str, Any]] = []
        self.gemini_key   = gemini_key
        self._gemini_client = None

        if gemini_key and _GENAI_AVAILABLE:
            try:
                self._gemini_client = google_genai.Client(api_key=gemini_key)
                logger.info("Gemini client initialised.")
            except Exception as e:
                logger.warning(f"Gemini init failed: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    # INTERNAL ADD
    # ──────────────────────────────────────────────────────────────────────────

    def _add(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """
        Add pre-computed embeddings to the FAISS index.

        Parameters
        ----------
        embeddings : np.ndarray, shape [N, embed_dim], dtype float32
            Will be L2-normalised in-place before indexing.
        metadata   : list of N dicts
        """
        assert embeddings.shape[0] == len(metadata), \
            "embeddings and metadata must have the same length"
        assert embeddings.shape[1] == self.embed_dim, \
            f"Expected embed_dim={self.embed_dim}, got {embeddings.shape[1]}"

        emb = embeddings.astype(np.float32)
        faiss.normalize_L2(emb)          # in-place L2 normalisation
        self.index.add(emb)
        self.metadata.extend(metadata)

    # ──────────────────────────────────────────────────────────────────────────
    # RETRIEVE
    # ──────────────────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Find the k most similar posts to the query embedding.

        Parameters
        ----------
        query_embedding : np.ndarray, shape [512] or [1, 512]
        k               : number of results

        Returns
        -------
        List of metadata dicts, each augmented with a 'similarity' float in [0, 1].
        """
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
            item['similarity'] = float(dist)   # cosine similarity in [0,1] after L2 norm
            results.append(item)

        return results

    # ──────────────────────────────────────────────────────────────────────────
    # GENERATE ADVISORY
    # ──────────────────────────────────────────────────────────────────────────

    def generate_advisory(
        self,
        query_emb: np.ndarray,
        caption: str,
        predicted_score: float,
        post_type: str,
        k: int = 5,
    ) -> str:
        """
        Retrieve similar posts and ask Gemini for an actionable virality advisory.

        Parameters
        ----------
        query_emb       : [512] embedding of the current post
        caption         : caption text of the current post
        predicted_score : model's predicted virality score (0-1)
        post_type       : 'image' or 'video'
        k               : number of retrieved examples to include in prompt

        Returns
        -------
        Advisory string (from Gemini, or a rule-based fallback).
        """
        similar = self.retrieve(query_emb, k=k)
        context_lines = []
        for i, item in enumerate(similar, 1):
            score_pct = f"{item['score'] * 100:.1f}%"
            sim_pct   = f"{item['similarity'] * 100:.1f}%"
            context_lines.append(
                f"{i}. [{item.get('post_type','?')}] "
                f"score={score_pct} | sim={sim_pct} | "
                f"caption: {str(item.get('caption',''))[:120]}"
            )
        context_str = '\n'.join(context_lines) if context_lines else "No similar posts found."

        prompt = (
            f"You are ViralVisions AI, an expert Instagram growth advisor.\n\n"
            f"A user wants feedback on their upcoming post:\n"
            f"  Type    : {post_type}\n"
            f"  Caption : {caption[:300]}\n"
            f"  Predicted virality score: {predicted_score * 100:.1f}% (0=low, 100=viral)\n\n"
            f"Here are the {len(similar)} most similar high-performing posts from our knowledge bank:\n"
            f"{context_str}\n\n"
            f"Based on this context, provide:\n"
            f"1. A one-sentence virality verdict.\n"
            f"2. Three specific, actionable improvements to the caption or content strategy.\n"
            f"3. The best time to post for maximum reach (Instagram-specific).\n"
            f"Keep the total response under 200 words. Be direct and practical."
        )

        if self._gemini_client:
            try:
                response = self._gemini_client.models.generate_content(
                    model='gemini-2.0-flash',
                    contents=prompt,
                )
                return response.text.strip()
            except Exception as e:
                logger.warning(f"Gemini call failed: {e}. Using rule-based fallback.")

        # ── Rule-based fallback (no Gemini key or API error) ─────────────────
        label = ("HIGH" if predicted_score >= 0.7
                 else "MEDIUM" if predicted_score >= 0.4
                 else "LOW")
        return (
            f"Virality outlook: {label} ({predicted_score * 100:.1f}%). "
            f"Found {len(similar)} similar posts in knowledge bank. "
            "Tip: use specific niche hashtags, post between 6-9pm local time, "
            "and open with a strong hook in the first line of your caption."
        )

    # ──────────────────────────────────────────────────────────────────────────
    # SAVE / LOAD
    # ──────────────────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Persist FAISS index and metadata.
        Creates {path}.index and {path}_metadata.pkl
        """
        faiss.write_index(self.index, f"{path}.index")
        with open(f"{path}_metadata.pkl", 'wb') as f:
            pickle.dump(self.metadata, f)
        logger.info(f"Knowledge bank saved to {path}  ({self.index.ntotal} entries)")

    def load(self, path: str) -> None:
        """Load FAISS index and metadata from disk."""
        self.index    = faiss.read_index(f"{path}.index")
        with open(f"{path}_metadata.pkl", 'rb') as f:
            self.metadata = pickle.load(f)
        logger.info(f"Knowledge bank loaded from {path}  ({self.index.ntotal} entries)")

    # ──────────────────────────────────────────────────────────────────────────
    # STATS
    # ──────────────────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """Return basic statistics about the knowledge bank."""
        n = self.index.ntotal
        if n == 0:
            return {'total': 0}
        scores    = [m.get('score', 0.0) for m in self.metadata]
        sources   = list({m.get('source', 'unknown') for m in self.metadata})
        post_types = list({m.get('post_type', 'unknown') for m in self.metadata})
        return {
            'total'      : n,
            'mean_score' : float(np.mean(scores)),
            'max_score'  : float(np.max(scores)),
            'sources'    : sources,
            'post_types' : post_types,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # BUILD FROM SMPD
    # ══════════════════════════════════════════════════════════════════════════

    def build_from_smpd(
        self,
        label_path: str,
        text_path: str,
        img_filepath_path: str,
        img_base_dir: str,
        text_encoder,          # ViralBertEncoder instance
        visual_encoder,        # ResNetEncoder instance (embed_dim=2048)
        device: str = 'cpu',
        top_n: Optional[int] = None,
        min_score: float = 0.5,
    ) -> None:
        """
        Index SMPD-Image posts into the knowledge bank.

        Reads the three SMPD-Image files (aligned by row order), normalises
        labels, filters by min_score, generates 512-dim fused embeddings, and
        adds them to the FAISS index.

        Embedding strategy
        ------------------
        • Text embedding : BERT pooler_output [768] → text_proj [512]
        • Visual embedding: ResNet features [2048] → img_proj [512]
        • Fused embedding : mean(text_proj, img_proj) if image exists,
                            else text_proj only.

        The projection layers use random (untrained) weights — that is
        intentional. We only need a consistent 512-dim metric space for
        approximate nearest-neighbour retrieval, not discriminative accuracy.

        Parameters
        ----------
        label_path        : path to train_label.txt
        text_path         : path to train_text.json
        img_filepath_path : path to train_img_filepath.txt
        img_base_dir      : base dir for images; images are at
                            {img_base_dir}/train/{uid}/{pid}.jpg
        text_encoder      : ViralBertEncoder (frozen)
        visual_encoder    : ResNetEncoder(embed_dim=2048)
        device            : 'cpu' or 'cuda'
        top_n             : if set, only index the top_n records after
                            score filtering (useful for quick testing)
        min_score         : minimum normalised score (0-1) to include a post
        """
        from torchvision import transforms
        from PIL import Image as PILImage
        from src.preprocessing.engine import ViralPreprocessor

        logger.info("Building knowledge bank from SMPD-Image...")

        # ── Load raw files ────────────────────────────────────────────────────
        with open(label_path, 'r') as f:
            raw_labels = [float(line.strip()) for line in f if line.strip()]

        with open(text_path, 'r') as f:
            text_meta = json_load(f)

        with open(img_filepath_path, 'r') as f:
            img_rel_paths = [line.strip() for line in f if line.strip()]

        n = len(raw_labels)
        assert len(text_meta) == n and len(img_rel_paths) == n, \
            "SMPD file length mismatch"
        logger.info(f"  Loaded {n:,} raw records")

        # ── Normalise labels ──────────────────────────────────────────────────
        label_arr = np.array(raw_labels, dtype=np.float32)
        label_min, label_max = label_arr.min(), label_arr.max()
        norm_labels = (label_arr - label_min) / (label_max - label_min + 1e-8)

        # ── Filter by min_score ───────────────────────────────────────────────
        keep_indices = [i for i, s in enumerate(norm_labels) if s >= min_score]
        if top_n is not None:
            keep_indices = keep_indices[:top_n]
        logger.info(f"  After min_score={min_score:.2f} filter: "
                    f"{len(keep_indices):,} records")

        # ── Projection layers (random, fixed — consistent 512-dim space) ──────
        text_proj = nn.Linear(768, 512, bias=False).to(device)
        img_proj  = nn.Linear(2048, 512, bias=False).to(device)
        text_proj.eval()
        img_proj.eval()

        # ── Image transform (eval mode, no augmentation) ──────────────────────
        img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        preprocessor = ViralPreprocessor(bert_model='bert-base-uncased', max_len=128)

        text_encoder.eval().to(device)
        visual_encoder.eval().to(device)

        all_embeddings: List[np.ndarray] = []
        all_metadata:   List[Dict]       = []

        with torch.no_grad():
            for count, i in enumerate(keep_indices):
                meta       = text_meta[i]
                caption    = f"{meta.get('Title', '')} {meta.get('Alltags', '')}".strip()
                pid        = str(meta.get('Pid', i))
                score_norm = float(norm_labels[i])
                rel_path   = img_rel_paths[i]
                abs_path   = os.path.join(img_base_dir, rel_path)

                # ── Text embedding ────────────────────────────────────────────
                text_dict  = preprocessor.process_text(caption)
                input_ids  = text_dict['input_ids'].unsqueeze(0).to(device)  # [1,128]
                attn_mask  = text_dict['attention_mask'].unsqueeze(0).to(device)
                text_emb   = text_encoder(input_ids, attn_mask)               # [1,768]
                text_512   = text_proj(text_emb)                               # [1,512]

                # ── Visual embedding (if image exists) ────────────────────────
                has_image = False
                fused_512 = text_512.clone()   # default: text-only

                if os.path.exists(abs_path):
                    try:
                        img      = PILImage.open(abs_path).convert('RGB')
                        img_t    = img_transform(img).unsqueeze(0).to(device)  # [1,3,224,224]
                        vis_emb  = visual_encoder(img_t)                        # [1,2048]
                        vis_512  = img_proj(vis_emb)                            # [1,512]
                        fused_512 = (text_512 + vis_512) / 2.0                 # mean fusion
                        has_image = True
                    except Exception as e:
                        logger.debug(f"Image load failed for pid={pid}: {e}")

                embedding = fused_512.squeeze(0).cpu().numpy()  # [512]
                all_embeddings.append(embedding)
                all_metadata.append({
                    'post_id'   : pid,
                    'caption'   : caption,
                    'score'     : score_norm,
                    'has_image' : has_image,
                    'post_type' : 'image',
                    'source'    : 'smpd',
                    'indexed_at': datetime.now(timezone.utc).isoformat(),
                })

                if (count + 1) % 1000 == 0:
                    logger.info(f"  Indexed {count + 1:,}/{len(keep_indices):,} posts")

        # ── Batch-add to FAISS ────────────────────────────────────────────────
        emb_matrix = np.stack(all_embeddings, axis=0)  # [N, 512]
        self._add(emb_matrix, all_metadata)
        logger.info(f"✓ build_from_smpd complete. "
                    f"Knowledge bank now has {self.index.ntotal:,} entries.")

    # ══════════════════════════════════════════════════════════════════════════
    # REFRESH FROM INSTAGRAM
    # ══════════════════════════════════════════════════════════════════════════

    def refresh_from_instagram(
        self,
        session_cookies: Dict[str, str],
        hashtags: List[str],
        posts_per_tag: int = 20,
        text_encoder=None,
        device: str = 'cpu',
    ) -> int:
        """
        Scrape recent Instagram posts via instaloader and add them to the index.

        Uses authenticated session cookies to access public hashtag feeds.
        No Meta Graph API required.

        Parameters
        ----------
        session_cookies : dict with keys 'sessionid' and 'csrftoken'.
                          ⚠ Cookies expire frequently.
                          Regenerate: instagram.com → F12 → Application → Cookies
                          Copy 'sessionid' and 'csrftoken' values.
        hashtags        : list of hashtag strings WITHOUT the '#' prefix,
                          e.g. ['reels', 'streetphotography', 'indiefilm']
        posts_per_tag   : max number of posts to collect per hashtag
        text_encoder    : ViralBertEncoder instance. If None, random [512]
                          embeddings are used (still useful for demo/testing).
        device          : 'cpu' or 'cuda'

        Returns
        -------
        Total number of new posts added to the knowledge bank.

        Notes
        -----
        • 403 errors are expected and normal. instaloader retries automatically
          with exponential backoff. Do NOT run as Administrator (breaks cookie
          decryption on Windows).
        • QueryReturnedNotFoundException is raised for private/deleted hashtags
          or when rate-limited — caught and skipped gracefully.
        • share_rate = likes / (likes + comments + 1) × 100   (proxy virality)
          Normalised to 0-1 by dividing by 100.
        """
        # ── Set up instaloader with session cookies ───────────────────────────
        L = instaloader.Instaloader(
            quiet=True,
            download_pictures=False,
            download_videos=False,
            download_video_thumbnails=False,
            download_geotags=False,
            download_comments=False,
            save_metadata=False,
        )

        # Inject cookies (same as being logged in via browser)
        L.context._session.cookies.set(
            'sessionid', session_cookies['sessionid'], domain='.instagram.com'
        )
        L.context._session.cookies.set(
            'csrftoken', session_cookies['csrftoken'], domain='.instagram.com'
        )
        # Setting username signals to instaloader that we're authenticated
        L.context.username = 'nash.inks'

        # ── Optional: text encoder setup ──────────────────────────────────────
        if text_encoder is not None:
            from src.preprocessing.engine import ViralPreprocessor
            preprocessor = ViralPreprocessor(bert_model='bert-base-uncased', max_len=128)
            text_encoder.eval().to(device)
        else:
            preprocessor = None

        # ── Projection layer (random fixed weights, same rationale as build_from_smpd)
        text_proj = nn.Linear(768, 512, bias=False).to(device)
        text_proj.eval()

        total_added = 0

        for tag in hashtags:
            logger.info(f"  Fetching hashtag: #{tag} (up to {posts_per_tag} posts)")
            new_posts = 0

            try:
                hashtag_obj = instaloader.Hashtag.from_name(L.context, tag)
                # get_posts_resumable avoids the 'more_available' key error
                # that affects get_posts() on newer instaloader versions
                post_gen = hashtag_obj.get_posts_resumable()

                while new_posts < posts_per_tag:
                    try:
                        post = next(post_gen)
                    except StopIteration:
                        break

                    try:
                        likes    = post.likes
                        comments = post.comments
                        caption  = post.caption or ''
                        is_video = post.is_video
                        shortcode = post.shortcode

                        # ── Proxy virality score ──────────────────────────────
                        share_rate       = likes / (likes + comments + 1) * 100.0
                        score_normalised = min(share_rate / 100.0, 1.0)

                        # ── Embedding ─────────────────────────────────────────
                        if text_encoder is not None and preprocessor is not None:
                            with torch.no_grad():
                                text_dict = preprocessor.process_text(caption)
                                ids  = text_dict['input_ids'].unsqueeze(0).to(device)
                                mask = text_dict['attention_mask'].unsqueeze(0).to(device)
                                emb  = text_encoder(ids, mask)      # [1,768]
                                emb  = text_proj(emb)               # [1,512]
                                embedding = emb.squeeze(0).cpu().numpy()  # [512]
                        else:
                            embedding = np.random.randn(self.embed_dim).astype(np.float32)

                        # ── Metadata ──────────────────────────────────────────
                        meta = {
                            'post_id'   : shortcode,
                            'caption'   : caption[:500],
                            'score'     : score_normalised,
                            'post_type' : 'video' if is_video else 'image',
                            'source'    : f'instagram_{tag}',
                            'indexed_at': datetime.now(timezone.utc).isoformat(),
                        }

                        self._add(
                            np.expand_dims(embedding, 0),  # [1, 512]
                            [meta]
                        )
                        new_posts  += 1
                        total_added += 1

                    except Exception as inner_e:
                        # Individual post failures are non-fatal
                        logger.debug(f"    Skipped a post in #{tag}: {inner_e}")
                        continue

            except QueryReturnedNotFoundException:
                logger.warning(f"  Hashtag '#{tag}' not found or rate-limited — skipping.")
                continue
            except Exception as e:
                logger.warning(f"  Error fetching #{tag}: {e} — skipping.")
                continue

            logger.info(f"  #{tag}: added {new_posts} posts")

        logger.info(f"✓ refresh_from_instagram complete. "
                    f"Added {total_added} posts. "
                    f"Bank total: {self.index.ntotal:,}")
        return total_added


# ──────────────────────────────────────────────────────────────────────────────
# Tiny helper: json.load that works on both file objects and strings
# ──────────────────────────────────────────────────────────────────────────────
import json as _json_module

def json_load(f):
    """Wrapper so the class body can call json_load without importing json at module scope clash."""
    return _json_module.load(f)
