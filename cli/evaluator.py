from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from .extractor import extract_attributes_batch
from .utils import openai_client_sync

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # keep evaluator import‑safe even if transformers not installed
    SentenceTransformer = None

# ---------------- Embedding helpers ----------------

_OPENAI_MODEL = "text-embedding-3-small"
_MAX_OPENAI_BATCH = 100

_local_st_model: SentenceTransformer | None = None


def _embed_openai(phrases: List[str]) -> np.ndarray:
    """Batch embed using OpenAI Embeddings API (synchronous)."""
    vectors: List[List[float]] = []
    for i in range(0, len(phrases), _MAX_OPENAI_BATCH):
        slice_ = phrases[i : i + _MAX_OPENAI_BATCH]
        resp = openai_client_sync.embeddings.create(model=_OPENAI_MODEL, input=slice_)
        # API returns in the same order
        vectors.extend([d.embedding for d in resp.data])
    return np.array(vectors, dtype=np.float32)


def _embed_local(phrases: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    global _local_st_model
    if _local_st_model is None:
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not installed – cannot use local mode")
        _local_st_model = SentenceTransformer(model_name)
    return _local_st_model.encode(phrases, normalize_embeddings=True)

# ---------------- Evaluation core ----------------

_DEF_THRESHOLD = 0.55


def evaluate_from_csv(
    evaluation_csv: str,
    similarity_threshold: float = _DEF_THRESHOLD,
    embedder: str = "openai",  # choices: "openai" / "local"
) -> Dict[str, Any]:
    """Run extractor on CSV and score with chosen embedding backend."""
    df = pd.read_csv(evaluation_csv)
    req = {"review_id", "body", "delight_attribute"}
    if not req.issubset(df.columns):
        raise ValueError(f"CSV must have columns {req}")

    # ------------- run extraction ----------------
    reviews = [
        {"review_id": r.review_id, "body": r.body or ""}
        for r in df.itertuples(index=False)
    ]
    extracted = extract_attributes_batch(reviews)
    id_to_pred = {r["review_id"]: r.get("delight_attributes", []) for r in extracted}

    correct = incorrect = total = 0
    failures: List[Dict[str, Any]] = []

    for row in tqdm(df.itertuples(index=False), desc="Scoring"):
        rid, body, gold_raw = row.review_id, row.body, row.delight_attribute
        try:
            gold = json.loads(gold_raw)
        except Exception:
            gold = [s.strip() for s in re.split(r"[;,]", str(gold_raw)) if s.strip()]
        gold_norm = [g.lower().strip() for g in gold]
        pred_norm = [p.lower().strip() for p in id_to_pred.get(rid, [])]

        if not gold_norm and not pred_norm:
            continue

        phrases = pred_norm + gold_norm
        if embedder == "openai":
            embeds = _embed_openai(phrases)
        else:
            embeds = _embed_local(phrases)

        pred_emb, gold_emb = embeds[: len(pred_norm)], embeds[len(pred_norm) :]

        for i, emb in enumerate(pred_emb):
            sims = cosine_similarity([emb], gold_emb)[0]
            max_sim = float(np.max(sims)) if sims.size else 0.0
            if max_sim >= similarity_threshold:
                correct += 1
            else:
                incorrect += 1
                failures.append({
                    "review_id": rid,
                    "pred": pred_norm[i],
                    "gold": gold_norm,
                    "max_sim": max_sim,
                })
            total += 1

    accuracy = correct / total if total else 0.0
    return {
        "correct_count": correct,
        "incorrect_count": incorrect,
        "accuracy": accuracy
    }