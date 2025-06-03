import logging
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from collections import defaultdict, Counter

# Initialize embedding model at module level to avoid reloading
_embedding_model: SentenceTransformer = None

def cluster_attributes(
    all_attributes: List[str],
    eps: float = 0.5,
    min_samples: int = 1,
    embedding_model_name: str = 'all-MiniLM-L6-v2'
) -> Dict[str, int]:
    """
    Cluster semantically similar attributes.
    Returns a mapping attribute -> cluster_id.
    """
    if not all_attributes:
        return {}

    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(embedding_model_name)

    embeddings = _embedding_model.encode(all_attributes, normalize_embeddings=True)
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    labels = clustering.fit_predict(embeddings)

    attr_to_cluster = {attr: int(label) for attr, label in zip(all_attributes, labels)}
    return attr_to_cluster


def get_cluster_representatives(
    attr_to_cluster: Dict[str, int],
    original_attrs: List[str]
) -> Dict[int, str]:
    """
    Given mapping of attribute -> cluster_id, choose a representative for each cluster.
    Uses original attributes list to count frequency.
    """
    cluster_to_attrs = defaultdict(list)
    for attr in original_attrs:
        cid = attr_to_cluster.get(attr, -1)
        cluster_to_attrs[cid].append(attr)

    representatives = {}
    for cid, attrs in cluster_to_attrs.items():
        counts = Counter(attrs)
        rep = sorted(counts.items(), key=lambda x: (-x[1], len(x[0])))[0][0]
        representatives[cid] = rep
    return representatives