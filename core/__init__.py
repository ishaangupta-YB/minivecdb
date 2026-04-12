"""
MiniVecDB Core Module
Contains: distance metrics, embeddings, vector store, collections
"""

from .distance_metrics import (
    cosine_similarity,
    euclidean_distance,
    dot_product_similarity,
    batch_cosine_similarity,
    batch_euclidean_distance,
    batch_dot_product,
    normalise_vector,
    get_metric,
    list_metrics,
)

__all__ = [
    "cosine_similarity", "euclidean_distance", "dot_product_similarity",
    "batch_cosine_similarity", "batch_euclidean_distance", "batch_dot_product",
    "normalise_vector", "get_metric", "list_metrics",
]

# Export embedding engine symbols only when they are available.
try:
    from . import embeddings as _embeddings
except Exception:
    _embeddings = None

if _embeddings is not None:
    if hasattr(_embeddings, "EmbeddingEngine"):
        EmbeddingEngine = _embeddings.EmbeddingEngine
        __all__.append("EmbeddingEngine")
    if hasattr(_embeddings, "SimpleEmbeddingEngine"):
        SimpleEmbeddingEngine = _embeddings.SimpleEmbeddingEngine
        __all__.append("SimpleEmbeddingEngine")
    if hasattr(_embeddings, "create_embedding_engine"):
        create_embedding_engine = _embeddings.create_embedding_engine
        __all__.append("create_embedding_engine")
