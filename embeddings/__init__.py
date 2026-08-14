"""
Embeddings module for VoiceRAG:
Fast multilingual local embedding models, ONNX runtime execution, and batch embedders.
"""

from .model import EmbeddingEngine, get_embedding_engine
from .embed import BatchEmbedder

__all__ = ["EmbeddingEngine", "get_embedding_engine", "BatchEmbedder"]
