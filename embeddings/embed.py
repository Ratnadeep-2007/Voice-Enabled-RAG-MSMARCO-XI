"""
Batch embedding pipeline for offline indexing.
"""

import time
import logging
from typing import List, Dict, Any, Tuple
from .model import EmbeddingEngine, get_embedding_engine

logger = logging.getLogger(__name__)

class BatchEmbedder:
    def __init__(self, engine: EmbeddingEngine = None):
        self.engine = engine or get_embedding_engine()

    def process_chunks(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: int = 32
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Embeds chunks in batches and attaches vector to each chunk dictionary.
        Returns (embedded_chunks, total_time_ms).
        """
        if not chunks:
            return [], 0.0

        texts_to_embed = [
            chunk.get("annotated_text", chunk.get("text", ""))
            for chunk in chunks
        ]

        t0 = time.perf_counter()
        vectors = self.engine.embed_batch(texts_to_embed, batch_size=batch_size)
        total_time_ms = (time.perf_counter() - t0) * 1000.0

        embedded_chunks = []
        for chunk, vector in zip(chunks, vectors):
            c = dict(chunk)
            c["vector"] = vector
            embedded_chunks.append(c)

        logger.info(f"Embedded {len(embedded_chunks)} chunks in {total_time_ms:.2f} ms ({total_time_ms/len(chunks):.2f} ms/chunk)")
        return embedded_chunks, total_time_ms
