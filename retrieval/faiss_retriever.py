"""
FAISS IVF-PQ + LMDB Zero-Copy Retriever.
Performs sub-2ms vector search and memory-mapped payload retrieval.
"""

import os
import sys
import time
import logging
import numpy as np
from typing import List, Dict, Any, Optional

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import faiss

from embeddings.model import EmbeddingEngine, get_embedding_engine
from indexing.lmdb_store import LMDBDocumentStore, get_lmdb_store

logger = logging.getLogger(__name__)

class FAISSRetriever:
    def __init__(
        self,
        index_path: str = "data/faiss_index/msmarco_xi_ivfpq.faiss",
        lmdb_path: str = "data/lmdb_store",
        nprobe: int = 8
    ):
        self.index_path = index_path
        self.lmdb_path = lmdb_path
        self.nprobe = nprobe
        self.index: Optional[faiss.Index] = None
        self.lmdb_store = get_lmdb_store(db_path=lmdb_path)
        self.embed_engine = get_embedding_engine()
        self._load_index()

    def _load_index(self):
        """Loads FAISS index into RAM for instant queries."""
        if os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
                if hasattr(self.index, "nprobe"):
                    self.index.nprobe = self.nprobe
                logger.info(f"FAISS index loaded into RAM from {self.index_path} (Total vectors: {self.index.ntotal})")
            except Exception as e:
                logger.error(f"Error loading FAISS index: {e}")
                self.index = None
        else:
            logger.warning(f"FAISS index not found at {self.index_path}. Build it first with FAISSIVFPQIndexer.")
            self.index = None

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        score_threshold: float = 0.30
    ) -> Dict[str, Any]:
        """
        Executes sub-2ms vector search in FAISS and zero-copy lookup in LMDB.
        """
        if self.index is None:
            self._load_index()
            if self.index is None:
                return {
                    "chunks": [],
                    "search_latency_ms": 0.0,
                    "lmdb_latency_ms": 0.0,
                    "total_retrieval_ms": 0.0,
                    "status": "error",
                    "message": "FAISS index not loaded"
                }

        t_search_start = time.perf_counter()
        
        # Format query vector
        q_vec = np.array([query_vector], dtype=np.float32)
        faiss.normalize_L2(q_vec)

        # Search FAISS in RAM (<1ms)
        scores, indices = self.index.search(q_vec, min(top_k, self.index.ntotal))
        search_latency_ms = (time.perf_counter() - t_search_start) * 1000.0

        # Hydrate text & metadata from LMDB (<0.2ms zero-copy)
        t_lmdb_start = time.perf_counter()
        raw_ids = [int(idx) for idx in indices[0] if idx >= 0]
        raw_scores = [float(score) for idx, score in zip(indices[0], scores[0]) if idx >= 0]
        
        docs = self.lmdb_store.get_batch(raw_ids)
        lmdb_latency_ms = (time.perf_counter() - t_lmdb_start) * 1000.0

        # Construct structured chunks
        retrieved_chunks = []
        for doc, score in zip(docs, raw_scores):
            if doc and score >= score_threshold:
                retrieved_chunks.append({
                    "chunk_id": doc.get("chunk_id", ""),
                    "doc_id": doc.get("doc_id", ""),
                    "text": doc.get("text", ""),
                    "score": round(score, 4),
                    "language": doc.get("language", "en"),
                    "title": doc.get("title", ""),
                    "source": doc.get("source", "MSMARCO-XI")
                })

        total_retrieval_ms = (time.perf_counter() - t_search_start) * 1000.0

        return {
            "chunks": retrieved_chunks,
            "search_latency_ms": round(search_latency_ms, 2),
            "lmdb_latency_ms": round(lmdb_latency_ms, 2),
            "total_retrieval_ms": round(total_retrieval_ms, 2),
            "index_type": "FAISS_IVF_PQ",
            "status": "success"
        }

_faiss_retriever_instance: Optional[FAISSRetriever] = None

def get_faiss_retriever() -> FAISSRetriever:
    global _faiss_retriever_instance
    if _faiss_retriever_instance is None:
        _faiss_retriever_instance = FAISSRetriever()
    return _faiss_retriever_instance
