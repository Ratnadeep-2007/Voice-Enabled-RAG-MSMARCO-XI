"""
Dense Vector Retriever & Hybrid Retriever.
Performs ANN search via Qdrant with HNSW search parameters (ef_search).
Matches PRD §13, §14, §15, and hhDesign §19, §20.
"""

import time
import math
import logging
from typing import List, Dict, Any, Optional
from qdrant_client.http import models

from indexing.qdrant_client import QdrantManager, get_qdrant_manager
from embeddings.model import EmbeddingEngine, get_embedding_engine
from .filters import RelevanceFilter
from .ranking import Ranker

logger = logging.getLogger(__name__)

class DenseRetriever:
    def __init__(
        self,
        qdrant_mgr: Optional[QdrantManager] = None,
        embedding_engine: Optional[EmbeddingEngine] = None,
        collection_name: str = "msmarco_xi_dense"
    ):
        self.qdrant_mgr = qdrant_mgr or get_qdrant_manager()
        self.embedding_engine = embedding_engine or get_embedding_engine()
        self.collection_name = collection_name

    def retrieve(
        self,
        query_vector: List[float],
        top_k: int = 5,
        ef_search: int = 32,
        score_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Executes HNSW search on Qdrant and records latency in milliseconds.
        """
        t0 = time.perf_counter()
        client = self.qdrant_mgr.get_client()

        search_params = models.SearchParams(
            hnsw_ef=ef_search,
            exact=False
        )

        try:
            # Execute search using query_points (qdrant-client >=1.10) or search (<1.10)
            if hasattr(client, "query_points"):
                response = client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=top_k,
                    search_params=search_params,
                    score_threshold=score_threshold
                )
                hits = getattr(response, "points", response)
            elif hasattr(client, "search"):
                hits = client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=top_k,
                    search_params=search_params,
                    score_threshold=score_threshold
                )
            else:
                hits = []

            qdrant_time_ms = (time.perf_counter() - t0) * 1000.0

            results = []
            for hit in hits:
                results.append({
                    "id": getattr(hit, "id", None),
                    "score": round(float(getattr(hit, "score", 0.0)), 4),
                    "payload": getattr(hit, "payload", {}) or {}
                })

            return {
                "results": results,
                "latency_ms": round(qdrant_time_ms, 2),
                "top_k": top_k,
                "ef_search": ef_search,
                "count": len(results)
            }
        except Exception as e:
            qdrant_time_ms = (time.perf_counter() - t0) * 1000.0
            logger.error(f"Error during Qdrant search: {e}")
            return {
                "results": [],
                "latency_ms": round(qdrant_time_ms, 2),
                "error": str(e),
                "count": 0
            }


class BM25Retriever:
    """
    Lightweight BM25 implementation for Hybrid Retrieval benchmarking (PRD §25 Exp 6).
    """
    def __init__(self, corpus: Optional[List[Dict[str, Any]]] = None):
        self.corpus = corpus or []
        self.doc_len = []
        self.avg_doc_len = 0.0
        self.doc_freqs = []
        self.idf = {}
        self.k1 = 1.5
        self.b = 0.75
        if self.corpus:
            self._fit()

    def fit(self, corpus: List[Dict[str, Any]]):
        self.corpus = corpus
        self._fit()

    def _fit(self):
        total_len = 0
        df = {}
        self.doc_len = []
        self.doc_freqs = []

        for doc in self.corpus:
            text = doc.get("text", "")
            tokens = text.lower().split()
            self.doc_len.append(len(tokens))
            total_len += len(tokens)
            freqs = {}
            for t in tokens:
                freqs[t] = freqs.get(t, 0) + 1
            self.doc_freqs.append(freqs)
            for t in set(tokens):
                df[t] = df.get(t, 0) + 1

        N = max(1, len(self.corpus))
        self.avg_doc_len = total_len / N
        self.idf = {
            t: math.log(1 + (N - freq + 0.5) / (freq + 0.5))
            for t, freq in df.items()
        }

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        tokens = query.lower().split()
        scores = []
        for i, freqs in enumerate(self.doc_freqs):
            score = 0.0
            L = self.doc_len[i]
            for t in tokens:
                if t in freqs:
                    tf = freqs[t]
                    idf = self.idf.get(t, 0.1)
                    denom = tf + self.k1 * (1 - self.b + self.b * (L / (self.avg_doc_len or 1.0)))
                    score += idf * (tf * (self.k1 + 1)) / (denom or 1.0)
            if score > 0:
                scores.append({
                    "id": i + 1,
                    "score": round(score, 4),
                    "payload": self.corpus[i]
                })

        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores[:top_k]


class HybridRetriever:
    """
    Combines Dense Vector Retrieval + BM25 Sparse Search using Reciprocal Rank Fusion.
    """
    def __init__(self, dense_retriever: DenseRetriever, bm25_retriever: Optional[BM25Retriever] = None):
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever or BM25Retriever()

    def retrieve(
        self,
        query: str,
        query_vector: List[float],
        top_k: int = 5,
        ef_search: int = 32,
        rrf_k: int = 60
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        # 1. Dense retrieval
        dense_res = self.dense_retriever.retrieve(
            query_vector=query_vector,
            top_k=top_k * 2,
            ef_search=ef_search
        )
        dense_hits = dense_res.get("results", [])

        # 2. Sparse BM25 retrieval
        sparse_hits = self.bm25_retriever.search(query, top_k=top_k * 2)

        # 3. Reciprocal Rank Fusion
        fused = Ranker.reciprocal_rank_fusion(
            dense_results=dense_hits,
            sparse_results=sparse_hits,
            k=rrf_k,
            top_k=top_k
        )
        total_time_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "results": fused,
            "latency_ms": round(total_time_ms, 2),
            "dense_count": len(dense_hits),
            "sparse_count": len(sparse_hits),
            "fused_count": len(fused)
        }
