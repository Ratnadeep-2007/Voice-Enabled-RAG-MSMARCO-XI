"""
Ranking and Score Fusion utilities (RRF for Dense + Sparse Hybrid search).
"""

from typing import List, Dict, Any

class Ranker:
    @staticmethod
    def top_k(results: List[Dict[str, Any]], k: int = 5) -> List[Dict[str, Any]]:
        """Sorts by score descending and returns top k results."""
        sorted_results = sorted(results, key=lambda x: x.get("score", 0.0), reverse=True)
        return sorted_results[:k]

    @staticmethod
    def reciprocal_rank_fusion(
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        k: int = 60,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        RRF: Score = SUM(1 / (k + rank_i))
        Combines rankings from dense vector retrieval and sparse lexical search.
        """
        scores: Dict[str, float] = {}
        payload_map: Dict[str, Dict[str, Any]] = {}

        # Dense ranks
        for rank, item in enumerate(dense_results):
            cid = item.get("payload", {}).get("chunk_id", str(item.get("id")))
            scores[cid] = scores.get(cid, 0.0) + (1.0 / (k + rank + 1))
            payload_map[cid] = item

        # Sparse ranks
        for rank, item in enumerate(sparse_results):
            cid = item.get("payload", {}).get("chunk_id", str(item.get("id")))
            scores[cid] = scores.get(cid, 0.0) + (1.0 / (k + rank + 1))
            if cid not in payload_map:
                payload_map[cid] = item

        # Re-rank combined
        fused = []
        for cid, rrf_score in scores.items():
            base_item = payload_map[cid]
            fused_item = dict(base_item)
            fused_item["rrf_score"] = round(rrf_score, 4)
            fused_item["score"] = round(rrf_score, 4)
            fused.append(fused_item)

        fused.sort(key=lambda x: x["score"], reverse=True)
        return fused[:top_k]
