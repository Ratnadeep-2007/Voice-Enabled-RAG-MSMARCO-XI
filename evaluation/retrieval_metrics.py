"""
Retrieval Metrics Evaluation (PRD §24.2).
Computes Recall@K, Mean Reciprocal Rank (MRR), and nDCG@K against ground truth labels.
"""

import math
from typing import List, Dict, Any, Set

class RetrievalEvaluator:
    @staticmethod
    def calculate_recall_at_k(
        retrieved_doc_ids: List[str],
        relevant_doc_ids: Set[str],
        k: int = 5
    ) -> float:
        if not relevant_doc_ids:
            return 1.0 # Vacuously true for out-of-domain queries
        retrieved_k = set(retrieved_doc_ids[:k])
        hits = len(retrieved_k.intersection(relevant_doc_ids))
        return hits / len(relevant_doc_ids)

    @staticmethod
    def calculate_mrr(
        retrieved_doc_ids: List[str],
        relevant_doc_ids: Set[str]
    ) -> float:
        if not relevant_doc_ids:
            return 1.0
        for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
            if doc_id in relevant_doc_ids:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def calculate_ndcg_at_k(
        retrieved_doc_ids: List[str],
        relevant_doc_ids: Set[str],
        k: int = 5
    ) -> float:
        if not relevant_doc_ids:
            return 1.0

        dcg = 0.0
        for i, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
            rel = 1.0 if doc_id in relevant_doc_ids else 0.0
            dcg += (2.0**rel - 1.0) / math.log2(i + 1)

        # Ideal DCG (all relevant docs ranked at top)
        idcg = 0.0
        for i in range(1, min(len(relevant_doc_ids), k) + 1):
            idcg += (2.0**1.0 - 1.0) / math.log2(i + 1)

        return (dcg / idcg) if idcg > 0 else 0.0

    @staticmethod
    def evaluate_query_batch(
        predictions: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        predictions: List of dicts with 'retrieved_doc_ids' and 'relevant_doc_ids'
        """
        if not predictions:
            return {"recall@5": 0.0, "recall@10": 0.0, "mrr": 0.0, "ndcg@5": 0.0}

        recalls_5 = []
        recalls_10 = []
        mrrs = []
        ndcgs = []

        for p in predictions:
            retrieved = p.get("retrieved_doc_ids", [])
            relevant = set(p.get("relevant_doc_ids", []))
            recalls_5.append(RetrievalEvaluator.calculate_recall_at_k(retrieved, relevant, k=5))
            recalls_10.append(RetrievalEvaluator.calculate_recall_at_k(retrieved, relevant, k=10))
            mrrs.append(RetrievalEvaluator.calculate_mrr(retrieved, relevant))
            ndcgs.append(RetrievalEvaluator.calculate_ndcg_at_k(retrieved, relevant, k=5))

        return {
            "recall@5": round(sum(recalls_5) / len(recalls_5) * 100, 2),
            "recall@10": round(sum(recalls_10) / len(recalls_10) * 100, 2),
            "mrr": round(sum(mrrs) / len(mrrs), 3),
            "ndcg@5": round(sum(ndcgs) / len(ndcgs), 3)
        }
