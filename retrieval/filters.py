"""
Relevance filtering and threshold validation for retrieved chunks.
PRD §20 & §28: prevents hallucination when retrieval evidence is insufficient.
"""

from typing import List, Dict, Any

class RelevanceFilter:
    @staticmethod
    def filter_by_score(
        results: List[Dict[str, Any]],
        score_threshold: float = 0.50
    ) -> List[Dict[str, Any]]:
        """Filters retrieved results below the similarity threshold."""
        return [r for r in results if r.get("score", 0.0) >= score_threshold]

    @staticmethod
    def filter_by_language(
        results: List[Dict[str, Any]],
        target_language: str
    ) -> List[Dict[str, Any]]:
        """Filters retrieved results to match query language if requested."""
        if not target_language or target_language == "unknown":
            return results
        return [r for r in results if r.get("payload", {}).get("language") == target_language]

    @staticmethod
    def deduplicate(
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Deduplicates results based on chunk text content."""
        seen_texts = set()
        unique = []
        for r in results:
            text = r.get("payload", {}).get("text", "")
            if text not in seen_texts:
                seen_texts.add(text)
                unique.append(r)
        return unique
