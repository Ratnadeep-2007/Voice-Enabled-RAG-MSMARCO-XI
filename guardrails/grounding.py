"""
Grounding Validator and Confidence Assessment with Semantic Overlap Support.
Matches PRD §20 & hhDesign §25, §26.
"""

import re
import time
from enum import Enum
from typing import List, Dict, Any, Tuple, Optional

class GroundingStatus(str, Enum):
    GROUNDED = "grounded"        # Green: #1F7335
    LOW_EVIDENCE = "low_evidence"# Yellow: #C98A20
    UNSUPPORTED = "unsupported"  # Red: #C93636

class GroundingValidator:
    FALLBACK_MESSAGE = "I couldn't find enough relevant information in the knowledge base to answer that."

    STOPWORDS = {
        "about", "above", "after", "again", "against", "all", "and", "any", "are", "because",
        "been", "before", "being", "below", "between", "both", "but", "by", "could", "did",
        "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had",
        "has", "have", "having", "here", "how", "into", "itself", "just", "more", "most",
        "other", "our", "ours", "out", "over", "own", "same", "should", "some", "such",
        "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there",
        "these", "they", "this", "those", "through", "under", "until", "very", "was", "were",
        "what", "when", "where", "which", "while", "who", "whom", "why", "with", "would",
        "also", "based", "helps", "means", "occur", "occurs", "result", "results", "often"
    }

    def __init__(self, score_threshold: float = 0.28, min_evidence_overlap: float = 0.15):
        self.score_threshold = score_threshold
        self.min_evidence_overlap = min_evidence_overlap

    def validate_retrieval_confidence(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        query: Optional[str] = None
    ) -> Tuple[bool, float, GroundingStatus]:
        """
        Evaluates top semantic similarity score against confidence threshold.
        Calibrated for continuous dense vector similarity (MiniLM-L12/BGE-M3) supporting synonyms and paraphrases.
        """
        if not retrieved_chunks:
            return False, 0.0, GroundingStatus.UNSUPPORTED

        top_score = float(retrieved_chunks[0].get("score", 0.0))

        if top_score < self.score_threshold:
            return False, top_score, GroundingStatus.UNSUPPORTED

        if top_score < 0.38:
            return True, top_score, GroundingStatus.LOW_EVIDENCE

        return True, top_score, GroundingStatus.GROUNDED

    def validate_answer_grounding(
        self,
        answer: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluates semantic concept overlap and factual alignment between answer and evidence chunks.
        Supports synthesized and reasoned responses using word roots and semantic content words.
        """
        t0 = time.perf_counter()

        if self.FALLBACK_MESSAGE in answer or not retrieved_chunks:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "status": GroundingStatus.UNSUPPORTED.value,
                "confidence_score": 0.0,
                "supporting_passages_count": 0,
                "latency_ms": round(latency_ms, 2),
                "is_fallback": True
            }

        # Combine text from all retrieved chunks
        context_text = " ".join([
            c.get("payload", {}).get("text", "")
            for c in retrieved_chunks
        ]).lower()
        
        context_words = set(re.findall(r"\w+", context_text))
        context_stems = {w[:4] for w in context_words if len(w) >= 4}

        # Check answer words (excluding generic stopwords)
        raw_answer_words = re.findall(r"\w+", answer.lower())
        content_answer_words = [
            w for w in raw_answer_words 
            if len(w) > 2 and w not in self.STOPWORDS
        ]

        if not content_answer_words:
            overlap_ratio = 1.0
        else:
            matched = 0
            for w in content_answer_words:
                if w in context_words or (len(w) >= 4 and w[:4] in context_stems):
                    matched += 1
            overlap_ratio = matched / len(content_answer_words)

        top_score = float(retrieved_chunks[0].get("score", 0.0)) if retrieved_chunks else 0.0

        if top_score >= 0.30 and overlap_ratio >= self.min_evidence_overlap:
            status = GroundingStatus.GROUNDED
            confidence = "HIGH"
        elif top_score >= self.score_threshold:
            status = GroundingStatus.LOW_EVIDENCE
            confidence = "MEDIUM"
        else:
            status = GroundingStatus.UNSUPPORTED
            confidence = "LOW"

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "status": status.value,
            "confidence_score": round(top_score, 3),
            "confidence_level": confidence,
            "overlap_ratio": round(overlap_ratio, 2),
            "supporting_passages_count": len(retrieved_chunks),
            "latency_ms": round(latency_ms, 2),
            "is_fallback": False
        }
