"""
Grounding Validator and Confidence Assessment.
Matches PRD §20 & hhDesign §25, §26.
"""

import re
import time
from enum import Enum
from typing import List, Dict, Any, Tuple, Optional

class GroundingStatus(str, Enum):
    GROUNDED = "grounded" # Green: #1F7335
    LOW_EVIDENCE = "low_evidence" # Yellow: #C98A20
    UNSUPPORTED = "unsupported" # Red: #C93636

class GroundingValidator:
    FALLBACK_MESSAGE = "I couldn't find enough relevant information in the knowledge base to answer that."

    def __init__(self, score_threshold: float = 0.32, min_evidence_overlap: float = 0.20):
        self.score_threshold = score_threshold
        self.min_evidence_overlap = min_evidence_overlap

    def validate_retrieval_confidence(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        query: Optional[str] = None
    ) -> Tuple[bool, float, GroundingStatus]:
        """
        Evaluates top similarity score and query-evidence alignment against threshold.
        """
        if not retrieved_chunks:
            return False, 0.0, GroundingStatus.UNSUPPORTED

        top_score = retrieved_chunks[0].get("score", 0.0)

        # Check if query keywords have sufficient presence in the top retrieved evidence
        if query and retrieved_chunks:
            q_clean = query.lower().replace("?", " ").replace("!", " ").replace(".", " ")
            stop_words = {"what", "which", "where", "when", "who", "whom", "whose", "why", "how", "is", "are", "was", "were", "the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "does", "did", "can", "could", "should", "would", "tell", "me", "about"}
            q_words = [w.strip() for w in q_clean.split() if len(w.strip()) > 2 and w.strip() not in stop_words]
            top_chunk_text = (retrieved_chunks[0].get("payload", {}).get("text", "") + " " + retrieved_chunks[0].get("payload", {}).get("title", "")).lower()
            if q_words:
                matched_top = sum(1 for w in q_words if w in top_chunk_text)
                if matched_top / len(q_words) < 0.25:
                    return False, top_score, GroundingStatus.UNSUPPORTED

        if top_score < self.score_threshold:
            return False, top_score, GroundingStatus.LOW_EVIDENCE

        return True, top_score, GroundingStatus.GROUNDED

    def validate_answer_grounding(
        self,
        answer: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluates token overlap and factual alignment between answer and evidence chunks.
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

        # Check answer words
        answer_words = re.findall(r"\w+", answer.lower())
        meaningful_answer_words = [w for w in answer_words if len(w) > 3]

        if not meaningful_answer_words:
            overlap_ratio = 1.0
        else:
            matched = sum(1 for w in meaningful_answer_words if w in context_words)
            overlap_ratio = matched / len(meaningful_answer_words)

        top_score = retrieved_chunks[0].get("score", 0.0) if retrieved_chunks else 0.0

        if top_score >= 0.35 and overlap_ratio >= self.min_evidence_overlap:
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
