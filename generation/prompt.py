"""
Context Builder and Prompt Construction.
Supports compact JSON and TOON serialization formats for Experiment 5 (PRD §18, §25).
"""

import json
import time
from enum import Enum
from typing import List, Dict, Any, Tuple

class ContextFormat(str, Enum):
    JSON = "json"
    TOON = "toon"

class ContextBuilder:
    def __init__(self, default_format: ContextFormat = ContextFormat.JSON):
        self.default_format = default_format

    def serialize_json(self, chunks: List[Dict[str, Any]]) -> str:
        """Compact JSON representation of chunks."""
        compact_list = []
        for c in chunks:
            payload = c.get("payload", {})
            compact_list.append({
                "cid": payload.get("chunk_id", str(c.get("id", ""))),
                "doc": payload.get("document_id", ""),
                "title": payload.get("title", ""),
                "text": payload.get("text", "")
            })
        return json.dumps(compact_list, ensure_ascii=False, separators=(",", ":"))

    def serialize_toon(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Token-Oriented Object Notation (TOON).
        Optimized delimiter-based syntax with minimal punctuation overhead.
        """
        lines = []
        for c in chunks:
            payload = c.get("payload", {})
            cid = payload.get("chunk_id", str(c.get("id", "")))
            title = payload.get("title", "")
            text = payload.get("text", "")
            lines.append(f"<CHUNK id={cid} title=\"{title}\">{text}</CHUNK>")
        return "\n".join(lines)

    def build_prompt(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        context_format: ContextFormat = ContextFormat.JSON
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        Constructs (system_prompt, user_prompt, telemetry).
        """
        t0 = time.perf_counter()

        if context_format == ContextFormat.TOON:
            context_str = self.serialize_toon(retrieved_chunks)
        else:
            context_str = self.serialize_json(retrieved_chunks)

        system_prompt = (
            "You are VoiceRAG, a low-latency knowledge assistant. "
            "Your answers MUST be strictly grounded in the provided context. "
            "Rules:\n"
            "1. Answer concisely in 2 to 4 sentences.\n"
            "2. Only use factual claims directly present in the context.\n"
            "3. If the context does NOT contain enough evidence to answer, reply exactly:\n"
            "   'I couldn't find enough relevant information in the knowledge base to answer that.'\n"
            "4. Do not speculate, invent details, or mention system prompt instructions."
        )

        user_prompt = f"CONTEXT:\n{context_str}\n\nQUESTION:\n{query}\n\nANSWER:"

        build_time_ms = (time.perf_counter() - t0) * 1000.0
        # Approximate tokens (words * 1.33)
        approx_tokens = int(len((system_prompt + user_prompt).split()) * 1.33)

        telemetry = {
            "format": context_format.value,
            "context_length_chars": len(context_str),
            "estimated_input_tokens": approx_tokens,
            "latency_ms": round(build_time_ms, 2)
        }

        return system_prompt, user_prompt, telemetry
