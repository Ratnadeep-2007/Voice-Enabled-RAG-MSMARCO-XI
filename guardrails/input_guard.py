"""
Lightweight deterministic input guardrails (PRD §20).
Enforces query length constraints and filters prompt injection attacks without LLM latency overhead.
"""

import re
import time
from typing import Dict, Any, Tuple

class InputGuard:
    INJECTION_PATTERNS = [
        r"ignore (all )?previous instructions",
        r"disregard (all )?above",
        r"system prompt",
        r"you are now (an evil|dan)",
        r"bypass security",
        r"drop table",
        r"<script.*?>",
    ]

    def __init__(
        self,
        min_length: int = 3,
        max_length: int = 500,
        check_injection: bool = True
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.check_injection = check_injection

    def validate_query(self, query: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validates user query.
        Returns (is_valid, error_or_cleaned_query, telemetry).
        """
        t0 = time.perf_counter()
        if not query or not query.strip():
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return False, "Please provide a question.", {"latency_ms": round(latency_ms, 2), "check": "empty"}

        clean = query.strip()

        if len(clean) < self.min_length:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return False, "Query is too short. Please provide a full question.", {"latency_ms": round(latency_ms, 2), "check": "min_length"}

        if len(clean) > self.max_length:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return False, f"Query exceeds maximum character limit of {self.max_length}.", {"latency_ms": round(latency_ms, 2), "check": "max_length"}

        if self.check_injection:
            for pattern in self.INJECTION_PATTERNS:
                if re.search(pattern, clean, re.IGNORECASE):
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                    return False, "Query contained restricted instructions.", {"latency_ms": round(latency_ms, 2), "check": "injection"}

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return True, clean, {"latency_ms": round(latency_ms, 2), "check": "passed"}
