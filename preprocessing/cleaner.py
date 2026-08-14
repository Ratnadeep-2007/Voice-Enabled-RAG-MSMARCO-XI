"""
Text cleaner and normalizer for multilingual retrieval and Indic NLP.
Preserves critical technical terms, punctuation, and diacritics.
"""

import re
import unicodedata
from typing import Optional

class TextCleaner:
    @staticmethod
    def normalize_text(text: str) -> str:
        if not text:
            return ""
        # Apply NFKC Unicode normalization (vital for Indic scripts and punctuation)
        text = unicodedata.normalize("NFKC", text)
        # Remove zero-width spaces and non-printable control characters except standard whitespace
        text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)
        # Normalize excessive whitespace, tabs, and newlines
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        return text.strip()

    @staticmethod
    def clean_for_query(query: str) -> str:
        if not query:
            return ""
        clean = TextCleaner.normalize_text(query)
        # Remove trailing repeated punctuation
        clean = re.sub(r"[?!.,]{2,}$", lambda m: m.group(0)[0], clean)
        return clean.strip()
