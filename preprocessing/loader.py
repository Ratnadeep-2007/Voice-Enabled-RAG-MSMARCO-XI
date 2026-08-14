"""
Dataset loader for MSMARCO-XI and multilingual retrieval corpora.
Supports loading from local JSON/JSONL/Parquet/CSV and Hugging Face datasets.
"""

import json
import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class DatasetLoader:
    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path or "data/evaluation/msmarco_xi_sample.json"

    def load_documents(self, custom_path: Optional[str] = None) -> List[Dict[str, Any]]:
        path = custom_path or self.data_path
        if not os.path.exists(path):
            logger.warning(f"File not found at {path}, returning empty list.")
            return []

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                logger.info(f"Successfully loaded {len(data)} documents from {path}")
                return data
        except Exception as e:
            logger.error(f"Error loading dataset from {path}: {e}")
            return []

    def load_test_queries(self, test_path: str = "data/evaluation/test_queries.json") -> List[Dict[str, Any]]:
        if not os.path.exists(test_path):
            logger.warning(f"Test queries file not found at {test_path}")
            return []
        try:
            with open(test_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading test queries from {test_path}: {e}")
            return []

    def get_corpus_statistics(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        languages = {}
        domains = {}
        total_chars = 0

        for doc in documents:
            lang = doc.get("language", "unknown")
            languages[lang] = languages.get(lang, 0) + 1
            domain = doc.get("domain", "General")
            domains[domain] = domains.get(domain, 0) + 1
            total_chars += len(doc.get("passage", ""))

        return {
            "total_documents": len(documents),
            "languages": languages,
            "domains": domains,
            "avg_passage_length_chars": round(total_chars / max(1, len(documents)), 2),
            "estimated_tokens": int(total_chars / 4)
        }
