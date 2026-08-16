"""
AI4Bharat MSMARCO-XI Dataset Processor.
Downloads, extracts, cleans, and standardizes MSMARCO-XI parquet files from Hugging Face:
https://huggingface.co/datasets/ai4bharat/MSMARCO-XI
"""

import os
import io
import json
import logging
from typing import List, Dict, Any, Optional
import httpx
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# Hugging Face Parquet URL mapping for MSMARCO-XI
HF_DATASET_BASE = "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main"

LANGUAGE_CONFIGS = {
    "hi": {"name": "Hindi", "val_file": "validation/hinval.parquet", "train_file": "train/hintrain.parquet"},
    "ta": {"name": "Tamil", "val_file": "validation/tamval.parquet", "train_file": "train/tamtrain.parquet"},
    "te": {"name": "Telugu", "val_file": "validation/telval.parquet", "train_file": "train/teltrain.parquet"},
    "bn": {"name": "Bengali", "val_file": "validation/benval.parquet", "train_file": "train/bentrain.parquet"},
    "mr": {"name": "Marathi", "val_file": "validation/marval.parquet", "train_file": "train/martrain.parquet"},
    "gu": {"name": "Gujarati", "val_file": "validation/gujval.parquet", "train_file": "train/gujtrain.parquet"},
    "kn": {"name": "Kannada", "val_file": "validation/kanval.parquet", "train_file": "train/kantrain.parquet"},
    "ml": {"name": "Malayalam", "val_file": "validation/malval.parquet", "train_file": "train/maltrain.parquet"},
    "pa": {"name": "Punjabi", "val_file": "validation/panval.parquet", "train_file": "train/pantrain.parquet"},
    "or": {"name": "Odia", "val_file": "validation/orival.parquet", "train_file": "train/oritrain.parquet"},
    "as": {"name": "Assamese", "val_file": "validation/asmval.parquet", "train_file": "train/asmtrain.parquet"},
    "ur": {"name": "Urdu", "val_file": "validation/urdval.parquet", "train_file": "train/urdtrain.parquet"}
}

class MSMARCOXIProcessor:
    def __init__(self, output_path: str = "data/msmarco_xi_dataset.json"):
        self.output_path = output_path

    def download_and_process(
        self,
        languages: Optional[List[str]] = None,
        samples_per_lang: int = 100,
        split: str = "validation"
    ) -> List[Dict[str, Any]]:
        """
        Downloads parquet files from HF repository and extracts normalized documents.
        """
        selected_langs = languages or list(LANGUAGE_CONFIGS.keys())
        all_documents: List[Dict[str, Any]] = []

        logger.info(f"Starting MSMARCO-XI dataset processing for languages: {selected_langs} (samples_per_lang={samples_per_lang}, split={split})")

        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            for lang_code in selected_langs:
                config = LANGUAGE_CONFIGS.get(lang_code)
                if not config:
                    logger.warning(f"Unknown language code '{lang_code}', skipping.")
                    continue

                rel_file = config["val_file"] if split == "validation" else config["train_file"]
                url = f"{HF_DATASET_BASE}/{rel_file}"

                logger.info(f"Fetching {config['name']} ({lang_code}) from {url}...")
                try:
                    resp = client.get(url)
                    if resp.status_code != 200:
                        logger.error(f"Failed to download {url}: HTTP {resp.status_code}")
                        continue

                    table = pq.read_table(io.BytesIO(resp.content))
                    col_names = table.column_names
                    num_rows = min(samples_per_lang, table.num_rows)

                    logger.info(f"Loaded {table.num_rows} rows for {lang_code}. Extracting {num_rows} passages...")

                    for i in range(num_rows):
                        row = {c: table.column(c)[i].as_py() for c in col_names}
                        
                        # Extract passage, query, doc_id based on parquet schema
                        passage_text = (
                            row.get("passage") 
                            or row.get("text") 
                            or row.get("translated_passage") 
                            or row.get("body")
                            or ""
                        )
                        query_text = (
                            row.get("query") 
                            or row.get("question") 
                            or row.get("translated_query") 
                            or ""
                        )
                        doc_id = str(row.get("doc_id") or row.get("id") or f"msmarco_{lang_code}_{i:05d}")
                        
                        if not passage_text or len(str(passage_text).strip()) < 20:
                            continue

                        # Extract title from first sentence
                        passage_str = str(passage_text).strip()
                        title = passage_str[:60] + "..." if len(passage_str) > 60 else passage_str

                        doc = {
                            "doc_id": f"{lang_code}_{doc_id}",
                            "language": lang_code,
                            "title": title,
                            "passage": passage_str,
                            "source": f"ai4bharat/MSMARCO-XI/{config['name']}",
                            "domain": "Indic Multilingual Knowledge",
                            "query_id": f"q_{lang_code}_{i:05d}",
                            "related_query": str(query_text).strip() if query_text else ""
                        }
                        all_documents.append(doc)

                except Exception as e:
                    logger.error(f"Error processing {lang_code} parquet: {e}")

        logger.info(f"Extracted {len(all_documents)} total standardized multilingual documents from MSMARCO-XI.")
        
        # Save to disk
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(all_documents, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved processed dataset to {self.output_path}")
        return all_documents
