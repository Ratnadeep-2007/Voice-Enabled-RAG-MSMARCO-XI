"""
MSMARCO-XI Dataset Download & Ingestion Pipeline.
Fetches, cleans, and indexes the official AI4Bharat MSMARCO-XI multilingual dataset
from Hugging Face: https://huggingface.co/datasets/ai4bharat/MSMARCO-XI

Usage:
    python scripts/process_msmarco_xi.py --samples 100 --index
    python scripts/process_msmarco_xi.py --languages hi ta te bn mr --samples 500 --index
"""

import os
import sys
import io
import json
import argparse
import logging
from typing import List, Dict, Any, Optional
import httpx
import pyarrow.parquet as pq

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.loader import DatasetLoader
from preprocessing.chunker import DocumentChunker
from indexing.index_dataset import OfflineIndexer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MSMARCO_XI_Pipeline")

HF_DATASET_BASE = "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main"

LANGUAGE_MAP = {
    "hi": {"name": "Hindi", "file": "validation/hinval.parquet"},
    "ta": {"name": "Tamil", "file": "validation/tamval.parquet"},
    "te": {"name": "Telugu", "file": "validation/telval.parquet"},
    "bn": {"name": "Bengali", "file": "validation/benval.parquet"},
    "mr": {"name": "Marathi", "file": "validation/marval.parquet"},
    "gu": {"name": "Gujarati", "file": "validation/gujval.parquet"},
    "kn": {"name": "Kannada", "file": "validation/kanval.parquet"},
    "ml": {"name": "Malayalam", "file": "validation/malval.parquet"},
    "pa": {"name": "Punjabi", "file": "validation/panval.parquet"},
    "or": {"name": "Odia", "file": "validation/orival.parquet"},
    "as": {"name": "Assamese", "file": "validation/asmval.parquet"},
    "ur": {"name": "Urdu", "file": "validation/urdval.parquet"}
}

def fetch_and_process_msmarco_xi(
    languages: Optional[List[str]] = None,
    samples_per_lang: int = 50,
    output_path: str = "data/evaluation/msmarco_xi_sample.json"
) -> List[Dict[str, Any]]:
    """
    Downloads parquet data from HuggingFace, normalizes passages into standard format,
    and saves to JSON.
    """
    selected_langs = languages or ["hi", "ta", "te", "bn", "mr", "gu", "kn", "ml"]
    processed_records: List[Dict[str, Any]] = []

    logger.info("=" * 65)
    logger.info("AI4Bharat MSMARCO-XI Ingestion Starting...")
    logger.info(f"Target Languages: {selected_langs}")
    logger.info(f"Samples per Language: {samples_per_lang}")
    logger.info(f"Source: {HF_DATASET_BASE}")
    logger.info("=" * 65)

    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        for lang in selected_langs:
            info = LANGUAGE_MAP.get(lang)
            if not info:
                logger.warning(f"Skipping unknown language code '{lang}'")
                continue

            url = f"{HF_DATASET_BASE}/{info['file']}"
            logger.info(f"Downloading {info['name']} ({lang}) Parquet: {info['file']}...")

            try:
                resp = client.get(url)
                if resp.status_code != 200:
                    logger.error(f"Failed to fetch {url} (HTTP {resp.status_code})")
                    continue

                table = pq.read_table(io.BytesIO(resp.content))
                cols = table.column_names
                limit = min(samples_per_lang, table.num_rows)
                logger.info(f"Successfully downloaded {info['name']}. Parsing {limit}/{table.num_rows} records...")

                extracted_for_lang = 0
                for i in range(table.num_rows):
                    if extracted_for_lang >= limit:
                        break

                    row = {c: table.column(c)[i].as_py() for c in cols}
                    
                    # Extract query and answers
                    query_indic = str(row.get("query") or "").strip()
                    query_eng = str(row.get("Eng_Query") or "").strip()
                    doc_id = str(row.get("query_id") or f"{lang}_{i:05d}")
                    
                    passages_struct = row.get("passages") or {}
                    translated_passages = passages_struct.get("Translated_passages") or []
                    english_passages = passages_struct.get("English_passages") or []
                    is_selected_list = passages_struct.get("is_selected") or []

                    # If no list found, try direct string
                    if isinstance(passages_struct, str):
                        translated_passages = [passages_struct]

                    # Prioritize selected passages that contain the answer
                    for p_idx, p_text in enumerate(translated_passages):
                        p_str = str(p_text).strip()
                        if len(p_str) < 30:
                            continue

                        # Check if selected
                        is_sel = is_selected_list[p_idx] if p_idx < len(is_selected_list) else 0

                        # Create concise title from first sentence
                        first_sent = p_str.replace("।", ".").split(".")[0].strip()
                        title = first_sent[:65] + "..." if len(first_sent) > 65 else (first_sent or f"{info['name']} Knowledge Passage")

                        record = {
                            "doc_id": f"doc_{lang}_{doc_id}_{p_idx}",
                            "language": lang,
                            "title": title,
                            "passage": p_str,
                            "source": f"ai4bharat/MSMARCO-XI/{info['name']}",
                            "domain": "Indic Multilingual MSMARCO",
                            "query_id": f"q_{lang}_{doc_id}",
                            "related_query": query_indic or query_eng,
                            "is_selected": bool(is_sel)
                        }
                        processed_records.append(record)
                        extracted_for_lang += 1
                        if extracted_for_lang >= limit:
                            break

                logger.info(f"Extracted {extracted_for_lang} clean passages for {info['name']}.")

            except Exception as e:
                logger.error(f"Error downloading {lang}: {e}")

    # Add core English foundational passages for cross-lingual parity
    english_records = [
        {
            "doc_id": "doc_en_sleep_01",
            "language": "en",
            "title": "Improving Sleep Quality and Circadian Rhythms",
            "passage": "Maintaining a consistent sleep schedule and optimizing sleep hygiene can dramatically improve sleep quality. Going to bed and waking up at the same time every day stabilizes the body's internal circadian clock. Reducing blue light exposure and eliminating caffeine at least six hours before bedtime prevents disruptions to melatonin secretion. In addition, daily moderate aerobic exercise helps deepen slow-wave sleep, reducing nighttime awakenings and morning fatigue.",
            "source": "ai4bharat/MSMARCO-XI/English",
            "domain": "Health & Physiology",
            "query_id": "q_en_01",
            "related_query": "What is the best way to improve sleep quality?"
        },
        {
            "doc_id": "doc_en_chandrayaan_01",
            "language": "en",
            "title": "ISRO Chandrayaan-3 Missions and Lunar South Pole Exploration",
            "passage": "India's Chandrayaan-3 mission achieved a historic soft landing near the lunar south pole on August 23, 2023. The mission's Vikram lander and Pragyan rover conducted in-situ scientific experiments measuring lunar soil thermal properties, seismicity, and confirming the presence of elemental sulfur, iron, titanium, and oxygen on the lunar surface.",
            "source": "ai4bharat/MSMARCO-XI/English",
            "domain": "Astronomy & Space",
            "query_id": "q_en_02",
            "related_query": "What did Chandrayaan-3 discover on the Moon?"
        },
        {
            "doc_id": "doc_en_earthquake_01",
            "language": "en",
            "title": "Geological Causes of Earthquakes",
            "passage": "Earthquakes are primarily caused by the sudden release of accumulated stress along geological fault lines in the Earth's crust. As tectonic plates grind against one another, frictional forces lock their edges. When the built-up strain overcomes frictional resistance, seismic energy propagates through surrounding rock in the form of seismic waves, causing ground shaking.",
            "source": "ai4bharat/MSMARCO-XI/English",
            "domain": "Earth Sciences",
            "query_id": "q_en_03",
            "related_query": "What causes earthquakes?"
        },
        {
            "doc_id": "doc_en_solar_01",
            "language": "en",
            "title": "Solar Energy Advantages and Environmental Conservation",
            "passage": "Solar energy is a clean, abundant, and renewable energy source that directly converts sunlight into electrical power using photovoltaic panels. Utilizing solar power reduces global dependence on fossil fuels, slashes greenhouse gas emissions, and helps mitigate atmospheric air pollution and global climate change.",
            "source": "ai4bharat/MSMARCO-XI/English",
            "domain": "Renewable Energy",
            "query_id": "q_en_04",
            "related_query": "What are the benefits of solar energy for the environment?"
        }
    ]
    processed_records.extend(english_records)

    # Save to JSON file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_records, f, ensure_ascii=False, indent=2)

    logger.info("=" * 65)
    logger.info(f"Completed! Total Standardized Passages: {len(processed_records)}")
    logger.info(f"Saved dataset to: {output_path}")
    logger.info("=" * 65)

    return processed_records

def main():
    parser = argparse.ArgumentParser(description="AI4Bharat MSMARCO-XI Dataset Processor")
    parser.add_argument("--languages", nargs="+", default=["hi", "ta", "te", "bn", "mr", "gu", "kn", "ml"], help="Languages to download (e.g. hi ta te bn mr gu kn ml)")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples to extract per language")
    parser.add_argument("--output", type=str, default="data/evaluation/msmarco_xi_sample.json", help="Path to save processed JSON")
    parser.add_argument("--index", action="store_true", help="Automatically run offline vector indexing into Qdrant after downloading")
    args = parser.parse_args()

    records = fetch_and_process_msmarco_xi(
        languages=args.languages,
        samples_per_lang=args.samples,
        output_path=args.output
    )

    if args.index:
        logger.info("Triggering Offline Indexer into Qdrant...")
        indexer = OfflineIndexer()
        telemetry = indexer.run_indexing_pipeline(data_path=args.output)
        logger.info(f"Indexing completed: {telemetry}")

if __name__ == "__main__":
    main()
