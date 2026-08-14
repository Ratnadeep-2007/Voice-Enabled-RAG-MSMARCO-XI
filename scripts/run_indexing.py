#!/usr/bin/env python3
"""
CLI script to run offline indexing pipeline on MSMARCO-XI dataset.
Matches PRD §12 & §27.
"""

import os
import sys
import argparse
import logging

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from indexing.index_dataset import OfflineIndexer
from preprocessing.chunker import ChunkingStrategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VoiceRAG_Indexer")

def main():
    parser = argparse.ArgumentParser(description="VoiceRAG Offline Indexing Pipeline")
    parser.add_argument("--data-path", type=str, default="data/evaluation/msmarco_xi_sample.json", help="Path to raw dataset JSON")
    parser.add_argument("--strategy", type=str, default="adaptive", choices=["fixed", "fixed_overlap", "sentence", "adaptive"], help="Chunking strategy")
    parser.add_argument("--m", type=int, default=16, help="HNSW connectivity M")
    parser.add_argument("--ef-construct", type=int, default=100, help="HNSW ef_construct")
    parser.add_argument("--quantization", type=str, default=None, choices=["scalar", "binary", None], help="Quantization mode")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    args = parser.parse_args()

    strat_map = {
        "fixed": ChunkingStrategy.FIXED,
        "fixed_overlap": ChunkingStrategy.FIXED_OVERLAP,
        "sentence": ChunkingStrategy.SENTENCE,
        "adaptive": ChunkingStrategy.ADAPTIVE
    }

    logger.info(f"Starting VoiceRAG Offline Indexing (strategy={args.strategy}, M={args.m}, ef={args.ef_construct})...")
    indexer = OfflineIndexer()
    results = indexer.run_indexing_pipeline(
        data_path=args.data_path,
        chunking_strategy=strat_map[args.strategy],
        hnsw_m=args.m,
        hnsw_ef_construct=args.ef_construct,
        quantization_type=args.quantization,
        batch_size=args.batch_size
    )
    logger.info(f"Indexing completed: {results}")

if __name__ == "__main__":
    main()
