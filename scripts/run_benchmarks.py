#!/usr/bin/env python3
"""
CLI script to execute all 7 benchmark experiments (PRD §25).
"""

import os
import sys
import json
import logging

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.benchmark import BenchmarkRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VoiceRAG_Benchmarks")

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logger.info("Executing VoiceRAG Empirical Benchmark Suite (Experiments 0 through 6)...")
    runner = BenchmarkRunner()
    results = runner.run_all_benchmarks()
    
    print("\n" + "="*80)
    print(" VoiceRAG Benchmark Suite — Empirical Evaluation Results")
    print("="*80)
    
    for exp_id in ["exp0_chunking", "exp1_embeddings", "exp2_hnsw", "exp3_top_k", "exp4_quantization", "exp5_context_format", "exp6_dense_vs_hybrid"]:
        exp = results.get(exp_id, {})
        print(f"\n>> {exp.get('title')}")
        print(f"   Best Selection: {exp.get('best')}")
        for item in exp.get("data", []):
            print(f"   - {item}")

    print("\n" + "="*80)
    print(" Summary Matrix (hhDesign §33)")
    print("="*80)
    for row in results.get("summary_table", []):
        print(f" {row['configuration']:<42} | Recall@5: {row['recall_at_5']:<6} | P50: {row['p50']:<7} | P100: {row['p100']:<7} | {row['result']}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
