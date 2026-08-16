"""
FAISS IVF-PQ (Inverted File Product Quantization) Offline Vector Indexer.
Builds highly compressed, sub-millisecond vector indices and persists them to NVMe.
"""

import os
import sys
import time
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import faiss

from embeddings.model import EmbeddingEngine, get_embedding_engine
from indexing.lmdb_store import LMDBDocumentStore, get_lmdb_store
from preprocessing.loader import DatasetLoader
from preprocessing.chunker import DocumentChunker, ChunkingStrategy

logger = logging.getLogger(__name__)

class FAISSIVFPQIndexer:
    def __init__(
        self,
        dimension: int = 384,
        index_path: str = "data/faiss_index/msmarco_xi_ivfpq.faiss",
        lmdb_path: str = "data/lmdb_store",
        nlist: int = 64,
        m_subvectors: int = 48,
        nbits: int = 8
    ):
        """
        :param dimension: Embedding dimensionality (384 for MiniLM-L12).
        :param index_path: Path to save the serialized .faiss index.
        :param lmdb_path: Path to save the LMDB document store.
        :param nlist: Number of Voronoi centroid clusters for IVF.
        :param m_subvectors: Number of sub-vector slices for Product Quantization (must divide dimension: 384/48 = 8 dims per slice).
        :param nbits: Number of bits per sub-vector quantization (default 8 = 256 centroids per sub-space).
        """
        self.dimension = dimension
        self.index_path = index_path
        self.lmdb_path = lmdb_path
        self.nlist = nlist
        self.m_subvectors = m_subvectors
        self.nbits = nbits
        
        self.embed_engine = get_embedding_engine()
        self.lmdb_store = get_lmdb_store(db_path=lmdb_path)
        self.index: Optional[faiss.Index] = None

    def build_index(self, vectors: np.ndarray, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Trains and populates the FAISS IVF-PQ index and LMDB document store.
        """
        t0 = time.perf_counter()
        num_points = vectors.shape[0]
        
        # Ensure float32 normalized vectors for Inner Product (Cosine Similarity)
        vectors = vectors.astype(np.float32)
        faiss.normalize_L2(vectors)

        # Select index type based on dataset size
        if num_points >= max(256, self.nlist * 4):
            logger.info(f"Training FAISS IndexIVFPQ (d={self.dimension}, nlist={self.nlist}, M={self.m_subvectors}, nbits={self.nbits}) on {num_points} vectors...")
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFPQ(
                quantizer,
                self.dimension,
                self.nlist,
                self.m_subvectors,
                self.nbits,
                faiss.METRIC_INNER_PRODUCT
            )
            self.index.train(vectors)
            self.index.add(vectors)
            self.index.nprobe = 8 # Probe top 8 Voronoi cells at search time
            index_type = "IVF-PQ"
        elif num_points >= 32:
            effective_nlist = min(16, num_points // 2)
            logger.info(f"Using FAISS IndexIVFFlat (d={self.dimension}, nlist={effective_nlist}) on {num_points} vectors...")
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, effective_nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.train(vectors)
            self.index.add(vectors)
            self.index.nprobe = 4
            index_type = "IVF-Flat"
        else:
            logger.info(f"Using FAISS IndexFlatIP (Exact Cosine, d={self.dimension}) on {num_points} vectors...")
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(vectors)
            index_type = "FlatIP"

        # Save to LMDB
        t_lmdb_start = time.perf_counter()
        self.lmdb_store.put_batch(documents, start_idx=0)
        lmdb_time_ms = (time.perf_counter() - t_lmdb_start) * 1000.0

        # Save .faiss file to disk
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        
        total_time_ms = (time.perf_counter() - t0) * 1000.0
        file_size_kb = os.path.getsize(self.index_path) / 1024.0

        telemetry = {
            "num_indexed_points": num_points,
            "index_type": index_type,
            "index_file": self.index_path,
            "file_size_kb": round(file_size_kb, 2),
            "lmdb_time_ms": round(lmdb_time_ms, 2),
            "total_indexing_time_ms": round(total_time_ms, 2),
            "status": "success"
        }
        logger.info(f"FAISS index built and written to NVMe: {telemetry}")
        return telemetry

    def run_pipeline(self, data_path: str = "data/evaluation/msmarco_xi_sample.json") -> Dict[str, Any]:
        """
        Runs full offline pipeline: Load -> Chunk -> Embed -> FAISS IVF-PQ Index -> Save NVMe.
        """
        logger.info(f"Running FAISS offline indexing pipeline for {data_path}...")
        t_total_start = time.perf_counter()

        loader = DatasetLoader(data_path)
        docs = loader.load_documents()
        if not docs:
            return {"status": "error", "message": "No documents found to index"}

        # Chunking
        chunker = DocumentChunker(strategy=ChunkingStrategy.ADAPTIVE)
        chunks = chunker.chunk_corpus(docs)
        logger.info(f"Generated {len(chunks)} chunks from {len(docs)} documents.")

        # Embedding
        t_emb_start = time.perf_counter()
        texts = [c["text"] for c in chunks]
        embeddings = self.embed_engine.embed_batch(texts)
        embed_time_ms = (time.perf_counter() - t_emb_start) * 1000.0

        # Build Index & LMDB
        vectors = np.array(embeddings, dtype=np.float32)
        telemetry = self.build_index(vectors, chunks)
        telemetry["embed_time_ms"] = round(embed_time_ms, 2)
        telemetry["total_pipeline_ms"] = round((time.perf_counter() - t_total_start) * 1000.0, 2)
        return telemetry

if __name__ == "__main__":
    indexer = FAISSIVFPQIndexer()
    res = indexer.run_pipeline()
    print("FAISS IVF-PQ Indexing Result:", json.dumps(res, indent=2))
