"""
Offline Indexing Pipeline.
Executes dataset loading -> adaptive chunking -> batch embedding -> Qdrant HNSW indexing.
Matches PRD §12 offline pipeline flow.
"""

import time
import logging
from typing import Dict, Any, Optional, List
from qdrant_client.http import models

from preprocessing.loader import DatasetLoader
from preprocessing.chunker import DocumentChunker, ChunkingStrategy
from preprocessing.metadata import MetadataBuilder
from embeddings.model import EmbeddingEngine, get_embedding_engine
from embeddings.embed import BatchEmbedder
from .qdrant_client import QdrantManager, get_qdrant_manager
from .create_collection import CollectionCreator

logger = logging.getLogger(__name__)

class OfflineIndexer:
    def __init__(
        self,
        qdrant_mgr: Optional[QdrantManager] = None,
        embedding_engine: Optional[EmbeddingEngine] = None,
        collection_name: str = "msmarco_xi_dense"
    ):
        self.qdrant_mgr = qdrant_mgr or get_qdrant_manager()
        self.embedding_engine = embedding_engine or get_embedding_engine()
        self.collection_name = collection_name
        self.loader = DatasetLoader()
        self.chunker = DocumentChunker(strategy=ChunkingStrategy.ADAPTIVE)
        self.batch_embedder = BatchEmbedder(engine=self.embedding_engine)

    def run_indexing_pipeline(
        self,
        data_path: Optional[str] = None,
        chunking_strategy: ChunkingStrategy = ChunkingStrategy.ADAPTIVE,
        hnsw_m: int = 16,
        hnsw_ef_construct: int = 100,
        quantization_type: Optional[str] = None,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Runs the complete offline indexing pipeline and returns execution telemetry.
        """
        t_start = time.perf_counter()
        telemetry = {}

        # 1. Load Dataset
        t0 = time.perf_counter()
        docs = self.loader.load_documents(data_path)
        telemetry["load_time_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        telemetry["loaded_documents"] = len(docs)

        if not docs:
            logger.error("No documents loaded for indexing.")
            return {"status": "error", "message": "No documents found."}

        # 2. Chunk Documents
        t0 = time.perf_counter()
        all_chunks: List[Dict[str, Any]] = []
        for doc in docs:
            chunks = self.chunker.chunk_document(doc, strategy=chunking_strategy)
            all_chunks.extend(chunks)
        telemetry["chunk_time_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        telemetry["total_chunks"] = len(all_chunks)

        # 3. Create / Recreate Qdrant Collection
        client = self.qdrant_mgr.get_client()
        CollectionCreator.create_collection(
            client=client,
            collection_name=self.collection_name,
            vector_size=self.embedding_engine.dimension,
            hnsw_m=hnsw_m,
            hnsw_ef_construct=hnsw_ef_construct,
            quantization_type=quantization_type,
            recreate=True
        )

        # 4. Batch Embed
        embedded_chunks, embed_time_ms = self.batch_embedder.process_chunks(
            all_chunks,
            batch_size=batch_size
        )
        telemetry["embed_time_ms"] = round(embed_time_ms, 2)

        # 5. Upload Points to Qdrant
        t0 = time.perf_counter()
        points = []
        for idx, chunk in enumerate(embedded_chunks):
            vector = chunk.pop("vector")
            points.append(
                models.PointStruct(
                    id=idx + 1,
                    vector=vector,
                    payload=chunk
                )
            )

        client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True
        )
        telemetry["upload_time_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        telemetry["total_indexed_points"] = len(points)
        telemetry["total_pipeline_ms"] = round((time.perf_counter() - t_start) * 1000.0, 2)
        telemetry["status"] = "success"

        logger.info(f"Offline indexing completed successfully: {telemetry}")
        return telemetry
