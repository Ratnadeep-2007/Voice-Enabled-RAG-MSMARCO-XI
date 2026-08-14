"""
Qdrant Collection Creation and HNSW / Quantization Configuration.
Supports tuning M, ef_construct, and quantization for benchmarking (PRD §14, §25).
"""

import logging
from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models

logger = logging.getLogger(__name__)

class CollectionCreator:
    @staticmethod
    def create_collection(
        client: QdrantClient,
        collection_name: str = "msmarco_xi_dense",
        vector_size: int = 384,
        distance: str = "Cosine",
        hnsw_m: int = 16,
        hnsw_ef_construct: int = 100,
        quantization_type: Optional[str] = None, # None, 'scalar', 'binary'
        recreate: bool = True
    ) -> bool:
        """
        Creates or updates a Qdrant collection with specified HNSW index parameters.
        """
        try:
            # Map distance string to Qdrant Distance enum
            dist_enum = models.Distance.COSINE
            if distance.lower() == "euclid":
                dist_enum = models.Distance.EUCLID
            elif distance.lower() == "dot":
                dist_enum = models.Distance.DOT

            # HNSW config
            hnsw_config = models.HnswConfigDiff(
                m=hnsw_m,
                ef_construct=hnsw_ef_construct,
                full_scan_threshold=1000
            )

            # Quantization config (Scalar or Binary)
            quant_config = None
            if quantization_type == "scalar":
                quant_config = models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True
                    )
                )
            elif quantization_type == "binary":
                quant_config = models.BinaryQuantization(
                    binary=models.BinaryQuantizationConfig(
                        always_ram=True
                    )
                )

            # Check if collection exists
            exists = False
            try:
                exists = client.collection_exists(collection_name)
            except Exception:
                try:
                    client.get_collection(collection_name)
                    exists = True
                except Exception:
                    exists = False

            if exists and recreate:
                logger.info(f"Recreating collection '{collection_name}'...")
                client.delete_collection(collection_name)
                exists = False

            if not exists:
                logger.info(f"Creating collection '{collection_name}' (dim={vector_size}, distance={distance}, HNSW m={hnsw_m}, ef_construct={hnsw_ef_construct}, quant={quantization_type})")
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=dist_enum
                    ),
                    hnsw_config=hnsw_config,
                    quantization_config=quant_config
                )
            return True
        except Exception as e:
            logger.error(f"Error creating collection '{collection_name}': {e}")
            return False
