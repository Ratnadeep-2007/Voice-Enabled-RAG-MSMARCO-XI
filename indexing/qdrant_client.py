"""
Qdrant Vector Database Client Manager.
Supports in-memory, local persistent disk, or remote cluster modes.
"""

import os
import logging
from typing import Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models

logger = logging.getLogger(__name__)

class QdrantManager:
    def __init__(
        self,
        storage_mode: str = "memory",
        local_path: str = "data/qdrant_db",
        host: str = "localhost",
        port: int = 6333,
        url: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        self.storage_mode = storage_mode
        self.local_path = local_path
        self.host = host
        self.port = port
        self.url = url
        self.api_key = api_key
        self.client = self._init_client()

    def _init_client(self) -> QdrantClient:
        try:
            if self.url:
                logger.info(f"Connecting to remote Qdrant at {self.url}...")
                return QdrantClient(url=self.url, api_key=self.api_key)
            elif self.storage_mode == "local":
                os.makedirs(self.local_path, exist_ok=True)
                logger.info(f"Connecting to local Qdrant at path: {self.local_path}")
                return QdrantClient(path=self.local_path)
            elif self.storage_mode == "remote":
                logger.info(f"Connecting to Qdrant at {self.host}:{self.port}...")
                return QdrantClient(host=self.host, port=self.port, api_key=self.api_key)
            else:
                logger.info("Initializing high-performance in-memory Qdrant client...")
                return QdrantClient(":memory:")
        except Exception as e:
            logger.warning(f"Error initializing Qdrant ({e}), falling back to in-memory mode.")
            return QdrantClient(":memory:")

    def get_client(self) -> QdrantClient:
        return self.client

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        try:
            info = self.client.get_collection(collection_name)
            points_count = getattr(info, "points_count", 0) or 0
            indexed_vectors_count = getattr(info, "indexed_vectors_count", points_count) or points_count
            status = getattr(info, "status", "green")
            vectors_config = getattr(info.config, "params", None)
            dim = 384
            if vectors_config and hasattr(vectors_config, "vectors"):
                v = vectors_config.vectors
                if hasattr(v, "size"):
                    dim = v.size
            return {
                "collection_name": collection_name,
                "points_count": points_count,
                "vectors_count": indexed_vectors_count,
                "status": str(status),
                "dimension": dim,
                "memory_kb": round((points_count * dim * 4) / 1024, 2)
            }
        except Exception as e:
            logger.warning(f"Error getting collection info for '{collection_name}': {e}")
            return {
                "collection_name": collection_name,
                "points_count": 0,
                "vectors_count": 0,
                "status": "not_found",
                "dimension": 384,
                "memory_kb": 0
            }


_qdrant_manager: Optional[QdrantManager] = None

def get_qdrant_manager(
    storage_mode: str = "memory",
    local_path: str = "data/qdrant_db"
) -> QdrantManager:
    global _qdrant_manager
    if _qdrant_manager is None:
        _qdrant_manager = QdrantManager(storage_mode=storage_mode, local_path=local_path)
    return _qdrant_manager
