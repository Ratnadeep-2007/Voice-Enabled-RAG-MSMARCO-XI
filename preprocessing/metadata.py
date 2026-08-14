"""
Metadata builder and schema definition for MSMARCO-XI chunk traceability.
"""

from typing import Dict, Any, Optional

class MetadataBuilder:
    @staticmethod
    def build_chunk_payload(
        chunk_id: str,
        doc_id: str,
        text: str,
        annotated_text: str,
        language: str,
        source: str,
        title: str,
        position: int,
        total_chunks: int,
        token_count: int,
        query_id: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        payload = {
            "chunk_id": chunk_id,
            "document_id": doc_id,
            "text": text,
            "annotated_text": annotated_text,
            "language": language,
            "source": source,
            "title": title,
            "position": position,
            "total_chunks": total_chunks,
            "token_count": token_count,
            "query_id": query_id or ""
        }
        if extra_metadata:
            payload.update(extra_metadata)
        return payload
