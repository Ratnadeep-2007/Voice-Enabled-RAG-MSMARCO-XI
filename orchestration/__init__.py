"""
Orchestration module for VoiceRAG:
End-to-end RAG harness, trace telemetry, and latency tracking.
"""

from .pipeline import VoiceRAGPipeline, get_rag_pipeline

__all__ = ["VoiceRAGPipeline", "get_rag_pipeline"]
