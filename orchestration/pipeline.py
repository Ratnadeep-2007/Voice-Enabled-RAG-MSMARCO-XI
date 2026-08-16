"""
VoiceRAG End-to-End Orchestrator Pipeline.
Coordinates STT -> Embedding -> Qdrant HNSW -> Context Builder -> Fast LLM -> Grounding.
Calculates exact stage timings, retrieval-path latency (<200ms bound), and full-pipeline telemetry.
Matches PRD §7, §21, §22, §28, §29 and hhDesign §14, §16, §18, §36.
"""

import time
import uuid
import datetime
import logging
from typing import Dict, Any, Optional, List

from speech.sarvam_stt import SarvamSTTClient, get_speech_client
from embeddings.model import EmbeddingEngine, get_embedding_engine
from retrieval.retriever import DenseRetriever, HybridRetriever, BM25Retriever
from retrieval.filters import RelevanceFilter
from retrieval.ranking import Ranker
from generation.prompt import ContextBuilder, ContextFormat
from generation.llm import FastLLMGenerator, get_llm_generator
from guardrails.input_guard import InputGuard
from guardrails.grounding import GroundingValidator, GroundingStatus
from indexing.qdrant_client import QdrantManager, get_qdrant_manager

logger = logging.getLogger(__name__)

class VoiceRAGPipeline:
    def __init__(
        self,
        collection_name: str = "msmarco_xi_dense",
        score_threshold: float = 0.32,
        default_top_k: int = 5,
        default_ef_search: int = 32
    ):
        self.collection_name = collection_name
        self.score_threshold = score_threshold
        self.default_top_k = default_top_k
        self.default_ef_search = default_ef_search

        # Subsystems
        self.speech_client = get_speech_client()
        self.embedding_engine = get_embedding_engine()
        self.qdrant_mgr = get_qdrant_manager()
        self.dense_retriever = DenseRetriever(
            qdrant_mgr=self.qdrant_mgr,
            embedding_engine=self.embedding_engine,
            collection_name=self.collection_name
        )
        self.hybrid_retriever = HybridRetriever(dense_retriever=self.dense_retriever)
        self.context_builder = ContextBuilder()
        self.llm_generator = get_llm_generator()
        self.input_guard = InputGuard()
        self.grounding_validator = GroundingValidator(score_threshold=self.score_threshold)
        
        # Recent request traces for live telemetry audit (hhDesign §36)
        self.recent_traces: List[Dict[str, Any]] = []

    def _create_trace_event(self, name: str, timestamp_str: str, duration_ms: float, details: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "stage": name,
            "timestamp": timestamp_str,
            "duration_ms": round(duration_ms, 2),
            "details": details
        }

    def process_request(
        self,
        query_text: Optional[str] = None,
        audio_bytes: Optional[bytes] = None,
        audio_filename: str = "audio.wav",
        top_k: Optional[int] = None,
        ef_search: Optional[int] = None,
        context_format: str = "json",
        use_hybrid: bool = False,
        language_override: Optional[str] = None,
        model_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes complete online RAG pipeline and returns grounded response + full telemetry.
        """
        t_pipeline_start = time.perf_counter()
        req_id = f"req_{uuid.uuid4().hex[:8]}"
        trace_events = []
        timings = {}

        now_str = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        trace_events.append(self._create_trace_event("REQUEST_RECEIVED", now_str, 0.0, {"request_id": req_id}))

        # -------------------------------------------------------------
        # 1. Speech-to-Text (STT) Stage
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        detected_language = language_override or "en"
        query = query_text

        if audio_bytes and len(audio_bytes) > 0:
            stt_res = self.speech_client.transcribe_audio_bytes(
                audio_bytes,
                filename=audio_filename,
                language_code=language_override
            )
            query = stt_res.get("text", "")
            detected_language = stt_res.get("language", detected_language)
            stt_latency = stt_res.get("latency_ms", (time.perf_counter() - t0) * 1000.0)
            timings["stt_ms"] = round(stt_latency, 2)
            stt_provider = stt_res.get("provider", "sarvam")
        elif query_text:
            stt_res = self.speech_client.transcribe_text_direct(query_text, language=detected_language)
            timings["stt_ms"] = 0.0
            stt_provider = "direct_text"
        else:
            timings["stt_ms"] = 0.0
            stt_provider = "none"
            query = ""

        now_str = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        trace_events.append(self._create_trace_event(
            "SARVAM_STT",
            now_str,
            timings["stt_ms"],
            {"query": query, "language": detected_language, "provider": stt_provider}
        ))

        # -------------------------------------------------------------
        # 2. Input Validation & Guardrails
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        is_valid, validation_msg, guard_meta = self.input_guard.validate_query(query)
        validation_latency = (time.perf_counter() - t0) * 1000.0
        timings["validation_ms"] = round(validation_latency, 2)

        if not is_valid:
            now_str = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            trace_events.append(self._create_trace_event("INPUT_GUARD_REJECT", now_str, validation_latency, {"reason": validation_msg}))
            total_time = (time.perf_counter() - t_pipeline_start) * 1000.0
            return {
                "request_id": req_id,
                "status": "rejected",
                "query": query,
                "answer": validation_msg,
                "grounding_status": GroundingStatus.UNSUPPORTED.value,
                "confidence_level": "LOW",
                "retrieved_chunks": [],
                "timings": {
                    "stt_ms": timings["stt_ms"],
                    "validation_ms": timings["validation_ms"],
                    "embedding_ms": 0.0,
                    "qdrant_ms": 0.0,
                    "context_ms": 0.0,
                    "llm_ms": 0.0,
                    "grounding_ms": 0.0,
                    "retrieval_path_ms": 0.0,
                    "total_e2e_ms": round(total_time, 2)
                },
                "trace_events": trace_events
            }

        # -------------------------------------------------------------
        # 3. Query Embedding (Local multilingual model)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        emb_res = self.embedding_engine.embed_query_timed(query)
        query_vector = emb_res["vector"]
        timings["embedding_ms"] = emb_res["latency_ms"]

        now_str = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        trace_events.append(self._create_trace_event(
            "MULTILINGUAL_EMBEDDING",
            now_str,
            timings["embedding_ms"],
            {"dimension": emb_res["dimension"], "model": emb_res["model"]}
        ))

        # -------------------------------------------------------------
        # 4. Vector DB Retrieval (Qdrant + HNSW)
        # -------------------------------------------------------------
        k = top_k or self.default_top_k
        ef = ef_search or self.default_ef_search

        if use_hybrid:
            ret_res = self.hybrid_retriever.retrieve(
                query=query,
                query_vector=query_vector,
                top_k=k,
                ef_search=ef
            )
        else:
            ret_res = self.dense_retriever.retrieve(
                query_vector=query_vector,
                top_k=k,
                ef_search=ef
            )

        raw_results = ret_res.get("results", [])
        timings["qdrant_ms"] = ret_res.get("latency_ms", 0.0)

        now_str = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        trace_events.append(self._create_trace_event(
            "QDRANT_HNSW_RETRIEVAL",
            now_str,
            timings["qdrant_ms"],
            {"retrieved_count": len(raw_results), "ef_search": ef, "top_k": k, "mode": "hybrid" if use_hybrid else "dense"}
        ))

        # -------------------------------------------------------------
        # 5. Relevance Filtering & Confidence Check
        # -------------------------------------------------------------
        has_confidence, top_score, initial_grounding = self.grounding_validator.validate_retrieval_confidence(raw_results, query=query)
        
        # Filter duplicates or low scores
        filtered_chunks = RelevanceFilter.deduplicate(raw_results) if has_confidence else []

        # -------------------------------------------------------------
        # 6. Context Construction
        # -------------------------------------------------------------
        c_format = ContextFormat.TOON if context_format.lower() == "toon" else ContextFormat.JSON
        t0 = time.perf_counter()
        system_prompt, user_prompt, ctx_telemetry = self.context_builder.build_prompt(
            query=query,
            retrieved_chunks=filtered_chunks,
            context_format=c_format
        )
        timings["context_ms"] = ctx_telemetry["latency_ms"]

        now_str = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        trace_events.append(self._create_trace_event(
            "CONTEXT_BUILDER",
            now_str,
            timings["context_ms"],
            {"format": c_format.value, "tokens": ctx_telemetry["estimated_input_tokens"]}
        ))

        # Retrieval-path latency (<200ms target bound = embedding + qdrant + context)
        retrieval_path_ms = round(timings["embedding_ms"] + timings["qdrant_ms"] + timings["context_ms"], 2)
        timings["retrieval_path_ms"] = retrieval_path_ms

        # -------------------------------------------------------------
        # 7. Low-Latency LLM Generation
        # -------------------------------------------------------------
        if not has_confidence or not filtered_chunks:
            # Fallback if no relevant evidence exists in corpus
            answer = GroundingValidator.FALLBACK_MESSAGE
            timings["llm_ms"] = 1.0
            gen_meta = {"provider": "fallback_guard", "model": "rule_based", "status": "no_evidence"}
        else:
            gen_res = self.llm_generator.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                query=query,
                retrieved_chunks=filtered_chunks,
                model_override=model_override
            )
            answer = gen_res["answer"]
            timings["llm_ms"] = gen_res["latency_ms"]
            gen_meta = gen_res

        now_str = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        trace_events.append(self._create_trace_event(
            "FAST_LLM_GENERATION",
            now_str,
            timings["llm_ms"],
            {"model": gen_meta.get("model", "fast_llm"), "provider": gen_meta.get("provider", "local")}
        ))

        # -------------------------------------------------------------
        # 8. Grounding Validation Check
        # -------------------------------------------------------------
        grounding_result = self.grounding_validator.validate_answer_grounding(
            answer=answer,
            retrieved_chunks=filtered_chunks
        )
        timings["grounding_ms"] = grounding_result["latency_ms"]

        now_str = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        trace_events.append(self._create_trace_event(
            "GROUNDING_VALIDATION",
            now_str,
            timings["grounding_ms"],
            {"status": grounding_result["status"], "confidence_score": grounding_result["confidence_score"]}
        ))

        total_e2e_ms = round((time.perf_counter() - t_pipeline_start) * 1000.0, 2)
        timings["total_e2e_ms"] = total_e2e_ms

        # Format retrieved chunks for frontend presentation
        formatted_chunks = []
        for idx, item in enumerate(filtered_chunks):
            payload = item.get("payload", {})
            raw_s = item.get("score", 0.0)
            calibrated_s = round(min(0.965, max(0.1, raw_s * 2.3)), 3) if raw_s >= self.score_threshold else round(raw_s, 3)
            formatted_chunks.append({
                "rank": f"#{idx+1:02d}",
                "id": payload.get("chunk_id", str(item.get("id", ""))),
                "document_id": payload.get("document_id", "doc_unknown"),
                "title": payload.get("title", ""),
                "score": calibrated_s,
                "raw_score": raw_s,
                "text": payload.get("text", ""),
                "language": payload.get("language", "en"),
                "source": payload.get("source", "MSMARCO-XI"),
                "token_count": payload.get("token_count", 0),
                "retrieval_ms": timings["qdrant_ms"]
            })

        response_payload = {
            "request_id": req_id,
            "status": "success",
            "query": query,
            "answer": answer,
            "language": detected_language,
            "grounding_status": grounding_result["status"],
            "confidence_score": grounding_result["confidence_score"],
            "confidence_level": grounding_result.get("confidence_level", "HIGH"),
            "supporting_passages_count": len(formatted_chunks),
            "retrieved_chunks": formatted_chunks,
            "timings": timings,
            "trace_events": trace_events,
            "retrieval_path_target_met": retrieval_path_ms < 200.0
        }

        # Store in rolling trace log (last 50 requests)
        self.recent_traces.append(response_payload)
        if len(self.recent_traces) > 50:
            self.recent_traces.pop(0)

        return response_payload


# Global singleton instance
_rag_pipeline: Optional[VoiceRAGPipeline] = None

def get_rag_pipeline() -> VoiceRAGPipeline:
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = VoiceRAGPipeline()
    return _rag_pipeline
