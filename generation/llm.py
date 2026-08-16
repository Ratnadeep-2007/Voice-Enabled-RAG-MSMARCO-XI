"""
Low-Latency LLM Generation Service.
Supports hosted fast APIs (Groq, OpenAI, Gemini) and local high-speed grounded synthesis fallback.
Matches PRD §19 & hhDesign §23.
"""

import os
import time
import json
import logging
from typing import Dict, Any, Optional, List
import httpx

logger = logging.getLogger(__name__)

class FastLLMGenerator:
    def __init__(
        self,
        provider: str = "fast_llm",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 90,
        temperature: float = 0.1,
        timeout: float = 2.5
    ):
        self.provider_choice = os.getenv("LLM_PROVIDER", "groq").strip().lower()
        self.cerebras_key = os.getenv("CEREBRAS_API_KEY", "").strip()
        self.groq_key = os.getenv("GROQ_API_KEY", "").strip()
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        self.openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.generic_key = os.getenv("LLM_API_KEY", "").strip()
        
        # Select active API key and provider
        if self.provider_choice == "openrouter" and self.openrouter_key:
            self.provider = "openrouter"
            self.api_key = self.openrouter_key
            self.model = model or "qwen/qwen-2.5-7b-instruct"
        elif self.provider_choice == "groq" and self.groq_key:
            self.provider = "groq"
            self.api_key = self.groq_key
            self.model = model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        elif self.provider_choice == "cerebras" and self.cerebras_key:
            self.provider = "cerebras"
            self.api_key = self.cerebras_key
            self.model = model or os.getenv("CEREBRAS_MODEL", "gpt-oss-120b")
        elif self.cerebras_key:
            self.provider = "cerebras"
            self.api_key = self.cerebras_key
            self.model = model or os.getenv("CEREBRAS_MODEL", "gpt-oss-120b")
        elif self.groq_key:
            self.provider = "groq"
            self.api_key = self.groq_key
            self.model = model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        elif self.openrouter_key:
            self.provider = "openrouter"
            self.api_key = self.openrouter_key
            self.model = model or "qwen/qwen-2.5-7b-instruct"
        elif self.openai_key:
            self.provider = "openai"
            self.api_key = self.openai_key
            self.model = model or "gpt-4o-mini"
        else:
            self.provider = "local"
            self.api_key = (api_key or self.generic_key).strip()
            self.model = model or "llama-3.1-8b-instant"

        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        
        # Persistent HTTP client with connection pooling & keep-alive
        limits = httpx.Limits(max_keepalive_connections=20, max_connections=50, keepalive_expiry=60.0)
        self._http_client = httpx.Client(timeout=self.timeout, limits=limits)

    def _local_grounded_synthesis(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> str:
        """
        Ultra-fast local grounded response synthesizer (<10ms).
        Synthesizes concise answers strictly extracted and normalized from top supporting chunks.
        """
        if not retrieved_chunks:
            return "I couldn't find enough relevant information in the knowledge base to answer that."

        top_chunk = retrieved_chunks[0].get("payload", {})
        passage = top_chunk.get("text", "")
        title = top_chunk.get("title", "")

        # Split passage into key sentences
        sentences = [s.strip() for s in passage.replace("।", ".").split(".") if s.strip()]
        if not sentences:
            return passage

        # Pick the most relevant sentences
        selected = sentences[:3]
        grounded_answer = ". ".join(selected)
        if not grounded_answer.endswith("."):
            grounded_answer += "."
        return grounded_answer

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        model_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes generation on Groq / Cerebras / OpenAI or local fallback with microsecond telemetry.
        """
        t0 = time.perf_counter()
        target_model = (model_override or self.model).strip()

        if target_model == "local_fast":
            local_ans = self._local_grounded_synthesis(query, retrieved_chunks)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "answer": local_ans,
                "latency_ms": round(max(0.8, latency_ms), 2),
                "provider": "local_synthesizer",
                "model": "local_fast",
                "status": "success"
            }

        # Resolve provider based on target_model or active API keys
        active_key = self.api_key
        headers = {
            "Content-Type": "application/json"
        }

        if "qwen" in target_model:
            url = "https://openrouter.ai/api/v1/chat/completions"
            provider_name = "openrouter"
            active_key = self.openrouter_key or self.api_key
            headers["HTTP-Referer"] = "https://github.com/Ratnadeep-2007/Voice-Enabled-RAG-MSMARCO-XI"
            headers["X-Title"] = "VoiceRAG Indic MSMARCO"
            if "qwen-3" in target_model or "qwen3" in target_model or "4b" in target_model:
                target_model = "qwen/qwen-2.5-7b-instruct" # Standard fast 7B/3B on OpenRouter
            elif not target_model.startswith("qwen/"):
                target_model = f"qwen/{target_model}"
        elif "gpt-oss" in target_model or target_model == "cerebras_gpt_oss_120b":
            url = "https://api.cerebras.ai/v1/chat/completions"
            provider_name = "cerebras_lpu"
            active_key = self.cerebras_key or self.api_key
            target_model = "gpt-oss-120b"
        elif target_model == "cerebras_llama_3.3_70b":
            url = "https://api.cerebras.ai/v1/chat/completions"
            provider_name = "cerebras_lpu"
            active_key = self.cerebras_key or self.api_key
            target_model = "llama-3.3-70b"
        elif target_model == "groq_llama_3.3_70b":
            url = "https://api.groq.com/openai/v1/chat/completions"
            provider_name = "groq_lpu"
            active_key = self.groq_key or self.api_key
            target_model = "llama-3.3-70b-versatile"
        elif "llama-3.1" in target_model or target_model == "groq_llama_3.1_8b":
            url = "https://api.groq.com/openai/v1/chat/completions"
            provider_name = "groq_lpu"
            active_key = self.groq_key or self.api_key
            target_model = "llama-3.1-8b-instant"
        elif "gpt-4" in target_model:
            url = "https://api.openai.com/v1/chat/completions"
            provider_name = "openai"
            active_key = self.openai_key or self.api_key
            target_model = "gpt-4o-mini"
        else:
            if self.groq_key or active_key.startswith("gsk_"):
                url = "https://api.groq.com/openai/v1/chat/completions"
                provider_name = "groq_lpu"
            elif self.cerebras_key or active_key.startswith("csk-") or active_key.startswith("csk_"):
                url = "https://api.cerebras.ai/v1/chat/completions"
                provider_name = "cerebras_lpu"
            elif self.openrouter_key or active_key.startswith("sk-or-"):
                url = "https://openrouter.ai/api/v1/chat/completions"
                provider_name = "openrouter"
                headers["HTTP-Referer"] = "https://github.com/Ratnadeep-2007/Voice-Enabled-RAG-MSMARCO-XI"
                headers["X-Title"] = "VoiceRAG Indic MSMARCO"
            else:
                url = "https://api.openai.com/v1/chat/completions"
                provider_name = "openai"

        # Check for external API key
        if active_key and len(active_key) > 8:
            try:
                headers["Authorization"] = f"Bearer {active_key}"
                body = {
                    "model": target_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature
                }

                # Reuse warm keep-alive connection
                resp = self._http_client.post(url, headers=headers, json=body)
                latency_ms = (time.perf_counter() - t0) * 1000.0
                if resp.status_code == 200:
                    data = resp.json()
                    answer = data["choices"][0]["message"]["content"].strip()
                    return {
                        "answer": answer,
                        "latency_ms": round(latency_ms, 2),
                        "provider": provider_name,
                        "model": target_model,
                        "status": "success"
                    }
                else:
                    logger.warning(f"LLM API ({provider_name}) returned {resp.status_code}: {resp.text}")
            except Exception as e:
                logger.error(f"Error connecting to LLM API: {e}")

        # High-speed local grounded synthesis fallback (<10ms)
        local_ans = self._local_grounded_synthesis(query, retrieved_chunks)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "answer": local_ans,
            "latency_ms": round(max(0.8, latency_ms), 2),
            "provider": "local_synthesizer",
            "model": "local_grounded_v2",
            "status": "fallback"
        }


_llm_generator: Optional[FastLLMGenerator] = None

def get_llm_generator(api_key: Optional[str] = None, provider: str = "fast_llm") -> FastLLMGenerator:
    global _llm_generator
    if _llm_generator is None:
        _llm_generator = FastLLMGenerator(api_key=api_key, provider=provider)
    return _llm_generator
