"""
Low-Latency LLM Generation Service.
Supports high-speed reasoning APIs (Groq LPU, CommandCode AI, Cerebras WSE-3, OpenRouter, OpenAI) with automatic multi-provider failover.
Default: Groq llama-3.1-8b-instant for sub-200ms total pipeline latency.
Matches PRD §19 & hhDesign §23.
"""

import os
import sys
import time
import json
import logging
from typing import Dict, Any, Optional, List
import httpx
from dotenv import load_dotenv

# Ensure .env is loaded from workspace root
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
if os.path.exists(env_path):
    load_dotenv(env_path, override=False)

logger = logging.getLogger(__name__)

class FastLLMGenerator:
    def __init__(
        self,
        provider: str = "fast_llm",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 180,
        temperature: float = 0.2,
        timeout: float = 5.0
    ):
        self.groq_key = os.getenv("GROQ_API_KEY", "").strip()
        self.cmd_key = (os.getenv("CMD_API_KEY", "") or os.getenv("COMMANDCODE_API_KEY", "")).strip()
        self.cmd_model = os.getenv("COMMANDCODE_MODEL", "deepseek/deepseek-v4-flash").strip()
        self.cerebras_key = os.getenv("CEREBRAS_API_KEY", "").strip()
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        self.openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.generic_key = os.getenv("LLM_API_KEY", "").strip()
        self.provider_choice = os.getenv("LLM_PROVIDER", "groq").strip().lower()

        # --- Primary provider selection based on LLM_PROVIDER ---
        # groq llama-3.1-8b-instant = ~160-220ms end-to-end (sub-200ms target)
        if self.provider_choice == "groq" and self.groq_key:
            self.provider = "groq"
            self.api_key = self.groq_key
            self.model = model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        elif self.provider_choice in ("commandcode", "cmd") and self.cmd_key:
            self.provider = "commandcode"
            self.api_key = self.cmd_key
            self.model = model or self.cmd_model
        elif self.provider_choice == "cerebras" and self.cerebras_key:
            self.provider = "cerebras"
            self.api_key = self.cerebras_key
            self.model = model or os.getenv("CEREBRAS_MODEL", "gpt-oss-120b")
        elif self.groq_key:
            self.provider = "groq"
            self.api_key = self.groq_key
            self.model = model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        elif self.cmd_key:
            self.provider = "commandcode"
            self.api_key = self.cmd_key
            self.model = model or self.cmd_model
        elif self.openrouter_key:
            self.provider = "openrouter"
            self.api_key = self.openrouter_key
            self.model = model or os.getenv("OPENROUTER_MODEL", "qwen/qwen-2.5-7b-instruct")
        elif self.cerebras_key:
            self.provider = "cerebras"
            self.api_key = self.cerebras_key
            self.model = model or os.getenv("CEREBRAS_MODEL", "gpt-oss-120b")
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

        # Persistent HTTP client with aggressive keep-alive for sub-200ms latency
        limits = httpx.Limits(max_keepalive_connections=20, max_connections=50, keepalive_expiry=60.0)
        self._http_client = httpx.Client(timeout=self.timeout, limits=limits)

    def _local_grounded_synthesis(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> str:
        """
        Local grounded response synthesizer fallback when external APIs are unreachable.
        Synthesizes a clean informative answer based on top supporting context.
        """
        if not retrieved_chunks:
            return "I couldn't find enough relevant information in the knowledge base to answer that."
        top_chunk = retrieved_chunks[0].get("payload", {})
        passage = top_chunk.get("text", "").strip()
        if not passage:
            return "I couldn't find enough relevant information in the knowledge base to answer that."
        sentences = [s.strip() for s in passage.replace("।", ".").split(".") if len(s.strip()) > 10]
        if not sentences:
            return passage
        grounded_answer = ". ".join(sentences[:2])
        if not grounded_answer.endswith("."):
            grounded_answer += "."
        return grounded_answer

    def _call_provider_api(
        self,
        url: str,
        headers: Dict[str, str],
        model_name: str,
        system_prompt: str,
        user_prompt: str
    ) -> Optional[str]:
        """Executes HTTP chat completion request against a specific provider."""
        # For reasoning/thinking models (DeepSeek), allocate extra token budget for reasoning tokens
        effective_max_tokens = max(512, self.max_tokens) if "deepseek" in model_name.lower() else self.max_tokens
        body = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": effective_max_tokens,
            "temperature": self.temperature
        }
        resp = self._http_client.post(url, headers=headers, json=body)
        if resp.status_code == 200:
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        else:
            logger.warning(f"Provider {url} ({model_name}) returned status {resp.status_code}: {resp.text[:200]}")
            return None

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        model_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes generation across Groq / CommandCode / Cerebras / OpenRouter / OpenAI
        with microsecond telemetry and automatic multi-provider failover.
        Default target: sub-200ms total pipeline latency with llama-3.1-8b-instant on Groq LPU.
        """
        t0 = time.perf_counter()
        target_model = (model_override or self.model).strip()

        # Local fast synthesizer — zero network, <1ms
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

        candidates = []

        # 1. Groq LPU — fastest cloud inference, primary sub-200ms path
        if "8b" in target_model or "instant" in target_model or "llama-3.1" in target_model or "groq" in target_model:
            if self.groq_key:
                candidates.append(("https://api.groq.com/openai/v1/chat/completions",
                    {"Authorization": f"Bearer {self.groq_key}", "Content-Type": "application/json"},
                    "llama-3.1-8b-instant", "groq_lpu"))
        if "70b" in target_model or "llama-3.3" in target_model or "versatile" in target_model:
            if self.groq_key:
                candidates.append(("https://api.groq.com/openai/v1/chat/completions",
                    {"Authorization": f"Bearer {self.groq_key}", "Content-Type": "application/json"},
                    "llama-3.3-70b-versatile", "groq_lpu"))

        # 2. CommandCode AI (DeepSeek V4 Flash / R1 — deep reasoning, higher latency ~3-5s)
        if "deepseek" in target_model or "commandcode" in target_model:
            if self.cmd_key:
                c_model = target_model if "/" in target_model and "deepseek" in target_model else self.cmd_model
                candidates.append(("https://api.commandcode.ai/provider/v1/chat/completions",
                    {"Authorization": f"Bearer {self.cmd_key}", "Content-Type": "application/json"},
                    c_model, "commandcode_ai"))

        # 3. Cerebras WSE-3 (if active key)
        if "cerebras" in target_model or "gpt-oss" in target_model:
            if self.cerebras_key:
                candidates.append(("https://api.cerebras.ai/v1/chat/completions",
                    {"Authorization": f"Bearer {self.cerebras_key}", "Content-Type": "application/json"},
                    "gpt-oss-120b", "cerebras_wse3"))

        # 4. OpenRouter (Qwen 2.5 multilingual)
        if "qwen" in target_model:
            if self.openrouter_key:
                candidates.append(("https://openrouter.ai/api/v1/chat/completions",
                    {"Authorization": f"Bearer {self.openrouter_key}", "Content-Type": "application/json",
                     "HTTP-Referer": "https://github.com/Ratnadeep-2007/Voice-Enabled-RAG-MSMARCO-XI",
                     "X-Title": "VoiceRAG Indic MSMARCO"},
                    "qwen/qwen-2.5-7b-instruct", "openrouter"))

        # 5. OpenAI
        if "gpt-4" in target_model:
            if self.openai_key:
                candidates.append(("https://api.openai.com/v1/chat/completions",
                    {"Authorization": f"Bearer {self.openai_key}", "Content-Type": "application/json"},
                    "gpt-4o-mini", "openai"))

        # --- Failover chain: Groq 8B → Groq 70B → CommandCode → OpenRouter → OpenAI ---
        if self.groq_key:
            candidates.append(("https://api.groq.com/openai/v1/chat/completions",
                {"Authorization": f"Bearer {self.groq_key}", "Content-Type": "application/json"},
                "llama-3.1-8b-instant", "groq_lpu"))
            candidates.append(("https://api.groq.com/openai/v1/chat/completions",
                {"Authorization": f"Bearer {self.groq_key}", "Content-Type": "application/json"},
                "llama-3.3-70b-versatile", "groq_lpu"))
        if self.cmd_key:
            candidates.append(("https://api.commandcode.ai/provider/v1/chat/completions",
                {"Authorization": f"Bearer {self.cmd_key}", "Content-Type": "application/json"},
                self.cmd_model, "commandcode_ai"))
        if self.cerebras_key:
            candidates.append(("https://api.cerebras.ai/v1/chat/completions",
                {"Authorization": f"Bearer {self.cerebras_key}", "Content-Type": "application/json"},
                "gpt-oss-120b", "cerebras_wse3"))
        if self.openrouter_key:
            candidates.append(("https://openrouter.ai/api/v1/chat/completions",
                {"Authorization": f"Bearer {self.openrouter_key}", "Content-Type": "application/json",
                 "HTTP-Referer": "https://github.com/Ratnadeep-2007/Voice-Enabled-RAG-MSMARCO-XI",
                 "X-Title": "VoiceRAG Indic MSMARCO"},
                "qwen/qwen-2.5-7b-instruct", "openrouter"))
        if self.openai_key:
            candidates.append(("https://api.openai.com/v1/chat/completions",
                {"Authorization": f"Bearer {self.openai_key}", "Content-Type": "application/json"},
                "gpt-4o-mini", "openai"))

        # Deduplicate while preserving priority order
        seen = set()
        dedup_candidates = []
        for c in candidates:
            key = (c[0], c[2])
            if key not in seen:
                seen.add(key)
                dedup_candidates.append(c)

        # Execute providers in sequence, return on first success
        for url, headers, model_name, provider_name in dedup_candidates:
            try:
                ans = self._call_provider_api(url, headers, model_name, system_prompt, user_prompt)
                if ans:
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                    return {
                        "answer": ans,
                        "latency_ms": round(latency_ms, 2),
                        "provider": provider_name,
                        "model": model_name,
                        "status": "success"
                    }
            except Exception as e:
                logger.warning(f"Error calling {provider_name} ({model_name}): {e}")

        # Local grounded synthesis fallback (zero-network)
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
