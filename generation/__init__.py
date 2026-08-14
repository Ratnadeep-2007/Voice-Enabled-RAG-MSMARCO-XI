"""
Generation module for VoiceRAG:
Context builders (JSON & TOON), prompt formatting, and low-latency LLM synthesis.
"""

from .prompt import ContextBuilder, ContextFormat
from .llm import FastLLMGenerator, get_llm_generator

__all__ = ["ContextBuilder", "ContextFormat", "FastLLMGenerator", "get_llm_generator"]
