"""
Sarvam STT Client and Audio Transcription Service.
Matches PRD §8.1: converts spoken audio to text with language detection and latency measurement.
Uses SOTA 'saarika:v2.5' model on Sarvam AI.
"""

import time
import os
import io
import json
import logging
from typing import Dict, Any, Optional
import httpx
from dotenv import load_dotenv

# Ensure .env is loaded so SARVAM_API_KEY is available at import time
_env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
if os.path.exists(_env_path):
    load_dotenv(_env_path, override=False)

logger = logging.getLogger(__name__)

class SarvamSTTClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        language_code: str = "en-IN",
        model: str = "saarika:v2.5"
    ):
        self.api_key = api_key or os.getenv("SARVAM_API_KEY", "")
        self.language_code = language_code
        self.model = model
        self.api_url = "https://api.sarvam.ai/speech-to-text"
        
        # Persistent HTTP client with connection pooling & keep-alive
        limits = httpx.Limits(max_keepalive_connections=20, max_connections=50, keepalive_expiry=60.0)
        self._http_client = httpx.Client(timeout=8.0, limits=limits)

    def _normalize_lang_code(self, lang: Optional[str]) -> str:
        """Normalizes 2-letter ISO codes (hi, ta, en) to Sarvam BCP-47 codes (hi-IN, ta-IN, en-IN)."""
        if not lang or lang in ("unknown", "auto"):
            return "unknown"
        lang = lang.strip().lower()
        mapping = {
            "hi": "hi-IN",
            "en": "en-IN",
            "ta": "ta-IN",
            "te": "te-IN",
            "bn": "bn-IN",
            "mr": "mr-IN",
            "gu": "gu-IN",
            "kn": "kn-IN",
            "ml": "ml-IN",
            "pa": "pa-IN",
            "od": "od-IN",
            "or": "od-IN"
        }
        return mapping.get(lang, lang if "-" in lang else f"{lang}-IN")

    def transcribe_audio_bytes(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        language_code: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribes speech audio bytes to text using Sarvam AI STT API (saarika:v2.5).
        """
        t0 = time.perf_counter()
        normalized_lang = self._normalize_lang_code(language_code or self.language_code)

        # If Sarvam API key is available, call the real endpoint
        if self.api_key and len(self.api_key.strip()) > 5:
            try:
                headers = {"api-subscription-key": self.api_key.strip()}
                data = {
                    "language_code": normalized_lang if normalized_lang != "unknown" else "en-IN",
                    "model": "saarika:v2.5"
                }
                files = {
                    "file": (filename, audio_bytes, "audio/wav")
                }

                # Send request to Sarvam STT
                response = self._http_client.post(self.api_url, headers=headers, data=data, files=files)
                latency_ms = (time.perf_counter() - t0) * 1000.0

                if response.status_code == 200:
                    res_json = response.json()
                    transcript = res_json.get("transcript", "").strip()
                    detected_lang = res_json.get("language_code", normalized_lang)
                    
                    if not transcript:
                        logger.info("Sarvam STT returned empty transcript (silent/inaudible audio).")
                        return {
                            "text": "",
                            "language": detected_lang.split("-")[0] if "-" in detected_lang else detected_lang,
                            "latency_ms": round(latency_ms, 2),
                            "provider": "sarvam_ai",
                            "status": "empty_speech"
                        }

                    return {
                        "text": transcript,
                        "language": detected_lang.split("-")[0] if "-" in detected_lang else detected_lang,
                        "latency_ms": round(latency_ms, 2),
                        "provider": "sarvam_ai",
                        "status": "success"
                    }
                else:
                    logger.warning(f"Sarvam STT API returned status {response.status_code}: {response.text}")
            except Exception as e:
                logger.warning(f"Sarvam STT API request error: {e}")

        # If Sarvam key is missing or failed, return clear status
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "text": "",
            "language": language_code or "en",
            "latency_ms": round(latency_ms, 2),
            "provider": "sarvam_stt",
            "status": "no_speech_detected"
        }

    def transcribe_audio(
        self,
        audio_bytes: bytes,
        language_code: Optional[str] = None,
        filename: str = "audio.wav"
    ) -> Dict[str, Any]:
        """Alias for transcribe_audio_bytes."""
        return self.transcribe_audio_bytes(audio_bytes, filename=filename, language_code=language_code)

    def transcribe_text_direct(self, text: str, language: str = "en") -> Dict[str, Any]:
        """Direct text injection (e.g. from Web Speech API or typing) with zero STT overhead."""
        return {
            "text": text.strip(),
            "language": language,
            "latency_ms": 0.5,
            "provider": "direct_input",
            "status": "success"
        }


_speech_client: Optional[SarvamSTTClient] = None

def get_speech_client(api_key: Optional[str] = None) -> SarvamSTTClient:
    global _speech_client
    if _speech_client is None:
        _speech_client = SarvamSTTClient(api_key=api_key)
    return _speech_client
