"""
Sarvam STT Client and Audio Transcription Service.
Matches PRD §8.1: converts spoken audio to text with language detection,
comprehensive pipeline observability, and latency measurement.
Uses SOTA 'saaras:v3' model on Sarvam AI.
"""

import time
import os
import io
import wave
import struct
import math
import logging
from typing import Dict, Any, Optional
import httpx
from dotenv import load_dotenv

# Ensure .env is loaded so SARVAM_API_KEY is available at import time
_env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
if os.path.exists(_env_path):
    load_dotenv(_env_path, override=False)

logger = logging.getLogger("VoiceRAG_STT")
logging.basicConfig(level=logging.INFO)


def normalize_audio_to_16k_wav(audio_bytes: bytes) -> bytes:
    """
    Decodes ANY audio format sent by browsers (WebM, Opus, MP4, AAC, OGG, WAV) using PyAV
    and standardizes it into 16kHz Mono 16-bit PCM WAV in <2ms before sending to Sarvam AI.
    """
    if len(audio_bytes) < 32:
        return audio_bytes
    try:
        import av
        in_buf = io.BytesIO(audio_bytes)
        out_buf = io.BytesIO()
        with av.open(in_buf) as container:
            if not container.streams.audio:
                return audio_bytes
            stream = container.streams.audio[0]
            resampler = av.AudioResampler(format='s16', layout='mono', rate=16000)
            with wave.open(out_buf, 'wb') as wav_out:
                wav_out.setnchannels(1)
                wav_out.setsampwidth(2)
                wav_out.setframerate(16000)
                for frame in container.decode(stream):
                    for rf in resampler.resample(frame):
                        wav_out.writeframes(rf.to_ndarray().tobytes())
        normalized = out_buf.getvalue()
        if len(normalized) > 44:
            return normalized
    except Exception as e:
        logger.debug("PyAV normalization fallback: %s", e)
    return audio_bytes


def inspect_wav_audio(audio_bytes: bytes) -> Dict[str, Any]:
    """Inspects raw WAV bytes to extract audio duration, sample rate, peak amplitude, and RMS energy."""
    info: Dict[str, Any] = {
        "size_bytes": len(audio_bytes),
        "is_wav": False,
        "duration_seconds": 0.0,
        "is_silent": True
    }
    if len(audio_bytes) < 44:
        return info
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as w:
            nchannels = w.getnchannels()
            sampwidth = w.getsampwidth()
            framerate = w.getframerate()
            nframes = w.getnframes()
            duration_sec = nframes / float(framerate) if framerate > 0 else 0.0
            frames = w.readframes(nframes)
            
            # Compute RMS and peak amplitude for 16-bit PCM
            if sampwidth == 2 and nframes > 0:
                samples = struct.unpack(f"<{nframes * nchannels}h", frames)
                peak = max(abs(s) for s in samples) if samples else 0
                rms = math.sqrt(sum(s * s for s in samples) / len(samples)) if samples else 0.0
                rms_db = 20 * math.log10(rms / 32768.0) if rms > 0 else -100.0
            else:
                peak = 0
                rms = 0.0
                rms_db = -100.0
                
            info.update({
                "is_wav": True,
                "channels": nchannels,
                "sample_width": sampwidth,
                "sample_rate": framerate,
                "frames": nframes,
                "duration_seconds": round(duration_sec, 2),
                "peak_amplitude": peak,
                "rms_amplitude": round(rms, 2),
                "rms_db": round(rms_db, 1),
                "is_silent": rms < 30.0  # Threshold for audible speech
            })
    except Exception as e:
        info["wav_parse_error"] = str(e)
    return info


class SarvamSTTClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        language_code: str = "unknown",
        model: Optional[str] = None
    ):
        self.api_key = (api_key or os.getenv("SARVAM_API_KEY", "")).strip()
        self.language_code = language_code
        self.model = model or os.getenv("SARVAM_MODEL", "saaras:v3")
        self.api_url = "https://api.sarvam.ai/speech-to-text"
        
        # Persistent HTTP client with connection pooling & keep-alive
        limits = httpx.Limits(max_keepalive_connections=20, max_connections=50, keepalive_expiry=60.0)
        self._http_client = httpx.Client(timeout=10.0, limits=limits)

    def _normalize_lang_code(self, lang: Optional[str]) -> str:
        """Normalizes 2-letter ISO codes (hi, ta, en) to Sarvam BCP-47 codes (hi-IN, ta-IN, en-IN, unknown)."""
        if not lang or lang.strip().lower() in ("unknown", "auto", "all"):
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
            "or": "od-IN",
            "as": "as-IN",
            "ur": "ur-IN"
        }
        return mapping.get(lang, lang if "-" in lang else f"{lang}-IN")

    def transcribe_audio_bytes(
        self,
        audio_bytes: bytes,
        filename: str = "speech.wav",
        language_code: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribes speech audio bytes to text using Sarvam AI STT API (saaras:v3).
        Includes audio normalization (PyAV) and extensive console diagnostics for observability.
        """
        t0 = time.perf_counter()
        normalized_lang = self._normalize_lang_code(language_code or self.language_code)
        
        # 1. Normalize audio format to 16kHz mono WAV (supports WebM, Opus, MP4, WAV)
        normalized_wav_bytes = normalize_audio_to_16k_wav(audio_bytes)
        
        # 2. Audio Inspection & Diagnostics
        audio_info = inspect_wav_audio(normalized_wav_bytes)
        print("=" * 68)
        print("[STT Pipeline] >>> Inbound Voice Audio Received")
        print(f"  Filename     : {filename}")
        print(f"  Original Size: {len(audio_bytes):,} bytes | Normalized WAV: {len(normalized_wav_bytes):,} bytes")
        if audio_info.get("is_wav"):
            print(f"  Duration     : {audio_info.get('duration_seconds')}s ({audio_info.get('frames')} frames)")
            print(f"  Format       : {audio_info.get('sample_rate')}Hz Mono 16-bit PCM")
            print(f"  Signal Level : Peak={audio_info.get('peak_amplitude')} | RMS={audio_info.get('rms_amplitude')} ({audio_info.get('rms_db')} dB)")
            print(f"  Silence Check: {'SILENT (No speech energy)' if audio_info.get('is_silent') else 'AUDIBLE SIGNAL DETECTED'}")
        else:
            print(f"  WAV Header   : Non-standard / raw ({audio_info.get('wav_parse_error', 'raw bytes')})")
        print(f"  Target Lang  : {normalized_lang} (Requested: {language_code or 'default'})")
        print(f"  STT Model    : {self.model}")
        print("=" * 68)

        # 3. Key Check
        if not self.api_key or len(self.api_key) < 5:
            print("[STT Pipeline ERROR] SARVAM_API_KEY is not set or invalid in .env!")
            return {
                "text": "",
                "language": normalized_lang if normalized_lang != "unknown" else "en",
                "latency_ms": 0.1,
                "provider": "sarvam_ai",
                "status": "missing_api_key",
                "audio_info": audio_info
            }

        # 4. Dispatch to Sarvam AI STT
        try:
            masked_key = f"{self.api_key[:6]}...{self.api_key[-4:]}"
            print(f"[STT Pipeline] Calling https://api.sarvam.ai/speech-to-text with key={masked_key}...")
            
            headers = {"api-subscription-key": self.api_key}
            data = {
                "language_code": normalized_lang,
                "model": self.model
            }
            files = {
                "file": ("speech.wav", normalized_wav_bytes, "audio/wav")
            }

            response = self._http_client.post(self.api_url, headers=headers, data=data, files=files)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            print(f"[STT Pipeline] <<< Sarvam AI HTTP Response {response.status_code} in {latency_ms:.1f}ms")

            if response.status_code == 200:
                res_json = response.json()
                transcript = res_json.get("transcript", "").strip()
                detected_lang = res_json.get("language_code", normalized_lang)
                req_id = res_json.get("request_id", "N/A")

                print(f"[STT Pipeline Result] Request ID  : {req_id}")
                print(f"[STT Pipeline Result] Language    : {detected_lang}")
                print(f"[STT Pipeline Result] Transcript  : \"{transcript}\"")

                if not transcript:
                    print("[STT Pipeline Result] [WARNING] Empty transcript returned by Sarvam STT.")
                    if audio_info.get("is_silent"):
                        print("  -> Cause: The microphone input contained near-zero RMS energy (silence).")
                    else:
                        print("  -> Cause: Audio was audible but Sarvam STT did not recognize clear phonemes.")

                    return {
                        "text": "",
                        "language": detected_lang.split("-")[0] if "-" in detected_lang else detected_lang,
                        "latency_ms": round(latency_ms, 2),
                        "provider": "sarvam_ai",
                        "status": "empty_speech",
                        "audio_info": audio_info
                    }

                return {
                    "text": transcript,
                    "language": detected_lang.split("-")[0] if "-" in detected_lang else detected_lang,
                    "latency_ms": round(latency_ms, 2),
                    "provider": "sarvam_ai",
                    "status": "success",
                    "audio_info": audio_info
                }
            else:
                print(f"[STT Pipeline ERROR] Sarvam AI returned HTTP {response.status_code}: {response.text}")
                return {
                    "text": "",
                    "language": normalized_lang,
                    "latency_ms": round(latency_ms, 2),
                    "provider": "sarvam_ai",
                    "status": f"http_error_{response.status_code}",
                    "error_detail": response.text,
                    "audio_info": audio_info
                }

        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[STT Pipeline EXCEPTION] Request to Sarvam AI failed: {e}")
            return {
                "text": "",
                "language": normalized_lang,
                "latency_ms": round(latency_ms, 2),
                "provider": "sarvam_ai",
                "status": "exception",
                "error_detail": str(e),
                "audio_info": audio_info
            }

    def transcribe_audio(
        self,
        audio_bytes: bytes,
        language_code: Optional[str] = None,
        filename: str = "speech.wav"
    ) -> Dict[str, Any]:
        """Alias for transcribe_audio_bytes."""
        return self.transcribe_audio_bytes(audio_bytes, filename=filename, language_code=language_code)

    def transcribe_text_direct(self, text: str, language: str = "en") -> Dict[str, Any]:
        """Direct text injection with zero STT overhead."""
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
