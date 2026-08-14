"""
Speech module for VoiceRAG:
Sarvam STT integration, audio processing, and speech recognition harness.
"""

from .sarvam_stt import SarvamSTTClient, get_speech_client

__all__ = ["SarvamSTTClient", "get_speech_client"]
