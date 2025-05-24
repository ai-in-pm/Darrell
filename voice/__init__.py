"""
Voice synthesis and processing components for Darrell Agent
ElevenLabs integration, audio processing, and speech recognition
"""

from .elevenlabs_integration import ElevenLabsVoiceCloner
from .audio_processor import AudioProcessor
from .speech_recognition import SpeechRecognitionEngine

__all__ = [
    "ElevenLabsVoiceCloner",
    "AudioProcessor",
    "SpeechRecognitionEngine"
]
