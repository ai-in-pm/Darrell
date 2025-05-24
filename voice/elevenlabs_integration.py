"""
ElevenLabs Voice Synthesis Integration for Darrell Agent
Advanced voice cloning and real-time speech synthesis
"""

import asyncio
import io
import time
import wave
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np

try:
    import elevenlabs
    from elevenlabs import Voice, VoiceSettings, generate, clone, set_api_key
    ELEVENLABS_AVAILABLE = True
except ImportError:
    print("Warning: ElevenLabs package not available. Install with: pip install elevenlabs")
    ELEVENLABS_AVAILABLE = False

import requests
import json
from ..utils.logger import DarrellLogger, LogCategory
from ..utils.config import VoiceConfig


@dataclass
class VoiceSample:
    """Voice sample for training"""
    file_path: str
    duration: float
    quality_score: float
    text_content: Optional[str] = None


@dataclass
class SynthesisRequest:
    """Voice synthesis request"""
    text: str
    voice_id: str
    settings: Dict[str, Any]
    priority: int = 1
    callback: Optional[callable] = None


class ElevenLabsVoiceCloner:
    """
    Advanced ElevenLabs voice synthesis integration
    Handles voice cloning, real-time synthesis, and audio optimization
    """
    
    def __init__(self, api_key: str, voice_id: str = None, config: VoiceConfig = None):
        """
        Initialize ElevenLabs voice cloner
        
        Args:
            api_key: ElevenLabs API key
            voice_id: Pre-trained voice ID (optional)
            config: Voice configuration
        """
        self.api_key = api_key
        self.voice_id = voice_id
        self.config = config or VoiceConfig()
        self.logger = DarrellLogger("ElevenLabsVoice")
        
        # Voice management
        self.available_voices = {}
        self.custom_voice_id = None
        self.voice_settings = None
        
        # Synthesis queue and caching
        self.synthesis_queue = asyncio.Queue()
        self.audio_cache = {}
        self.cache_max_size = 100
        
        # Performance tracking
        self.synthesis_times = []
        self.total_characters_synthesized = 0
        self.total_audio_generated = 0.0  # seconds
        
        # State management
        self.is_initialized = False
        self.is_processing = False
        self._synthesis_task = None
        
        if not ELEVENLABS_AVAILABLE:
            self.logger.error("ElevenLabs package not available", LogCategory.VOICE)
            raise ImportError("ElevenLabs package required for voice synthesis")
    
    async def initialize(self) -> bool:
        """Initialize ElevenLabs integration"""
        try:
            self.logger.info("Initializing ElevenLabs voice cloner...", LogCategory.VOICE)
            
            # Set API key
            set_api_key(self.api_key)
            
            # Load available voices
            await self._load_available_voices()
            
            # Setup voice settings
            self._setup_voice_settings()
            
            # Validate voice ID
            if self.voice_id and not await self._validate_voice_id(self.voice_id):
                self.logger.warning(f"Voice ID {self.voice_id} not found, will use default", 
                                  LogCategory.VOICE)
                self.voice_id = None
            
            # Start synthesis processing task
            self._synthesis_task = asyncio.create_task(self._process_synthesis_queue())
            
            self.is_initialized = True
            self.logger.info("ElevenLabs voice cloner initialized successfully", LogCategory.VOICE)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ElevenLabs: {e}", LogCategory.VOICE, error=e)
            return False
    
    async def _load_available_voices(self):
        """Load available voices from ElevenLabs"""
        try:
            # Get voices using ElevenLabs API
            voices = elevenlabs.voices()
            
            for voice in voices:
                self.available_voices[voice.voice_id] = {
                    'name': voice.name,
                    'category': voice.category,
                    'description': getattr(voice, 'description', ''),
                    'preview_url': getattr(voice, 'preview_url', ''),
                    'settings': voice.settings.__dict__ if voice.settings else {}
                }
            
            self.logger.info(f"Loaded {len(self.available_voices)} available voices", 
                           LogCategory.VOICE)
            
        except Exception as e:
            self.logger.error(f"Failed to load available voices: {e}", LogCategory.VOICE, error=e)
    
    def _setup_voice_settings(self):
        """Setup voice synthesis settings"""
        self.voice_settings = VoiceSettings(
            stability=self.config.voice_stability,
            similarity_boost=self.config.voice_similarity_boost,
            style=self.config.voice_style,
            use_speaker_boost=self.config.voice_use_speaker_boost
        )
        
        self.logger.info("Voice settings configured", LogCategory.VOICE,
                        performance_data={
                            "stability": self.config.voice_stability,
                            "similarity_boost": self.config.voice_similarity_boost,
                            "style": self.config.voice_style,
                            "speaker_boost": self.config.voice_use_speaker_boost
                        })
    
    async def _validate_voice_id(self, voice_id: str) -> bool:
        """Validate if voice ID exists"""
        return voice_id in self.available_voices
    
    async def clone_voice_from_samples(self, sample_files: List[str], 
                                     voice_name: str, description: str = "") -> Optional[str]:
        """
        Clone voice from audio samples
        
        Args:
            sample_files: List of audio file paths
            voice_name: Name for the cloned voice
            description: Description of the voice
        
        Returns:
            Voice ID of cloned voice or None if failed
        """
        try:
            self.logger.info(f"Starting voice cloning with {len(sample_files)} samples", 
                           LogCategory.VOICE)
            
            # Validate sample files
            validated_samples = []
            for sample_file in sample_files:
                if await self._validate_audio_sample(sample_file):
                    validated_samples.append(sample_file)
                else:
                    self.logger.warning(f"Invalid audio sample: {sample_file}", LogCategory.VOICE)
            
            if len(validated_samples) < 1:
                self.logger.error("No valid audio samples provided", LogCategory.VOICE)
                return None
            
            # Clone voice using ElevenLabs
            voice = clone(
                name=voice_name,
                description=description,
                files=validated_samples
            )
            
            self.custom_voice_id = voice.voice_id
            self.voice_id = voice.voice_id
            
            # Update available voices
            self.available_voices[voice.voice_id] = {
                'name': voice_name,
                'category': 'cloned',
                'description': description,
                'preview_url': '',
                'settings': {}
            }
            
            self.logger.info(f"Voice cloned successfully: {voice.voice_id}", LogCategory.VOICE,
                           performance_data={
                               "voice_id": voice.voice_id,
                               "samples_used": len(validated_samples)
                           })
            
            return voice.voice_id
            
        except Exception as e:
            self.logger.error(f"Voice cloning failed: {e}", LogCategory.VOICE, error=e)
            return None
    
    async def _validate_audio_sample(self, file_path: str) -> bool:
        """Validate audio sample for voice cloning"""
        try:
            path = Path(file_path)
            if not path.exists():
                return False
            
            # Check file size (should be reasonable for voice sample)
            file_size = path.stat().st_size
            if file_size < 1024 or file_size > 50 * 1024 * 1024:  # 1KB to 50MB
                return False
            
            # Check file extension
            if path.suffix.lower() not in ['.wav', '.mp3', '.m4a', '.flac']:
                return False
            
            return True
            
        except Exception:
            return False
    
    async def synthesize_speech(self, text: str, voice_id: str = None, 
                              priority: int = 1, use_cache: bool = True) -> Optional[bytes]:
        """
        Synthesize speech from text
        
        Args:
            text: Text to synthesize
            voice_id: Voice ID to use (defaults to configured voice)
            priority: Synthesis priority (1=highest, 5=lowest)
            use_cache: Whether to use cached audio
        
        Returns:
            Audio data as bytes or None if failed
        """
        try:
            # Use configured voice if none specified
            if not voice_id:
                voice_id = self.voice_id
            
            if not voice_id:
                self.logger.error("No voice ID available for synthesis", LogCategory.VOICE)
                return None
            
            # Check cache first
            cache_key = self._get_cache_key(text, voice_id)
            if use_cache and cache_key in self.audio_cache:
                self.logger.debug(f"Using cached audio for text: {text[:50]}...", LogCategory.VOICE)
                return self.audio_cache[cache_key]
            
            # Create synthesis request
            request = SynthesisRequest(
                text=text,
                voice_id=voice_id,
                settings=self.voice_settings.__dict__,
                priority=priority
            )
            
            # Add to queue for processing
            await self.synthesis_queue.put(request)
            
            # For now, process immediately (could be made async with callbacks)
            return await self._synthesize_immediate(text, voice_id)
            
        except Exception as e:
            self.logger.error(f"Speech synthesis failed: {e}", LogCategory.VOICE, error=e)
            return None
    
    async def _synthesize_immediate(self, text: str, voice_id: str) -> Optional[bytes]:
        """Perform immediate synthesis"""
        try:
            start_time = time.time()
            
            # Generate audio using ElevenLabs
            audio = generate(
                text=text,
                voice=Voice(voice_id=voice_id, settings=self.voice_settings),
                model="eleven_multilingual_v2"
            )
            
            # Convert to bytes if needed
            if isinstance(audio, np.ndarray):
                # Convert numpy array to bytes
                audio_bytes = audio.tobytes()
            else:
                audio_bytes = audio
            
            synthesis_time = time.time() - start_time
            
            # Update performance metrics
            self.synthesis_times.append(synthesis_time)
            self.total_characters_synthesized += len(text)
            
            # Cache the result
            cache_key = self._get_cache_key(text, voice_id)
            self._add_to_cache(cache_key, audio_bytes)
            
            self.logger.log_voice_activity("Speech synthesized", {
                "text_length": len(text),
                "synthesis_time": synthesis_time,
                "voice_id": voice_id
            })
            
            return audio_bytes
            
        except Exception as e:
            self.logger.error(f"Immediate synthesis failed: {e}", LogCategory.VOICE, error=e)
            return None
    
    async def _process_synthesis_queue(self):
        """Process synthesis requests from queue"""
        self.logger.info("Starting synthesis queue processor", LogCategory.VOICE)
        
        while True:
            try:
                # Get request from queue
                request = await self.synthesis_queue.get()
                
                if request is None:  # Shutdown signal
                    break
                
                # Process synthesis
                audio_data = await self._synthesize_immediate(request.text, request.voice_id)
                
                # Call callback if provided
                if request.callback and audio_data:
                    try:
                        await request.callback(audio_data)
                    except Exception as e:
                        self.logger.error(f"Synthesis callback failed: {e}", LogCategory.VOICE, error=e)
                
                # Mark task as done
                self.synthesis_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error in synthesis queue processor: {e}", LogCategory.VOICE, error=e)
                await asyncio.sleep(1.0)
    
    def _get_cache_key(self, text: str, voice_id: str) -> str:
        """Generate cache key for text and voice combination"""
        import hashlib
        content = f"{text}:{voice_id}:{self.voice_settings.__dict__}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _add_to_cache(self, key: str, audio_data: bytes):
        """Add audio to cache with size management"""
        # Remove oldest entries if cache is full
        if len(self.audio_cache) >= self.cache_max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.audio_cache))
            del self.audio_cache[oldest_key]
        
        self.audio_cache[key] = audio_data
    
    async def get_voice_info(self, voice_id: str = None) -> Optional[Dict[str, Any]]:
        """Get information about a voice"""
        if not voice_id:
            voice_id = self.voice_id
        
        if voice_id in self.available_voices:
            return self.available_voices[voice_id]
        
        return None
    
    async def list_available_voices(self) -> Dict[str, Dict[str, Any]]:
        """List all available voices"""
        return self.available_voices.copy()
    
    async def test_voice_synthesis(self, test_text: str = "Hello, this is a test of the voice synthesis system.") -> bool:
        """Test voice synthesis functionality"""
        try:
            self.logger.info("Testing voice synthesis...", LogCategory.VOICE)
            
            audio_data = await self.synthesize_speech(test_text)
            
            if audio_data and len(audio_data) > 0:
                self.logger.info("Voice synthesis test successful", LogCategory.VOICE,
                               performance_data={"audio_size": len(audio_data)})
                return True
            else:
                self.logger.error("Voice synthesis test failed - no audio generated", LogCategory.VOICE)
                return False
                
        except Exception as e:
            self.logger.error(f"Voice synthesis test failed: {e}", LogCategory.VOICE, error=e)
            return False
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get voice synthesis performance metrics"""
        avg_synthesis_time = sum(self.synthesis_times) / len(self.synthesis_times) if self.synthesis_times else 0
        
        return {
            "total_syntheses": len(self.synthesis_times),
            "total_characters": self.total_characters_synthesized,
            "total_audio_seconds": self.total_audio_generated,
            "average_synthesis_time": avg_synthesis_time,
            "cache_size": len(self.audio_cache),
            "available_voices": len(self.available_voices),
            "custom_voice_id": self.custom_voice_id,
            "current_voice_id": self.voice_id
        }
    
    async def cleanup(self):
        """Cleanup voice synthesis resources"""
        self.logger.info("Cleaning up ElevenLabs voice cloner...", LogCategory.VOICE)
        
        # Stop synthesis task
        if self._synthesis_task:
            await self.synthesis_queue.put(None)  # Shutdown signal
            self._synthesis_task.cancel()
            try:
                await self._synthesis_task
            except asyncio.CancelledError:
                pass
        
        # Clear cache
        self.audio_cache.clear()
        
        self.is_initialized = False
        self.logger.info("ElevenLabs voice cloner cleanup completed", LogCategory.VOICE)
