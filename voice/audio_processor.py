"""
Audio Processing System for Darrell Agent
Handles real-time audio I/O, microphone control, and audio stream management
"""

import asyncio
import threading
import time
import wave
import io
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from pathlib import Path
import numpy as np

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    print("Warning: PyAudio not available. Install with: pip install pyaudio")
    PYAUDIO_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    print("Warning: SoundFile not available. Install with: pip install soundfile")
    SOUNDFILE_AVAILABLE = False

from ..utils.logger import DarrellLogger, LogCategory


@dataclass
class AudioDevice:
    """Audio device information"""
    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_sample_rate: float
    is_default_input: bool = False
    is_default_output: bool = False


@dataclass
class AudioStream:
    """Audio stream configuration"""
    device_index: int
    channels: int
    sample_rate: int
    chunk_size: int
    format: int
    input: bool = False
    output: bool = False


class AudioProcessor:
    """
    Advanced audio processing system for Darrell Agent
    Handles microphone input, speaker output, and real-time audio processing
    """
    
    def __init__(self, sample_rate: int = 22050, chunk_size: int = 1024,
                 default_microphone: str = "", default_speaker: str = ""):
        """
        Initialize audio processor
        
        Args:
            sample_rate: Audio sample rate
            chunk_size: Audio chunk size for processing
            default_microphone: Preferred microphone device name
            default_speaker: Preferred speaker device name
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.default_microphone = default_microphone
        self.default_speaker = default_speaker
        self.logger = DarrellLogger("AudioProcessor")
        
        # PyAudio instance
        self.pyaudio_instance = None
        
        # Audio devices
        self.input_devices = []
        self.output_devices = []
        self.selected_input_device = None
        self.selected_output_device = None
        
        # Audio streams
        self.input_stream = None
        self.output_stream = None
        self.recording_stream = None
        
        # Audio buffers
        self.input_buffer = []
        self.output_buffer = []
        self.recording_buffer = []
        
        # Stream control
        self.is_recording = False
        self.is_playing = False
        self.is_monitoring = False
        
        # Callbacks
        self.audio_callback = None
        self.recording_callback = None
        
        # Threading
        self._audio_thread = None
        self._stop_event = threading.Event()
        
        # Performance tracking
        self.audio_stats = {
            'total_input_frames': 0,
            'total_output_frames': 0,
            'buffer_overruns': 0,
            'buffer_underruns': 0,
            'average_latency': 0.0
        }
        
        self.is_initialized = False
        
        if not PYAUDIO_AVAILABLE:
            self.logger.error("PyAudio not available", LogCategory.VOICE)
            raise ImportError("PyAudio required for audio processing")
    
    async def initialize(self) -> bool:
        """Initialize audio processor"""
        try:
            self.logger.info("Initializing audio processor...", LogCategory.VOICE)
            
            # Initialize PyAudio
            self.pyaudio_instance = pyaudio.PyAudio()
            
            # Discover audio devices
            await self._discover_audio_devices()
            
            # Select default devices
            await self._select_default_devices()
            
            # Test audio system
            test_success = await self._test_audio_system()
            
            if test_success:
                self.is_initialized = True
                self.logger.info("Audio processor initialized successfully", LogCategory.VOICE,
                               performance_data={
                                   "sample_rate": self.sample_rate,
                                   "chunk_size": self.chunk_size,
                                   "input_device": self.selected_input_device.name if self.selected_input_device else None,
                                   "output_device": self.selected_output_device.name if self.selected_output_device else None
                               })
                return True
            else:
                self.logger.error("Audio system test failed", LogCategory.VOICE)
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to initialize audio processor: {e}", LogCategory.VOICE, error=e)
            return False
    
    async def _discover_audio_devices(self):
        """Discover available audio devices"""
        try:
            device_count = self.pyaudio_instance.get_device_count()
            default_input = self.pyaudio_instance.get_default_input_device_info()
            default_output = self.pyaudio_instance.get_default_output_device_info()
            
            for i in range(device_count):
                try:
                    device_info = self.pyaudio_instance.get_device_info_by_index(i)
                    
                    device = AudioDevice(
                        index=i,
                        name=device_info['name'],
                        max_input_channels=device_info['maxInputChannels'],
                        max_output_channels=device_info['maxOutputChannels'],
                        default_sample_rate=device_info['defaultSampleRate'],
                        is_default_input=(i == default_input['index']),
                        is_default_output=(i == default_output['index'])
                    )
                    
                    if device.max_input_channels > 0:
                        self.input_devices.append(device)
                    
                    if device.max_output_channels > 0:
                        self.output_devices.append(device)
                        
                except Exception as e:
                    self.logger.warning(f"Failed to get info for device {i}: {e}", LogCategory.VOICE)
            
            self.logger.info(f"Discovered {len(self.input_devices)} input and {len(self.output_devices)} output devices", 
                           LogCategory.VOICE)
            
        except Exception as e:
            self.logger.error(f"Failed to discover audio devices: {e}", LogCategory.VOICE, error=e)
    
    async def _select_default_devices(self):
        """Select default input and output devices"""
        try:
            # Select input device
            if self.default_microphone:
                # Try to find specified microphone
                for device in self.input_devices:
                    if self.default_microphone.lower() in device.name.lower():
                        self.selected_input_device = device
                        break
            
            if not self.selected_input_device:
                # Use default input device
                for device in self.input_devices:
                    if device.is_default_input:
                        self.selected_input_device = device
                        break
            
            if not self.selected_input_device and self.input_devices:
                # Use first available input device
                self.selected_input_device = self.input_devices[0]
            
            # Select output device
            if self.default_speaker:
                # Try to find specified speaker
                for device in self.output_devices:
                    if self.default_speaker.lower() in device.name.lower():
                        self.selected_output_device = device
                        break
            
            if not self.selected_output_device:
                # Use default output device
                for device in self.output_devices:
                    if device.is_default_output:
                        self.selected_output_device = device
                        break
            
            if not self.selected_output_device and self.output_devices:
                # Use first available output device
                self.selected_output_device = self.output_devices[0]
            
            self.logger.info("Audio devices selected", LogCategory.VOICE,
                           performance_data={
                               "input_device": self.selected_input_device.name if self.selected_input_device else "None",
                               "output_device": self.selected_output_device.name if self.selected_output_device else "None"
                           })
            
        except Exception as e:
            self.logger.error(f"Failed to select default devices: {e}", LogCategory.VOICE, error=e)
    
    async def _test_audio_system(self) -> bool:
        """Test audio system functionality"""
        try:
            self.logger.info("Testing audio system...", LogCategory.VOICE)
            
            # Test input device
            if self.selected_input_device:
                try:
                    test_stream = self.pyaudio_instance.open(
                        format=pyaudio.paInt16,
                        channels=1,
                        rate=self.sample_rate,
                        input=True,
                        input_device_index=self.selected_input_device.index,
                        frames_per_buffer=self.chunk_size
                    )
                    
                    # Read a small amount of data
                    test_data = test_stream.read(self.chunk_size, exception_on_overflow=False)
                    test_stream.stop_stream()
                    test_stream.close()
                    
                    if test_data:
                        self.logger.info("Input device test passed", LogCategory.VOICE)
                    else:
                        self.logger.warning("Input device test failed - no data", LogCategory.VOICE)
                        
                except Exception as e:
                    self.logger.warning(f"Input device test failed: {e}", LogCategory.VOICE)
            
            # Test output device
            if self.selected_output_device:
                try:
                    test_stream = self.pyaudio_instance.open(
                        format=pyaudio.paInt16,
                        channels=1,
                        rate=self.sample_rate,
                        output=True,
                        output_device_index=self.selected_output_device.index,
                        frames_per_buffer=self.chunk_size
                    )
                    
                    # Generate and play a brief silence
                    silence = np.zeros(self.chunk_size, dtype=np.int16).tobytes()
                    test_stream.write(silence)
                    test_stream.stop_stream()
                    test_stream.close()
                    
                    self.logger.info("Output device test passed", LogCategory.VOICE)
                    
                except Exception as e:
                    self.logger.warning(f"Output device test failed: {e}", LogCategory.VOICE)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Audio system test failed: {e}", LogCategory.VOICE, error=e)
            return False
    
    async def start_recording(self, callback: Callable = None) -> bool:
        """
        Start audio recording
        
        Args:
            callback: Optional callback for audio data
        
        Returns:
            Success status
        """
        try:
            if self.is_recording:
                self.logger.warning("Recording already in progress", LogCategory.VOICE)
                return False
            
            if not self.selected_input_device:
                self.logger.error("No input device available", LogCategory.VOICE)
                return False
            
            self.recording_callback = callback
            self.recording_buffer.clear()
            
            # Create recording stream
            self.recording_stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.selected_input_device.index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._recording_callback
            )
            
            self.recording_stream.start_stream()
            self.is_recording = True
            
            self.logger.info("Audio recording started", LogCategory.VOICE)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start recording: {e}", LogCategory.VOICE, error=e)
            return False
    
    def _recording_callback(self, in_data, frame_count, time_info, status):
        """PyAudio recording callback"""
        try:
            if status:
                self.audio_stats['buffer_overruns'] += 1
            
            # Add to buffer
            self.recording_buffer.append(in_data)
            self.audio_stats['total_input_frames'] += frame_count
            
            # Call user callback if provided
            if self.recording_callback:
                try:
                    self.recording_callback(in_data, frame_count)
                except Exception as e:
                    self.logger.error(f"Recording callback error: {e}", LogCategory.VOICE, error=e)
            
            return (None, pyaudio.paContinue)
            
        except Exception as e:
            self.logger.error(f"Recording callback error: {e}", LogCategory.VOICE, error=e)
            return (None, pyaudio.paAbort)
    
    async def stop_recording(self) -> Optional[bytes]:
        """
        Stop audio recording
        
        Returns:
            Recorded audio data as bytes
        """
        try:
            if not self.is_recording:
                return None
            
            self.is_recording = False
            
            if self.recording_stream:
                self.recording_stream.stop_stream()
                self.recording_stream.close()
                self.recording_stream = None
            
            # Combine recorded data
            if self.recording_buffer:
                audio_data = b''.join(self.recording_buffer)
                self.recording_buffer.clear()
                
                self.logger.info(f"Recording stopped, captured {len(audio_data)} bytes", LogCategory.VOICE)
                return audio_data
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to stop recording: {e}", LogCategory.VOICE, error=e)
            return None
    
    async def play_audio(self, audio_data: bytes, sample_rate: int = None) -> bool:
        """
        Play audio data
        
        Args:
            audio_data: Audio data to play
            sample_rate: Sample rate of audio data
        
        Returns:
            Success status
        """
        try:
            if not self.selected_output_device:
                self.logger.error("No output device available", LogCategory.VOICE)
                return False
            
            if not sample_rate:
                sample_rate = self.sample_rate
            
            # Create output stream
            output_stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                output=True,
                output_device_index=self.selected_output_device.index,
                frames_per_buffer=self.chunk_size
            )
            
            # Play audio in chunks
            chunk_size_bytes = self.chunk_size * 2  # 2 bytes per sample for paInt16
            for i in range(0, len(audio_data), chunk_size_bytes):
                chunk = audio_data[i:i + chunk_size_bytes]
                output_stream.write(chunk)
            
            output_stream.stop_stream()
            output_stream.close()
            
            self.logger.log_voice_activity("Audio played", {
                "data_size": len(audio_data),
                "sample_rate": sample_rate
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to play audio: {e}", LogCategory.VOICE, error=e)
            return False
    
    async def save_audio(self, audio_data: bytes, file_path: str, sample_rate: int = None) -> bool:
        """
        Save audio data to file
        
        Args:
            audio_data: Audio data to save
            file_path: Output file path
            sample_rate: Sample rate of audio data
        
        Returns:
            Success status
        """
        try:
            if not SOUNDFILE_AVAILABLE:
                # Fallback to wave module
                return await self._save_audio_wave(audio_data, file_path, sample_rate)
            
            if not sample_rate:
                sample_rate = self.sample_rate
            
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Save using soundfile
            sf.write(file_path, audio_array, sample_rate)
            
            self.logger.info(f"Audio saved to {file_path}", LogCategory.VOICE)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save audio: {e}", LogCategory.VOICE, error=e)
            return False
    
    async def _save_audio_wave(self, audio_data: bytes, file_path: str, sample_rate: int = None) -> bool:
        """Save audio using wave module (fallback)"""
        try:
            if not sample_rate:
                sample_rate = self.sample_rate
            
            with wave.open(file_path, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 2 bytes per sample
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save audio with wave: {e}", LogCategory.VOICE, error=e)
            return False
    
    async def load_audio(self, file_path: str) -> Optional[Tuple[bytes, int]]:
        """
        Load audio from file
        
        Args:
            file_path: Audio file path
        
        Returns:
            Tuple of (audio_data, sample_rate) or None if failed
        """
        try:
            if SOUNDFILE_AVAILABLE:
                # Use soundfile
                audio_array, sample_rate = sf.read(file_path, dtype=np.int16)
                audio_data = audio_array.tobytes()
                return audio_data, sample_rate
            else:
                # Use wave module
                with wave.open(file_path, 'rb') as wav_file:
                    sample_rate = wav_file.getframerate()
                    audio_data = wav_file.readframes(wav_file.getnframes())
                    return audio_data, sample_rate
                    
        except Exception as e:
            self.logger.error(f"Failed to load audio: {e}", LogCategory.VOICE, error=e)
            return None
    
    async def get_audio_devices(self) -> Dict[str, List[AudioDevice]]:
        """Get available audio devices"""
        return {
            'input_devices': self.input_devices.copy(),
            'output_devices': self.output_devices.copy()
        }
    
    async def set_input_device(self, device_index: int) -> bool:
        """Set input device by index"""
        try:
            for device in self.input_devices:
                if device.index == device_index:
                    self.selected_input_device = device
                    self.logger.info(f"Input device set to: {device.name}", LogCategory.VOICE)
                    return True
            
            self.logger.error(f"Input device index {device_index} not found", LogCategory.VOICE)
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to set input device: {e}", LogCategory.VOICE, error=e)
            return False
    
    async def set_output_device(self, device_index: int) -> bool:
        """Set output device by index"""
        try:
            for device in self.output_devices:
                if device.index == device_index:
                    self.selected_output_device = device
                    self.logger.info(f"Output device set to: {device.name}", LogCategory.VOICE)
                    return True
            
            self.logger.error(f"Output device index {device_index} not found", LogCategory.VOICE)
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to set output device: {e}", LogCategory.VOICE, error=e)
            return False
    
    async def get_audio_stats(self) -> Dict[str, Any]:
        """Get audio processing statistics"""
        return self.audio_stats.copy()
    
    async def cleanup(self):
        """Cleanup audio processor resources"""
        self.logger.info("Cleaning up audio processor...", LogCategory.VOICE)
        
        # Stop recording if active
        if self.is_recording:
            await self.stop_recording()
        
        # Close streams
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
        
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
        
        if self.recording_stream:
            self.recording_stream.stop_stream()
            self.recording_stream.close()
        
        # Terminate PyAudio
        if self.pyaudio_instance:
            self.pyaudio_instance.terminate()
        
        self.is_initialized = False
        self.logger.info("Audio processor cleanup completed", LogCategory.VOICE)
