"""
Configuration Management for Darrell Agent
Handles all configuration settings for voice synthesis, meeting automation, and AI models
"""

import os
import yaml
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class VoiceConfig:
    """Voice synthesis configuration"""
    elevenlabs_api_key: str = ""
    voice_id: str = ""
    voice_stability: float = 0.75
    voice_similarity_boost: float = 0.75
    voice_style: float = 0.0
    voice_use_speaker_boost: bool = True
    sample_rate: int = 22050
    audio_format: str = "mp3"
    chunk_size: int = 1024
    voice_samples_path: str = "voice_samples/"
    voice_model_path: str = "voice_models/"


@dataclass
class MeetingConfig:
    """Meeting automation configuration"""
    zoom_executable_path: str = ""
    zoom_data_dir: str = ""
    auto_join_audio: bool = True
    auto_join_video: bool = False
    default_microphone: str = ""
    default_speaker: str = ""
    meeting_timeout: int = 7200  # 2 hours
    response_delay_min: float = 1.0
    response_delay_max: float = 3.0
    participation_frequency: float = 0.3  # 30% participation rate
    politeness_level: str = "professional"  # casual, professional, formal


@dataclass
class AIModelConfig:
    """AI model configuration"""
    florence_model_path: str = "microsoft/Florence-2-large"
    florence_device: str = "auto"
    florence_precision: str = "fp16"
    xlam_model: str = "Salesforce/xLAM-2-1b-fc-r"
    xlam_base_url: str = "http://localhost:8000/v1/"
    conversation_model: str = "gpt-4"
    conversation_api_key: str = ""
    whisper_model: str = "base"
    embedding_dim: int = 1024
    max_context_length: int = 4096


@dataclass
class SecurityConfig:
    """Security and privacy configuration"""
    encrypt_credentials: bool = True
    credential_storage_path: str = "credentials/"
    session_timeout: int = 3600
    max_login_attempts: int = 3
    enable_meeting_recording: bool = False
    data_retention_days: int = 7
    privacy_mode: bool = True
    audit_logging: bool = True


@dataclass
class AutomationConfig:
    """Automation behavior configuration"""
    screenshot_interval: float = 0.5
    ui_detection_confidence: float = 0.8
    click_delay: float = 0.1
    typing_delay: float = 0.05
    window_focus_timeout: float = 5.0
    retry_attempts: int = 3
    error_recovery_enabled: bool = True
    backup_automation_method: str = "coordinates"


@dataclass
class DarrellConfig:
    """Main configuration class for Darrell Agent"""
    
    # Sub-configurations
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    meeting: MeetingConfig = field(default_factory=MeetingConfig)
    ai_models: AIModelConfig = field(default_factory=AIModelConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    automation: AutomationConfig = field(default_factory=AutomationConfig)
    
    # General settings
    agent_name: str = "Darrell"
    debug_mode: bool = False
    log_level: str = "INFO"
    log_file: str = "darrell_agent.log"
    config_version: str = "1.0.0"
    
    # Paths
    base_path: str = field(default_factory=lambda: str(Path.cwd()))
    data_path: str = "data/"
    models_path: str = "models/"
    logs_path: str = "logs/"
    temp_path: str = "temp/"
    
    @classmethod
    def from_file(cls, config_path: str) -> 'DarrellConfig':
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            return cls.from_dict(config_data)
        except Exception as e:
            raise ValueError(f"Failed to load configuration from {config_path}: {e}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DarrellConfig':
        """Create configuration from dictionary"""
        config = cls()
        
        # Update voice config
        if 'voice' in config_dict:
            for key, value in config_dict['voice'].items():
                if hasattr(config.voice, key):
                    setattr(config.voice, key, value)
        
        # Update meeting config
        if 'meeting' in config_dict:
            for key, value in config_dict['meeting'].items():
                if hasattr(config.meeting, key):
                    setattr(config.meeting, key, value)
        
        # Update AI models config
        if 'ai_models' in config_dict:
            for key, value in config_dict['ai_models'].items():
                if hasattr(config.ai_models, key):
                    setattr(config.ai_models, key, value)
        
        # Update security config
        if 'security' in config_dict:
            for key, value in config_dict['security'].items():
                if hasattr(config.security, key):
                    setattr(config.security, key, value)
        
        # Update automation config
        if 'automation' in config_dict:
            for key, value in config_dict['automation'].items():
                if hasattr(config.automation, key):
                    setattr(config.automation, key, value)
        
        # Update general settings
        for key in ['agent_name', 'debug_mode', 'log_level', 'log_file', 'config_version',
                   'base_path', 'data_path', 'models_path', 'logs_path', 'temp_path']:
            if key in config_dict:
                setattr(config, key, config_dict[key])
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'voice': {
                'elevenlabs_api_key': self.voice.elevenlabs_api_key,
                'voice_id': self.voice.voice_id,
                'voice_stability': self.voice.voice_stability,
                'voice_similarity_boost': self.voice.voice_similarity_boost,
                'voice_style': self.voice.voice_style,
                'voice_use_speaker_boost': self.voice.voice_use_speaker_boost,
                'sample_rate': self.voice.sample_rate,
                'audio_format': self.voice.audio_format,
                'chunk_size': self.voice.chunk_size,
                'voice_samples_path': self.voice.voice_samples_path,
                'voice_model_path': self.voice.voice_model_path
            },
            'meeting': {
                'zoom_executable_path': self.meeting.zoom_executable_path,
                'zoom_data_dir': self.meeting.zoom_data_dir,
                'auto_join_audio': self.meeting.auto_join_audio,
                'auto_join_video': self.meeting.auto_join_video,
                'default_microphone': self.meeting.default_microphone,
                'default_speaker': self.meeting.default_speaker,
                'meeting_timeout': self.meeting.meeting_timeout,
                'response_delay_min': self.meeting.response_delay_min,
                'response_delay_max': self.meeting.response_delay_max,
                'participation_frequency': self.meeting.participation_frequency,
                'politeness_level': self.meeting.politeness_level
            },
            'ai_models': {
                'florence_model_path': self.ai_models.florence_model_path,
                'florence_device': self.ai_models.florence_device,
                'florence_precision': self.ai_models.florence_precision,
                'xlam_model': self.ai_models.xlam_model,
                'xlam_base_url': self.ai_models.xlam_base_url,
                'conversation_model': self.ai_models.conversation_model,
                'conversation_api_key': self.ai_models.conversation_api_key,
                'whisper_model': self.ai_models.whisper_model,
                'embedding_dim': self.ai_models.embedding_dim,
                'max_context_length': self.ai_models.max_context_length
            },
            'security': {
                'encrypt_credentials': self.security.encrypt_credentials,
                'credential_storage_path': self.security.credential_storage_path,
                'session_timeout': self.security.session_timeout,
                'max_login_attempts': self.security.max_login_attempts,
                'enable_meeting_recording': self.security.enable_meeting_recording,
                'data_retention_days': self.security.data_retention_days,
                'privacy_mode': self.security.privacy_mode,
                'audit_logging': self.security.audit_logging
            },
            'automation': {
                'screenshot_interval': self.automation.screenshot_interval,
                'ui_detection_confidence': self.automation.ui_detection_confidence,
                'click_delay': self.automation.click_delay,
                'typing_delay': self.automation.typing_delay,
                'window_focus_timeout': self.automation.window_focus_timeout,
                'retry_attempts': self.automation.retry_attempts,
                'error_recovery_enabled': self.automation.error_recovery_enabled,
                'backup_automation_method': self.automation.backup_automation_method
            },
            'agent_name': self.agent_name,
            'debug_mode': self.debug_mode,
            'log_level': self.log_level,
            'log_file': self.log_file,
            'config_version': self.config_version,
            'base_path': self.base_path,
            'data_path': self.data_path,
            'models_path': self.models_path,
            'logs_path': self.logs_path,
            'temp_path': self.temp_path
        }
    
    def save_to_file(self, config_path: str) -> None:
        """Save configuration to YAML file"""
        try:
            config_dict = self.to_dict()
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        except Exception as e:
            raise ValueError(f"Failed to save configuration to {config_path}: {e}")
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        # Validate voice configuration
        if not self.voice.elevenlabs_api_key:
            errors.append("ElevenLabs API key is required")
        
        # Validate paths
        for path_attr in ['data_path', 'models_path', 'logs_path', 'temp_path']:
            path = getattr(self, path_attr)
            if not os.path.isabs(path):
                full_path = os.path.join(self.base_path, path)
                if not os.path.exists(full_path):
                    try:
                        os.makedirs(full_path, exist_ok=True)
                    except Exception as e:
                        errors.append(f"Cannot create directory {full_path}: {e}")
        
        # Validate AI model settings
        if self.ai_models.embedding_dim <= 0:
            errors.append("Embedding dimension must be positive")
        
        if self.ai_models.max_context_length <= 0:
            errors.append("Max context length must be positive")
        
        # Validate meeting settings
        if self.meeting.meeting_timeout <= 0:
            errors.append("Meeting timeout must be positive")
        
        if not (0.0 <= self.meeting.participation_frequency <= 1.0):
            errors.append("Participation frequency must be between 0.0 and 1.0")
        
        return errors
    
    def get_full_path(self, relative_path: str) -> str:
        """Get full path from relative path"""
        if os.path.isabs(relative_path):
            return relative_path
        return os.path.join(self.base_path, relative_path)


def load_default_config() -> DarrellConfig:
    """Load default configuration"""
    return DarrellConfig()


def load_config_from_env() -> DarrellConfig:
    """Load configuration from environment variables"""
    config = DarrellConfig()
    
    # Voice configuration from environment
    if os.getenv('ELEVENLABS_API_KEY'):
        config.voice.elevenlabs_api_key = os.getenv('ELEVENLABS_API_KEY')
    
    if os.getenv('ELEVENLABS_VOICE_ID'):
        config.voice.voice_id = os.getenv('ELEVENLABS_VOICE_ID')
    
    # AI model configuration from environment
    if os.getenv('OPENAI_API_KEY'):
        config.ai_models.conversation_api_key = os.getenv('OPENAI_API_KEY')
    
    if os.getenv('XLAM_BASE_URL'):
        config.ai_models.xlam_base_url = os.getenv('XLAM_BASE_URL')
    
    # Meeting configuration from environment
    if os.getenv('ZOOM_EXECUTABLE_PATH'):
        config.meeting.zoom_executable_path = os.getenv('ZOOM_EXECUTABLE_PATH')
    
    # Debug mode from environment
    if os.getenv('DARRELL_DEBUG'):
        config.debug_mode = os.getenv('DARRELL_DEBUG').lower() in ['true', '1', 'yes']
    
    return config
