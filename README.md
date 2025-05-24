# 🤖 Darrell Agent - Sophisticated AI for Autonomous Zoom Meeting Attendance

**Darrell** is an advanced AI agent designed to autonomously attend Zoom meetings on your behalf, featuring voice cloning, natural conversation capabilities, and intelligent meeting participation using cutting-edge AI technologies.

## 🌟 Key Features

### 🎙️ Advanced Voice Synthesis
- **ElevenLabs Integration**: High-fidelity voice cloning from audio samples
- **Real-time Speech Generation**: Natural, human-like speech synthesis
- **Emotional Modulation**: Context-aware emotional expression
- **Voice Customization**: Personalized voice characteristics and speaking patterns

### 🤖 Intelligent Meeting Participation
- **xLAM-Powered Decision Making**: Advanced action planning and decision making
- **Florence-2 Computer Vision**: Sophisticated UI detection and screen understanding
- **Natural Conversation Flow**: Context-aware responses and appropriate participation timing
- **Meeting Context Understanding**: Agenda analysis and topic-aware contributions

### 🔒 Security & Privacy
- **Encrypted Credential Storage**: Secure handling of login credentials and API keys
- **Session Management**: Secure session handling with timeout controls
- **Privacy Protection**: Data sanitization and retention policies
- **Audit Logging**: Comprehensive security and activity logging

### 🎯 Automation Capabilities
- **Zoom Integration**: Automated login, meeting joining, and UI navigation
- **Microphone Control**: Intelligent muting/unmuting for natural conversation
- **Screen Analysis**: Real-time meeting state detection and participant recognition
- **Error Recovery**: Robust error handling and recovery mechanisms

## 🏗️ Architecture

```
Darrell Agent Architecture:
├── Core Engine/
│   ├── DarrellCore - Main orchestration
│   ├── MeetingCoordinator - Meeting lifecycle management
│   └── ConversationEngine - Natural conversation AI
├── Voice Pipeline/
│   ├── ElevenLabs Integration - Voice synthesis
│   ├── AudioProcessor - Audio I/O management
│   └── SpeechRecognition - Speech-to-text processing
├── Vision System/
│   ├── Florence-2 Integration - Computer vision
│   ├── ZoomUIDetector - UI element detection
│   └── MeetingAnalyzer - Meeting content analysis
├── Automation Layer/
│   ├── ZoomController - Meeting automation
│   ├── MeetingNavigator - Navigation control
│   └── CredentialManager - Secure authentication
└── Security & Utils/
    ├── SecurityManager - Security framework
    ├── ConfigManager - Configuration handling
    └── Logger - Advanced logging system
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- ElevenLabs API key
- OpenAI API key (for conversation AI)
- Zoom desktop application
- Git (for Florence-2 model cloning)

### Installation

1. **Clone and Setup**:
```bash
git clone https://github.com/your-repo/darrell-agent
cd darrell-agent
python setup_darrell.py
```

2. **Configure Environment**:
```bash
cp .env.template .env
# Edit .env with your API keys and settings
```

3. **Test Installation**:
```bash
python darrell_demo.py
```

### Configuration

Edit `darrell_config.yaml` to customize:

```yaml
# Voice Configuration
voice:
  elevenlabs_api_key: "your_api_key_here"
  voice_id: "your_voice_id"
  voice_stability: 0.75
  voice_similarity_boost: 0.75

# Meeting Configuration  
meeting:
  auto_join_audio: true
  auto_join_video: false
  participation_frequency: 0.3  # 30% participation rate
  politeness_level: "professional"

# AI Models
ai_models:
  florence_model_path: "microsoft/Florence-2-large"
  conversation_model: "gpt-4"
  xlam_model: "Salesforce/xLAM-2-1b-fc-r"
```

## 📖 Usage

### Basic Meeting Attendance

```python
import asyncio
from darrell_agent import DarrellAgent
from darrell_agent.utils.config import DarrellConfig

async def attend_meeting():
    # Load configuration
    config = DarrellConfig.from_file("darrell_config.yaml")
    
    # Create and initialize agent
    darrell = DarrellAgent(config)
    await darrell.initialize()
    
    # Join meeting
    success = await darrell.join_meeting(
        meeting_url="https://zoom.us/j/123456789",
        meeting_password="password123",
        duration_minutes=60,
        agenda="Weekly team standup meeting"
    )
    
    if success:
        print("✅ Successfully joined meeting!")
        
        # Agent will automatically participate
        # Monitor status
        while True:
            status = await darrell.get_status()
            print(f"Meeting status: {status}")
            await asyncio.sleep(30)
    
    # Cleanup
    await darrell.stop()

# Run the meeting attendance
asyncio.run(attend_meeting())
```

### Voice Cloning Setup

1. **Prepare Voice Samples**:
   - Record 10-15 minutes of high-quality audio
   - Use clear, natural speech
   - Include various emotions and tones
   - Save as WAV or MP3 files in `voice_samples/`

2. **Clone Voice**:
```python
from darrell_agent.voice import ElevenLabsVoiceCloner

async def setup_voice():
    cloner = ElevenLabsVoiceCloner(api_key="your_key")
    await cloner.initialize()
    
    # Clone voice from samples
    voice_id = await cloner.clone_voice_from_samples(
        sample_files=["voice_samples/sample1.wav", "voice_samples/sample2.wav"],
        voice_name="My Voice Clone",
        description="Professional meeting voice"
    )
    
    print(f"Voice cloned successfully: {voice_id}")
```

### Advanced Configuration

```python
# Custom meeting behavior
config.meeting.response_delay_min = 2.0  # Minimum response delay
config.meeting.response_delay_max = 5.0  # Maximum response delay
config.meeting.participation_frequency = 0.4  # 40% participation

# Security settings
config.security.encrypt_credentials = True
config.security.session_timeout = 3600  # 1 hour
config.security.data_retention_days = 7

# AI model settings
config.ai_models.florence_device = "cuda"  # Use GPU
config.ai_models.florence_precision = "fp16"  # Half precision
```

## 🔧 Components

### Core Components

- **DarrellAgent**: Main orchestration and lifecycle management
- **MeetingCoordinator**: Meeting-specific coordination and participation
- **ConversationEngine**: Natural conversation and response generation

### Voice System

- **ElevenLabsVoiceCloner**: Advanced voice synthesis and cloning
- **AudioProcessor**: Real-time audio processing and microphone control
- **SpeechRecognitionEngine**: Speech-to-text for meeting understanding

### Vision System

- **Florence2Integration**: Computer vision using Microsoft's Florence-2-large
- **ZoomUIDetector**: Zoom interface detection and navigation
- **MeetingAnalyzer**: Meeting content and participant analysis

### Automation

- **ZoomController**: Zoom application automation and control
- **MeetingNavigator**: Meeting navigation and interaction
- **CredentialManager**: Secure credential storage and management

## 📊 Performance Metrics

Darrell Agent provides comprehensive performance monitoring:

```python
# Get performance metrics
status = await darrell.get_status()
print(f"Voice synthesis ready: {status['voice_ready']}")
print(f"AI models ready: {status['ai_models_ready']}")
print(f"Automation ready: {status['automation_ready']}")

# Get detailed metrics
voice_metrics = await darrell.voice_cloner.get_performance_metrics()
vision_metrics = await darrell.florence_integration.get_performance_metrics()
```

## 🛡️ Security Features

### Credential Protection
- Encrypted storage of API keys and passwords
- System keyring integration
- Secure session management

### Privacy Controls
- Data sanitization for sensitive information
- Configurable data retention policies
- Audit logging for compliance

### Access Control
- Session-based authentication
- Permission-based operation validation
- Rate limiting and lockout protection

## 🔍 Troubleshooting

### Common Issues

1. **Voice Synthesis Fails**:
   - Check ElevenLabs API key
   - Verify voice ID exists
   - Check internet connection

2. **Meeting Join Fails**:
   - Verify Zoom executable path
   - Check meeting URL format
   - Ensure Zoom is updated

3. **Florence-2 Model Issues**:
   - Check GPU memory availability
   - Verify model download completed
   - Try CPU mode if GPU fails

### Debug Mode

Enable debug logging:
```python
config.debug_mode = True
config.log_level = "DEBUG"
```

### Log Analysis

Check logs in the `logs/` directory:
- `darrell_core.log` - Main agent logs
- `darrell_audit_*.log` - Audit trails
- `darrell_voice.log` - Voice system logs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE.txt) file for details.

## ⚠️ Ethical Considerations

### Transparency
- Inform meeting participants when AI attendance is used
- Comply with recording and consent regulations
- Respect privacy and confidentiality

### Professional Use
- Use appropriate response boundaries
- Implement escalation for complex decisions
- Maintain professional standards

### Data Privacy
- Protect meeting content and participant information
- Follow data retention policies
- Implement proper security measures

## 🆘 Support

- **Documentation**: Check the `docs/` directory
- **Issues**: Report bugs on GitHub Issues
- **Discussions**: Join community discussions
- **Email**: Contact support team

## 🔮 Roadmap

### Upcoming Features
- Multi-language support
- Advanced emotion recognition
- Integration with calendar systems
- Mobile device support
- Custom AI model training

### Performance Improvements
- Faster voice synthesis
- Reduced memory usage
- Better error recovery
- Enhanced UI detection

---

**Darrell Agent** - Bringing sophisticated AI to meeting automation while maintaining the highest standards of security, privacy, and professional conduct.
