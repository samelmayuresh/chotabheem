# 🧠 Emotion AI - Advanced Emotional Intelligence Platform

A comprehensive AI-powered emotional wellness companion that analyzes emotions through voice and text, provides therapeutic support, and tracks emotional patterns over time.

## ✨ Key Features

### 🎙️ **Multi-Modal Emotion Analysis**
- **Voice Analysis**: Upload audio files or record directly for emotion detection
- **Text Analysis**: Analyze written content for emotional patterns
- **Combined Analysis**: Merge voice and text insights for comprehensive understanding

### 🧠 **AI Therapy Assistant**
- **Context-Aware Responses**: Personalized therapeutic support based on your emotional state
- **Multiple Techniques**: CBT, Mindfulness, DBT, and Solution-Focused approaches
- **Session Tracking**: Continuous context awareness across conversations
- **Therapeutic Exercises**: Personalized recommendations based on detected emotions

### 📊 **Advanced Analytics**
- **Emotion Tracking**: Comprehensive mood history with confidence scores
- **Pattern Recognition**: Identify emotional trends and triggers
- **Personalized Insights**: AI-generated recommendations based on your data
- **Interactive Dashboards**: Beautiful visualizations of your emotional journey

### 🎯 **Wellness Features**
- **Breathing Exercises**: Animated guided breathing for stress relief
- **Weather Integration**: Mood suggestions based on current weather
- **GIF Therapy**: Emotion-based GIF recommendations for mood lifting
- **Progress Tracking**: Monitor your emotional growth over time

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd emotion-ai

# Run the setup script
python setup.py

# Copy and configure environment variables
cp .env.template .env
# Edit .env with your API keys

# Start the application
./run_app.sh  # Linux/Mac
# or
run_app.bat   # Windows
```

### Option 2: Manual Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app_enhanced.py
```

### Option 3: Docker

```bash
# Build the image
docker build -t emotion-ai .

# Run the container
docker run -p 8501:8501 emotion-ai
```

## 🔧 Configuration

### Required API Keys

Create a `.env` file with the following keys:

```env
# OpenRouter API Key (for AI therapy features)
OPENROUTER_KEY=your_openrouter_key_here

# Tenor API Key (for GIF features)
TENOR_API_KEY=your_tenor_key_here

# Weather API Key (for weather-based suggestions)
WEATHER_API_KEY=your_weather_key_here

# Supabase Configuration (for data storage)
SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_key_here
```

### Getting API Keys

1. **OpenRouter**: Sign up at [openrouter.ai](https://openrouter.ai) for AI therapy features
2. **Tenor**: Get a free key at [tenor.com/gifapi](https://tenor.com/gifapi) for GIF features
3. **WeatherAPI**: Register at [weatherapi.com](https://weatherapi.com) for weather integration
4. **Supabase**: Create a project at [supabase.com](https://supabase.com) for data storage

## 📱 Usage Guide

### 1. Voice Analysis
- **Upload Audio**: Support for WAV, MP3, M4A, OGG formats
- **Record Live**: Use your microphone for real-time analysis
- **Get Insights**: View emotion breakdown with confidence scores

### 2. Text Analysis
- **Input Text**: Type or paste text up to 5,000 characters
- **Emotion Detection**: Advanced BERT-based emotion classification
- **Contextual Insights**: Understand emotional complexity and patterns

### 3. AI Therapy
- **Start Conversation**: Share what's on your mind
- **Receive Support**: Get personalized therapeutic responses
- **Track Progress**: Monitor your emotional journey across sessions
- **Practice Exercises**: Follow suggested therapeutic activities

### 4. Analytics Dashboard
- **View Trends**: Track emotional patterns over time
- **Understand Patterns**: Identify triggers and positive influences
- **Get Recommendations**: Receive personalized wellness suggestions

## 🏗️ Architecture

### Core Components

```
emotion-ai/
├── app_enhanced.py          # Main application with enhanced UI
├── config.py               # Centralized configuration management
├── utils/
│   ├── emotion_analyzer.py # Advanced emotion analysis engine
│   ├── therapy_assistant.py # AI therapy with context awareness
│   └── database.py         # Enhanced database operations
├── requirements.txt        # Python dependencies
├── Dockerfile             # Container configuration
└── setup.py              # Automated setup script
```

### Technology Stack

- **Frontend**: Streamlit with custom CSS and animations
- **AI Models**: 
  - Whisper (OpenAI) for speech-to-text
  - BERT for emotion classification
  - GPT-4o-mini for therapy responses
- **Database**: Supabase for data persistence
- **Visualization**: Plotly for interactive charts
- **Audio Processing**: librosa for audio analysis
- **Voice Synthesis**: pyttsx3 for text-to-speech

## 🔬 Advanced Features

### Emotion Analysis Engine
- **Multi-source Analysis**: Combines text and voice emotion detection
- **Confidence Scoring**: Provides reliability metrics for each prediction
- **Pattern Recognition**: Identifies emotional complexity and trends
- **Contextual Understanding**: Considers conversation history and user patterns

### AI Therapy System
- **Therapeutic Techniques**: Implements CBT, DBT, Mindfulness approaches
- **Session Continuity**: Maintains context across conversations
- **Progress Tracking**: Monitors therapeutic outcomes
- **Crisis Detection**: Identifies when additional support may be needed

### Analytics Platform
- **Real-time Insights**: Live emotion tracking and analysis
- **Predictive Patterns**: Identifies potential emotional trends
- **Personalized Recommendations**: AI-generated wellness suggestions
- **Export Capabilities**: Download your data for external analysis

## 🛡️ Privacy & Security

- **Local Processing**: Emotion analysis runs locally when possible
- **Encrypted Storage**: All data encrypted in transit and at rest
- **User Control**: Full control over data retention and deletion
- **No Personal Data**: Only emotional patterns are stored, not personal content
- **GDPR Compliant**: Designed with privacy regulations in mind

## 🚀 Performance Optimizations

- **Model Caching**: AI models cached for faster response times
- **Batch Processing**: Efficient handling of multiple requests
- **Resource Management**: Optimized for both CPU and GPU environments
- **Progressive Loading**: Lazy loading of heavy components
- **Session Management**: Efficient state management across interactions

## 🧪 Development

### Running Tests

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Run with coverage
python -m pytest --cov=utils tests/
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Use type hints where possible
- Add docstrings for all functions
- Include unit tests for new features

## 📊 System Requirements

### Minimum Requirements
- **Python**: 3.8+
- **RAM**: 4GB
- **Storage**: 2GB free space
- **Internet**: Required for AI features

### Recommended Requirements
- **Python**: 3.11+
- **RAM**: 8GB+
- **GPU**: CUDA-compatible (optional, for faster processing)
- **Storage**: 5GB free space

## 🔧 Troubleshooting

### Common Issues

**1. Audio Processing Errors**
```bash
# Install FFmpeg
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

**2. Model Loading Issues**
```bash
# Clear cache and reinstall
pip cache purge
pip install --force-reinstall transformers torch
```

**3. Database Connection Problems**
- Check your Supabase credentials in `.env`
- Verify network connectivity
- Ensure Supabase project is active

**4. API Rate Limits**
- Check your API key quotas
- Implement request throttling if needed
- Consider upgrading API plans for higher limits

### Performance Tips

1. **Use GPU**: Install CUDA for faster model inference
2. **Optimize Memory**: Close other applications during heavy processing
3. **Cache Models**: Let the app cache models on first run
4. **Batch Requests**: Process multiple texts together when possible

## 📈 Roadmap

### Upcoming Features
- [ ] Mobile app companion
- [ ] Voice-only interaction mode
- [ ] Group therapy sessions
- [ ] Integration with wearable devices
- [ ] Multi-language support
- [ ] Advanced crisis intervention
- [ ] Therapist dashboard for professionals

### Long-term Vision
- [ ] Real-time emotion detection from video
- [ ] Integration with smart home devices
- [ ] Predictive mental health modeling
- [ ] Community support features
- [ ] Research collaboration tools

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Support

- **Documentation**: Check this README and inline code comments
- **Issues**: Report bugs via GitHub Issues
- **Discussions**: Join community discussions in GitHub Discussions
- **Email**: Contact the maintainers for urgent issues

## 🙏 Acknowledgments

- **OpenAI** for Whisper speech recognition
- **Hugging Face** for transformer models and hosting
- **Streamlit** for the amazing web framework
- **Supabase** for backend infrastructure
- **The open-source community** for countless contributions

---

**Built with ❤️ for mental health and emotional wellness**

*Remember: This tool is designed to support your emotional wellness journey, but it's not a replacement for professional mental health care. If you're experiencing a mental health crisis, please contact a qualified professional or emergency services.*