# 🚀 Streamlit Cloud Deployment Guide

## Quick Deployment Steps

### 1. Prepare Your Repository
- Ensure your code is in a GitHub repository
- Make sure `app.py` is in the root directory or specify the correct path
- Verify `requirements.txt` is present and optimized

### 2. Deploy to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository: `your-username/your-repo-name`
5. Set main file path: `chotabheem/app.py`
6. Click "Deploy!"

### 3. Configure Secrets (Optional)
For enhanced features, add these secrets in Streamlit Cloud:

1. Go to your app settings
2. Click "Secrets"
3. Add the following (replace with your actual keys):

```toml
# Required for mood history
SUPABASE_URL = "https://grhrzxgxazzvnouxczbb.supabase.co"
SUPABASE_KEY = "your_supabase_key"

# Optional for AI therapy
OPENROUTER_KEY = "your_openrouter_key"

# Weather API (free tier)
WEATHER_API_KEY = "24b359c76d994182864153220251507"
```

## Features Available in Cloud Deployment

### ✅ Working Features
- 🎙️ **Voice Analysis**: Upload audio files for emotion detection
- 📝 **Text Analysis**: Analyze text for emotional content
- 🎵 **Music Therapy**: Get mood-based music recommendations
- 🧠 **AI Therapist**: Chat with AI for emotional support (requires API key)
- 📊 **Mood History**: Track emotions over time (requires Supabase)
- 😊 **Face Emotion**: Analyze facial expressions in images
- 🌤️ **Weather Integration**: Mood suggestions based on weather

### ⚠️ Limited Features
- 🎤 **Voice Synthesis**: Disabled for cloud compatibility
- 🎭 **GIF Generation**: Requires API key configuration

## Troubleshooting

### Common Issues

1. **App won't start**
   - Check that `requirements.txt` has correct package versions
   - Ensure no local file dependencies (like `key.txt`)

2. **Missing features**
   - Add required API keys to Streamlit secrets
   - Check that all imports are available

3. **Performance issues**
   - Large models may take time to load initially
   - Consider using lighter model alternatives

### Performance Optimization

The app is optimized for Streamlit Cloud with:
- Cached model loading
- Optimized package versions
- Removed heavy dependencies
- Error handling for missing components

## API Keys Setup

### OpenRouter (AI Therapy)
1. Sign up at [openrouter.ai](https://openrouter.ai)
2. Get your API key
3. Add as `OPENROUTER_KEY` in Streamlit secrets

### Supabase (Mood History)
1. Create account at [supabase.com](https://supabase.com)
2. Create new project
3. Get URL and anon key from project settings
4. Add as `SUPABASE_URL` and `SUPABASE_KEY`

### Weather API
- Free tier available at [weatherapi.com](https://weatherapi.com)
- Add as `WEATHER_API_KEY`

## Support

If you encounter issues:
1. Check Streamlit Cloud logs
2. Verify all secrets are properly configured
3. Ensure repository is public or properly shared
4. Check that all required files are committed

## App URL
Once deployed, your app will be available at:
`https://your-app-name.streamlit.app`

Share this URL with others to let them use your emotion detection app!