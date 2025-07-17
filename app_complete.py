# app_complete.py - Complete Emotion AI with all original features
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import requests
import json

# Import our enhanced modules
from config import api_config, app_config, validate_api_keys, get_model_config
from utils.emotion_analyzer import EmotionAnalyzer
from utils.therapy_assistant import TherapyAssistant
from utils.hybrid_database import HybridEmotionDatabase as EmotionDatabase

# Configure logging
logging.basicConfig(level=logging.INFO)

# Page configuration
st.set_page_config(
    page_title=app_config.page_title,
    page_icon=app_config.page_icon,
    layout=app_config.layout,
    initial_sidebar_state="collapsed"
)

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
if "theme" not in st.session_state:
    st.session_state.theme = app_config.theme

# Enhanced CSS
def get_enhanced_css():
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --primary: #6366f1;
        --secondary: #8b5cf6;
        --accent: #06b6d4;
        --success: #10b981;
        --warning: #f59e0b;
        --error: #ef4444;
        --bg-primary: #0f172a;
        --bg-secondary: #1e293b;
        --bg-card: #334155;
        --text-primary: #f8fafc;
        --text-secondary: #cbd5e1;
        --border: #475569;
    }
    
    .stApp {
        background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
        font-family: 'Inter', sans-serif;
        color: var(--text-primary);
    }
    
    .emotion-display {
        background: var(--bg-card);
        border-radius: 1rem;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid var(--primary);
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .emotion-icon {
        font-size: 3rem;
        filter: drop-shadow(0 0 10px rgba(99, 102, 241, 0.5));
    }
    
    .confidence-bar {
        background: var(--bg-secondary);
        border-radius: 1rem;
        height: 0.5rem;
        overflow: hidden;
        margin-top: 0.5rem;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--primary), var(--accent));
        border-radius: 1rem;
        transition: width 1s ease;
    }
    
    .breathing-circle {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        animation: breathe 4s ease-in-out infinite;
        box-shadow: 0 0 30px rgba(99, 102, 241, 0.5);
        margin: 0 auto;
    }
    
    @keyframes breathe {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.2); }
    }
    </style>
    """

st.markdown(get_enhanced_css(), unsafe_allow_html=True)

# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize all components with caching"""
    model_config = get_model_config()
    
    # Initialize emotion analyzer
    emotion_analyzer = EmotionAnalyzer(model_config)
    
    # Initialize therapy assistant
    therapy_assistant = None
    if api_config.openrouter_key:
        therapy_assistant = TherapyAssistant(api_config.openrouter_key)
    
    # Initialize database
    database = None
    if api_config.supabase_url and api_config.supabase_key:
        database = EmotionDatabase(api_config.supabase_url, api_config.supabase_key)
    
    return emotion_analyzer, therapy_assistant, database

emotion_analyzer, therapy_assistant, database = initialize_components()

# Header
st.title("🧠 Emotion AI - Complete Edition")
st.markdown("Advanced emotional intelligence with music therapy, GIFs, and voice assistant")

# Weather integration
st.markdown("### 🌤️ Weather-Based Mood Suggestions")
city = st.text_input("Enter your city:", value="Mumbai", key="weather_city")
if city:
    try:
        api_key = "24b359c76d994182864153220251507"
        weather_url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={city}"
        response = requests.get(weather_url, timeout=10)
        response.raise_for_status()
        weather_data = response.json()
        
        if weather_data.get("current"):
            weather_main = weather_data["current"]["condition"]["text"]
            temp = weather_data["current"]["temp_c"]
            st.info(f"🌤️ Current weather in {city}: {weather_main}, {temp}°C")
            
            # Mood suggestions based on weather
            suggestions_map = {
                "Sunny": "It's a bright day! Perfect time for outdoor activities and positive vibes. 😊",
                "Partly cloudy": "A bit cloudy today. Great day for reflection and cozy indoor activities. ☁️",
                "Cloudy": "Cloudy weather. A good day to relax indoors and reflect. ☁️",
                "Rain": "Rainy weather can be calming. Maybe enjoy some relaxing music or a good book. 🌧️",
                "Light rain": "Light rain outside. A perfect time for a warm drink and some mindfulness. ☕",
            }
            suggestion = suggestions_map.get(weather_main, "Enjoy your day and take care of yourself! 🌟")
            st.success(f"💡 Mood Suggestion: {suggestion}")
    except Exception as e:
        st.warning(f"Could not fetch weather data: {str(e)}")

# Main tabs with ALL original features
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎙️ Voice Analysis", 
    "📝 Text Analysis", 
    "🎵 Music Therapy",
    "🧠 AI Therapist",
    "🎤 Voice Assistant",
    "📊 Analytics"
])

# Helper functions
def display_emotion_with_gif(emotion: str, confidence: float, text: str = ""):
    """Display emotion with GIF integration"""
    # Emotion icons
    emotion_icons = {
        "joy": "😊", "sadness": "😢", "anger": "😠", "fear": "😨",
        "surprise": "😲", "disgust": "🤢", "love": "❤️", "excitement": "🤩",
        "gratitude": "🙏", "pride": "😤", "optimism": "🌟", "caring": "🤗",
        "neutral": "😐", "anxiety": "😰", "stress": "😫", "confusion": "😕"
    }
    
    icon = emotion_icons.get(emotion, "🤔")
    
    # Display emotion
    st.markdown(f"""
    <div class="emotion-display">
        <div class="emotion-icon">{icon}</div>
        <div class="emotion-details">
            <h3>{emotion.title()}</h3>
            <p>Confidence: {confidence:.1%}</p>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {confidence * 100}%"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Add GIF display
    try:
        from gif_generator import add_gif_to_emotion_display
        add_gif_to_emotion_display(emotion, confidence, show_button=False)
    except ImportError:
        st.info("💡 Install gif_generator for emotion GIFs")
    
    # Store for other tabs
    st.session_state.last_emotion = emotion
    st.session_state.last_confidence = confidence
    
    # Log to database
    if database and database.available:
        emotion_data = {
            "primary": emotion,
            "confidence": confidence,
            "source": "analysis"
        }
        user_context = {"session_id": st.session_state.session_id}
        database.log_emotion(emotion_data, user_context)

# Tab 1: Voice Analysis
with tab1:
    st.markdown("### 🎙️ Voice Emotion Detection")
    
    col1, col2 = st.columns(2)
    with col1:
        audio_file = st.file_uploader("Upload Audio File", type=["wav", "mp3", "m4a"])
    with col2:
        mic_audio = st.audio_input("🎤 Record Voice")
    
    audio_source = mic_audio or audio_file
    if audio_source:
        with st.spinner("🎵 Processing audio..."):
            try:
                # Process audio (simplified for demo)
                transcript, emotions = emotion_analyzer.analyze_audio(audio_source)
                
                if transcript:
                    st.success(f"**Transcript:** {transcript}")
                
                if emotions:
                    primary_emotion = emotions[0]
                    display_emotion_with_gif(
                        primary_emotion["label"], 
                        primary_emotion["score"], 
                        transcript
                    )
                    
            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")

# Tab 2: Text Analysis
with tab2:
    st.markdown("### 📝 Text Emotion Analysis")
    
    text_input = st.text_area("Enter your text:", height=150)
    
    if st.button("🔍 Analyze Emotions") and text_input.strip():
        with st.spinner("🧠 Analyzing emotions..."):
            try:
                emotions = emotion_analyzer.analyze_text(text_input)
                
                if emotions:
                    primary_emotion = emotions[0]
                    display_emotion_with_gif(
                        primary_emotion["label"], 
                        primary_emotion["score"], 
                        text_input
                    )
                    
                    # Breathing exercise for negative emotions
                    if primary_emotion["label"] in ["anxiety", "stress", "anger", "sadness", "fear"]:
                        st.markdown("---")
                        st.markdown("#### 🫁 Recommended: Breathing Exercise")
                        
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.markdown("""
                            <div style="text-align: center; padding: 2rem;">
                                <div class="breathing-circle"></div>
                                <p><strong>Breathe with the circle</strong><br>
                                <small>4 seconds in, 4 seconds out</small></p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("""
                            **How to practice:**
                            1. Watch the circle expand and contract
                            2. Breathe in as it grows (4 seconds)
                            3. Breathe out as it shrinks (4 seconds)
                            4. Continue for 2-3 minutes
                            """)
                            
            except Exception as e:
                st.error(f"Error analyzing text: {str(e)}")

# Tab 3: Music Therapy
with tab3:
    st.markdown("### 🎵 Personalized Music Therapy")
    st.markdown("Get music recommendations and play them directly in the app based on your detected emotions.")
    
    if "last_emotion" not in st.session_state:
        st.info("🎭 Analyze some audio or text first to get personalized music recommendations")
    else:
        emotion = st.session_state.last_emotion
        confidence = st.session_state.last_confidence
        
        # Display current emotion
        st.markdown(f"**Current Emotion:** {emotion.title()} ({confidence:.1%} confidence)")
        
        # Modern Hindi music search mapping based on emotions (2023-2024 focus)
        hindi_search_map = {
            "admiration": "latest hindi motivational songs 2024 bollywood new inspirational",
            "anger": "new hindi peaceful songs 2024 calming bollywood meditation latest",
            "annoyance": "latest hindi soothing songs 2024 relaxing bollywood new",
            "approval": "new hindi feel good songs 2024 happy bollywood latest hits",
            "caring": "latest hindi emotional songs 2024 bollywood heart touching new",
            "confusion": "new hindi thoughtful songs 2024 bollywood latest deep",
            "curiosity": "latest hindi adventure songs 2024 bollywood new discovery",
            "desire": "new hindi romantic songs 2024 bollywood love latest pyaar",
            "disappointment": "latest hindi motivational songs 2024 bollywood uplifting new",
            "disapproval": "new hindi positive songs 2024 bollywood encouraging latest",
            "disgust": "latest hindi spiritual songs 2024 bollywood cleansing new",
            "embarrassment": "new hindi confidence songs 2024 bollywood empowering latest",
            "excitement": "latest hindi dance songs 2024 bollywood party new energetic hits",
            "fear": "new hindi courage songs 2024 bollywood brave empowering latest",
            "gratitude": "latest hindi devotional songs 2024 bollywood thankful new grateful",
            "grief": "new hindi sad songs 2024 bollywood emotional latest dukh",
            "joy": "latest hindi happy songs 2024 bollywood celebration new khushi hits",
            "love": "new hindi love songs 2024 bollywood romantic latest pyaar hits",
            "nervousness": "latest hindi calming songs 2024 bollywood peaceful new meditation",
            "optimism": "new hindi positive songs 2024 bollywood hopeful latest motivational",
            "pride": "latest hindi achievement songs 2024 bollywood success new victory",
            "realization": "new hindi spiritual songs 2024 bollywood awakening latest",
            "relief": "latest hindi peaceful songs 2024 bollywood relaxation new",
            "remorse": "new hindi emotional songs 2024 bollywood healing latest forgiveness",
            "sadness": "latest hindi sad songs 2024 bollywood emotional new dukh gham",
            "surprise": "new hindi amazing songs 2024 bollywood wonder latest",
            "neutral": "latest hindi melodious songs 2024 bollywood soothing new hits",
        }
        
        # English fallback for broader results
        english_search_map = {
            "admiration": "inspiring uplifting songs",
            "anger": "calming peaceful music",
            "annoyance": "soothing relaxation music",
            "approval": "feel-good positive songs",
            "caring": "warm loving songs",
            "confusion": "clarity focus music",
            "curiosity": "discovery adventure music",
            "desire": "romantic ambient songs",
            "disappointment": "uplifting motivational music",
            "disapproval": "positive affirmation music",
            "disgust": "cleansing purifying music",
            "embarrassment": "confidence-building music",
            "excitement": "energetic celebration music",
            "fear": "courage empowerment music",
            "gratitude": "thankfulness appreciation music",
            "grief": "healing comfort music",
            "joy": "happy celebration music",
            "love": "romantic love songs",
            "nervousness": "calming anxiety relief music",
            "optimism": "uplifting positive music",
            "pride": "achievement celebration music",
            "realization": "enlightenment discovery music",
            "relief": "peaceful relaxation music",
            "remorse": "forgiveness healing music",
            "sadness": "gentle uplifting music",
            "surprise": "wonder amazement music",
            "neutral": "balanced ambient music",
        }
        
        # Language preference
        language_pref = st.radio(
            "🎵 Music Language Preference:",
            ["Hindi/Bollywood (Recommended)", "English", "Mixed"],
            index=0
        )
        
        if language_pref == "Hindi/Bollywood (Recommended)":
            search_map = hindi_search_map
        elif language_pref == "English":
            search_map = english_search_map
        else:  # Mixed
            # Combine both for variety
            hindi_query = hindi_search_map.get(emotion, f"hindi {emotion} songs bollywood")
            english_query = english_search_map.get(emotion, f"{emotion} music")
            search_map = {emotion: f"{hindi_query} OR {english_query}"}
        
        query = search_map.get(emotion, f"{emotion} therapeutic music")
        st.markdown(f"**Recommended search:** *{query}*")
        
        # Direct music playback using yt-dlp
        col1, col2 = st.columns([3, 1])
        
        with col1:
            play_button = st.button("🎵 Find & Play Healing Music", use_container_width=True)
        
        with col2:
            debug_mode = st.checkbox("🔧 Debug", help="Show detailed error information")
        
        if play_button:
            with st.spinner("🎵 Searching for therapeutic music..."):
                try:
                    import yt_dlp
                    
                    if debug_mode:
                        st.info(f"🔍 Searching for: {query}")
                    
                    # Enhanced yt-dlp options for better compatibility
                    ydl_opts = {
                        "format": "bestaudio[ext=m4a]/bestaudio/best",
                        "quiet": not debug_mode,
                        "no_warnings": not debug_mode,
                        "noplaylist": True,
                        "extract_flat": False,
                        "writethumbnail": False,
                        "writeinfojson": False,
                        "ignoreerrors": True,
                        "age_limit": None,
                        "cookiefile": None,
                        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    }
                    
                    # Multiple search attempts with different queries
                    search_queries = [
                        f"ytsearch1:{query}",
                        f"ytsearch1:{query} song",
                        f"ytsearch1:{query} music",
                    ]
                    
                    success = False
                    
                    for search_query in search_queries:
                        try:
                            if debug_mode:
                                st.info(f"🔍 Trying: {search_query}")
                            
                            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                                results = ydl.extract_info(search_query, download=False)
                                
                                if debug_mode:
                                    st.json({"results_keys": list(results.keys()) if results else "No results"})
                                
                                if results and "entries" in results and results["entries"]:
                                    info = results["entries"][0]
                                    
                                    if debug_mode:
                                        st.json({"info_keys": list(info.keys())})
                                    
                                    # Try multiple URL formats
                                    url = None
                                    for url_key in ["url", "webpage_url", "original_url"]:
                                        if info.get(url_key):
                                            url = info[url_key]
                                            break
                                    
                                    # Get additional formats if main URL fails
                                    if not url and "formats" in info:
                                        for fmt in info["formats"]:
                                            if fmt.get("url") and "audio" in fmt.get("format", "").lower():
                                                url = fmt["url"]
                                                break
                                    
                                    title = info.get("title", "Unknown Title")
                                    duration = info.get("duration", 0)
                                    uploader = info.get("uploader", "Unknown Artist")
                                    
                                    if debug_mode:
                                        st.json({
                                            "title": title,
                                            "url_found": bool(url),
                                            "duration": duration,
                                            "uploader": uploader
                                        })
                                    
                                    if url:
                                        # Display music info
                                        st.success(f"🎵 **Now Playing:** {title}")
                                        st.info(f"👤 **Artist/Channel:** {uploader}")
                                        
                                        if duration:
                                            minutes = duration // 60
                                            seconds = duration % 60
                                            st.info(f"⏱️ **Duration:** {minutes}:{seconds:02d}")
                                        
                                        # Try to play the audio
                                        try:
                                            st.audio(url, format="audio/mp3")
                                            
                                            # Store current playing info
                                            st.session_state.current_music = {
                                                "title": title,
                                                "url": url,
                                                "emotion": emotion,
                                                "query": query
                                            }
                                            
                                            success = True
                                            break
                                            
                                        except Exception as audio_error:
                                            if debug_mode:
                                                st.error(f"Audio playback error: {audio_error}")
                                            continue
                                    else:
                                        if debug_mode:
                                            st.warning("No playable URL found in this result")
                                        continue
                                else:
                                    if debug_mode:
                                        st.warning(f"No entries found for: {search_query}")
                                    continue
                                    
                        except Exception as search_error:
                            if debug_mode:
                                st.error(f"Search error for {search_query}: {search_error}")
                            continue
                    
                    if not success:
                        st.error("❌ Could not find or play music with current search")
                        st.info("💡 Try using external links below or check your internet connection")
                        
                        # Show external links as fallback
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            youtube_url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
                            st.markdown(f"[🎵 YouTube]({youtube_url})")
                        with col2:
                            spotify_url = f"https://open.spotify.com/search/{query.replace(' ', '%20')}"
                            st.markdown(f"[🎵 Spotify]({spotify_url})")
                        with col3:
                            apple_url = f"https://music.apple.com/search?term={query.replace(' ', '+')}"
                            st.markdown(f"[🎵 Apple Music]({apple_url})")
                            
                except ImportError:
                    st.error("❌ yt-dlp not installed. Install with: `pip install yt-dlp`")
                    st.info("💡 Using external links instead:")
                    
                    # Fallback to external links
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        youtube_url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
                        st.markdown(f"[🎵 YouTube]({youtube_url})")
                    with col2:
                        spotify_url = f"https://open.spotify.com/search/{query.replace(' ', '%20')}"
                        st.markdown(f"[🎵 Spotify]({spotify_url})")
                    with col3:
                        apple_url = f"https://music.apple.com/search?term={query.replace(' ', '+')}"
                        st.markdown(f"[🎵 Apple Music]({apple_url})")
                        
                except Exception as e:
                    st.error(f"❌ Unable to load music: {str(e)}")
                    if debug_mode:
                        st.exception(e)
                    st.info("💡 Try the external links below or enable debug mode for more details")
                    
                    # Show external links as fallback
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        youtube_url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
                        st.markdown(f"[🎵 YouTube]({youtube_url})")
                    with col2:
                        spotify_url = f"https://open.spotify.com/search/{query.replace(' ', '%20')}"
                        st.markdown(f"[🎵 Spotify]({spotify_url})")
                    with col3:
                        apple_url = f"https://music.apple.com/search?term={query.replace(' ', '+')}"
                        st.markdown(f"[🎵 Apple Music]({apple_url})")
        
        # Show currently playing music info
        if "current_music" in st.session_state:
            st.markdown("---")
            st.markdown("#### 🎶 Currently Playing")
            current = st.session_state.current_music
            st.markdown(f"**{current['title']}**")
            st.caption(f"Selected for: {current['emotion'].title()} • Search: {current['query']}")
            
            # Option to find another song
            if st.button("🔄 Find Another Song", key="find_another"):
                if "current_music" in st.session_state:
                    del st.session_state.current_music
                st.rerun()
        
        # Curated recommendations by emotion category
        st.markdown("---")
        st.markdown("#### 🎼 Curated Therapeutic Playlists")
        
        if emotion in ["sadness", "grief", "disappointment", "remorse"]:
            st.markdown("""
            **🌧️ For Healing & Comfort:**
            - Ludovico Einaudi - Nuvole Bianche (Peaceful piano)
            - Max Richter - On The Nature of Daylight (Ambient classical)
            - Ólafur Arnalds - Near Light (Emotional instrumental)
            - Kiasmos - Blurred EP (Minimal electronic)
            """)
        elif emotion in ["anger", "annoyance", "stress", "nervousness"]:
            st.markdown("""
            **🧘 For Calming & Centering:**
            - Brian Eno - Music for Airports (Ambient soundscapes)
            - Marconi Union - Weightless (Scientifically proven to reduce anxiety)
            - Nils Frahm - Says (Minimalist piano)
            - Stars of the Lid - The Tired Sounds Of (Drone ambient)
            """)
        elif emotion in ["joy", "excitement", "gratitude", "pride"]:
            st.markdown("""
            **🎉 For Celebration & Energy:**
            - Pharrell Williams - Happy (Feel-good pop)
            - Bob Marley - Three Little Birds (Uplifting reggae)
            - Earth, Wind & Fire - September (Classic funk)
            - Stevie Wonder - Sir Duke (Joyful soul)
            """)
        elif emotion in ["fear", "anxiety", "confusion"]:
            st.markdown("""
            **💪 For Courage & Clarity:**
            - Emancipator - Soon It Will Be Cold Enough (Downtempo)
            - Bonobo - Kong (Chillout electronic)
            - Tycho - A Walk (Ambient electronic)
            - GoGo Penguin - Hopopono (Modern jazz)
            """)
        else:
            st.markdown("""
            **⚖️ For Balance & Focus:**
            - Kiasmos - Swept (Minimal techno)
            - GoGo Penguin - Raven (Modern jazz)
            - Emancipator - Dusk to Dawn (Downtempo electronic)
            - Ott - Queen of All Everything (Psychedelic dub)
            """)

# Tab 4: AI Therapist
with tab4:
    st.markdown("### 🧠 AI Therapeutic Support")
    
    if not therapy_assistant:
        st.warning("⚠️ AI Therapy requires OpenRouter API key")
    else:
        # Initialize chat history
        if "therapy_messages" not in st.session_state:
            st.session_state.therapy_messages = []
        
        # Display chat history
        for message in st.session_state.therapy_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Share what's on your mind..."):
            # Add user message
            st.session_state.therapy_messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate response
            with st.spinner("💭 Generating therapeutic response..."):
                try:
                    emotions = emotion_analyzer.analyze_text(prompt)
                    emotion_data = emotion_analyzer.get_emotion_insights(emotions)
                    
                    therapy_response = therapy_assistant.get_therapeutic_response(
                        prompt, emotion_data, st.session_state.session_id
                    )
                    
                    # Display response
                    with st.chat_message("assistant"):
                        st.write(therapy_response["response"])
                    
                    # Add to history
                    st.session_state.therapy_messages.append({
                        "role": "assistant", 
                        "content": therapy_response["response"]
                    })
                    
                except Exception as e:
                    st.error(f"Therapy assistant error: {str(e)}")

# Tab 5: Voice Assistant
with tab5:
    st.markdown("### 🎤 Voice Assistant")
    st.markdown("Text-to-speech functionality with emotion-based responses.")
    
    # Alternative TTS approaches
    tts_method = st.radio(
        "Choose TTS Method:",
        ["Browser TTS (Recommended)", "System TTS (pyttsx3)", "Online TTS"],
        index=0
    )
    
    if tts_method == "Browser TTS (Recommended)":
        st.markdown("#### 🌐 Browser-Based Text-to-Speech")
        st.info("💡 This uses your browser's built-in speech synthesis - no installation required!")
        
        # Voice settings
        col1, col2 = st.columns(2)
        with col1:
            speech_rate = st.slider("Speech Rate", 0.5, 2.0, 1.0, 0.1)
        with col2:
            speech_pitch = st.slider("Speech Pitch", 0.0, 2.0, 1.0, 0.1)
        
        # Text to speak
        tts_text = st.text_area("Enter text to speak:", height=100, 
                               placeholder="Type something for the voice assistant to say...")
        
        if st.button("🗣️ Speak Text (Browser)", use_container_width=True) and tts_text:
            # Generate JavaScript for browser TTS
            js_code = f"""
            <script>
            function speakText() {{
                if ('speechSynthesis' in window) {{
                    // Cancel any ongoing speech
                    window.speechSynthesis.cancel();
                    
                    const utterance = new SpeechSynthesisUtterance(`{tts_text.replace('`', '\\`')}`);
                    utterance.rate = {speech_rate};
                    utterance.pitch = {speech_pitch};
                    utterance.volume = 0.8;
                    
                    utterance.onstart = function() {{
                        console.log('Speech started');
                    }};
                    
                    utterance.onend = function() {{
                        console.log('Speech ended');
                    }};
                    
                    utterance.onerror = function(event) {{
                        console.error('Speech error:', event.error);
                    }};
                    
                    window.speechSynthesis.speak(utterance);
                }} else {{
                    alert('Speech synthesis not supported in this browser');
                }}
            }}
            
            // Auto-trigger speech
            speakText();
            </script>
            """
            
            st.components.v1.html(js_code, height=0)
            st.success("✅ Speech triggered! (Check browser audio)")
            
    elif tts_method == "System TTS (pyttsx3)":
        st.markdown("#### 🖥️ System Text-to-Speech")
        
        try:
            import pyttsx3
            import threading
            import queue
            
            # Thread-safe TTS function
            def speak_text_threaded(text, rate=150, volume=0.8):
                """Thread-safe TTS function"""
                try:
                    # Create new engine instance for thread safety
                    engine = pyttsx3.init()
                    engine.setProperty('rate', rate)
                    engine.setProperty('volume', volume)
                    
                    # Use threading to avoid run loop conflicts
                    def tts_worker():
                        engine.say(text)
                        engine.runAndWait()
                        engine.stop()
                    
                    thread = threading.Thread(target=tts_worker)
                    thread.daemon = True
                    thread.start()
                    thread.join(timeout=10)  # 10 second timeout
                    
                    return True
                except Exception as e:
                    st.error(f"TTS Error: {str(e)}")
                    return False
            
            # Voice settings
            col1, col2 = st.columns(2)
            with col1:
                rate = st.slider("Speech Rate", 100, 300, 150)
            with col2:
                volume = st.slider("Volume", 0.0, 1.0, 0.8)
            
            # Text to speak
            tts_text = st.text_area("Enter text to speak:", height=100)
            
            if st.button("🗣️ Speak Text (System)", use_container_width=True) and tts_text:
                with st.spinner("🎤 Speaking..."):
                    if speak_text_threaded(tts_text, rate, volume):
                        st.success("✅ Speech completed!")
                    else:
                        st.error("❌ Speech failed - try Browser TTS instead")
        
        except ImportError:
            st.error("❌ pyttsx3 not installed. Install with: `pip install pyttsx3`")
            st.info("💡 Use Browser TTS instead - it works without installation!")
    
    else:  # Online TTS
        st.markdown("#### 🌍 Online Text-to-Speech")
        st.info("💡 This uses online TTS services (requires internet)")
        
        # Text to speak
        tts_text = st.text_area("Enter text to speak:", height=100)
        
        if st.button("🗣️ Generate Speech (Online)", use_container_width=True) and tts_text:
            try:
                # Use gTTS (Google Text-to-Speech) as fallback
                from gtts import gTTS
                import io
                import base64
                
                with st.spinner("🎤 Generating speech..."):
                    # Create TTS object
                    tts = gTTS(text=tts_text, lang='en', slow=False)
                    
                    # Save to bytes buffer
                    audio_buffer = io.BytesIO()
                    tts.write_to_fp(audio_buffer)
                    audio_buffer.seek(0)
                    
                    # Play audio
                    st.audio(audio_buffer, format='audio/mp3')
                    st.success("✅ Speech generated!")
                    
            except ImportError:
                st.error("❌ gTTS not installed. Install with: `pip install gtts`")
                st.info("💡 Use Browser TTS instead!")
            except Exception as e:
                st.error(f"❌ Online TTS error: {str(e)}")
    
    # Emotion-based responses
    st.markdown("---")
    st.markdown("#### 🎭 Emotion-Based Voice Responses")
    
    if "last_emotion" in st.session_state:
        emotion = st.session_state.last_emotion
        confidence = st.session_state.last_confidence
        
        st.markdown(f"**Current Emotion:** {emotion.title()} ({confidence:.1%} confidence)")
        
        # Enhanced emotion responses
        emotion_responses = {
            "joy": "I can sense your happiness radiating through your words! That's absolutely wonderful to hear. Keep embracing those positive feelings and let them lift you up!",
            "sadness": "I understand you're feeling down right now, and that's completely okay. Remember, sadness is a natural part of the human experience. You're not alone in this journey, and brighter days are ahead.",
            "anger": "I can feel the frustration and intensity in your emotions. Take a deep breath with me. Let's work through these feelings together, one step at a time. You have the strength to overcome this.",
            "fear": "I sense some worry and uncertainty in your voice. It takes courage to acknowledge these feelings, and you're being incredibly brave by sharing them. Let's find some calm and clarity together.",
            "excitement": "Your excitement is absolutely contagious! I love hearing such vibrant, positive energy from you. This enthusiasm is a beautiful thing - let it fuel your passions!",
            "stress": "I can tell you're feeling overwhelmed right now. Remember, it's okay to feel stressed sometimes. Take things one moment at a time. You've got this, and you're stronger than you know.",
            "anxiety": "I recognize the anxious feelings you're experiencing. Anxiety can feel overwhelming, but remember that you've overcome challenges before. Let's focus on breathing and finding your center.",
            "gratitude": "Your sense of gratitude is truly beautiful. Appreciation and thankfulness are powerful emotions that can transform your entire perspective. Thank you for sharing this positive energy!",
            "neutral": "Thank you for sharing with me. I'm here to listen and support you in whatever way you need. Your thoughts and feelings are always valid and important."
        }
        
        response_text = emotion_responses.get(emotion, emotion_responses["neutral"])
        
        st.markdown("**Suggested empathetic response:**")
        st.info(response_text)
        
        # Quick response buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button(f"🗣️ Speak {emotion.title()} Response", use_container_width=True):
                if tts_method == "Browser TTS (Recommended)":
                    # Browser TTS for emotion response
                    js_code = f"""
                    <script>
                    function speakEmotionResponse() {{
                        if ('speechSynthesis' in window) {{
                            window.speechSynthesis.cancel();
                            const utterance = new SpeechSynthesisUtterance(`{response_text.replace('`', '\\`')}`);
                            utterance.rate = 0.9;
                            utterance.pitch = 1.0;
                            utterance.volume = 0.8;
                            window.speechSynthesis.speak(utterance);
                        }}
                    }}
                    speakEmotionResponse();
                    </script>
                    """
                    st.components.v1.html(js_code, height=0)
                    st.success("✅ Emotional response delivered!")
                else:
                    st.info("💡 Switch to Browser TTS for instant responses!")
        
        with col2:
            if st.button("📋 Copy Response Text", use_container_width=True):
                st.code(response_text)
                st.success("✅ Response text displayed above!")
        
        # Therapeutic suggestions based on emotion
        st.markdown("#### 💡 Therapeutic Suggestions")
        
        if emotion in ["sadness", "grief", "disappointment"]:
            st.markdown("""
            **🌱 For healing and growth:**
            - Practice self-compassion and gentle self-talk
            - Consider journaling your thoughts and feelings
            - Reach out to supportive friends or family
            - Engage in activities that bring you comfort
            """)
        elif emotion in ["anger", "annoyance", "frustration"]:
            st.markdown("""
            **🧘 For managing intensity:**
            - Try the 4-7-8 breathing technique
            - Take a short walk or do physical exercise
            - Practice progressive muscle relaxation
            - Express feelings through creative outlets
            """)
        elif emotion in ["anxiety", "fear", "nervousness"]:
            st.markdown("""
            **🛡️ For building calm and confidence:**
            - Use grounding techniques (5-4-3-2-1 method)
            - Practice mindfulness meditation
            - Challenge anxious thoughts with facts
            - Create a calming environment around you
            """)
        elif emotion in ["joy", "excitement", "gratitude"]:
            st.markdown("""
            **✨ For amplifying positivity:**
            - Share your joy with others
            - Practice gratitude journaling
            - Engage in activities you love
            - Use this energy for creative projects
            """)
    else:
        st.info("🎭 Analyze some emotions first to get personalized voice responses!")
        
        # Demo responses
        st.markdown("#### 🎯 Try These Demo Responses:")
        
        demo_responses = [
            "Welcome to your personal emotion AI assistant!",
            "I'm here to support your emotional wellness journey.",
            "Remember, every emotion is valid and important.",
            "You have the strength to overcome any challenge."
        ]
        
        for i, demo_text in enumerate(demo_responses):
            if st.button(f"🗣️ Demo Response {i+1}", key=f"demo_{i}"):
                if tts_method == "Browser TTS (Recommended)":
                    js_code = f"""
                    <script>
                    if ('speechSynthesis' in window) {{
                        window.speechSynthesis.cancel();
                        const utterance = new SpeechSynthesisUtterance(`{demo_text}`);
                        utterance.rate = 1.0;
                        utterance.pitch = 1.0;
                        utterance.volume = 0.8;
                        window.speechSynthesis.speak(utterance);
                    }}
                    </script>
                    """
                    st.components.v1.html(js_code, height=0)
                    st.success(f"✅ Playing: '{demo_text}'")

# Tab 6: Analytics
with tab6:
    st.markdown("### 📊 Emotion Analytics")
    
    if not database or not database.available:
        st.warning("⚠️ Analytics require database connection")
    else:
        # Time range
        days = st.selectbox("Time Range", [7, 14, 30, 90], index=2)
        
        # Get analytics
        analytics = database.get_emotion_analytics(days)
        
        if "error" not in analytics:
            # Basic stats
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Entries", analytics.get("total_entries", 0))
            with col2:
                st.metric("Avg Confidence", f"{analytics.get('average_confidence', 0):.1%}")
            with col3:
                st.metric("Top Emotion", analytics.get("most_common_emotion", "neutral").title())
            
            # Emotion distribution
            if analytics.get("emotion_distribution"):
                st.markdown("#### Emotion Distribution")
                
                emotion_dist = analytics["emotion_distribution"]
                fig = px.pie(
                    values=list(emotion_dist.values()),
                    names=list(emotion_dist.keys()),
                    title=f"Emotions Over Last {days} Days"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Insights
            insights = database.get_personalized_insights(days)
            if insights:
                st.markdown("#### 💡 Personalized Insights")
                for insight in insights:
                    st.markdown(f"• {insight}")
        else:
            st.info("No data available yet. Start tracking emotions to see analytics!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; padding: 2rem;">
    <p>🧠 Emotion AI - Complete Edition with Music Therapy, GIFs & Voice Assistant</p>
    <p><small>Built with ❤️ using Streamlit and advanced AI</small></p>
</div>
""", unsafe_allow_html=True)