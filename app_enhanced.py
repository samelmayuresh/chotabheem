# app_enhanced.py - Improved Emotion AI with advanced features
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import asyncio
import logging
from typing import Dict, List, Optional

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

# Enhanced CSS with better responsiveness and animations
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
    
    .main-header {
        background: rgba(15, 23, 42, 0.95);
        backdrop-filter: blur(10px);
        border-bottom: 1px solid var(--border);
        padding: 1rem 2rem;
        margin: -1rem -2rem 2rem -2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .logo-section {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .logo-icon {
        font-size: 2rem;
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    .status-indicator {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 2rem;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    .status-connected {
        background: rgba(16, 185, 129, 0.1);
        color: var(--success);
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .hero-section {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        border-radius: 1rem;
        padding: 3rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="50" cy="50" r="1" fill="white" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
        opacity: 0.1;
    }
    
    .hero-content {
        position: relative;
        z-index: 1;
    }
    
    .feature-card {
        background: var(--bg-card);
        border-radius: 1rem;
        padding: 1.5rem;
        border: 1px solid var(--border);
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        border-color: var(--primary);
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
        transition: all 0.3s ease;
    }
    
    .emotion-icon {
        font-size: 3rem;
        filter: drop-shadow(0 0 10px rgba(99, 102, 241, 0.5));
    }
    
    .emotion-details h3 {
        margin: 0 0 0.5rem 0;
        color: var(--primary);
        font-weight: 600;
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
    
    .metric-card {
        background: var(--bg-card);
        border-radius: 0.75rem;
        padding: 1rem;
        text-align: center;
        border: 1px solid var(--border);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary);
        margin-bottom: 0.25rem;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .breathing-exercise {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 2rem;
        background: var(--bg-card);
        border-radius: 1rem;
        margin: 2rem 0;
    }
    
    .breathing-circle {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        animation: breathe 4s ease-in-out infinite;
        box-shadow: 0 0 30px rgba(99, 102, 241, 0.5);
    }
    
    @keyframes breathe {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.2); }
    }
    
    .stButton > button {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        color: white;
        border: none;
        border-radius: 0.75rem;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(99, 102, 241, 0.3);
    }
    
    .insight-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(139, 92, 246, 0.1));
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 1rem;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .warning-card {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(239, 68, 68, 0.1));
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 1rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    @media (max-width: 768px) {
        .main-header {
            flex-direction: column;
            gap: 1rem;
            text-align: center;
        }
        
        .hero-section {
            padding: 2rem 1rem;
        }
        
        .emotion-display {
            flex-direction: column;
            text-align: center;
        }
    }
    </style>
    """

# Apply enhanced CSS
st.markdown(get_enhanced_css(), unsafe_allow_html=True)

# Validate API keys and show warnings
warnings = validate_api_keys()
if warnings:
    for warning in warnings:
        st.warning(warning)

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
st.markdown("""
<div class="main-header">
    <div class="logo-section">
        <div class="logo-icon">🧠</div>
        <div>
            <h1 style="margin: 0; font-size: 1.5rem;">Emotion AI</h1>
            <p style="margin: 0; color: var(--text-secondary); font-size: 0.875rem;">Advanced Emotional Intelligence</p>
        </div>
    </div>
    <div class="status-indicator status-connected">
        <span>●</span>
        <span>System Online</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Hero section with dynamic content
current_hour = datetime.now().hour
if 5 <= current_hour < 12:
    greeting = "Good Morning"
    emoji = "🌅"
elif 12 <= current_hour < 17:
    greeting = "Good Afternoon"
    emoji = "☀️"
elif 17 <= current_hour < 21:
    greeting = "Good Evening"
    emoji = "🌆"
else:
    greeting = "Good Night"
    emoji = "🌙"

st.markdown(f"""
<div class="hero-section">
    <div class="hero-content">
        <h1>{emoji} {greeting}!</h1>
        <p>Welcome to your personal AI-powered emotional wellness companion. Let's explore your emotional landscape together.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Quick stats if database is available
if database and database.available:
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        
        # Get recent analytics
        analytics = database.get_emotion_analytics(days=7)
        
        with col1:
            total_entries = analytics.get("total_entries", 0)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total_entries}</div>
                <div class="metric-label">This Week</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            avg_confidence = analytics.get("average_confidence", 0)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{avg_confidence:.1%}</div>
                <div class="metric-label">Avg Confidence</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            most_common = analytics.get("most_common_emotion", "neutral")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{most_common.title()}</div>
                <div class="metric-label">Top Emotion</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            stability = analytics.get("emotional_stability", {}).get("stability", "unknown")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{stability.title()}</div>
                <div class="metric-label">Stability</div>
            </div>
            """, unsafe_allow_html=True)

# Main tabs with enhanced features
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🎙️ Voice Analysis", 
    "📝 Text Analysis", 
    "🎵 Music Therapy",
    "🧠 AI Therapist",
    "🎤 Voice Assistant",
    "📊 Analytics",
    "🎯 Insights",
    "⚙️ Settings"
])

# Enhanced emotion display function
def display_enhanced_emotion_results(emotions: List[Dict], source: str = "text"):
    """Display emotion results with enhanced UI"""
    if not emotions:
        st.warning("No emotions detected")
        return
    
    primary_emotion = emotions[0]
    
    # Emotion icons mapping
    emotion_icons = {
        "joy": "😊", "sadness": "😢", "anger": "😠", "fear": "😨",
        "surprise": "😲", "disgust": "🤢", "love": "❤️", "excitement": "🤩",
        "gratitude": "🙏", "pride": "😤", "optimism": "🌟", "caring": "🤗",
        "neutral": "😐", "anxiety": "😰", "stress": "😫", "confusion": "😕"
    }
    
    icon = emotion_icons.get(primary_emotion["label"], "🤔")
    confidence = primary_emotion["score"]
    
    # Primary emotion display
    st.markdown(f"""
    <div class="emotion-display">
        <div class="emotion-icon">{icon}</div>
        <div class="emotion-details">
            <h3>{primary_emotion["label"].title()}</h3>
            <p>Confidence: {confidence:.1%}</p>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {confidence * 100}%"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Top emotions chart
    if len(emotions) > 1:
        st.subheader("Emotion Breakdown")
        
        # Create interactive chart
        df = pd.DataFrame(emotions[:8])  # Top 8 emotions
        
        fig = px.bar(
            df, 
            x="score", 
            y="label",
            orientation='h',
            title="Detected Emotions",
            color="score",
            color_continuous_scale="viridis"
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Log to database if available
    if database and database.available:
        emotion_data = {
            "primary": primary_emotion["label"],
            "confidence": confidence,
            "all_scores": emotions,
            "source": source
        }
        
        user_context = {
            "session_id": st.session_state.session_id,
            "processing_time": 0.0  # Would be calculated in real implementation
        }
        
        if database.log_emotion(emotion_data, user_context):
            st.success("✅ Emotion logged successfully")

# Tab 1: Enhanced Voice Analysis
with tab1:
    st.markdown("### 🎙️ Advanced Voice Emotion Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Upload Audio File")
        audio_file = st.file_uploader(
            "Choose an audio file", 
            type=["wav", "mp3", "m4a", "ogg"],
            help="Upload an audio file for emotion analysis"
        )
    
    with col2:
        st.markdown("#### Record Voice")
        mic_audio = st.audio_input("🎤 Record your voice")
    
    audio_source = mic_audio or audio_file
    
    if audio_source:
        with st.spinner("🎵 Analyzing audio..."):
            try:
                # Process audio
                transcript, emotions = emotion_analyzer.analyze_audio(audio_source)
                
                if transcript:
                    st.markdown("#### 📝 Transcript")
                    st.info(f'"{transcript}"')
                
                if emotions:
                    st.markdown("#### 🎭 Detected Emotions")
                    display_enhanced_emotion_results(emotions, "audio")
                    
                    # Generate insights
                    insights = emotion_analyzer.get_emotion_insights(emotions)
                    if insights["insights"]:
                        st.markdown("#### 💡 Insights")
                        for insight in insights["insights"]:
                            st.markdown(f"• {insight}")
                
            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")

# Tab 2: Enhanced Text Analysis
with tab2:
    st.markdown("### 📝 Advanced Text Emotion Analysis")
    
    # Text input with character counter
    text_input = st.text_area(
        "Enter your text for analysis:",
        height=150,
        max_chars=app_config.max_text_length,
        help=f"Maximum {app_config.max_text_length} characters"
    )
    
    if text_input:
        char_count = len(text_input)
        st.caption(f"Characters: {char_count}/{app_config.max_text_length}")
        
        if st.button("🔍 Analyze Text", use_container_width=True):
            with st.spinner("🧠 Analyzing emotions..."):
                try:
                    emotions = emotion_analyzer.analyze_text(text_input)
                    
                    if emotions:
                        # Store emotions for other tabs
                        st.session_state.last_emotions = emotions
                        
                        display_enhanced_emotion_results(emotions, "text")
                        
                        # Add GIF display
                        primary_emotion = emotions[0]["label"]
                        confidence = emotions[0]["score"]
                        
                        # Import and display GIF
                        try:
                            from gif_generator import add_gif_to_emotion_display
                            add_gif_to_emotion_display(primary_emotion, confidence, show_button=False)
                        except ImportError:
                            st.info("💡 GIF functionality not available. Install required packages for emotion GIFs.")
                        
                        # Generate insights
                        insights = emotion_analyzer.get_emotion_insights(emotions)
                        if insights["insights"]:
                            st.markdown("#### 💡 Analysis Insights")
                            for insight in insights["insights"]:
                                st.markdown(f"• {insight}")
                        
                        # Suggest breathing exercise for negative emotions
                        primary_emotion = emotions[0]["label"]
                        if primary_emotion in ["anxiety", "stress", "anger", "sadness", "fear"]:
                            st.markdown("---")
                            st.markdown("#### 🫁 Recommended: Breathing Exercise")
                            
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                st.markdown("""
                                <div class="breathing-exercise">
                                    <div class="breathing-circle"></div>
                                    <p style="margin-top: 1rem; text-align: center;">
                                        <strong>Breathe with the circle</strong><br>
                                        <small>4 seconds in, 4 seconds out</small>
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown("""
                                **How to practice:**
                                1. Watch the circle expand and contract
                                2. Breathe in as it grows (4 seconds)
                                3. Breathe out as it shrinks (4 seconds)
                                4. Continue for 2-3 minutes
                                
                                This technique helps activate your parasympathetic nervous system and reduce stress.
                                """)
                
                except Exception as e:
                    st.error(f"Error analyzing text: {str(e)}")

# Tab 3: Enhanced AI Therapist
with tab3:
    st.markdown("### 🧠 AI Therapy Assistant")
    
    if not therapy_assistant:
        st.warning("⚠️ AI Therapy requires OpenRouter API key. Please add your key to enable this feature.")
    else:
        # Session info
        st.markdown(f"**Session ID:** `{st.session_state.session_id}`")
        
        # Chat interface
        if "therapy_messages" not in st.session_state:
            st.session_state.therapy_messages = []
        
        # Display chat history
        for message in st.session_state.therapy_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if "technique" in message:
                    st.caption(f"Technique: {message['technique']}")
        
        # Chat input
        if prompt := st.chat_input("Share what's on your mind..."):
            # Add user message
            st.session_state.therapy_messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.write(prompt)
            
            # Analyze emotion in user input
            with st.spinner("🧠 Understanding your emotions..."):
                emotions = emotion_analyzer.analyze_text(prompt)
                emotion_data = emotion_analyzer.get_emotion_insights(emotions)
            
            # Generate therapy response
            with st.spinner("💭 Generating therapeutic response..."):
                therapy_response = therapy_assistant.get_therapeutic_response(
                    prompt, 
                    emotion_data, 
                    st.session_state.session_id
                )
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.write(therapy_response["response"])
                
                # Show technique used
                technique_info = therapy_assistant.techniques.get(therapy_response["technique"], {})
                st.caption(f"**Approach:** {technique_info.get('name', 'General Support')}")
                
                # Show suggested exercises
                if therapy_response["exercises"]:
                    with st.expander("💪 Suggested Exercises"):
                        for exercise in therapy_response["exercises"]:
                            st.markdown(f"""
                            **{exercise['name']}** ({exercise['duration']})
                            
                            {exercise['description']}
                            """)
                
                # Show session insights
                insights = therapy_response["session_insights"]
                if insights["insights"]:
                    with st.expander("📊 Session Insights"):
                        for insight in insights["insights"]:
                            st.markdown(f"• {insight}")
            
            # Add assistant message to history
            st.session_state.therapy_messages.append({
                "role": "assistant", 
                "content": therapy_response["response"],
                "technique": therapy_response["technique"]
            })

# Tab 3: Music Therapy
with tab3:
    st.markdown("### 🎵 Personalized Music Therapy")
    st.markdown("Get music recommendations based on your detected emotions.")
    
    # Check if we have recent emotion data
    if "last_emotions" not in st.session_state:
        st.info("🎭 Analyze some audio or text first to get personalized music recommendations")
    else:
        emotions = st.session_state.last_emotions
        primary_emotion = emotions[0]["label"] if emotions else "neutral"
        confidence = emotions[0]["score"] if emotions else 0.5
        
        # Display current emotion
        st.markdown(f"**Current Emotion:** {primary_emotion.title()} ({confidence:.1%} confidence)")
        
        # Music search mapping
        search_map = {
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
        
        query = search_map.get(primary_emotion, f"{primary_emotion} therapeutic music")
        st.markdown(f"**Recommended search:** *{query}*")
        
        # Music platform links
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
        
        # Mood-based recommendations
        st.markdown("#### 🎼 Curated Playlists")
        
        if primary_emotion in ["sadness", "grief", "disappointment"]:
            st.markdown("""
            **For healing and comfort:**
            - Ludovico Einaudi - Peaceful piano melodies
            - Max Richter - Ambient classical compositions
            - Ólafur Arnalds - Emotional instrumental pieces
            """)
        elif primary_emotion in ["anger", "annoyance", "stress"]:
            st.markdown("""
            **For calming and centering:**
            - Brian Eno - Ambient soundscapes
            - Marconi Union - Weightless (scientifically proven to reduce anxiety)
            - Nils Frahm - Minimalist piano
            """)
        elif primary_emotion in ["joy", "excitement", "gratitude"]:
            st.markdown("""
            **For celebration and energy:**
            - Pharrell Williams - Happy
            - Bob Marley - Three Little Birds
            - Earth, Wind & Fire - September
            """)
        else:
            st.markdown("""
            **For balance and focus:**
            - Kiasmos - Minimal techno
            - GoGo Penguin - Modern jazz
            - Emancipator - Downtempo electronic
            """)

# Tab 4: AI Therapist
with tab4:
    st.markdown("### 🧠 AI Therapy Assistant")
    
    if not therapy_assistant:
        st.warning("⚠️ AI Therapy requires OpenRouter API key. Please add your key to enable this feature.")
    else:
        # Session info
        st.markdown(f"**Session ID:** `{st.session_state.session_id}`")
        
        # Chat interface
        if "therapy_messages" not in st.session_state:
            st.session_state.therapy_messages = []
        
        # Display chat history
        for message in st.session_state.therapy_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if "technique" in message:
                    st.caption(f"Technique: {message['technique']}")
        
        # Chat input
        if prompt := st.chat_input("Share what's on your mind..."):
            # Add user message
            st.session_state.therapy_messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.write(prompt)
            
            # Analyze emotion in user input
            with st.spinner("🧠 Understanding your emotions..."):
                emotions = emotion_analyzer.analyze_text(prompt)
                emotion_data = emotion_analyzer.get_emotion_insights(emotions)
            
            # Generate therapy response
            with st.spinner("💭 Generating therapeutic response..."):
                therapy_response = therapy_assistant.get_therapeutic_response(
                    prompt, 
                    emotion_data, 
                    st.session_state.session_id
                )
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.write(therapy_response["response"])
                
                # Show technique used
                technique_info = therapy_assistant.techniques.get(therapy_response["technique"], {})
                st.caption(f"**Approach:** {technique_info.get('name', 'General Support')}")
                
                # Show suggested exercises
                if therapy_response["exercises"]:
                    with st.expander("💪 Suggested Exercises"):
                        for exercise in therapy_response["exercises"]:
                            st.markdown(f"""
                            **{exercise['name']}** ({exercise['duration']})
                            
                            {exercise['description']}
                            """)
                
                # Show session insights
                insights = therapy_response["session_insights"]
                if insights["insights"]:
                    with st.expander("📊 Session Insights"):
                        for insight in insights["insights"]:
                            st.markdown(f"• {insight}")
            
            # Add assistant message to history
            st.session_state.therapy_messages.append({
                "role": "assistant", 
                "content": therapy_response["response"],
                "technique": therapy_response["technique"]
            })

# Tab 5: Voice Assistant
with tab5:
    st.markdown("### 🎤 Voice Assistant")
    st.markdown("Text-to-speech functionality for emotion feedback and guidance.")
    
    # Initialize TTS engine
    try:
        import pyttsx3
        
        if "tts_engine" not in st.session_state:
            st.session_state.tts_engine = pyttsx3.init()
            
            # Configure voice settings
            voices = st.session_state.tts_engine.getProperty('voices')
            if voices:
                st.session_state.tts_engine.setProperty('voice', voices[0].id)
            st.session_state.tts_engine.setProperty('rate', 150)
            st.session_state.tts_engine.setProperty('volume', 0.8)
        
        # Voice settings
        col1, col2 = st.columns(2)
        
        with col1:
            speech_rate = st.slider("Speech Rate", 100, 300, 150)
            st.session_state.tts_engine.setProperty('rate', speech_rate)
        
        with col2:
            volume = st.slider("Volume", 0.0, 1.0, 0.8)
            st.session_state.tts_engine.setProperty('volume', volume)
        
        # Text input for TTS
        tts_text = st.text_area(
            "Enter text to speak:",
            height=100,
            placeholder="Type something for the voice assistant to say..."
        )
        
        if st.button("🗣️ Speak Text", use_container_width=True) and tts_text:
            with st.spinner("🎤 Speaking..."):
                try:
                    st.session_state.tts_engine.say(tts_text)
                    st.session_state.tts_engine.runAndWait()
                    st.success("✅ Speech completed!")
                except Exception as e:
                    st.error(f"Speech error: {str(e)}")
        
        # Emotion-based responses
        st.markdown("#### 🎭 Emotion-Based Voice Responses")
        
        if "last_emotions" in st.session_state:
            emotions = st.session_state.last_emotions
            primary_emotion = emotions[0]["label"] if emotions else "neutral"
            
            # Predefined responses for emotions
            emotion_responses = {
                "joy": "I can sense your happiness! That's wonderful to hear. Keep embracing those positive feelings!",
                "sadness": "I understand you're feeling down. Remember, it's okay to feel sad sometimes. You're not alone in this.",
                "anger": "I can feel your frustration. Take a deep breath with me. Let's work through this together.",
                "fear": "I sense some worry in your voice. You're brave for sharing this. Let's find some calm together.",
                "excitement": "Your excitement is contagious! I love hearing such positive energy from you!",
                "stress": "I can tell you're feeling overwhelmed. Let's take this one step at a time. You've got this.",
                "neutral": "Thank you for sharing with me. I'm here to listen and support you however you need."
            }
            
            response_text = emotion_responses.get(primary_emotion, emotion_responses["neutral"])
            
            st.markdown(f"**Suggested response for {primary_emotion}:**")
            st.info(response_text)
            
            if st.button(f"🗣️ Speak {primary_emotion.title()} Response", use_container_width=True):
                with st.spinner("🎤 Speaking emotional response..."):
                    try:
                        st.session_state.tts_engine.say(response_text)
                        st.session_state.tts_engine.runAndWait()
                        st.success("✅ Emotional response delivered!")
                    except Exception as e:
                        st.error(f"Speech error: {str(e)}")
        else:
            st.info("Analyze some emotions first to get personalized voice responses!")
    
    except ImportError:
        st.error("❌ Text-to-speech not available. Install pyttsx3: `pip install pyttsx3`")
    except Exception as e:
        st.error(f"❌ Voice assistant error: {str(e)}")

# Tab 6: Analytics Dashboard
with tab6:
    st.markdown("### 📊 Emotion Analytics Dashboard")
    
    if not database or not database.available:
        st.warning("⚠️ Analytics require database connection. Please check your configuration.")
    else:
        # Time range selector
        col1, col2 = st.columns(2)
        with col1:
            days_range = st.selectbox("Time Range", [7, 14, 30, 90], index=2)
        with col2:
            if st.button("🔄 Refresh Data"):
                st.cache_data.clear()
        
        # Get analytics data
        analytics = database.get_emotion_analytics(days_range)
        
        if "error" not in analytics:
            # Emotion distribution pie chart
            st.markdown("#### Emotion Distribution")
            emotion_dist = analytics["emotion_distribution"]
            
            fig_pie = px.pie(
                values=list(emotion_dist.values()),
                names=list(emotion_dist.keys()),
                title=f"Emotions Over Last {days_range} Days"
            )
            fig_pie.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Mood trajectory
            mood_trajectory = analytics.get("mood_trajectory", {})
            if mood_trajectory.get("trajectory") != "insufficient_data":
                st.markdown("#### Mood Trajectory")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Trend", 
                        mood_trajectory["trajectory"].title(),
                        delta="Improving" if mood_trajectory.get("improvement") else "Stable"
                    )
                with col2:
                    recent_score = mood_trajectory.get("recent_mood_score", 0.0)
                    historical_score = mood_trajectory.get("historical_mood_score", 0.0)
                    st.metric(
                        "Recent Score", 
                        f"{recent_score:.2f}",
                        delta=f"{recent_score - historical_score:.2f}"
                    )
                with col3:
                    stability = analytics.get("emotional_stability", {})
                    st.metric(
                        "Stability", 
                        stability.get("stability", "Unknown").title(),
                        delta=f"Diversity: {stability.get('emotion_diversity', 0):.1f}"
                    )
            
            # Weekly patterns
            weekly_patterns = analytics.get("weekly_patterns", {})
            if weekly_patterns.get("pattern") != "insufficient_data":
                st.markdown("#### Weekly Patterns")
                
                weekly_emotions = weekly_patterns.get("weekly_dominant_emotions", {})
                if weekly_emotions:
                    weeks = list(weekly_emotions.keys())
                    emotions = list(weekly_emotions.values())
                    
                    fig_weekly = px.bar(
                        x=weeks,
                        y=[1] * len(weeks),  # Placeholder values
                        color=emotions,
                        title="Dominant Emotions by Week"
                    )
                    fig_weekly.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='white',
                        showlegend=True
                    )
                    st.plotly_chart(fig_weekly, use_container_width=True)
        else:
            st.info("📈 Start tracking your emotions to see analytics here!")

# Tab 5: Personalized Insights
with tab5:
    st.markdown("### 🎯 Personalized Insights")
    
    if database and database.available:
        insights = database.get_personalized_insights(30)
        
        st.markdown("#### Your Emotional Journey")
        for i, insight in enumerate(insights):
            st.markdown(f"""
            <div class="insight-card">
                <strong>Insight #{i+1}</strong><br>
                {insight}
            </div>
            """, unsafe_allow_html=True)
        
        # Recommendations based on patterns
        st.markdown("#### Personalized Recommendations")
        
        analytics = database.get_emotion_analytics(30)
        most_common = analytics.get("most_common_emotion", "neutral")
        
        recommendations = {
            "anxiety": [
                "🧘 Practice daily mindfulness meditation (10-15 minutes)",
                "📱 Try anxiety management apps like Headspace or Calm",
                "🚶 Regular physical exercise can reduce anxiety symptoms",
                "💤 Maintain consistent sleep schedule (7-9 hours)"
            ],
            "sadness": [
                "🌞 Spend time outdoors in natural light",
                "👥 Connect with supportive friends or family",
                "📝 Keep a gratitude journal",
                "🎨 Engage in creative activities you enjoy"
            ],
            "stress": [
                "⏰ Practice time management techniques",
                "🫁 Learn deep breathing exercises",
                "🎵 Listen to calming music",
                "🛁 Create relaxing evening routines"
            ],
            "anger": [
                "🏃 Use physical exercise to release tension",
                "📝 Practice journaling to process emotions",
                "🧘 Try progressive muscle relaxation",
                "💬 Consider talking to a counselor"
            ]
        }
        
        if most_common in recommendations:
            st.markdown(f"**Based on your frequent {most_common} emotions:**")
            for rec in recommendations[most_common]:
                st.markdown(f"• {rec}")
    else:
        st.info("🎯 Connect your database to get personalized insights!")

# Tab 6: Settings and Configuration
with tab6:
    st.markdown("### ⚙️ Settings & Configuration")
    
    # API Configuration
    st.markdown("#### API Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Current Status:**")
        if api_config.openrouter_key:
            st.success("✅ OpenRouter API configured")
        else:
            st.error("❌ OpenRouter API not configured")
        
        if api_config.tenor_key:
            st.success("✅ Tenor API configured")
        else:
            st.error("❌ Tenor API not configured")
    
    with col2:
        st.markdown("**Database Status:**")
        if database and database.available:
            st.success("✅ Database connected")
        else:
            st.error("❌ Database not connected")
    
    # Theme settings
    st.markdown("#### Appearance")
    
    theme_option = st.selectbox(
        "Theme",
        ["Dark", "Light"],
        index=0 if st.session_state.theme == "dark" else 1
    )
    
    if theme_option.lower() != st.session_state.theme:
        st.session_state.theme = theme_option.lower()
        st.rerun()
    
    # Data management
    st.markdown("#### Data Management")
    
    if database and database.available:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Export Data"):
                df = database.get_mood_history(90)
                if not df.empty:
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "Download CSV",
                        csv,
                        "emotion_data.csv",
                        "text/csv"
                    )
                else:
                    st.warning("No data to export")
        
        with col2:
            if st.button("🗑️ Clear Session Data"):
                st.session_state.clear()
                st.success("Session data cleared")
                st.rerun()
    
    # System information
    with st.expander("🔧 System Information"):
        st.markdown(f"""
        - **Session ID:** `{st.session_state.session_id}`
        - **Theme:** {st.session_state.theme.title()}
        - **Model Device:** {'GPU' if get_model_config()['device'] != -1 else 'CPU'}
        - **Cache TTL:** {app_config.cache_ttl} seconds
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: var(--text-secondary); padding: 2rem;">
    <p>🧠 Emotion AI - Your Personal Emotional Intelligence Assistant</p>
    <p><small>Built with ❤️ using Streamlit, Transformers, and advanced AI</small></p>
</div>
""", unsafe_allow_html=True)