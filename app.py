# app.py – Modern UI Rich emotion + YouTube + AI therapy + Mindful-Breathing + Voice Assistant
import streamlit as st
import librosa, torch, numpy as np, pandas as pd
import altair as alt
from transformers import pipeline
import os
import pyttsx3  # Free text-to-speech
import threading
import queue
import requests
import json
from datetime import datetime, timedelta
from supabase import create_client, Client
import datetime as dt
import tempfile, wave, os, io
# Add this import after your existing imports
from gif_generator import display_emotion_gif, create_gif_tab, add_gif_to_emotion_display
from face_emotion_detector import analyze_face, draw_emotion_on_face
# Modern UI Configuration
st.set_page_config(
    page_title="🧠 Emotion AI - Voice & Text Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# Theme toggle CSS variables and styles
def get_css(theme):
    if theme == "light":
        return """
        <style>
        :root {
        --primary: #7f5af0;
        --secondary: #ff6ec4;
        --accent: #ff9f43;
        --bg-dark: #1e1e2f;
        --bg-card: #2a2a3d;
        --text: #e0def4;
        --text-muted: #a6a1b2;
        --border: rgba(127, 90, 240, 0.3);
        --success: #00e676;
        --warning: #ffea00;
        --error: #ff3d00;
        --gradient: linear-gradient(135deg, var(--primary), var(--secondary));
        --glow: 0 0 50px rgba(255, 110, 196, 0.7);
    }
    .stApp {
        background: var(--bg-dark);
        color: var(--text);
        font-family: 'Poppins', 'Inter', sans-serif;
        transition: background 0.5s ease, color 0.5s ease;
        padding: 15px 20px;
    }
    .custom-header {
        background: rgba(42, 42, 61, 0.95);
        border-bottom: 2px solid var(--primary);
        color: var(--text);
        box-shadow: 0 4px 20px rgba(255, 110, 196, 0.4);
        padding: 15px 25px;
        font-weight: 700;
        font-size: 1.4rem;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .status-connected {
        background: var(--success);
        color: white;
        border-radius: 25px;
        padding: 6px 18px;
        font-weight: 800;
        box-shadow: 0 0 15px var(--success);
        transition: background 0.3s ease;
    }
    .status-disconnected {
        background: var(--error);
        color: white;
        border-radius: 25px;
        padding: 6px 18px;
        font-weight: 800;
        box-shadow: 0 0 15px var(--error);
        transition: background 0.3s ease;
    }
    .logo {
        background: var(--gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
        font-size: 2rem;
        letter-spacing: 0.12em;
        user-select: none;
        text-shadow: 0 0 15px rgba(255, 110, 196, 0.9);
    }
    .hero {
        padding: 50px 30px 30px 30px;
        text-align: center;
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(255, 110, 196, 0.5);
        color: white;
        font-weight: 700;
        font-size: 1.8rem;
        letter-spacing: 0.07em;
        margin-bottom: 40px;
        user-select: none;
        text-shadow: 0 0 10px rgba(255, 110, 196, 0.8);
        margin-left: 10px;
        margin-right: 10px;
    }
    .hero p {
        color: rgba(255, 255, 255, 0.9);
        font-weight: 500;
        font-size: 1.2rem;
        margin-top: 15px;
        margin-bottom: 0;
        text-shadow: 0 0 8px rgba(255, 110, 196, 0.7);
    }
    button, .stButton>button {
        background: var(--primary);
        color: white;
        border: none;
        border-radius: 16px;
        padding: 12px 30px;
        font-weight: 800;
        font-size: 1.1rem;
        cursor: pointer;
        box-shadow: 0 8px 20px rgba(255, 110, 196, 0.5);
        transition: background 0.4s ease, box-shadow 0.4s ease;
        margin: 10px 0;
        width: 100%;
        max-width: 300px;
    }
    button:hover, .stButton>button:hover {
        background: var(--secondary);
        box-shadow: 0 10px 30px rgba(255, 159, 67, 0.7);
        transform: translateY(-2px);
    }
    button:active, .stButton>button:active {
        background: var(--accent);
        box-shadow: 0 6px 15px rgba(255, 159, 67, 0.7);
        transform: translateY(0);
    }
    .emotion-score {
        display: flex;
        align-items: center;
        background: var(--bg-card);
        border-radius: 20px;
        padding: 20px 25px;
        box-shadow: 0 8px 25px rgba(255, 110, 196, 0.15);
        margin-bottom: 25px;
        transition: box-shadow 0.4s ease;
    }
    .emotion-score:hover {
        box-shadow: 0 12px 35px rgba(255, 159, 67, 0.3);
    }
    .emotion-icon {
        font-size: 3.5rem;
        margin-right: 25px;
        user-select: none;
        text-shadow: 0 0 10px rgba(255, 110, 196, 0.8);
    }
    .emotion-details {
        flex-grow: 1;
    }
    .emotion-label {
        font-weight: 800;
        font-size: 1.5rem;
        color: var(--primary);
        user-select: none;
    }
    .emotion-percentage {
        font-weight: 700;
        color: var(--text-muted);
    }
    .breathing-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin-top: 30px;
        margin-bottom: 30px;
    }
    .breathing-circle {
        width: 140px;
        height: 140px;
        border-radius: 50%;
        background: var(--primary);
        animation: breathe 5s ease-in-out infinite;
        box-shadow: 0 0 30px var(--primary);
    }
    @keyframes breathe {
        0%, 100% {
            transform: scale(1);
            box-shadow: 0 0 30px var(--primary);
        }
        50% {
            transform: scale(1.3);
            box-shadow: 0 0 60px var(--secondary);
        }
    }
    .breathing-text {
        margin-top: 20px;
        font-weight: 700;
        color: var(--primary);
        user-select: none;
    }
    .breathing-instructions {
        margin-top: 8px;
        font-size: 1rem;
        color: var(--text-muted);
        user-select: none;
    }
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .stApp {
            padding: 10px 15px;
        }
        .custom-header {
            flex-direction: column;
            align-items: flex-start;
            font-size: 1.2rem;
            padding: 10px 15px;
        }
        .hero {
            font-size: 1.4rem;
            padding: 30px 15px 20px 15px;
            margin-bottom: 30px;
        }
        button, .stButton>button {
            font-size: 1rem;
            padding: 10px 20px;
            max-width: 100%;
            width: 100%;
        }
        .emotion-score {
            flex-direction: column;
            align-items: flex-start;
            padding: 15px 20px;
            margin-bottom: 20px;
        }
        .emotion-icon {
            font-size: 2.5rem;
            margin-bottom: 10px;
            margin-right: 0;
        }
        .emotion-label {
            font-size: 1.2rem;
        }
        .breathing-circle {
            width: 100px;
            height: 100px;
        }
        .breathing-text {
            font-size: 1rem;
        }
        .breathing-instructions {
            font-size: 0.9rem;
        }
    }
+    @media (min-width: 769px) and (max-width: 1024px) {
+        .stApp {
+            padding: 15px 25px;
+            max-width: 900px;
+            margin: 0 auto;
+        }
+        .custom-header {
+            font-size: 1.3rem;
+            padding: 15px 20px;
+        }
+        .hero {
+            font-size: 1.6rem;
+            padding: 40px 20px 25px 20px;
+            margin-bottom: 35px;
+        }
+        button, .stButton>button {
+            font-size: 1.05rem;
+            padding: 11px 25px;
+            max-width: 280px;
+            width: 100%;
+        }
+        .emotion-score {
+            padding: 18px 22px;
+            margin-bottom: 22px;
+        }
+        .emotion-icon {
+            font-size: 3rem;
+        }
+        .emotion-label {
+            font-size: 1.3rem;
+        }
+        .breathing-circle {
+            width: 120px;
+            height: 120px;
+        }
+        .breathing-text {
+            font-size: 1.1rem;
+        }
+        .breathing-instructions {
+            font-size: 1rem;
+        }
+    }
+    @media (min-width: 1025px) {
+        .stApp {
+            max-width: 1200px;
+            margin: 0 auto;
+            padding: 20px 30px;
+        }
+        .custom-header {
+            font-size: 1.4rem;
+            padding: 20px 25px;
+        }
+        .hero {
+            font-size: 1.8rem;
+            padding: 50px 30px 30px 30px;
+            margin-bottom: 40px;
+        }
+        button, .stButton>button {
+            font-size: 1.1rem;
+            padding: 12px 30px;
+            max-width: 300px;
+            width: 100%;
+        }
+        .emotion-score {
+            padding: 20px 25px;
+            margin-bottom: 25px;
+        }
+        .emotion-icon {
+            font-size: 3.5rem;
+        }
+        .emotion-label {
+            font-size: 1.5rem;
+        }
+        .breathing-circle {
+            width: 140px;
+            height: 140px;
+        }
+        .breathing-text {
+            font-size: 1.2rem;
+        }
+        .breathing-instructions {
+            font-size: 1rem;
+        }
    </style>
    """

# Initialize theme in session state
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# Theme toggle button in header
# Removed the entire block to keep only one database connected message

# Handle theme toggle button click
if st.button("Toggle Theme", key="toggle_theme_1"):
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
    try:
        st.experimental_rerun()
    except AttributeError:
        st.rerun()

# Initialize theme in session state
if "theme" not in st.session_state:
    st.session_state.theme = "dark"


# Inject CSS based on theme
st.markdown(get_css(st.session_state.theme), unsafe_allow_html=True)

# Font Awesome icons
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
""", unsafe_allow_html=True)

import streamlit as st
import librosa, torch, numpy as np, pandas as pd
import altair as alt
from transformers import pipeline
import os
import pyttsx3  # Free text-to-speech
import threading
import requests
import json
from datetime import datetime, timedelta
from supabase import create_client, Client
import datetime as dt
import tempfile, wave, os, io
# Add this import after your existing imports
from gif_generator import display_emotion_gif, create_gif_tab, add_gif_to_emotion_display

# Supabase configuration
SUPABASE_URL = "https://grhrzxgxazzvnouxczbb.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImdyaHJ6eGd4YXp6dm5vdXhjemJiIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDk2MTkyNDQsImV4cCI6MjA2NTE5NTI0NH0.9Kk47EZJGDrSGWK5m8ntfmF694Z-EulSjzd7DVvBMw8"
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    SUPABASE_AVAILABLE = True
except Exception as e:
    SUPABASE_AVAILABLE = False

# Custom header
st.markdown(f"""
<div class="custom-header">
    <div class="header-content">
        <div class="logo">
            <i class="fas fa-brain"></i>
            Emotion AI
        </div>
        <div class="status-badge status-connected">
            <i class="fas fa-database"></i>
            Database Connected
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Insert weather feature above the "Advanced Emotion Detection" hero section

st.markdown('<div class="modern-card">', unsafe_allow_html=True)
city = st.text_input("Enter your city for weather-based mood suggestions:", value="Mumbai", key="weather_city")
api_key = "24b359c76d994182864153220251507"
weather_url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={city}"
try:
    response = requests.get(weather_url, timeout=10)
    response.raise_for_status()
    weather_data = response.json()
    if weather_data.get("current"):
        weather_main = weather_data["current"]["condition"]["text"]
        temp = weather_data["current"]["temp_c"]
        st.markdown(f"### 🌤️ Current weather in {city.title()}: {weather_main}, {temp}°C")
        # Mood suggestions based on weather
        suggestions_map = {
            "Sunny": "It's a bright day! Perfect time for outdoor activities and positive vibes. 😊",
            "Partly cloudy": "A bit cloudy today. Great day for reflection and cozy indoor activities. ☁️",
            "Cloudy": "Cloudy weather. A good day to relax indoors and reflect. ☁️",
            "Rain": "Rainy weather can be calming. Maybe enjoy some relaxing music or a good book. 🌧️",
            "Light rain": "Light rain outside. A perfect time for a warm drink and some mindfulness. ☕",
            "Thunderstorm": "Stormy weather! Stay safe and consider some calming breathing exercises. ⚡",
            "Snow": "Snowy day! Enjoy the beauty and maybe try some creative indoor hobbies. ❄️",
            "Mist": "Misty weather. Take a slow walk and enjoy the peaceful atmosphere. 🌫️",
            "Fog": "Foggy outside. Be cautious if going out and enjoy some quiet time indoors. 🌁",
            "Haze": "Hazy day. Keep hydrated and take it easy. 🌫️",
        }
        suggestion = suggestions_map.get(weather_main, "Enjoy your day and take care of yourself! 🌟")
        st.info(f"💡 Mood Suggestion: {suggestion}")
    else:
        st.error("Could not retrieve weather information. Please check the city name.")
except Exception as e:
    st.error(f"Error fetching weather data: {str(e)}")
# Removed extra markdown closing div to avoid duplicate text

# Hero section
st.markdown("""
<div class="hero">
    <h1>Advanced Emotion Detection</h1>
    <p>Harness the power of AI to understand emotions through voice and text analysis. Get real-time insights, therapeutic support, and personalized recommendations.</p>
</div>
""", unsafe_allow_html=True)

def log_emotion(label: str, score: float):
    """Insert a single emotion record with error handling."""
    if not SUPABASE_AVAILABLE:
        return
    try:
        data = {
            "created_at": dt.datetime.utcnow().isoformat(),
            "emotion": label,
            "score": float(score),
        }
        supabase.table("mood_history").insert(data).execute()
        st.success(f"✅ Emotion logged: {label} (confidence: {score:.1%})")
    except Exception as e:
        st.warning(f"Could not save to database. App will continue working without saving mood history.")

def get_top_emotion(scores):
    """Get the top emotion from scores with debugging."""
    if not scores:
        return "neutral", 0.0
    # Convert to DataFrame and sort by score
    df = pd.DataFrame(scores).sort_values("score", ascending=False)
    top_emotion = df.iloc[0]["label"]
    top_score = df.iloc[0]["score"]
    return top_emotion, top_score

def display_emotion_score_with_smart_voice(emotion, score, voice_engine=None, user_context=""):
    """Display emotion with AI-powered voice assistant – duplicate-safe keys"""
    import time
    emotion_icons = {
        "joy": "😊", "sadness": "😢", "anger": "😠", "fear": "😨",
        "surprise": "😲", "disgust": "🤢", "love": "❤️", "excitement": "🤩",
        "gratitude": "🙏", "pride": "😤", "optimism": "🌟", "caring": "🤗",
        "neutral": "😐", "admiration": "👏", "approval": "👍", "curiosity": "🤔",
        "desire": "💭", "disappointment": "😞", "disapproval": "👎", "embarrassment": "😳",
        "grief": "😭", "nervousness": "😰", "realization": "💡", "relief": "😌",
        "remorse": "😔", "annoyance": "😤", "confusion": "😕"
    }
    icon = emotion_icons.get(emotion, "🤔")
    
    # Unique key for this button
    unique_key = f"voice_{emotion}_{score}_{int(time.time()*1000)}"

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"""
        <div class="emotion-score">
            <div class="emotion-icon">{icon}</div>
            <div class="emotion-details">
                <div class="emotion-label">{emotion.title()}</div>
                <div class="emotion-percentage">{score:.1%} confidence</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        if st.button("🎙️ Voice Support", key=unique_key):
            with st.spinner("🧠 Generating response..."):
                response_data = get_smart_voice_response(emotion, score, user_context)
            if response_data['ai_powered']:
                st.success("🤖 **AI Voice Assistant Response:**")
            else:
                st.info("💬 **Voice Assistant Response:**")
            st.write(response_data['message'])
            st.info(f"💡 **Suggestion:** {response_data['suggestion']}")
            if voice_engine:
                full_response = f"{response_data['message']} {response_data['suggestion']}"
                speak_text(full_response, voice_engine)
                st.success("🔊 Playing voice response...")
            else:
                st.warning("🔇 Voice engine not available, but text support is shown above.")
                # ---------- Cached models ----------
@st.cache_resource
def load_asr():
    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-base",
        device=0 if torch.cuda.is_available() else -1,
    )

@st.cache_resource
def load_emotion():
    return pipeline(
        "text-classification",
        model="bhadresh-savani/bert-base-go-emotion",
        return_all_scores=True,
        device=0 if torch.cuda.is_available() else -1,
    )

asr_pipe = load_asr()
emo_pipe = load_emotion()
@st.cache_resource
def load_voice_engine():
    """Initialize text-to-speech engine with error logging"""
    import logging
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        # Set a female voice if available
        for voice in voices:
            if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                engine.setProperty('voice', voice.id)
                break
        engine.setProperty('rate', 160)  # Speed
        engine.setProperty('volume', 0.8)  # Volume
        return engine
    except Exception as e:
        logging.error(f"Failed to initialize pyttsx3 voice engine: {e}")
        return None

def get_ai_voice_response(emotion, confidence_score, user_context=""):
    """Get AI-powered emotional support using minimal tokens"""
    # Read your existing OpenRouter key
    try:
        with open("key.txt", encoding="utf-8") as f:
            OPENROUTER_KEY = f.read().strip()
    except FileNotFoundError:
        return None
    # Ultra-compact prompt to minimize token usage
    prompt = f"""You are a compassionate voice assistant. User emotion: {emotion} (confidence: {confidence_score:.1%}). 
    Provide:
    1. Brief supportive message (max 30 words)
    2. One practical suggestion (max 20 words)
    Format: "SUPPORT: [message] | SUGGESTION: [suggestion]"
    Context: {user_context}"""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "openai/gpt-4o-mini",  # Use cheaper model for voice responses
        "messages": [
            {
                "role": "system",
                "content": "You are a warm, supportive voice assistant. Keep responses very brief and actionable."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ],
        "max_tokens": 80,  # Very low token limit
        "temperature": 0.7
    }
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        ai_response = response.json()["choices"][0]["message"]["content"]
        # Parse the structured response
        if "SUPPORT:" in ai_response and "SUGGESTION:" in ai_response:
            parts = ai_response.split("SUGGESTION:")
            support_msg = parts[0].replace("SUPPORT:", "").strip()
            suggestion = parts[1].strip()
        else:
            # Fallback parsing
            support_msg = ai_response[:100] + "..." if len(ai_response) > 100 else ai_response
            suggestion = "Take a moment to breathe and be kind to yourself."
        return {
            "message": support_msg,
            "suggestion": suggestion,
            "ai_powered": True
        }
    except Exception as e:
        return None

# Enhanced fallback responses (if AI fails)
FALLBACK_RESPONSES = {
    "sadness": {
        "message": "I understand you're feeling sad. These feelings are valid and temporary. You have the strength to get through this.",
        "suggestion": "Try deep breathing or reach out to someone who cares about you."
    },
    "anger": {
        "message": "I can sense your anger. It's okay to feel this way. Let's work together to process it healthily.",
        "suggestion": "Take 10 slow breaths or go for a short walk to release this energy."
    },
    "fear": {
        "message": "Fear can be overwhelming, but you're not alone. You've overcome challenges before.",
        "suggestion": "Ground yourself: name 5 things you see, 4 you can touch, 3 you hear."
    },
    "anxiety": {
        "message": "Anxiety feels scary, but you are safe right now. Let's bring you back to the present.",
        "suggestion": "Try 4-7-8 breathing: inhale 4, hold 7, exhale 8 counts."
    },
    "stress": {
        "message": "Stress is your body responding to challenges. You're doing better than you think.",
        "suggestion": "Break tasks into smaller pieces and take regular breaks."
    },
    "joy": {
        "message": "I'm glad you're feeling joyful! This positive energy is wonderful and well-deserved.",
        "suggestion": "Share this joy with someone or take a moment to appreciate what brought it."
    },
    "neutral": {
        "message": "You seem balanced, which is actually quite peaceful. This gives you space to reflect.",
        "suggestion": "Use this calm moment for gentle self-reflection or planning something positive."
    }
}

def get_smart_voice_response(emotion, confidence_score, user_context=""):
    """Smart voice assistant that uses AI when available, fallback otherwise"""
    # Try AI response first (using your existing OpenRouter key)
    ai_response = get_ai_voice_response(emotion, confidence_score, user_context)
    if ai_response:
        return ai_response
    # Fallback to pre-programmed responses
    emotion_mapping = {
        "sadness": "sadness", "grief": "sadness", "disappointment": "sadness",
        "anger": "anger", "annoyance": "anger", "disapproval": "anger",
        "fear": "fear", "nervousness": "anxiety", "confusion": "anxiety",
        "joy": "joy", "excitement": "joy", "optimism": "joy", "gratitude": "joy",
        "love": "joy", "admiration": "joy", "pride": "joy", "approval": "joy",
        "caring": "joy", "curiosity": "neutral", "surprise": "neutral",
        "neutral": "neutral", "realization": "neutral"
    }
    mapped_emotion = emotion_mapping.get(emotion.lower(), "neutral")
    fallback_data = FALLBACK_RESPONSES.get(mapped_emotion, FALLBACK_RESPONSES["neutral"])
    return {
        "message": fallback_data["message"],
        "suggestion": fallback_data["suggestion"],
        "ai_powered": False
    }

# Speech queue and background thread for pyttsx3 to avoid "run loop already started" error
speech_queue = queue.Queue()

def speech_worker(voice_engine):
    import logging
    while True:
        text = speech_queue.get()
        if text is None:
            break  # Exit signal
        try:
            voice_engine.say(text)
            voice_engine.runAndWait()
        except Exception as e:
            logging.error(f"Error during speech synthesis: {e}")
        speech_queue.task_done()

# Start speech worker thread once
speech_thread = None
def start_speech_thread(voice_engine):
    global speech_thread
    if speech_thread is None or not speech_thread.is_alive():
        speech_thread = threading.Thread(target=speech_worker, args=(voice_engine,), daemon=True)
        speech_thread.start()

def speak_text(text, voice_engine):
    """Enqueue text to be spoken by the speech worker thread"""
    if voice_engine:
        if speech_thread is None or not speech_thread.is_alive():
            start_speech_thread(voice_engine)
        speech_queue.put(text)

def add_voice_to_chat_response(chat_response, voice_engine):
    """Add voice synthesis to chat responses"""
    if voice_engine and st.button("🎙️ Read Response Aloud"):
        speak_text(chat_response, voice_engine)
        st.success("🔊 Playing response...")

# Initialize voice engine
voice_engine = load_voice_engine()

import logging
if voice_engine is None:
    st.warning("🔇 Voice engine failed to initialize. Please ensure pyttsx3 is installed and working.")
else:
    st.info("✅ Voice engine initialized successfully.")

# ---------- Chart styling ----------
def show_emotion_chart(scores, top_k=8):
    df = pd.DataFrame(scores).sort_values("score", ascending=False).head(top_k)
    # Create gradient colors
    colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#00f2fe', 
              '#43e97b', '#38f9d7', '#ffecd2']
    bars = (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("score:Q", scale=alt.Scale(domain=[0, 1]), title="Confidence Score"),
            y=alt.Y("label:N", sort="-x", title="Emotions"),
            color=alt.Color("label:N", 
                          scale=alt.Scale(range=colors), 
                          legend=None),
            tooltip=[
                alt.Tooltip("label:N", title="Emotion"),
                alt.Tooltip("score:Q", format=".1%", title="Confidence")
            ],
        )
        .properties(
            height=400,
            width=600,
            title=alt.TitleParams(
                text="Emotion Analysis Results",
                fontSize=16,
                fontWeight='bold',
                color='#ffffff'
            )
        )
        .configure_axis(
            labelColor='#a0a0a0',
            titleColor='#ffffff',
            gridColor='rgba(255,255,255,0.1)',
            domainColor='rgba(255,255,255,0.1)'
        )
        .configure_view(
            strokeWidth=0
        )
        .configure(
            background='transparent'
        )
    )
    st.altair_chart(bars, use_container_width=True)

# ---------- Mindful-Breathing component ----------
def show_breathing_exercise():
    st.markdown("""
    <div class="breathing-container">
        <div class="breathing-circle"></div>
        <div class="breathing-text">Breathe with the circle</div>
        <div class="breathing-instructions">
            Follow the gentle expansion and contraction to find your calm center
        </div>
    </div>
    """, unsafe_allow_html=True)

trigger_emotions = {"stress", "anger", "sadness", "fear", "nervousness", "grief", "remorse"}

def maybe_breathing_popup(scores):
    top_emotion = pd.DataFrame(scores).iloc[0]["label"]
    if top_emotion.lower() in trigger_emotions:
        st.markdown("---")
        st.subheader("🫁 Mindful Breathing Exercise")
        st.info("We detected you might benefit from a moment of calm. Try this breathing exercise:")
        show_breathing_exercise()

# ---------- Content with modern container ----------
st.markdown('<div class="content-container">', unsafe_allow_html=True)

# ---------- Tabs ----------
tab_audio, tab_text, tab_yt, tab_chat, tab_voice, tab_history, tab_face = st.tabs([
    "🎙️ Voice Analysis",
    "📝 Text Analysis",
    "🎵 Music Therapy",
    "🧠 AI Therapist",
    "🎤 Voice Assistant",
    "📊 Mood History",
    "😊 Face Emotion"
])

# ---------------- TAB 1 : Audio ----------------
with tab_audio:
    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st.markdown("### 🎙️ Voice Emotion Detection")
    st.markdown("Upload an audio file or record your voice to analyze emotional content.")
    col1, col2 = st.columns(2)
    with col1:
        audio_file = st.file_uploader("Upload Audio File", type=["wav", "mp3", "m4a"])
    with col2:
        mic_audio = st.audio_input("🎤 Record Voice")
        st.caption("📱 On mobile: tap the red 🎤 button → allow microphone → tap again to record.")
    st.markdown('</div>', unsafe_allow_html=True) 

    audio_bytes = mic_audio or audio_file
    if audio_bytes:
        st.markdown('<div class="modern-card">', unsafe_allow_html=True)
        with st.spinner("🎵 Processing audio..."):
            y, sr = librosa.load(audio_bytes, sr=16000)
        with st.spinner("🗣️ Transcribing speech..."):
            transcript = asr_pipe(y)["text"].strip()
        st.success(f"**Transcript:** {transcript}")
        with st.spinner("🧠 Analyzing emotions..."):
            scores = emo_pipe(transcript)[0]
        st.session_state.last_scores = scores
        top_emotion, top_score = get_top_emotion(scores)
        # Display top emotion with modern styling
        display_emotion_score_with_smart_voice(top_emotion, top_score, voice_engine, transcript)
  
        # Add GIF display to audio tab
        add_gif_to_emotion_display(top_emotion, top_score, show_button=False)

        # Log to database
        log_emotion(top_emotion, top_score)
        # Show chart
        show_emotion_chart(scores)
        # Breathing exercise if needed
        maybe_breathing_popup(scores)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("🎤 Choose an audio file or record your voice to get started")

        #----------------------------#
        # Add this after tab_voice section


# ---------------- TAB 2 : Text ----------------
with tab_text:
    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st.markdown("### 📝 Text Emotion Analysis")
    st.markdown("Analyze the emotional tone of any text content.")
    raw_text = st.text_area("Enter your text here:", height=150, placeholder="Type or paste your text here...")
    if st.button("🔍 Analyze Emotions", use_container_width=True) and raw_text.strip():
        with st.spinner("🧠 Processing text..."):
            scores = emo_pipe(raw_text.strip())[0]
        st.session_state.last_scores = scores
        top_emotion, top_score = get_top_emotion(scores)
        # Display results
        display_emotion_score_with_smart_voice(top_emotion, top_score, voice_engine, raw_text)
        # Add this line after emotion display in audio and text tabs
        add_gif_to_emotion_display(top_emotion, top_score, show_button=False)
        log_emotion(top_emotion, top_score)
        show_emotion_chart(scores)
        maybe_breathing_popup(scores)
    elif not raw_text.strip():
        st.info("📝 Enter some text to analyze its emotional content")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- TAB 3 : Music Therapy ----------------
with tab_yt:
    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st.markdown("### 🎵 Personalized Music Therapy")
    st.markdown("Get music recommendations based on your detected emotions.")
    
    if "last_scores" not in st.session_state:
        st.info("🎭 Analyze some audio or text first to get personalized music recommendations")
    else:
        top_emotion = pd.DataFrame(st.session_state.last_scores).sort_values("score", ascending=False).iloc[0]["label"]
        # Display current emotion
        top_score = pd.DataFrame(st.session_state.last_scores).sort_values("score", ascending=False).iloc[0]["score"]
        display_emotion_score_with_smart_voice(top_emotion, top_score, voice_engine)
        
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
        query = search_map.get(top_emotion, f"{top_emotion} therapeutic music")
        st.markdown(f"**Recommended search:** *{query}*")
        
        if st.button("🎵 Find Healing Music", use_container_width=True):
            with st.spinner("🎵 Searching for therapeutic music..."):
                try:
                    import yt_dlp
                    ydl_opts = {
                        "format": "bestaudio/best",
                        "quiet": True,
                        "noplaylist": True,
                        "extractaudio": True,
                        "audioformat": "mp3",
                    }
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        results = ydl.extract_info(f"ytsearch1:{query}", download=False)
                        info = results["entries"][0]
                        url = info["url"]
                        title = info["title"]
                    st.success(f"🎵 Now playing: **{title}**")
                    st.audio(url, format="audio/mp3")
                except Exception as e:
                    st.error(f"Unable to load music: {str(e)}")
                    st.info("Try searching manually on your preferred music platform")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- TAB 4 : AI Therapist ----------------
with tab_chat:
    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st.markdown("### 🧠 AI Therapeutic Support")
    st.markdown("Get personalized emotional support and cognitive behavioral therapy guidance.")
    
    # Read OpenRouter key
    try:
        with open("key.txt", encoding="utf-8") as f:
            OPENROUTER_KEY = f.read().strip()
    except FileNotFoundError:
        st.error("🔑 API key not found. Create a 'key.txt' file with your OpenRouter API key.")
        st.stop()
    
    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
    MODEL = "openai/gpt-4o"
    
    # Initialize session state
    if "chat_msgs" not in st.session_state:
        st.session_state.chat_msgs = [
            {
                "role": "system",
                "content": (
                    "You are a compassionate AI therapist specializing in emotional wellness. "
                    "Use evidence-based CBT techniques: validate emotions, explore triggers gently, "
                    "offer reframing perspectives, and suggest practical coping strategies. "
                    "Keep responses warm, supportive, and concise (≤150 words). "
                    "Always prioritize user safety and well-being."
                )
            }
        ]
    
    def get_ai_response(messages):
        headers = {
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": MODEL, 
            "messages": messages, 
            "max_tokens": 250,
            "temperature": 0.7
        }
        try:
            response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"I'm experiencing technical difficulties. Please try again. Error: {str(e)}"
    
    # Display chat history
    for msg in st.session_state.chat_msgs[1:]:  # Skip system message
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    # Chat input
    if prompt := st.chat_input("Share what's on your mind..."):
        # Add user message
        st.session_state.chat_msgs.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = get_ai_response(st.session_state.chat_msgs)
            st.write(response)
            # Add voice synthesis button
            add_voice_to_chat_response(response, voice_engine)
        
        # Add assistant response
        st.session_state.chat_msgs.append({"role": "assistant", "content": response})
    
    # Clear chat button
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_msgs = st.session_state.chat_msgs[:1]  # Keep system message
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- TAB 5 : Voice Assistant ----------------
with tab_voice:
    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st.markdown("### 🎙️ AI Voice Assistant")
    st.markdown("Get personalized emotional support with AI-powered responses and voice synthesis.")
    
    # Quick emotion support
    st.markdown("#### Quick Emotional Support")
    col1, col2 = st.columns(2)
    with col1:
        quick_emotion = st.selectbox(
            "How are you feeling?",
            ["sadness", "anger", "fear", "anxiety", "stress", "disappointment", "joy", "neutral", "confusion", "excitement"]
        )
    with col2:
        confidence = st.slider("How strongly?", 0.1, 1.0, 0.7)
    
    user_context = st.text_area("Tell me more about your situation (optional):", height=100)
    
    if st.button("🎙️ Get AI Voice Support", use_container_width=True):
        with st.spinner("🧠 AI is crafting your response..."):
            response_data = get_smart_voice_response(quick_emotion, confidence, user_context)
        
        # Display AI or fallback indicator
        if response_data['ai_powered']:
            st.success("🤖 **AI-Powered Response:**")
        else:
            st.info("💬 **Support Response:**")
        
        st.write(response_data['message'])
        st.info(f"💡 **Suggestion:** {response_data['suggestion']}")  
        # Auto-play voice response
        if voice_engine:
            full_response = f"{response_data['message']} {response_data['suggestion']}"
            speak_text(full_response, voice_engine)
            st.success("🔊 Playing voice response...")
        else:
            st.warning("🔇 Install `pyttsx3` for voice support: `pip install pyttsx3`")
    # Voice settings
    st.markdown("#### Voice Settings")
    if voice_engine:
        st.success("✅ Voice engine ready")
        if st.button("🎤 Test Voice"):
            speak_text("Hello! I'm your AI emotional support assistant. I'm here to help you feel better.", voice_engine)
    else:
        st.error("❌ Voice engine not available")
        st.info("Install: `pip install pyttsx3`")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- TAB 6 : Mood History ----------------
with tab_history:
    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Mood Analytics Dashboard")
    st.markdown("Track your emotional patterns and progress over time.")
    
    if not SUPABASE_AVAILABLE:
        st.error("📊 Database connection required for mood tracking")
        st.info("Please check your database configuration to view mood history.")
    else:
        try:
            # Fetch mood history
            response = supabase.table("mood_history").select("*").order("created_at", desc=True).limit(100).execute()
            rows = response.data
            
            if rows:
                df_hist = pd.DataFrame(rows)
                df_hist['created_at'] = pd.to_datetime(df_hist['created_at'])
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Entries", len(df_hist))
                with col2:
                    most_common = df_hist['emotion'].mode().iloc[0] if not df_hist.empty else "N/A"
                    st.metric("Most Common Emotion", most_common)
                with col3:
                    avg_score = df_hist['score'].mean() if not df_hist.empty else 0
                    st.metric("Average Confidence", f"{avg_score:.1%}")
                with col4:
                    days_tracked = (df_hist['created_at'].max() - df_hist['created_at'].min()).days + 1
                    st.metric("Days Tracked", days_tracked)
                
                # Recent entries table
                st.markdown("#### Recent Mood Entries")
                display_df = df_hist.head(10)[['created_at', 'emotion', 'score']].copy()
                display_df['created_at'] = display_df['created_at'].dt.strftime('%Y-%m-%d %H:%M')
                display_df['score'] = display_df['score'].apply(lambda x: f"{x:.1%}")
                display_df.columns = ['Timestamp', 'Emotion', 'Confidence']
                st.dataframe(display_df, use_container_width=True)
                
                # Mood timeline chart
                st.markdown("#### Mood Timeline")
                if len(df_hist) > 1:
                    df_chart = df_hist.sort_values('created_at')
                    
                    chart = alt.Chart(df_chart).mark_line(
                        point=alt.OverlayMarkDef(filled=True, size=100)
                    ).encode(
                        x=alt.X('created_at:T', title='Time'),
                        y=alt.Y('score:Q', title='Confidence Score', scale=alt.Scale(domain=[0, 1])),
                        color=alt.Color('emotion:N', title='Emotion'),
                        tooltip=['created_at:T', 'emotion:N', 'score:Q']
                    ).properties(
                        height=400,
                        title='Mood Score Over Time'
                    ).configure_axis(
                        labelColor='#a0a0a0',
                        titleColor='#ffffff',
                        gridColor='rgba(255,255,255,0.1)'
                    ).configure_view(
                        strokeWidth=0
                    ).configure(
                        background='transparent'
                    )
                    st.altair_chart(chart, use_container_width=True)
                
                # Emotion distribution
                st.markdown("#### Emotion Distribution")
                emotion_counts = df_hist['emotion'].value_counts()
                pie_chart = alt.Chart(
                    pd.DataFrame({
                        'emotion': emotion_counts.index,
                        'count': emotion_counts.values
                    })
                ).mark_arc(innerRadius=50).encode(
                    theta=alt.Theta('count:Q'),
                    color=alt.Color('emotion:N', legend=alt.Legend(orient='right')),
                    tooltip=['emotion:N', 'count:Q']
                ).properties(
                    height=400,
                    title='Emotion Distribution'
                ).configure_axis(
                    labelColor='#a0a0a0',
                    titleColor='#ffffff'
                ).configure_view(
                    strokeWidth=0
                ).configure(
                    background='transparent'
                )
                st.altair_chart(pie_chart, use_container_width=True)
            else:
                st.info("🎭 No mood entries yet. Start analyzing some audio or text to build your mood history!")
        except Exception as e:
            st.error(f"Database error: {str(e)}")
            st.info("**Troubleshooting:**")
            st.info("• Check if 'mood_history' table exists")
            st.info("• Verify database permissions")
            st.info("• Ensure table structure: id, created_at, emotion, score")
    st.markdown('</div>', unsafe_allow_html=True)

# Remove the weather tab section and add weather feature in sidebar top right area
# Remove the entire block of weather tab

# Remove weather feature from sidebar and add to top right of main screen
# Use columns layout to place weather feature on right side

# Place weather feature at the top right above main content
col_top_left, col_top_right = st.columns([4, 1])
# Remove duplicate weather feature at bottom if any
# The weather feature is now only present above the "Advanced Emotion Detection" heading

# ---------------- TAB 7 : Face Emotion Detector ----------------
with tab_face:
    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st.markdown("### 😊 Face Emotion Detection")
    st.markdown("Upload an image to detect emotions from faces.")

    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        if st.button("Analyze Face", use_container_width=True):
            with st.spinner("Analyzing face..."):
                # To avoid re-reading the file, we use the `getvalue()` method
                image_bytes = uploaded_image.getvalue()

                # We need to pass a file-like object to the analyze_face function

                result = analyze_face(io.BytesIO(image_bytes))

                if result:
                    st.success("Analysis complete!")

                    # We need to pass a file-like object to the draw_emotion_on_face function
                    processed_image = draw_emotion_on_face(io.BytesIO(image_bytes), result)

                    if processed_image is not None:
                        st.image(processed_image, caption="Processed Image", use_column_width=True)

                    st.write("### Analysis Results")
                    for i, face in enumerate(result):
                        st.write(f"**Face {i+1}:**")
                        st.write(f"- Emotion: {face['dominant_emotion']}")
                        st.write("- Region:")
                        st.json(face['region'])

    st.markdown('</div>', unsafe_allow_html=True)

# No extra closing div markdown at the bottom to avoid duplicate text or errors



    



    #-----------------------------------------------------------#
    # ---------------- SIDEBAR : Mood-Food Deep-Link ----------------
with st.sidebar.expander("🍽️ Mood Food", expanded=True):
    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st.subheader("Order by Mood")

    if "last_scores" not in st.session_state:
        st.info("🎭 Detect emotion first to see food suggestions.")
    else:
        top_emotion = pd.DataFrame(st.session_state.last_scores).iloc[0]["label"]
        dish_map = {
            "sadness": "comfort pasta",
            "anger": "spicy wings",
            "joy": "celebration cake",
            "stress": "sushi bowl",
            "fear": "warm soup",
            "neutral": "balanced wrap"
        }
        # Fix default dish to string
        dish = st.selectbox("Pick a dish", [dish_map.get(top_emotion, "chef special")])
        city = st.text_input("City", value="Mumbai")

        # Format city for URL
        city_url = city.strip().lower().replace(" ", "-")

        col1, col2 = st.columns(2)
        # Use anchor tags with target _blank to open in new tabs
        swiggy_url = f"https://www.swiggy.com/search?q={dish.replace(' ', '+')}+{city_url}"
        zomato_url = f"https://www.zomato.com/{city_url}/search?q={dish.replace(' ', '+')}"

        if col1.button("🛒 Swiggy"):
            st.markdown(f'<a href="{swiggy_url}" target="_blank" rel="noopener noreferrer">Open Swiggy</a>', unsafe_allow_html=True)
        if col2.button("🛒 Zomato"):
            st.markdown(f'<a href="{zomato_url}" target="_blank" rel="noopener noreferrer">Open Zomato</a>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
