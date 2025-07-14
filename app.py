    # app.py – Modern UI Rich emotion + YouTube + AI therapy + Mindful-Breathing
import streamlit as st
import librosa, torch, numpy as np, pandas as pd
import altair as alt
from transformers import pipeline
import os

from supabase import create_client, Client
import datetime as dt

# Modern UI Configuration
st.set_page_config(
    page_title="🧠 Emotion AI - Voice & Text Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    :root {
        --primary: #667eea;
        --secondary: #764ba2;
        --accent: #f093fb;
        --bg-dark: #0a0a0f;
        --bg-card: rgba(255, 255, 255, 0.05);
        --text: #ffffff;
        --text-muted: #a0a0a0;
        --border: rgba(255, 255, 255, 0.1);
        --success: #00d4aa;
        --warning: #ffd700;
        --error: #ff6b6b;
        --gradient: linear-gradient(135deg, var(--primary), var(--secondary));
        --glow: 0 0 30px rgba(102, 126, 234, 0.3);
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container */
    .main .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }
    
    /* Background animation */
    .stApp {
        background: var(--bg-dark);
        font-family: 'Inter', sans-serif;
    }
    
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            radial-gradient(circle at 20% 30%, rgba(102, 126, 234, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 70%, rgba(118, 75, 162, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 50% 50%, rgba(240, 147, 251, 0.05) 0%, transparent 50%);
        animation: float 20s ease-in-out infinite;
        z-index: -1;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-20px) rotate(180deg); }
    }

    /* Custom header */
    .custom-header {
        background: rgba(10, 10, 15, 0.8);
        backdrop-filter: blur(20px);
        border-bottom: 1px solid var(--border);
        padding: 20px 40px;
        position: sticky;
        top: 0;
        z-index: 100;
        margin-bottom: 40px;
    }
    
    .header-content {
        display: flex;
        align-items: center;
        justify-content: space-between;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    .logo {
        display: flex;
        align-items: center;
        gap: 15px;
        font-size: 28px;
        font-weight: 700;
        background: var(--gradient);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .logo i {
        background: var(--gradient);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 32px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.1); }
    }
    
    .status-badge {
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .status-connected {
        background: rgba(0, 212, 170, 0.2);
        color: var(--success);
        border: 1px solid var(--success);
    }
    
    .status-disconnected {
        background: rgba(255, 107, 107, 0.2);
        color: var(--error);
        border: 1px solid var(--error);
    }

    /* Hero section */
    .hero {
        text-align: center;
        padding: 60px 40px;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    .hero h1 {
        font-size: clamp(2.5rem, 5vw, 4rem);
        font-weight: 800;
        margin-bottom: 20px;
        background: linear-gradient(135deg, #667eea, #764ba2, #f093fb);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 3s ease-in-out infinite;
    }
    
    @keyframes shimmer {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .hero p {
        font-size: 1.25rem;
        color: var(--text-muted);
        max-width: 600px;
        margin: 0 auto 40px;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent;
        border-bottom: 1px solid var(--border);
        gap: 0;
        justify-content: center;
        margin-bottom: 40px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border: none;
        color: var(--text-muted);
        padding: 15px 25px;
        font-size: 16px;
        font-weight: 500;
        transition: all 0.3s ease;
        position: relative;
        height: auto;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text);
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] {
        color: var(--primary) !important;
        background: transparent !important;
    }
    
    .stTabs [aria-selected="true"]::after {
        content: '';
        position: absolute;
        bottom: -1px;
        left: 0;
        right: 0;
        height: 2px;
        background: var(--gradient);
        border-radius: 2px 2px 0 0;
    }

    /* Content containers */
    .content-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 40px;
    }

    /* Modern cards */
    .modern-card {
        background: var(--bg-card);
        border-radius: 20px;
        padding: 30px;
        margin-bottom: 30px;
        border: 1px solid var(--border);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .modern-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: var(--gradient);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .modern-card:hover {
        transform: translateY(-5px);
        box-shadow: var(--glow);
        border-color: var(--primary);
    }
    
    .modern-card:hover::before {
        opacity: 1;
    }

    /* Buttons */
    .stButton button {
        background: var(--gradient);
        border: none;
        border-radius: 12px;
        color: white;
        font-weight: 600;
        padding: 12px 24px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }

    /* File uploader */
    .stFileUploader {
        background: var(--bg-card);
        border-radius: 15px;
        border: 2px dashed var(--border);
        padding: 30px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: var(--primary);
        background: rgba(102, 126, 234, 0.05);
    }

    /* Audio input */
    .stAudioInput {
        background: var(--bg-card);
        border-radius: 15px;
        border: 2px solid var(--border);
        padding: 20px;
        margin: 20px 0;
    }

    /* Text area */
    .stTextArea textarea {
        background: var(--bg-card);
        border: 2px solid var(--border);
        border-radius: 15px;
        color: var(--text);
        font-family: 'Inter', sans-serif;
        padding: 15px;
        transition: all 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }

    /* Charts */
    .stPlotlyChart {
        background: var(--bg-card);
        border-radius: 15px;
        border: 1px solid var(--border);
        padding: 20px;
        margin: 20px 0;
    }

    /* Dataframe */
    .stDataFrame {
        background: var(--bg-card);
        border-radius: 15px;
        border: 1px solid var(--border);
        overflow: hidden;
    }

    /* Success/Error messages */
    .stSuccess {
        background: rgba(0, 212, 170, 0.1);
        border: 1px solid var(--success);
        border-radius: 10px;
        color: var(--success);
    }
    
    .stError {
        background: rgba(255, 107, 107, 0.1);
        border: 1px solid var(--error);
        border-radius: 10px;
        color: var(--error);
    }
    
    .stWarning {
        background: rgba(255, 215, 0, 0.1);
        border: 1px solid var(--warning);
        border-radius: 10px;
        color: var(--warning);
    }
    
    .stInfo {
        background: rgba(102, 126, 234, 0.1);
        border: 1px solid var(--primary);
        border-radius: 10px;
        color: var(--primary);
    }

    /* Breathing animation */
    .breathing-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 40px;
        background: var(--bg-card);
        border-radius: 20px;
        border: 1px solid var(--border);
        margin: 20px 0;
    }
    
    .breathing-circle {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        background: radial-gradient(circle, #a1c4fd, #c2e9fb);
        margin-bottom: 20px;
        animation: breathe 4s ease-in-out infinite;
    }
    
    @keyframes breathe {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.2); }
    }
    
    .breathing-text {
        font-size: 20px;
        font-weight: 600;
        color: var(--text);
        margin-bottom: 10px;
    }
    
    .breathing-instructions {
        color: var(--text-muted);
        text-align: center;
        max-width: 300px;
    }

    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .custom-header {
            padding: 15px 20px;
        }
        
        .header-content {
            flex-direction: column;
            gap: 15px;
        }
        
        .logo {
            font-size: 24px;
        }
        
        .hero {
            padding: 40px 20px;
        }
        
        .hero h1 {
            font-size: 2rem;
        }
        
        .hero p {
            font-size: 1rem;
        }
        
        .content-container {
            padding: 0 20px;
        }
        
        .modern-card {
            padding: 20px;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 12px 16px;
            font-size: 14px;
        }
    }

    /* Loading spinner */
    .stSpinner {
        border: 3px solid rgba(102, 126, 234, 0.3);
        border-top: 3px solid var(--primary);
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* Chat messages */
    .stChatMessage {
        background: var(--bg-card);
        border-radius: 15px;
        border: 1px solid var(--border);
        margin: 10px 0;
    }
    
    .stChatInput {
        background: var(--bg-card);
        border: 2px solid var(--border);
        border-radius: 15px;
        transition: all 0.3s ease;
    }
    
    .stChatInput:focus-within {
        border-color: var(--primary);
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }

    /* Emotion score display */
    .emotion-score {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 15px;
        background: var(--bg-card);
        border-radius: 10px;
        border: 1px solid var(--border);
        margin: 10px 0;
    }
    
    .emotion-icon {
        font-size: 24px;
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: var(--gradient);
        border-radius: 50%;
        color: white;
    }
    
    .emotion-details {
        flex: 1;
    }
    
    .emotion-label {
        font-weight: 600;
        color: var(--text);
        margin-bottom: 5px;
    }
    
    .emotion-percentage {
        color: var(--text-muted);
        font-size: 14px;
    }

    /* Progress bars */
    .stProgress > div > div {
        background: var(--gradient);
        border-radius: 10px;
    }
    
    .stProgress > div {
        background: var(--bg-card);
        border-radius: 10px;
        border: 1px solid var(--border);
    }
</style>
""", unsafe_allow_html=True)

# Font Awesome icons
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
""", unsafe_allow_html=True)

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
        <div class="status-badge {'status-connected' if SUPABASE_AVAILABLE else 'status-disconnected'}">
            <i class="fas fa-{'database' if SUPABASE_AVAILABLE else 'exclamation-triangle'}"></i>
            {'Database Connected' if SUPABASE_AVAILABLE else 'Database Offline'}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

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

def display_emotion_score(emotion, score):
    """Display emotion with modern styling"""
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
    
    st.markdown(f"""
    <div class="emotion-score">
        <div class="emotion-icon">{icon}</div>
        <div class="emotion-details">
            <div class="emotion-label">{emotion.title()}</div>
            <div class="emotion-percentage">{score:.1%} confidence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

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
tab_audio, tab_text, tab_yt, tab_chat, tab_history = st.tabs([
    "🎙️ Voice Analysis", 
    "📝 Text Analysis", 
    "🎵 Music Therapy",
    "🧠 AI Therapist", 
    "📊 Mood History"
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
        display_emotion_score(top_emotion, top_score)
        
        # Log to database
        log_emotion(top_emotion, top_score)
        
        # Show chart
        show_emotion_chart(scores)
        
        # Breathing exercise if needed
        maybe_breathing_popup(scores)
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("🎤 Choose an audio file or record your voice to get started")

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
        display_emotion_score(top_emotion, top_score)
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
        display_emotion_score(top_emotion, top_score)
        
        search_map = {
            "admiration": "inspiring uplifting songs",
            "anger": "calming peaceful music",
            "annoyance": "soothing relaxation music",
            "approval": "feel good positive songs",
            "caring": "warm loving songs",
            "confusion": "clarity focus music",
            "curiosity": "discovery adventure music",
            "desire": "romantic ambient songs",
            "disappointment": "uplifting motivational music",
            "disapproval": "positive affirmation music",
            "disgust": "cleansing purifying music",
            "embarrassment": "confidence building music",
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
        import requests
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
        
        # Add assistant response
        st.session_state.chat_msgs.append({"role": "assistant", "content": response})

    # Clear chat button
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_msgs = st.session_state.chat_msgs[:1]  # Keep system message
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- TAB 5 : Mood History ----------------
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
                if len(df_hist) > 1:
                    st.markdown("#### Mood Timeline")
                    df_chart = df_hist.sort_values('created_at')
                    
                    # Create timeline chart
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

# Close content container
st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; padding: 40px 0; color: var(--text-muted); border-top: 1px solid var(--border); margin-top: 60px;">
    <p>Built with ❤️ using Streamlit • Powered by AI for emotional wellness</p>
</div>
""", unsafe_allow_html=True)

# Add motivation functionality
motiv_map = {
    "sadness": "motivational podcast for sadness",
    "disappointment": "overcoming disappointment motivational",
    "grief": "grief healing motivational talk",
    "remorse": "forgiveness motivational video",
    "stress": "stress relief motivational podcast",
    "anger": "calming anger motivational talk",
    "fear": "face fear motivational video",
}

def maybe_motivation(scores):
    top_emotion = pd.DataFrame(scores).iloc[0]["label"]
    if top_emotion in motiv_map:
        query = motiv_map[top_emotion]
        st.markdown("---")
        st.markdown("### 🎙️ Motivational Support")
        st.info(f"Based on your {top_emotion} emotion, here's some motivational content:")
        
        if st.button("▶️ Find Motivation", use_container_width=True):
            with st.spinner("🎯 Finding motivational content..."):
                try:
                    import yt_dlp
                    ydl_opts = {
                        "format": "best",
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
                    
                    st.success(f"🎥 Now playing: **{title}**")
                    st.video(url)
                    
                except Exception as e:
                    st.error(f"Unable to load video: {str(e)}")
                    st.info("Try searching manually on YouTube or other platforms")

# Add JavaScript for enhanced interactivity
st.markdown("""
<script>
    // Add smooth scrolling
    document.addEventListener('DOMContentLoaded', function() {
        // Smooth scroll for internal links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                document.querySelector(this.getAttribute('href')).scrollIntoView({
                    behavior: 'smooth'
                });
            });
        });
        
        // Add loading states to buttons
        document.querySelectorAll('button').forEach(button => {
            button.addEventListener('click', function() {
                if (!this.disabled) {
                    this.style.opacity = '0.7';
                    setTimeout(() => {
                        this.style.opacity = '1';
                    }, 2000);
                }
            });
        });
    });
</script>
""", unsafe_allow_html=True)