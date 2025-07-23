# app_enhanced_music.py - Enhanced app with improved music playback system
import streamlit as st
import pandas as pd
import numpy as np
import librosa
import torch
from datetime import datetime
import logging

# Import enhanced music components
from utils.enhanced_music_ui import create_enhanced_music_therapy_tab
from utils.emotion_music_integration import EmotionMusicIntegrator
from utils.session_manager import SessionManager

# Configure logging
logging.basicConfig(level=logging.INFO)

# Page configuration
st.set_page_config(
    page_title="Enhanced Emotion AI with Music Therapy",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .emotion-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .music-player {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'user_id' not in st.session_state:
    st.session_state.user_id = f"user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

if 'emotions_detected' not in st.session_state:
    st.session_state.emotions_detected = []

# Main app header
st.markdown('<h1 class="main-header">🎵 Enhanced Emotion AI with Music Therapy</h1>', unsafe_allow_html=True)

# Sidebar for navigation and settings
with st.sidebar:
    st.markdown("### 🎛️ Settings")
    
    # User preferences
    st.markdown("#### User Preferences")
    vocal_preference = st.slider("Vocal tracks preference:", 0.0, 1.0, 0.8, 0.1)
    auto_advance = st.checkbox("Auto-advance songs", value=True)
    session_length = st.selectbox("Preferred session length:", ["short", "medium", "long"])
    
    # Save preferences to session state
    st.session_state.user_preferences = {
        "vocal_ratio": vocal_preference,
        "auto_advance": auto_advance,
        "session_length": session_length
    }
    
    st.markdown("---")
    
    # Quick stats
    if 'session_manager' in st.session_state:
        session_manager = st.session_state.session_manager
        user_session = session_manager.get_session(st.session_state.user_id)
        
        st.markdown("#### 📊 Your Stats")
        st.metric("Sessions", user_session.listening_stats.get("total_sessions", 0))
        
        total_time = user_session.listening_stats.get("total_listening_time", 0)
        st.metric("Listening Time", f"{total_time//3600:.0f}h {(total_time%3600)//60:.0f}m")
        
        favorite_emotions = user_session.listening_stats.get("favorite_emotions", {})
        if favorite_emotions:
            top_emotion = max(favorite_emotions.items(), key=lambda x: x[1])[0]
            st.metric("Favorite Emotion", top_emotion.title())

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["🎭 Emotion Analysis", "🎵 Enhanced Music Therapy", "📊 Analytics", "⚙️ Settings"])

# Tab 1: Emotion Analysis
with tab1:
    st.markdown("### 🎭 Emotion Analysis")
    st.markdown("Analyze your emotions through text or voice to get personalized music therapy recommendations.")
    
    # Analysis method selection
    analysis_method = st.radio(
        "Choose analysis method:",
        ["Text Analysis", "Voice Analysis", "Combined Analysis"],
        horizontal=True
    )
    
    emotions_detected = []
    
    if analysis_method == "Text Analysis":
        st.markdown("#### 📝 Text Emotion Analysis")
        
        text_input = st.text_area(
            "How are you feeling today?",
            placeholder="I'm feeling excited about my new project but also a bit nervous about the challenges ahead...",
            height=100
        )
        
        if text_input and st.button("🔍 Analyze Text", type="primary"):
            with st.spinner("Analyzing your emotions..."):
                try:
                    # Initialize integrator if not exists
                    if 'music_integrator' not in st.session_state:
                        st.session_state.music_integrator = EmotionMusicIntegrator()
                    
                    integrator = st.session_state.music_integrator
                    
                    # Analyze emotions
                    session = integrator.analyze_and_create_session(
                        text=text_input,
                        therapy_mode="targeted",
                        user_preferences=st.session_state.user_preferences
                    )
                    
                    # Extract emotions for display
                    primary_emotion = session.emotion_analysis.primary_emotion
                    emotions_detected = [
                        {
                            "label": primary_emotion.label,
                            "score": primary_emotion.confidence,
                            "confidence": primary_emotion.confidence
                        }
                    ]
                    
                    # Add secondary emotions
                    for emotion in session.emotion_analysis.all_emotions[1:3]:
                        if emotion.confidence > 0.3:
                            emotions_detected.append({
                                "label": emotion.label,
                                "score": emotion.confidence,
                                "confidence": emotion.confidence
                            })
                    
                    st.session_state.emotions_detected = emotions_detected
                    st.session_state.current_music_session = session
                    
                    # Display results
                    st.success("✅ Emotion analysis completed!")
                    
                    for i, emotion in enumerate(emotions_detected):
                        if i == 0:
                            st.markdown(f"""
                            <div class="emotion-card">
                                <h4>🎯 Primary Emotion: {emotion['label'].title()}</h4>
                                <p>Confidence: {emotion['confidence']:.1%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.info(f"🔄 Secondary: {emotion['label'].title()} ({emotion['confidence']:.1%})")
                    
                except Exception as e:
                    st.error(f"Error analyzing emotions: {e}")
                    logging.error(f"Emotion analysis error: {e}")
    
    elif analysis_method == "Voice Analysis":
        st.markdown("#### 🎤 Voice Emotion Analysis")
        st.info("Voice analysis requires audio input. Upload an audio file or record your voice.")
        
        # Audio upload
        uploaded_audio = st.file_uploader(
            "Upload audio file",
            type=['wav', 'mp3', 'm4a'],
            help="Upload a short audio clip (max 30 seconds recommended)"
        )
        
        if uploaded_audio:
            st.audio(uploaded_audio)
            
            if st.button("🔍 Analyze Voice", type="primary"):
                with st.spinner("Analyzing voice emotions..."):
                    try:
                        # Load audio
                        audio_data, sample_rate = librosa.load(uploaded_audio, sr=16000)
                        
                        # Initialize integrator
                        if 'music_integrator' not in st.session_state:
                            st.session_state.music_integrator = EmotionMusicIntegrator()
                        
                        integrator = st.session_state.music_integrator
                        
                        # Analyze emotions
                        session = integrator.analyze_and_create_session(
                            audio_data=audio_data,
                            sample_rate=sample_rate,
                            therapy_mode="targeted",
                            user_preferences=st.session_state.user_preferences
                        )
                        
                        # Extract emotions for display
                        primary_emotion = session.emotion_analysis.primary_emotion
                        emotions_detected = [
                            {
                                "label": primary_emotion.label,
                                "score": primary_emotion.confidence,
                                "confidence": primary_emotion.confidence
                            }
                        ]
                        
                        st.session_state.emotions_detected = emotions_detected
                        st.session_state.current_music_session = session
                        
                        st.success("✅ Voice emotion analysis completed!")
                        st.markdown(f"""
                        <div class="emotion-card">
                            <h4>🎯 Detected Emotion: {primary_emotion.label.title()}</h4>
                            <p>Confidence: {primary_emotion.confidence:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Error analyzing voice: {e}")
                        logging.error(f"Voice analysis error: {e}")
    
    else:  # Combined Analysis
        st.markdown("#### 🔄 Combined Text & Voice Analysis")
        st.info("Combine text and voice input for more accurate emotion detection.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            text_input = st.text_area("Text input:", height=100)
        
        with col2:
            uploaded_audio = st.file_uploader("Audio input:", type=['wav', 'mp3', 'm4a'])
            if uploaded_audio:
                st.audio(uploaded_audio)
        
        if text_input and uploaded_audio and st.button("🔍 Analyze Combined", type="primary"):
            with st.spinner("Performing multimodal emotion analysis..."):
                try:
                    # Load audio
                    audio_data, sample_rate = librosa.load(uploaded_audio, sr=16000)
                    
                    # Initialize integrator
                    if 'music_integrator' not in st.session_state:
                        st.session_state.music_integrator = EmotionMusicIntegrator()
                    
                    integrator = st.session_state.music_integrator
                    
                    # Analyze emotions
                    session = integrator.analyze_and_create_session(
                        text=text_input,
                        audio_data=audio_data,
                        sample_rate=sample_rate,
                        therapy_mode="full_session",
                        user_preferences=st.session_state.user_preferences
                    )
                    
                    # Extract emotions for display
                    primary_emotion = session.emotion_analysis.primary_emotion
                    emotions_detected = [
                        {
                            "label": primary_emotion.label,
                            "score": primary_emotion.confidence,
                            "confidence": primary_emotion.confidence
                        }
                    ]
                    
                    st.session_state.emotions_detected = emotions_detected
                    st.session_state.current_music_session = session
                    
                    st.success("✅ Multimodal emotion analysis completed!")
                    st.markdown(f"""
                    <div class="emotion-card">
                        <h4>🎯 Combined Analysis Result: {primary_emotion.label.title()}</h4>
                        <p>Confidence: {primary_emotion.confidence:.1%}</p>
                        <p>Analysis Type: Multimodal (Text + Voice)</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error in combined analysis: {e}")
                    logging.error(f"Combined analysis error: {e}")

# Tab 2: Enhanced Music Therapy
with tab2:
    # Use our enhanced music therapy UI
    create_enhanced_music_therapy_tab(st.session_state.emotions_detected)

# Tab 3: Analytics
with tab3:
    st.markdown("### 📊 Analytics & Insights")
    
    if 'session_manager' in st.session_state:
        session_manager = st.session_state.session_manager
        user_session = session_manager.get_session(st.session_state.user_id)
        
        # User analytics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Sessions",
                user_session.listening_stats.get("total_sessions", 0)
            )
        
        with col2:
            total_time = user_session.listening_stats.get("total_listening_time", 0)
            hours = total_time // 3600
            minutes = (total_time % 3600) // 60
            st.metric("Listening Time", f"{hours}h {minutes}m")
        
        with col3:
            st.metric("Playlists Explored", len(user_session.playlist_history))
        
        with col4:
            feedback_count = len(user_session.feedback_history)
            st.metric("Feedback Given", feedback_count)
        
        # Emotion distribution
        favorite_emotions = user_session.listening_stats.get("favorite_emotions", {})
        if favorite_emotions:
            st.markdown("#### 🎭 Your Emotion Distribution")
            
            emotion_df = pd.DataFrame(
                list(favorite_emotions.items()),
                columns=["Emotion", "Sessions"]
            )
            
            st.bar_chart(emotion_df.set_index("Emotion"))
        
        # Recent activity
        if user_session.playlist_history:
            st.markdown("#### 📝 Recent Activity")
            
            for playlist_id in user_session.playlist_history[-5:]:
                if 'music_integrator' in st.session_state:
                    playlist = st.session_state.music_integrator.music_controller.playlist_manager.get_playlist(playlist_id)
                    if playlist:
                        st.write(f"🎵 {playlist.name} - {len(playlist.songs)} songs")
        
        # System analytics
        if 'music_integrator' in st.session_state:
            st.markdown("#### 🌐 System Analytics")
            
            integration_stats = st.session_state.music_integrator.get_integration_statistics()
            if "total_sessions" in integration_stats:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.json({
                        "Total Users": integration_stats.get("total_users", 0),
                        "Total Sessions": integration_stats.get("total_sessions", 0),
                        "Active Sessions": integration_stats.get("active_sessions", 0)
                    })
                
                with col2:
                    if "emotion_distribution" in integration_stats:
                        st.write("**Popular Emotions:**")
                        for emotion, count in integration_stats["emotion_distribution"].items():
                            st.write(f"• {emotion.title()}: {count}")
    
    else:
        st.info("Start using the music therapy feature to see your analytics!")

# Tab 4: Settings
with tab4:
    st.markdown("### ⚙️ Settings & Configuration")
    
    # User preferences
    st.markdown("#### 👤 User Preferences")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Music preferences
        st.markdown("**Music Preferences:**")
        default_therapy_mode = st.selectbox(
            "Default therapy mode:",
            ["targeted", "full_session", "custom"],
            format_func=lambda x: {
                "targeted": "🎯 Targeted Therapy",
                "full_session": "🧘 Full Session",
                "custom": "🎨 Custom Playlist"
            }[x]
        )
        
        content_filter = st.selectbox(
            "Content filter:",
            ["clean", "mild", "all"],
            format_func=lambda x: {
                "clean": "Clean content only",
                "mild": "Allow mild content",
                "all": "All content"
            }[x]
        )
        
        language_preference = st.selectbox(
            "Language preference:",
            ["english", "mixed"],
            format_func=lambda x: {
                "english": "English only",
                "mixed": "Mixed languages"
            }[x]
        )
    
    with col2:
        # System settings
        st.markdown("**System Settings:**")
        
        auto_save = st.checkbox("Auto-save playlists", value=True)
        error_reporting = st.checkbox("Enable error reporting", value=True)
        analytics_tracking = st.checkbox("Enable analytics tracking", value=True)
        
        # Session timeout
        session_timeout = st.slider("Session timeout (hours):", 1, 48, 24)
    
    # Save settings
    if st.button("💾 Save Settings"):
        # Update user preferences
        updated_preferences = {
            "vocal_ratio": vocal_preference,
            "auto_advance": auto_advance,
            "session_length": session_length,
            "default_therapy_mode": default_therapy_mode,
            "content_filter": content_filter,
            "language_preference": language_preference,
            "auto_save": auto_save,
            "error_reporting": error_reporting,
            "analytics_tracking": analytics_tracking,
            "session_timeout": session_timeout
        }
        
        if 'session_manager' in st.session_state:
            success = st.session_state.session_manager.update_session_preferences(
                st.session_state.user_id, updated_preferences
            )
            if success:
                st.success("✅ Settings saved successfully!")
            else:
                st.error("❌ Failed to save settings")
        
        st.session_state.user_preferences = updated_preferences
    
    # Data management
    st.markdown("---")
    st.markdown("#### 📁 Data Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📤 Export Data"):
            if 'session_manager' in st.session_state:
                user_data = st.session_state.session_manager.export_user_data(st.session_state.user_id)
                if user_data:
                    st.download_button(
                        "📥 Download Data",
                        data=str(user_data),
                        file_name=f"user_data_{st.session_state.user_id}.json",
                        mime="application/json"
                    )
    
    with col2:
        if st.button("🗑️ Clear History"):
            if 'session_manager' in st.session_state:
                # Clear user history
                user_session = st.session_state.session_manager.get_session(st.session_state.user_id)
                user_session.playlist_history = []
                user_session.feedback_history = []
                user_session.listening_stats = {
                    "total_sessions": 0,
                    "total_listening_time": 0,
                    "favorite_emotions": {},
                    "preferred_therapy_mode": "targeted"
                }
                st.success("✅ History cleared!")
    
    with col3:
        if st.button("🔄 Reset All"):
            # Clear all session state
            for key in list(st.session_state.keys()):
                if key.startswith(('music_', 'session_', 'current_', 'emotions_')):
                    del st.session_state[key]
            st.success("✅ All data reset!")
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>🎵 Enhanced Emotion AI with Music Therapy</p>
    <p>Powered by advanced emotion detection and personalized music recommendations</p>
</div>
""", unsafe_allow_html=True)

# Cleanup on app close
def cleanup():
    if 'enhanced_music_ui' in st.session_state:
        st.session_state.enhanced_music_ui.cleanup()

# Register cleanup
import atexit
atexit.register(cleanup)