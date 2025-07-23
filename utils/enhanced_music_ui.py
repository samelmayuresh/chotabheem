# utils/enhanced_music_ui.py - Enhanced Streamlit UI for music playback
import streamlit as st
import time
from typing import List, Dict, Optional, Any
from datetime import datetime

from utils.emotion_music_integration import EmotionMusicIntegrator, EmotionMusicSession
from utils.enhanced_music_controller import EnhancedMusicController
from utils.session_manager import SessionManager
from utils.playback_error_handler import PlaybackErrorHandler
from utils.player_engine import PlaybackStatus

class EnhancedMusicUI:
    """Enhanced Streamlit UI for music therapy with full playback controls"""
    
    def __init__(self):
        # Initialize components
        if 'music_integrator' not in st.session_state:
            st.session_state.music_integrator = EmotionMusicIntegrator()
        
        if 'session_manager' not in st.session_state:
            st.session_state.session_manager = SessionManager()
        
        if 'error_handler' not in st.session_state:
            st.session_state.error_handler = PlaybackErrorHandler()
        
        self.integrator = st.session_state.music_integrator
        self.session_manager = st.session_state.session_manager
        self.error_handler = st.session_state.error_handler
    
    def render_enhanced_music_therapy_tab(self, emotions: List[Dict] = None):
        """Render the enhanced music therapy tab with full controls"""
        
        st.markdown("### 🎵 Enhanced Music Therapy")
        st.markdown("Experience therapeutic music with continuous playback and intelligent emotion integration")
        
        # Get or create user session
        user_id = st.session_state.get('user_id', 'default_user')
        user_session = self.session_manager.get_session(user_id)
        
        # Display user recommendations if available
        if user_session.listening_stats.get("total_sessions", 0) > 0:
            self._render_user_recommendations(user_session)
        
        # Main therapy session creation
        self._render_session_creation(emotions, user_session)
        
        # Current session controls
        if 'current_music_session' in st.session_state:
            self._render_playback_controls()
            self._render_playlist_display()
        
        # Session history and analytics
        self._render_session_history(user_session)
    
    def _render_user_recommendations(self, user_session):
        """Render personalized recommendations"""
        
        recommendations = self.session_manager.get_user_recommendations(user_session.user_id)
        
        if recommendations.get("favorite_emotions"):
            st.info(f"🎯 Your favorite emotions: {', '.join(recommendations['favorite_emotions'])}")
        
        if recommendations.get("listening_insights"):
            with st.expander("📊 Your Listening Insights"):
                for insight in recommendations["listening_insights"]:
                    st.write(f"• {insight}")
    
    def _render_session_creation(self, emotions, user_session):
        """Render session creation interface"""
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### Create New Therapy Session")
            
            # Input method selection
            input_method = st.radio(
                "How would you like to analyze your emotions?",
                ["Use detected emotions", "Text input", "Manual selection"],
                horizontal=True
            )
            
            primary_emotion = None
            secondary_emotions = []
            
            if input_method == "Use detected emotions" and emotions:
                primary_emotion = emotions[0]["label"]
                secondary_emotions = [e["label"] for e in emotions[1:3] if e["score"] > 0.3]
                
                st.success(f"🎯 Primary emotion: **{primary_emotion}** ({emotions[0]['score']:.1%} confidence)")
                if secondary_emotions:
                    st.info(f"🔄 Secondary emotions: {', '.join(secondary_emotions)}")
            
            elif input_method == "Text input":
                text_input = st.text_area(
                    "Describe how you're feeling:",
                    placeholder="I'm feeling happy and excited about today..."
                )
                
                if text_input and st.button("Analyze Text"):
                    with st.spinner("Analyzing your emotions..."):
                        try:
                            session = self.integrator.analyze_and_create_session(
                                text=text_input,
                                therapy_mode="targeted"
                            )
                            st.session_state.current_music_session = session
                            primary_emotion = session.emotion_analysis.primary_emotion.label
                            st.success(f"✅ Created session for emotion: {primary_emotion}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error analyzing emotions: {e}")
            
            elif input_method == "Manual selection":
                available_emotions = ["joy", "sadness", "anger", "fear", "anxiety", "love", "neutral"]
                primary_emotion = st.selectbox("Select primary emotion:", available_emotions)
                
                secondary_emotions = st.multiselect(
                    "Select secondary emotions (optional):",
                    [e for e in available_emotions if e != primary_emotion]
                )
        
        with col2:
            st.markdown("#### Session Settings")
            
            # Therapy mode
            therapy_mode = st.selectbox(
                "Therapy Mode:",
                ["targeted", "full_session", "custom"],
                format_func=lambda x: {
                    "targeted": "🎯 Targeted (5 songs)",
                    "full_session": "🧘 Full Session (10+ songs)", 
                    "custom": "🎨 Custom Playlist"
                }[x]
            )
            
            # User preferences
            with st.expander("⚙️ Preferences"):
                vocal_ratio = st.slider("Vocal tracks ratio:", 0.0, 1.0, 0.8, 0.1)
                auto_advance = st.checkbox("Auto-advance songs", value=True)
                volume = st.slider("Volume:", 0.0, 1.0, 0.7, 0.1)
                
                if therapy_mode == "custom":
                    song_count = st.number_input("Number of songs:", 3, 20, 7)
                else:
                    song_count = 5 if therapy_mode == "targeted" else 10
        
        # Create session button
        if primary_emotion and st.button("🎵 Create Therapy Session", type="primary"):
            user_preferences = {
                "vocal_ratio": vocal_ratio,
                "auto_advance": auto_advance,
                "volume": volume
            }
            
            with st.spinner("Creating your personalized therapy session..."):
                try:
                    if input_method == "Manual selection":
                        # Create session directly with music controller
                        music_session = self.integrator.music_controller.create_continuous_session(
                            emotion=primary_emotion,
                            mode=therapy_mode,
                            song_count=song_count,
                            secondary_emotions=secondary_emotions,
                            user_preferences=user_preferences
                        )
                        
                        # Create a mock emotion-music session
                        from utils.text_emotion_ensemble import EmotionScore
                        from utils.enhanced_emotion_analyzer import EmotionResult
                        
                        mock_emotion = EmotionScore(
                            label=primary_emotion,
                            score=0.8,
                            confidence=0.8,
                            source="manual",
                            model_name="user_selection",
                            processing_time=0.0,
                            metadata={}
                        )
                        
                        mock_result = EmotionResult(
                            primary_emotion=mock_emotion,
                            all_emotions=[mock_emotion],
                            confidence_level="high",
                            uncertainty_score=0.2,
                            processing_metadata={"source": "manual"},
                            insights=[],
                            recommendations=[]
                        )
                        
                        from utils.emotion_music_integration import EmotionMusicSession
                        session = EmotionMusicSession(
                            session_id=f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            emotion_analysis=mock_result,
                            music_session=music_session,
                            integration_metadata={"analysis_type": "manual"},
                            created_at=datetime.now()
                        )
                    else:
                        # Use emotion analysis
                        session = self.integrator.analyze_and_create_session(
                            text=f"I'm feeling {primary_emotion}",
                            therapy_mode=therapy_mode,
                            user_preferences=user_preferences
                        )
                    
                    st.session_state.current_music_session = session
                    
                    # Update user session
                    self.session_manager.add_playlist_to_history(
                        user_id, session.music_session.playlist.id
                    )
                    
                    st.success(f"✅ Created {therapy_mode} session with {len(session.music_session.playlist.songs)} songs!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error creating session: {e}")
                    self.error_handler.handle_error(e)
    
    def _render_playback_controls(self):
        """Render playback controls interface"""
        
        session = st.session_state.current_music_session
        controller = self.integrator.music_controller
        
        st.markdown("---")
        st.markdown("### 🎮 Playback Controls")
        
        # Get current playback state
        playback_state = controller.get_playback_state()
        
        # Main control buttons
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if st.button("⏮️ Previous"):
                controller.handle_playback_control("skip_previous")
                st.rerun()
        
        with col2:
            if playback_state.status == PlaybackStatus.PLAYING:
                if st.button("⏸️ Pause"):
                    controller.handle_playback_control("pause")
                    st.rerun()
            else:
                if st.button("▶️ Play"):
                    controller.handle_playback_control("play")
                    st.rerun()
        
        with col3:
            if st.button("⏹️ Stop"):
                controller.handle_playback_control("stop")
                st.rerun()
        
        with col4:
            if st.button("⏭️ Next"):
                controller.handle_playback_control("skip_next")
                st.rerun()
        
        with col5:
            if st.button("🔄 Shuffle"):
                current_shuffle = playback_state.shuffle
                controller.handle_playback_control("shuffle", enabled=not current_shuffle)
                st.rerun()
        
        # Progress and volume controls
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if playback_state.current_song:
                # Progress bar
                progress = playback_state.get_progress_percentage()
                st.progress(progress / 100)
                
                # Time display
                current_time = playback_state.get_formatted_position()
                total_time = playback_state.get_formatted_duration()
                st.caption(f"{current_time} / {total_time}")
        
        with col2:
            # Volume control
            new_volume = st.slider("🔊", 0.0, 1.0, playback_state.volume, key="volume_slider")
            if new_volume != playback_state.volume:
                controller.handle_playback_control("volume", volume=new_volume)
        
        # Repeat mode
        repeat_modes = ["none", "song", "playlist"]
        current_repeat = playback_state.repeat_mode
        new_repeat = st.selectbox(
            "Repeat mode:",
            repeat_modes,
            index=repeat_modes.index(current_repeat),
            format_func=lambda x: {"none": "🔁 No repeat", "song": "🔂 Repeat song", "playlist": "🔁 Repeat playlist"}[x]
        )
        
        if new_repeat != current_repeat:
            controller.handle_playback_control("repeat", mode=new_repeat)
            st.rerun()
    
    def _render_playlist_display(self):
        """Render current playlist with song information"""
        
        session = st.session_state.current_music_session
        controller = self.integrator.music_controller
        playback_state = controller.get_playback_state()
        
        st.markdown("### 📋 Current Playlist")
        
        # Playlist info
        playlist = session.music_session.playlist
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Songs", len(playlist.songs))
        
        with col2:
            st.metric("Duration", playlist.get_duration_formatted())
        
        with col3:
            vocal_percentage = playlist.vocal_instrumental_ratio * 100
            st.metric("Vocal Tracks", f"{vocal_percentage:.0f}%")
        
        # Current song highlight
        if playback_state.current_song:
            st.markdown(f"**🎵 Now Playing:** {playback_state.current_song.title} - {playback_state.current_song.artist}")
        
        # Song list
        for i, song in enumerate(playlist.songs):
            is_current = (playback_state.current_song and 
                         song.title == playback_state.current_song.title and 
                         song.artist == playback_state.current_song.artist)
            
            # Song container
            with st.container():
                col1, col2, col3, col4 = st.columns([1, 3, 2, 1])
                
                with col1:
                    if is_current:
                        st.markdown("▶️")
                    else:
                        if st.button("▶️", key=f"play_{i}"):
                            # Jump to this song
                            controller.player_engine.state.playlist_position = i
                            controller.player_engine.state.current_song = song
                            controller.handle_playback_control("play")
                            st.rerun()
                
                with col2:
                    style = "background-color: #e6f3ff;" if is_current else ""
                    st.markdown(f"<div style='{style} padding: 5px; border-radius: 5px;'>"
                              f"<strong>{song.title}</strong><br>"
                              f"<small>{song.artist}</small></div>", 
                              unsafe_allow_html=True)
                
                with col3:
                    st.caption(f"{song.genre} • {song.mood}")
                    if song.has_vocals:
                        st.caption("🎤 Vocal")
                    else:
                        st.caption("🎼 Instrumental")
                
                with col4:
                    duration_min = song.duration // 60
                    duration_sec = song.duration % 60
                    st.caption(f"{duration_min}:{duration_sec:02d}")
        
        # Playlist actions
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Save Playlist"):
                success = controller.save_current_session()
                if success:
                    st.success("Playlist saved!")
                else:
                    st.error("Failed to save playlist")
        
        with col2:
            if st.button("🔄 Shuffle Playlist"):
                # This would require playlist modification
                st.info("Shuffle feature coming soon!")
        
        with col3:
            if st.button("📤 Export Playlist"):
                export_data = controller.playlist_manager.export_playlist(
                    playlist.id, format="text"
                )
                if export_data:
                    st.download_button(
                        "📥 Download",
                        export_data,
                        file_name=f"{playlist.name}.txt",
                        mime="text/plain"
                    )
    
    def _render_session_history(self, user_session):
        """Render user's session history and analytics"""
        
        if user_session.playlist_history:
            with st.expander("📊 Your Session History"):
                
                # Quick stats
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Sessions", user_session.listening_stats.get("total_sessions", 0))
                
                with col2:
                    total_time = user_session.listening_stats.get("total_listening_time", 0)
                    hours = total_time // 3600
                    st.metric("Listening Time", f"{hours:.1f}h")
                
                with col3:
                    favorite_emotions = user_session.listening_stats.get("favorite_emotions", {})
                    if favorite_emotions:
                        top_emotion = max(favorite_emotions.items(), key=lambda x: x[1])[0]
                        st.metric("Top Emotion", top_emotion.title())
                
                # Recent playlists
                st.markdown("**Recent Playlists:**")
                for playlist_id in user_session.playlist_history[-5:]:
                    playlist = self.integrator.music_controller.playlist_manager.get_playlist(playlist_id)
                    if playlist:
                        if st.button(f"🎵 {playlist.name}", key=f"load_{playlist_id}"):
                            success = self.integrator.music_controller.load_saved_playlist(playlist_id)
                            if success:
                                st.session_state.current_music_session = self.integrator.active_sessions[
                                    list(self.integrator.active_sessions.keys())[-1]
                                ]
                                st.success(f"Loaded playlist: {playlist.name}")
                                st.rerun()
    
    def render_feedback_interface(self):
        """Render user feedback interface"""
        
        if 'current_music_session' not in st.session_state:
            return
        
        session = st.session_state.current_music_session
        
        with st.expander("💬 Session Feedback"):
            st.markdown("Help us improve your music therapy experience!")
            
            # Rating
            rating = st.slider("Rate this session:", 1, 5, 3)
            
            # Feedback categories
            col1, col2 = st.columns(2)
            
            with col1:
                liked_aspects = st.multiselect(
                    "What did you like?",
                    ["Song selection", "Vocal/instrumental balance", "Emotional match", "Playlist flow", "Duration"]
                )
            
            with col2:
                improvements = st.multiselect(
                    "What could be improved?",
                    ["More variety", "Better emotional match", "Different genres", "Shorter/longer playlist", "Technical issues"]
                )
            
            # Comments
            comments = st.text_area("Additional comments (optional):")
            
            if st.button("Submit Feedback"):
                feedback = {
                    "rating": rating,
                    "liked_aspects": liked_aspects,
                    "improvements": improvements,
                    "comments": comments,
                    "session_id": session.session_id
                }
                
                success = self.integrator.provide_user_feedback(session.session_id, feedback)
                if success:
                    st.success("Thank you for your feedback!")
                else:
                    st.error("Failed to submit feedback")
    
    def cleanup(self):
        """Cleanup UI resources"""
        if hasattr(self, 'integrator'):
            self.integrator.cleanup()
        if hasattr(self, 'session_manager'):
            self.session_manager.cleanup()
        if hasattr(self, 'error_handler'):
            self.error_handler.cleanup()

def create_enhanced_music_therapy_tab(emotions: List[Dict] = None):
    """Create enhanced music therapy tab - main entry point"""
    
    # Initialize UI
    if 'enhanced_music_ui' not in st.session_state:
        st.session_state.enhanced_music_ui = EnhancedMusicUI()
    
    ui = st.session_state.enhanced_music_ui
    
    # Render main interface
    ui.render_enhanced_music_therapy_tab(emotions)
    
    # Render feedback interface
    ui.render_feedback_interface()
    
    # Auto-refresh for playback updates (every 5 seconds)
    if 'current_music_session' in st.session_state:
        time.sleep(0.1)  # Small delay to prevent too frequent updates