# utils/enhanced_music_therapy.py - Enhanced music therapy with actual song playback
import streamlit as st
import requests
import json
import random
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import logging

@dataclass
class Song:
    """Represents a song with metadata"""
    title: str
    artist: str
    url: str
    duration: int  # in seconds
    genre: str
    mood: str
    preview_url: Optional[str] = None
    thumbnail: Optional[str] = None
    description: Optional[str] = None

class EnhancedMusicTherapy:
    """Enhanced music therapy system with actual song playback"""
    
    def __init__(self):
        self.emotion_playlists = self._create_emotion_playlists()
        self.current_playlist = []
        self.current_song_index = 0
        
    def _create_emotion_playlists(self) -> Dict[str, List[Song]]:
        """Create curated playlists for different emotions with actual songs"""
        
        playlists = {
            "joy": [
                Song("Happy", "Pharrell Williams", "https://www.youtube.com/watch?v=ZbZSe6N_BXs", 233, "Pop", "Uplifting", 
                     description="Upbeat and infectious happiness"),
                Song("Good as Hell", "Lizzo", "https://www.youtube.com/watch?v=SmbmeOgWsqE", 219, "Pop", "Empowering",
                     description="Self-love and confidence booster"),
                Song("Can't Stop the Feeling", "Justin Timberlake", "https://www.youtube.com/watch?v=ru0K8uYEZWw", 236, "Pop", "Energetic",
                     description="Pure joy and celebration"),
                Song("Walking on Sunshine", "Katrina and the Waves", "https://www.youtube.com/watch?v=iPUmE-tne5U", 239, "Pop", "Uplifting",
                     description="Classic feel-good anthem"),
                Song("Three Little Birds", "Bob Marley", "https://www.youtube.com/watch?v=LanCLS_hIo4", 180, "Reggae", "Peaceful",
                     description="Everything's gonna be alright"),
            ],
            
            "sadness": [
                Song("Someone Like You", "Adele", "https://www.youtube.com/watch?v=hLQl3WQQoQ0", 285, "Pop", "Healing",
                     description="Emotional healing and acceptance"),
                Song("The Sound of Silence", "Simon & Garfunkel", "https://www.youtube.com/watch?v=4fWyzwo1xg0", 200, "Folk", "Contemplative",
                     description="Gentle reflection and comfort"),
                Song("Mad World", "Gary Jules", "https://www.youtube.com/watch?v=4N3N1MlvVc4", 191, "Alternative", "Melancholic",
                     description="Understanding sadness"),
                Song("Breathe Me", "Sia", "https://www.youtube.com/watch?v=ghPcYqn0p4Y", 273, "Pop", "Vulnerable",
                     description="Finding strength in vulnerability"),
                Song("Hurt", "Johnny Cash", "https://www.youtube.com/watch?v=8AHCfZTRGiI", 218, "Country", "Reflective",
                     description="Deep emotional processing"),
            ],
            
            "anger": [
                Song("Let It Go", "Idina Menzel", "https://www.youtube.com/watch?v=L0MK7qz13bU", 225, "Pop", "Release",
                     description="Letting go of anger and frustration"),
                Song("Stronger", "Kelly Clarkson", "https://www.youtube.com/watch?v=Xn676-fLq7I", 222, "Pop", "Empowering",
                     description="Finding strength through challenges"),
                Song("Roar", "Katy Perry", "https://www.youtube.com/watch?v=CevxZvSJLk8", 223, "Pop", "Empowering",
                     description="Finding your voice and power"),
                Song("Fight Song", "Rachel Platten", "https://www.youtube.com/watch?v=xo1VInw-SKc", 204, "Pop", "Motivational",
                     description="Inner strength and resilience"),
                Song("Calm Down", "Rema & Selena Gomez", "https://www.youtube.com/watch?v=WcIcVapfqXw", 239, "Afrobeats", "Soothing",
                     description="Calming anger with rhythm"),
            ],
            
            "fear": [
                Song("Brave", "Sara Bareilles", "https://www.youtube.com/watch?v=QUQsqBqxoR4", 239, "Pop", "Courage",
                     description="Finding courage to speak up"),
                Song("Stronger (What Doesn't Kill You)", "Kelly Clarkson", "https://www.youtube.com/watch?v=Xn676-fLq7I", 222, "Pop", "Resilience",
                     description="Building resilience and strength"),
                Song("Confident", "Demi Lovato", "https://www.youtube.com/watch?v=9f06QZCVUHg", 216, "Pop", "Confidence",
                     description="Building self-confidence"),
                Song("Titanium", "David Guetta ft. Sia", "https://www.youtube.com/watch?v=JRfuAukYTKg", 245, "Electronic", "Strength",
                     description="Unbreakable inner strength"),
                Song("Eye of the Tiger", "Survivor", "https://www.youtube.com/watch?v=btPJPFnesV4", 245, "Rock", "Motivational",
                     description="Classic courage anthem"),
            ],
            
            "anxiety": [
                Song("Weightless", "Marconi Union", "https://www.youtube.com/watch?v=UfcAVejslrU", 515, "Ambient", "Calming",
                     description="Scientifically proven to reduce anxiety"),
                Song("Clair de Lune", "Claude Debussy", "https://www.youtube.com/watch?v=CvFH_6DNRCY", 300, "Classical", "Peaceful",
                     description="Gentle classical for relaxation"),
                Song("Aqueous Transmission", "Incubus", "https://www.youtube.com/watch?v=eQK7KSTQfaw", 456, "Alternative", "Meditative",
                     description="Long-form meditation music"),
                Song("River", "Joni Mitchell", "https://www.youtube.com/watch?v=3NH-ctddY9o", 240, "Folk", "Soothing",
                     description="Gentle folk for calming"),
                Song("The Night We Met", "Lord Huron", "https://www.youtube.com/watch?v=KtlgYxa6BMU", 207, "Indie", "Contemplative",
                     description="Peaceful reflection"),
            ],
            
            "love": [
                Song("All of Me", "John Legend", "https://www.youtube.com/watch?v=450p7goxZqg", 269, "R&B", "Romantic",
                     description="Deep love and devotion"),
                Song("Perfect", "Ed Sheeran", "https://www.youtube.com/watch?v=2Vv-BfVoq4g", 263, "Pop", "Romantic",
                     description="Perfect love song"),
                Song("At Last", "Etta James", "https://www.youtube.com/watch?v=S-cbOl96RFM", 180, "Soul", "Classic Romance",
                     description="Timeless love classic"),
                Song("Make You Feel My Love", "Adele", "https://www.youtube.com/watch?v=0put0_a--Ng", 213, "Pop", "Heartfelt",
                     description="Deep emotional connection"),
                Song("Thinking Out Loud", "Ed Sheeran", "https://www.youtube.com/watch?v=lp-EO5I60KA", 281, "Pop", "Tender",
                     description="Growing old together"),
            ],
            
            "neutral": [
                Song("Weightless", "Marconi Union", "https://www.youtube.com/watch?v=UfcAVejslrU", 515, "Ambient", "Balanced",
                     description="Perfect for focus and balance"),
                Song("Clair de Lune", "Claude Debussy", "https://www.youtube.com/watch?v=CvFH_6DNRCY", 300, "Classical", "Peaceful",
                     description="Timeless classical beauty"),
                Song("Gymnopédie No. 1", "Erik Satie", "https://www.youtube.com/watch?v=S-Xm7s9eGxU", 210, "Classical", "Meditative",
                     description="Gentle piano meditation"),
                Song("Porcelain", "Moby", "https://www.youtube.com/watch?v=IJWlBfo5Oj0", 240, "Electronic", "Ambient",
                     description="Electronic ambient calm"),
                Song("Holocene", "Bon Iver", "https://www.youtube.com/watch?v=TWcyIpul8OE", 337, "Indie", "Contemplative",
                     description="Peaceful indie reflection"),
            ]
        }
        
        return playlists
    
    def get_playlist_for_emotion(self, emotion: str, count: int = 5) -> List[Song]:
        """Get a playlist for a specific emotion"""
        
        # Map similar emotions
        emotion_mapping = {
            "happiness": "joy",
            "excited": "joy",
            "optimism": "joy",
            "depression": "sadness",
            "grief": "sadness",
            "disappointment": "sadness",
            "rage": "anger",
            "frustration": "anger",
            "annoyance": "anger",
            "worry": "anxiety",
            "nervousness": "anxiety",
            "stress": "anxiety",
            "scared": "fear",
            "terror": "fear",
            "panic": "fear",
            "romance": "love",
            "affection": "love",
            "caring": "love",
        }
        
        # Get the mapped emotion or use the original
        mapped_emotion = emotion_mapping.get(emotion.lower(), emotion.lower())
        
        # Get songs for the emotion
        if mapped_emotion in self.emotion_playlists:
            songs = self.emotion_playlists[mapped_emotion].copy()
        else:
            # Default to neutral if emotion not found
            songs = self.emotion_playlists["neutral"].copy()
        
        # Shuffle and return requested count
        random.shuffle(songs)
        return songs[:count]
    
    def create_therapy_session(self, primary_emotion: str, secondary_emotions: List[str] = None) -> List[Song]:
        """Create a therapeutic music session based on emotions"""
        
        session_playlist = []
        
        # Start with songs for the primary emotion (3 songs)
        primary_songs = self.get_playlist_for_emotion(primary_emotion, 3)
        session_playlist.extend(primary_songs)
        
        # Add songs for secondary emotions if provided (2 songs each)
        if secondary_emotions:
            for emotion in secondary_emotions[:2]:  # Limit to 2 secondary emotions
                secondary_songs = self.get_playlist_for_emotion(emotion, 2)
                session_playlist.extend(secondary_songs)
        
        # Add some neutral/balancing songs at the end (2 songs)
        neutral_songs = self.get_playlist_for_emotion("neutral", 2)
        session_playlist.extend(neutral_songs)
        
        return session_playlist
    
    def display_music_player(self, songs: List[Song], emotion: str):
        """Display an enhanced music player interface"""
        
        if not songs:
            st.warning("No songs available for this emotion.")
            return
        
        st.markdown(f"### 🎵 Music Therapy for {emotion.title()}")
        st.markdown(f"**{len(songs)} songs** curated for your emotional well-being")
        
        # Player controls
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.button("⏮️ Previous"):
                if "current_song_index" not in st.session_state:
                    st.session_state.current_song_index = 0
                st.session_state.current_song_index = max(0, st.session_state.current_song_index - 1)
                st.rerun()
        
        with col2:
            # Song selection
            if "current_song_index" not in st.session_state:
                st.session_state.current_song_index = 0
            
            current_index = st.session_state.current_song_index
            if current_index >= len(songs):
                current_index = 0
                st.session_state.current_song_index = 0
            
            song_options = [f"{i+1}. {song.title} - {song.artist}" for i, song in enumerate(songs)]
            selected_index = st.selectbox(
                "Select Song:",
                range(len(songs)),
                index=current_index,
                format_func=lambda x: song_options[x]
            )
            
            if selected_index != current_index:
                st.session_state.current_song_index = selected_index
                st.rerun()
        
        with col3:
            if st.button("⏭️ Next"):
                if "current_song_index" not in st.session_state:
                    st.session_state.current_song_index = 0
                st.session_state.current_song_index = min(len(songs) - 1, st.session_state.current_song_index + 1)
                st.rerun()
        
        # Current song display
        current_song = songs[st.session_state.current_song_index]
        
        # Song info card
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 15px;
            color: white;
            margin: 20px 0;
            text-align: center;
        ">
            <h3 style="margin: 0; color: white;">🎵 {current_song.title}</h3>
            <h4 style="margin: 5px 0; color: #e0e0e0;">by {current_song.artist}</h4>
            <p style="margin: 10px 0; color: #f0f0f0;"><em>{current_song.description}</em></p>
            <p style="margin: 5px 0; color: #d0d0d0;">
                {current_song.genre} • {current_song.mood} • {current_song.duration//60}:{current_song.duration%60:02d}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # YouTube embed
        st.markdown("### 🎬 Now Playing")
        
        # Extract YouTube video ID
        video_id = self._extract_youtube_id(current_song.url)
        if video_id:
            # Embed YouTube video
            st.markdown(f"""
            <div style="text-align: center; margin: 20px 0;">
                <iframe width="100%" height="400" 
                        src="https://www.youtube.com/embed/{video_id}?autoplay=1&rel=0" 
                        frameborder="0" 
                        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                        allowfullscreen>
                </iframe>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Fallback link
            st.markdown(f"[🎵 Play on YouTube]({current_song.url})")
        
        # Playlist overview
        st.markdown("### 📋 Full Playlist")
        
        for i, song in enumerate(songs):
            is_current = i == st.session_state.current_song_index
            icon = "▶️" if is_current else "🎵"
            style = "background-color: #f0f8ff; border-left: 4px solid #4CAF50;" if is_current else ""
            
            st.markdown(f"""
            <div style="padding: 10px; margin: 5px 0; border-radius: 8px; {style}">
                {icon} <strong>{song.title}</strong> - {song.artist}<br>
                <small style="color: #666;">{song.description} • {song.duration//60}:{song.duration%60:02d}</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Therapeutic benefits
        st.markdown("### 🧠 Therapeutic Benefits")
        benefits = self._get_therapeutic_benefits(emotion)
        for benefit in benefits:
            st.markdown(f"• {benefit}")
        
        # Auto-advance option
        if st.checkbox("🔄 Auto-advance to next song", value=False):
            st.info("Auto-advance enabled! Songs will play continuously.")
    
    def _extract_youtube_id(self, url: str) -> Optional[str]:
        """Extract YouTube video ID from URL"""
        try:
            if "youtube.com/watch?v=" in url:
                return url.split("v=")[1].split("&")[0]
            elif "youtu.be/" in url:
                return url.split("youtu.be/")[1].split("?")[0]
            return None
        except:
            return None
    
    def _get_therapeutic_benefits(self, emotion: str) -> List[str]:
        """Get therapeutic benefits for the emotion"""
        
        benefits_map = {
            "joy": [
                "Amplifies positive emotions and reinforces happiness",
                "Releases endorphins and dopamine for natural mood boost",
                "Creates positive associations and memories",
                "Encourages social connection and sharing joy"
            ],
            "sadness": [
                "Provides emotional validation and understanding",
                "Facilitates healthy emotional processing and release",
                "Offers comfort and companionship during difficult times",
                "Helps transition from sadness to acceptance and healing"
            ],
            "anger": [
                "Provides healthy outlet for intense emotions",
                "Helps process and transform anger into empowerment",
                "Reduces stress hormones and physical tension",
                "Promotes emotional regulation and self-control"
            ],
            "fear": [
                "Builds courage and confidence through empowering messages",
                "Reduces anxiety through rhythmic breathing synchronization",
                "Provides strength and motivation to face challenges",
                "Creates sense of safety and emotional support"
            ],
            "anxiety": [
                "Activates parasympathetic nervous system for relaxation",
                "Reduces cortisol levels and stress response",
                "Promotes mindfulness and present-moment awareness",
                "Improves sleep quality and emotional regulation"
            ],
            "love": [
                "Enhances feelings of connection and bonding",
                "Releases oxytocin for increased well-being",
                "Strengthens emotional intimacy and relationships",
                "Promotes self-love and acceptance"
            ],
            "neutral": [
                "Promotes emotional balance and stability",
                "Enhances focus and concentration",
                "Provides calming background for meditation",
                "Supports overall mental wellness and peace"
            ]
        }
        
        return benefits_map.get(emotion.lower(), [
            "Provides emotional support and validation",
            "Promotes relaxation and stress relief",
            "Enhances mood and well-being",
            "Supports healthy emotional processing"
        ])

def create_enhanced_music_therapy_tab(emotions: List[Dict]):
    """Create enhanced music therapy tab with actual song playback"""
    
    if not emotions:
        st.info("🎭 Analyze some audio or text first to get personalized music therapy")
        return
    
    # Initialize music therapy system
    music_therapy = EnhancedMusicTherapy()
    
    # Get primary and secondary emotions
    primary_emotion = emotions[0]["label"]
    secondary_emotions = [e["label"] for e in emotions[1:3] if e["score"] > 0.3]
    
    st.markdown("### 🎵 Enhanced Music Therapy")
    st.markdown("Experience therapeutic music with actual songs tailored to your emotions")
    
    # Display current emotional state
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"**Primary Emotion:** {primary_emotion.title()} ({emotions[0]['score']:.1%} confidence)")
        if secondary_emotions:
            st.markdown(f"**Secondary Emotions:** {', '.join([e.title() for e in secondary_emotions])}")
    
    with col2:
        therapy_mode = st.selectbox(
            "Therapy Mode:",
            ["Targeted Therapy", "Full Session", "Custom Playlist"],
            help="Choose how you want your music therapy session structured"
        )
    
    # Generate playlist based on mode
    if therapy_mode == "Targeted Therapy":
        # Focus on primary emotion only
        playlist = music_therapy.get_playlist_for_emotion(primary_emotion, 5)
        st.markdown(f"🎯 **Targeted therapy for {primary_emotion}** - 5 carefully selected songs")
        
    elif therapy_mode == "Full Session":
        # Create comprehensive therapy session
        playlist = music_therapy.create_therapy_session(primary_emotion, secondary_emotions)
        st.markdown(f"🧘 **Complete therapy session** - {len(playlist)} songs for holistic emotional support")
        
    else:  # Custom Playlist
        st.markdown("🎨 **Custom Playlist Builder**")
        
        # Let user select emotions
        available_emotions = list(music_therapy.emotion_playlists.keys())
        selected_emotions = st.multiselect(
            "Select emotions for your playlist:",
            available_emotions,
            default=[primary_emotion] if primary_emotion in available_emotions else ["neutral"]
        )
        
        songs_per_emotion = st.slider("Songs per emotion:", 1, 5, 3)
        
        playlist = []
        for emotion in selected_emotions:
            emotion_songs = music_therapy.get_playlist_for_emotion(emotion, songs_per_emotion)
            playlist.extend(emotion_songs)
        
        st.markdown(f"🎵 **Custom playlist** - {len(playlist)} songs from {len(selected_emotions)} emotional themes")
    
    # Display the music player
    if playlist:
        music_therapy.display_music_player(playlist, primary_emotion)
        
        # Additional features
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Shuffle Playlist"):
                random.shuffle(playlist)
                st.success("Playlist shuffled!")
                st.rerun()
        
        with col2:
            if st.button("💾 Save Playlist"):
                # Save to session state
                st.session_state.saved_playlist = playlist
                st.success("Playlist saved to session!")
        
        with col3:
            if st.button("📤 Export Playlist"):
                # Create exportable playlist
                playlist_text = f"# Music Therapy Playlist for {primary_emotion.title()}\n\n"
                for i, song in enumerate(playlist, 1):
                    playlist_text += f"{i}. {song.title} - {song.artist}\n"
                    playlist_text += f"   {song.url}\n\n"
                
                st.download_button(
                    label="📥 Download Playlist",
                    data=playlist_text,
                    file_name=f"music_therapy_{primary_emotion}.txt",
                    mime="text/plain"
                )
    
    else:
        st.warning("No songs available. Please try a different emotion or mode.")