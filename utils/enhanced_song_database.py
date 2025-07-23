# utils/enhanced_song_database.py - Enhanced song database with vocal tracks and therapeutic metadata
import random
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

@dataclass
class EnhancedSong:
    """Enhanced song model with vocal/instrumental metadata and therapeutic properties"""
    # Existing fields from original Song class
    title: str
    artist: str
    url: str
    duration: int  # in seconds
    genre: str
    mood: str
    preview_url: Optional[str] = None
    thumbnail: Optional[str] = None
    description: Optional[str] = None
    
    # New enhanced fields
    has_vocals: bool = True
    lyrics_theme: str = ""
    energy_level: int = 5  # 1-10 scale (1=very calm, 10=very energetic)
    therapeutic_benefits: List[str] = field(default_factory=list)
    content_rating: str = "clean"  # clean, explicit, mild
    popularity_score: int = 50  # 1-100 scale
    language: str = "english"
    release_year: Optional[int] = None
    emotional_intensity: int = 5  # 1-10 scale for therapeutic matching

class EnhancedSongDatabase:
    """Enhanced song database with expanded English vocal library and therapeutic categorization"""
    
    def __init__(self):
        self.emotion_playlists = self._create_enhanced_emotion_playlists()
        self.english_vocal_library = self._create_english_vocal_library()
        
    def _create_enhanced_emotion_playlists(self) -> Dict[str, List[EnhancedSong]]:
        """Create enhanced playlists with therapeutic metadata and vocal variety"""
        
        playlists = {
            "joy": [
                # Popular vocal tracks for joy
                EnhancedSong(
                    "Happy", "Pharrell Williams", "https://www.youtube.com/watch?v=ZbZSe6N_BXs", 
                    233, "Pop", "Uplifting", has_vocals=True, lyrics_theme="celebration",
                    energy_level=9, therapeutic_benefits=["mood elevation", "positive reinforcement"],
                    popularity_score=95, release_year=2013, emotional_intensity=8,
                    description="Infectious happiness and celebration of life"
                ),
                EnhancedSong(
                    "Good as Hell", "Lizzo", "https://www.youtube.com/watch?v=SmbmeOgWsqE",
                    219, "Pop", "Empowering", has_vocals=True, lyrics_theme="self-love",
                    energy_level=8, therapeutic_benefits=["self-confidence", "empowerment"],
                    popularity_score=88, release_year=2019, emotional_intensity=7,
                    description="Self-love anthem and confidence booster"
                ),
                EnhancedSong(
                    "Can't Stop the Feeling", "Justin Timberlake", "https://www.youtube.com/watch?v=ru0K8uYEZWw",
                    236, "Pop", "Energetic", has_vocals=True, lyrics_theme="joy",
                    energy_level=9, therapeutic_benefits=["energy boost", "mood lifting"],
                    popularity_score=92, release_year=2016, emotional_intensity=8,
                    description="Pure joy and celebration energy"
                ),
                EnhancedSong(
                    "Uptown Funk", "Mark Ronson ft. Bruno Mars", "https://www.youtube.com/watch?v=OPf0YbXqDm0",
                    270, "Funk", "Energetic", has_vocals=True, lyrics_theme="confidence",
                    energy_level=9, therapeutic_benefits=["confidence building", "energy boost"],
                    popularity_score=96, release_year=2014, emotional_intensity=8,
                    description="High-energy funk that builds confidence"
                ),
                EnhancedSong(
                    "Walking on Sunshine", "Katrina and the Waves", "https://www.youtube.com/watch?v=iPUmE-tne5U",
                    239, "Pop", "Uplifting", has_vocals=True, lyrics_theme="optimism",
                    energy_level=8, therapeutic_benefits=["optimism", "mood elevation"],
                    popularity_score=85, release_year=1985, emotional_intensity=7,
                    description="Classic feel-good anthem of pure optimism"
                ),
            ],
            
            "sadness": [
                # Healing vocal tracks for sadness
                EnhancedSong(
                    "Someone Like You", "Adele", "https://www.youtube.com/watch?v=hLQl3WQQoQ0",
                    285, "Pop", "Healing", has_vocals=True, lyrics_theme="acceptance",
                    energy_level=3, therapeutic_benefits=["emotional processing", "acceptance"],
                    popularity_score=94, release_year=2011, emotional_intensity=8,
                    description="Emotional healing through acceptance and letting go"
                ),
                EnhancedSong(
                    "Fix You", "Coldplay", "https://www.youtube.com/watch?v=k4V3Mo61fJM",
                    295, "Alternative", "Healing", has_vocals=True, lyrics_theme="support",
                    energy_level=4, therapeutic_benefits=["comfort", "hope"],
                    popularity_score=91, release_year=2005, emotional_intensity=7,
                    description="Gentle support and promise of healing"
                ),
                EnhancedSong(
                    "Breathe Me", "Sia", "https://www.youtube.com/watch?v=ghPcYqn0p4Y",
                    273, "Pop", "Vulnerable", has_vocals=True, lyrics_theme="vulnerability",
                    energy_level=3, therapeutic_benefits=["emotional release", "self-acceptance"],
                    popularity_score=78, release_year=2004, emotional_intensity=9,
                    description="Raw vulnerability leading to strength"
                ),
                EnhancedSong(
                    "Mad World", "Gary Jules", "https://www.youtube.com/watch?v=4N3N1MlvVc4",
                    191, "Alternative", "Melancholic", has_vocals=True, lyrics_theme="isolation",
                    energy_level=2, therapeutic_benefits=["emotional validation", "processing"],
                    popularity_score=82, release_year=2001, emotional_intensity=8,
                    description="Gentle acknowledgment of sadness and isolation"
                ),
                EnhancedSong(
                    "The Sound of Silence", "Simon & Garfunkel", "https://www.youtube.com/watch?v=4fWyzwo1xg0",
                    200, "Folk", "Contemplative", has_vocals=True, lyrics_theme="reflection",
                    energy_level=3, therapeutic_benefits=["contemplation", "inner peace"],
                    popularity_score=89, release_year=1964, emotional_intensity=6,
                    description="Peaceful reflection and gentle comfort"
                ),
            ],
            
            "anger": [
                # Empowering vocal tracks for anger transformation
                EnhancedSong(
                    "Stronger (What Doesn't Kill You)", "Kelly Clarkson", "https://www.youtube.com/watch?v=Xn676-fLq7I",
                    222, "Pop", "Empowering", has_vocals=True, lyrics_theme="resilience",
                    energy_level=8, therapeutic_benefits=["empowerment", "resilience building"],
                    popularity_score=87, release_year=2011, emotional_intensity=7,
                    description="Transforming challenges into strength"
                ),
                EnhancedSong(
                    "Roar", "Katy Perry", "https://www.youtube.com/watch?v=CevxZvSJLk8",
                    223, "Pop", "Empowering", has_vocals=True, lyrics_theme="self-empowerment",
                    energy_level=8, therapeutic_benefits=["confidence", "self-advocacy"],
                    popularity_score=90, release_year=2013, emotional_intensity=7,
                    description="Finding your voice and inner power"
                ),
                EnhancedSong(
                    "Fight Song", "Rachel Platten", "https://www.youtube.com/watch?v=xo1VInw-SKc",
                    204, "Pop", "Motivational", has_vocals=True, lyrics_theme="perseverance",
                    energy_level=7, therapeutic_benefits=["motivation", "inner strength"],
                    popularity_score=84, release_year=2014, emotional_intensity=8,
                    description="Inner strength and determination anthem"
                ),
                EnhancedSong(
                    "Let It Go", "Idina Menzel", "https://www.youtube.com/watch?v=L0MK7qz13bU",
                    225, "Pop", "Release", has_vocals=True, lyrics_theme="liberation",
                    energy_level=6, therapeutic_benefits=["emotional release", "freedom"],
                    popularity_score=93, release_year=2013, emotional_intensity=8,
                    description="Powerful release of anger and embracing freedom"
                ),
                EnhancedSong(
                    "Shake It Off", "Taylor Swift", "https://www.youtube.com/watch?v=nfWlot6h_JM",
                    219, "Pop", "Resilient", has_vocals=True, lyrics_theme="resilience",
                    energy_level=8, therapeutic_benefits=["resilience", "positivity"],
                    popularity_score=91, release_year=2014, emotional_intensity=6,
                    description="Shaking off negativity with resilient positivity"
                ),
            ],
            
            "fear": [
                # Courage-building vocal tracks
                EnhancedSong(
                    "Brave", "Sara Bareilles", "https://www.youtube.com/watch?v=QUQsqBqxoR4",
                    239, "Pop", "Courage", has_vocals=True, lyrics_theme="courage",
                    energy_level=6, therapeutic_benefits=["courage building", "self-expression"],
                    popularity_score=81, release_year=2013, emotional_intensity=7,
                    description="Finding courage to speak your truth"
                ),
                EnhancedSong(
                    "Confident", "Demi Lovato", "https://www.youtube.com/watch?v=9f06QZCVUHg",
                    216, "Pop", "Confidence", has_vocals=True, lyrics_theme="self-confidence",
                    energy_level=8, therapeutic_benefits=["confidence", "fearlessness"],
                    popularity_score=79, release_year=2015, emotional_intensity=7,
                    description="Building unshakeable self-confidence"
                ),
                EnhancedSong(
                    "Titanium", "David Guetta ft. Sia", "https://www.youtube.com/watch?v=JRfuAukYTKg",
                    245, "Electronic", "Strength", has_vocals=True, lyrics_theme="invincibility",
                    energy_level=7, therapeutic_benefits=["inner strength", "resilience"],
                    popularity_score=88, release_year=2011, emotional_intensity=8,
                    description="Unbreakable inner strength and resilience"
                ),
                EnhancedSong(
                    "Eye of the Tiger", "Survivor", "https://www.youtube.com/watch?v=btPJPFnesV4",
                    245, "Rock", "Motivational", has_vocals=True, lyrics_theme="determination",
                    energy_level=9, therapeutic_benefits=["motivation", "courage"],
                    popularity_score=92, release_year=1982, emotional_intensity=8,
                    description="Classic courage and determination anthem"
                ),
                EnhancedSong(
                    "Unstoppable", "Sia", "https://www.youtube.com/watch?v=cxjvTXo9WWM",
                    217, "Pop", "Empowering", has_vocals=True, lyrics_theme="invincibility",
                    energy_level=7, therapeutic_benefits=["empowerment", "fearlessness"],
                    popularity_score=85, release_year=2016, emotional_intensity=7,
                    description="Feeling unstoppable and fearless"
                ),
            ],
            
            "anxiety": [
                # Calming vocal tracks with soothing messages
                EnhancedSong(
                    "Breathe", "Taylor Swift", "https://www.youtube.com/watch?v=COWkaLm4zIE",
                    263, "Country", "Calming", has_vocals=True, lyrics_theme="reassurance",
                    energy_level=3, therapeutic_benefits=["anxiety relief", "reassurance"],
                    popularity_score=76, release_year=2008, emotional_intensity=4,
                    description="Gentle reassurance and breathing guidance"
                ),
                EnhancedSong(
                    "Weightless", "Marconi Union", "https://www.youtube.com/watch?v=UfcAVejslrU",
                    515, "Ambient", "Calming", has_vocals=False, lyrics_theme="meditation",
                    energy_level=1, therapeutic_benefits=["anxiety reduction", "relaxation"],
                    popularity_score=72, release_year=2011, emotional_intensity=2,
                    description="Scientifically designed to reduce anxiety"
                ),
                EnhancedSong(
                    "Three Little Birds", "Bob Marley", "https://www.youtube.com/watch?v=LanCLS_hIo4",
                    180, "Reggae", "Peaceful", has_vocals=True, lyrics_theme="reassurance",
                    energy_level=4, therapeutic_benefits=["peace", "reassurance"],
                    popularity_score=88, release_year=1977, emotional_intensity=3,
                    description="Everything's gonna be alright - gentle reassurance"
                ),
                EnhancedSong(
                    "Don't Stop Believin'", "Journey", "https://www.youtube.com/watch?v=1k8craCGpgs",
                    251, "Rock", "Hopeful", has_vocals=True, lyrics_theme="hope",
                    energy_level=6, therapeutic_benefits=["hope", "perseverance"],
                    popularity_score=94, release_year=1981, emotional_intensity=6,
                    description="Maintaining hope and belief in better times"
                ),
                EnhancedSong(
                    "Here Comes the Sun", "The Beatles", "https://www.youtube.com/watch?v=KQetemT1sWc",
                    185, "Pop", "Hopeful", has_vocals=True, lyrics_theme="optimism",
                    energy_level=5, therapeutic_benefits=["optimism", "hope"],
                    popularity_score=91, release_year=1969, emotional_intensity=4,
                    description="Gentle optimism and hope for brighter days"
                ),
            ],
            
            "love": [
                # Romantic and loving vocal tracks
                EnhancedSong(
                    "All of Me", "John Legend", "https://www.youtube.com/watch?v=450p7goxZqg",
                    269, "R&B", "Romantic", has_vocals=True, lyrics_theme="devotion",
                    energy_level=4, therapeutic_benefits=["love expression", "connection"],
                    popularity_score=93, release_year=2013, emotional_intensity=8,
                    description="Deep love and complete devotion"
                ),
                EnhancedSong(
                    "Perfect", "Ed Sheeran", "https://www.youtube.com/watch?v=2Vv-BfVoq4g",
                    263, "Pop", "Romantic", has_vocals=True, lyrics_theme="perfect love",
                    energy_level=4, therapeutic_benefits=["love celebration", "intimacy"],
                    popularity_score=95, release_year=2017, emotional_intensity=7,
                    description="Celebrating perfect love and partnership"
                ),
                EnhancedSong(
                    "At Last", "Etta James", "https://www.youtube.com/watch?v=S-cbOl96RFM",
                    180, "Soul", "Classic Romance", has_vocals=True, lyrics_theme="fulfillment",
                    energy_level=3, therapeutic_benefits=["emotional fulfillment", "joy"],
                    popularity_score=89, release_year=1960, emotional_intensity=8,
                    description="Timeless expression of love found"
                ),
                EnhancedSong(
                    "Make You Feel My Love", "Adele", "https://www.youtube.com/watch?v=0put0_a--Ng",
                    213, "Pop", "Heartfelt", has_vocals=True, lyrics_theme="devotion",
                    energy_level=3, therapeutic_benefits=["emotional connection", "comfort"],
                    popularity_score=87, release_year=2008, emotional_intensity=9,
                    description="Deep emotional connection and unwavering support"
                ),
                EnhancedSong(
                    "Thinking Out Loud", "Ed Sheeran", "https://www.youtube.com/watch?v=lp-EO5I60KA",
                    281, "Pop", "Tender", has_vocals=True, lyrics_theme="eternal love",
                    energy_level=3, therapeutic_benefits=["love appreciation", "commitment"],
                    popularity_score=92, release_year=2014, emotional_intensity=7,
                    description="Tender expression of growing old together"
                ),
            ],
            
            "neutral": [
                # Balanced vocal and instrumental tracks
                EnhancedSong(
                    "Weightless", "Marconi Union", "https://www.youtube.com/watch?v=UfcAVejslrU",
                    515, "Ambient", "Balanced", has_vocals=False, lyrics_theme="meditation",
                    energy_level=2, therapeutic_benefits=["balance", "focus"],
                    popularity_score=72, release_year=2011, emotional_intensity=1,
                    description="Perfect for focus and emotional balance"
                ),
                EnhancedSong(
                    "Clair de Lune", "Claude Debussy", "https://www.youtube.com/watch?v=CvFH_6DNRCY",
                    300, "Classical", "Peaceful", has_vocals=False, lyrics_theme="serenity",
                    energy_level=2, therapeutic_benefits=["peace", "contemplation"],
                    popularity_score=85, release_year=1905, emotional_intensity=2,
                    description="Timeless classical beauty and serenity"
                ),
                EnhancedSong(
                    "Holocene", "Bon Iver", "https://www.youtube.com/watch?v=TWcyIpul8OE",
                    337, "Indie", "Contemplative", has_vocals=True, lyrics_theme="reflection",
                    energy_level=3, therapeutic_benefits=["contemplation", "peace"],
                    popularity_score=78, release_year=2011, emotional_intensity=4,
                    description="Peaceful indie reflection on existence"
                ),
                EnhancedSong(
                    "River", "Joni Mitchell", "https://www.youtube.com/watch?v=3NH-ctddY9o",
                    240, "Folk", "Soothing", has_vocals=True, lyrics_theme="introspection",
                    energy_level=3, therapeutic_benefits=["introspection", "calm"],
                    popularity_score=81, release_year=1971, emotional_intensity=5,
                    description="Gentle folk introspection and emotional processing"
                ),
                EnhancedSong(
                    "Mad About You", "Sting", "https://www.youtube.com/watch?v=65Z8qHi7zKA",
                    231, "Pop", "Mellow", has_vocals=True, lyrics_theme="contentment",
                    energy_level=4, therapeutic_benefits=["contentment", "balance"],
                    popularity_score=79, release_year=1991, emotional_intensity=4,
                    description="Mellow contentment and emotional balance"
                ),
            ]
        }
        
        return playlists
    
    def _create_english_vocal_library(self) -> Dict[str, List[EnhancedSong]]:
        """Create expanded English vocal library organized by therapeutic purpose"""
        
        return {
            "empowerment": [
                EnhancedSong(
                    "Stronger", "Kelly Clarkson", "https://www.youtube.com/watch?v=Xn676-fLq7I",
                    222, "Pop", "Empowering", has_vocals=True, lyrics_theme="resilience",
                    energy_level=8, therapeutic_benefits=["empowerment", "resilience"],
                    popularity_score=87, release_year=2011, emotional_intensity=7
                ),
                EnhancedSong(
                    "Girl on Fire", "Alicia Keys", "https://www.youtube.com/watch?v=J91ti_MpdHA",
                    224, "R&B", "Empowering", has_vocals=True, lyrics_theme="self-empowerment",
                    energy_level=8, therapeutic_benefits=["confidence", "self-worth"],
                    popularity_score=83, release_year=2012, emotional_intensity=8
                ),
            ],
            "healing": [
                EnhancedSong(
                    "Scars to Your Beautiful", "Alessia Cara", "https://www.youtube.com/watch?v=MWASeaYuHZo",
                    230, "Pop", "Healing", has_vocals=True, lyrics_theme="self-acceptance",
                    energy_level=5, therapeutic_benefits=["self-acceptance", "healing"],
                    popularity_score=82, release_year=2015, emotional_intensity=6
                ),
                EnhancedSong(
                    "Beautiful", "Christina Aguilera", "https://www.youtube.com/watch?v=eAfyFTzZDMM",
                    238, "Pop", "Healing", has_vocals=True, lyrics_theme="self-worth",
                    energy_level=4, therapeutic_benefits=["self-worth", "acceptance"],
                    popularity_score=86, release_year=2002, emotional_intensity=7
                ),
            ],
            "motivation": [
                EnhancedSong(
                    "High Hopes", "Panic! At The Disco", "https://www.youtube.com/watch?v=IPXIgEAGe4U",
                    191, "Pop", "Motivational", has_vocals=True, lyrics_theme="ambition",
                    energy_level=8, therapeutic_benefits=["motivation", "optimism"],
                    popularity_score=89, release_year=2018, emotional_intensity=7
                ),
                EnhancedSong(
                    "Count on Me", "Bruno Mars", "https://www.youtube.com/watch?v=CRLlbYuQUWg",
                    195, "Pop", "Supportive", has_vocals=True, lyrics_theme="friendship",
                    energy_level=5, therapeutic_benefits=["support", "connection"],
                    popularity_score=84, release_year=2010, emotional_intensity=5
                ),
            ]
        }
    
    def get_vocal_songs_for_emotion(self, emotion: str, count: int = 5) -> List[EnhancedSong]:
        """Get vocal songs specifically for an emotion with enhanced English content"""
        
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
        
        mapped_emotion = emotion_mapping.get(emotion.lower(), emotion.lower())
        
        # Get songs for the emotion, prioritizing vocal tracks
        if mapped_emotion in self.emotion_playlists:
            all_songs = self.emotion_playlists[mapped_emotion].copy()
            # Filter for vocal tracks first
            vocal_songs = [song for song in all_songs if song.has_vocals]
            
            # If we have enough vocal songs, use them; otherwise mix with instrumental
            if len(vocal_songs) >= count:
                songs = vocal_songs
            else:
                songs = vocal_songs + [song for song in all_songs if not song.has_vocals]
        else:
            # Default to neutral if emotion not found
            songs = self.emotion_playlists["neutral"].copy()
        
        # Shuffle and return requested count
        random.shuffle(songs)
        return songs[:count]
    
    def get_mixed_playlist(self, emotion: str, vocal_ratio: float = 0.7, count: int = 5) -> List[EnhancedSong]:
        """Get a mixed playlist with specified vocal/instrumental ratio"""
        
        all_songs = self.get_songs_for_emotion(emotion, count * 2)  # Get more to choose from
        
        vocal_songs = [song for song in all_songs if song.has_vocals]
        instrumental_songs = [song for song in all_songs if not song.has_vocals]
        
        vocal_count = int(count * vocal_ratio)
        instrumental_count = count - vocal_count
        
        # Select songs based on ratio, but ensure we get the requested count
        selected_vocal = random.sample(vocal_songs, min(vocal_count, len(vocal_songs)))
        selected_instrumental = random.sample(instrumental_songs, min(instrumental_count, len(instrumental_songs)))
        
        # If we don't have enough songs of one type, fill with the other type
        total_selected = len(selected_vocal) + len(selected_instrumental)
        if total_selected < count:
            remaining_needed = count - total_selected
            # Try to fill with remaining vocal songs first, then instrumental
            remaining_vocal = [s for s in vocal_songs if s not in selected_vocal]
            remaining_instrumental = [s for s in instrumental_songs if s not in selected_instrumental]
            
            additional_songs = (remaining_vocal + remaining_instrumental)[:remaining_needed]
            selected_vocal.extend([s for s in additional_songs if s.has_vocals])
            selected_instrumental.extend([s for s in additional_songs if not s.has_vocals])
        
        # Combine and shuffle
        playlist = selected_vocal + selected_instrumental
        random.shuffle(playlist)
        
        return playlist[:count]
    
    def get_songs_for_emotion(self, emotion: str, count: int = 5) -> List[EnhancedSong]:
        """Get songs for emotion (includes both vocal and instrumental)"""
        
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
        
        mapped_emotion = emotion_mapping.get(emotion.lower(), emotion.lower())
        
        if mapped_emotion in self.emotion_playlists:
            songs = self.emotion_playlists[mapped_emotion].copy()
        else:
            songs = self.emotion_playlists["neutral"].copy()
        
        random.shuffle(songs)
        return songs[:count]
    
    def filter_content(self, songs: List[EnhancedSong], rating: str = "clean") -> List[EnhancedSong]:
        """Filter songs by content rating"""
        
        if rating == "all":
            return songs
        
        return [song for song in songs if song.content_rating == rating]
    
    def get_songs_by_energy_level(self, min_energy: int, max_energy: int, count: int = 5) -> List[EnhancedSong]:
        """Get songs within specific energy level range"""
        
        all_songs = []
        for emotion_songs in self.emotion_playlists.values():
            all_songs.extend(emotion_songs)
        
        filtered_songs = [
            song for song in all_songs 
            if min_energy <= song.energy_level <= max_energy
        ]
        
        random.shuffle(filtered_songs)
        return filtered_songs[:count]
    
    def get_therapeutic_songs(self, benefit: str, count: int = 5) -> List[EnhancedSong]:
        """Get songs that provide specific therapeutic benefits"""
        
        all_songs = []
        for emotion_songs in self.emotion_playlists.values():
            all_songs.extend(emotion_songs)
        
        therapeutic_songs = [
            song for song in all_songs 
            if benefit.lower() in [b.lower() for b in song.therapeutic_benefits]
        ]
        
        random.shuffle(therapeutic_songs)
        return therapeutic_songs[:count]