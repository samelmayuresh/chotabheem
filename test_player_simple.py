# test_player_simple.py - Simple test for player engine functionality
from utils.player_engine import PlayerEngine, PlaybackState, PlaybackStatus
from utils.enhanced_song_database import EnhancedSong
from utils.playlist_manager import TherapeuticPlaylist

def test_basic_functionality():
    """Test basic player engine functionality"""
    
    # Create test data
    test_songs = [
        EnhancedSong(
            title="Song 1", artist="Artist 1", url="https://example.com/1",
            duration=180, genre="Pop", mood="Happy"
        ),
        EnhancedSong(
            title="Song 2", artist="Artist 2", url="https://example.com/2",
            duration=200, genre="Rock", mood="Energetic"
        )
    ]
    
    test_playlist = TherapeuticPlaylist(
        id="test-playlist",
        name="Test Playlist",
        songs=test_songs,
        emotion_focus="mixed"
    )
    
    # Create player engine
    engine = PlayerEngine()
    
    # Test loading playlist
    success = engine.load_playlist(test_playlist)
    assert success, "Failed to load playlist"
    assert engine.state.playlist == test_playlist
    assert engine.state.current_song == test_songs[0]
    print("✅ Playlist loading works")
    
    # Test playing song
    success = engine.play_song()
    assert success, "Failed to play song"
    assert engine.state.status == PlaybackStatus.PLAYING
    assert engine.state.current_song == test_songs[0]
    print("✅ Song playback works")
    
    # Test pause/resume
    success = engine.pause()
    assert success, "Failed to pause"
    assert engine.state.status == PlaybackStatus.PAUSED
    
    success = engine.resume()
    assert success, "Failed to resume"
    assert engine.state.status == PlaybackStatus.PLAYING
    print("✅ Pause/resume works")
    
    # Test skip to next
    success = engine.skip_to_next()
    assert success, "Failed to skip to next"
    assert engine.state.playlist_position == 1
    assert engine.state.current_song == test_songs[1]
    print("✅ Skip to next works")
    
    # Test skip to previous
    success = engine.skip_to_previous()
    assert success, "Failed to skip to previous"
    assert engine.state.playlist_position == 0
    assert engine.state.current_song == test_songs[0]
    print("✅ Skip to previous works")
    
    # Test volume control
    success = engine.set_volume(0.5)
    assert success, "Failed to set volume"
    assert engine.state.volume == 0.5
    print("✅ Volume control works")
    
    # Test repeat mode
    success = engine.set_repeat_mode("playlist")
    assert success, "Failed to set repeat mode"
    assert engine.state.repeat_mode == "playlist"
    print("✅ Repeat mode works")
    
    # Test shuffle
    success = engine.set_shuffle(True)
    assert success, "Failed to set shuffle"
    assert engine.state.shuffle == True
    print("✅ Shuffle mode works")
    
    # Test stop
    success = engine.stop()
    assert success, "Failed to stop"
    assert engine.state.status == PlaybackStatus.STOPPED
    print("✅ Stop works")
    
    # Test session statistics
    stats = engine.get_session_statistics()
    assert isinstance(stats, dict)
    assert "session_duration" in stats
    assert "user_interactions" in stats
    print("✅ Session statistics work")
    
    # Cleanup
    engine.cleanup()
    print("✅ Cleanup works")
    
    print("\n🎉 All basic player engine tests passed!")

if __name__ == "__main__":
    test_basic_functionality()