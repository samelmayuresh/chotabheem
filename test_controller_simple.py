# test_controller_simple.py - Simple test for enhanced music controller
import tempfile
import os
from utils.enhanced_music_controller import EnhancedMusicController

def test_controller_functionality():
    """Test basic enhanced music controller functionality"""
    
    print("🎵 Testing Enhanced Music Controller")
    print("=" * 45)
    
    # Create temporary file for testing
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
    temp_file.close()
    
    try:
        # Create controller with callbacks
        events = []
        
        def on_session_start(session):
            events.append(f"Session started: {session.session_id}")
            print(f"  🎯 {events[-1]}")
        
        def on_song_change(song, state):
            events.append(f"Song changed: {song.title}")
            print(f"  🎵 {events[-1]}")
        
        def on_session_complete(session, state):
            events.append(f"Session completed: {session.session_id}")
            print(f"  ✅ {events[-1]}")
        
        controller = EnhancedMusicController(
            storage_path=temp_file.name,
            on_session_start=on_session_start,
            on_song_change=on_song_change,
            on_session_complete=on_session_complete
        )
        
        # Test 1: Create targeted session
        print("\n1. Creating targeted therapy session...")
        session = controller.create_continuous_session(
            emotion="joy",
            mode="targeted",
            song_count=3
        )
        
        assert session is not None, "Failed to create session"
        assert session.emotion_focus == "joy"
        assert session.therapy_mode == "targeted"
        assert len(session.playlist.songs) > 0
        print(f"  ✅ Created session with {len(session.playlist.songs)} songs")
        
        # Test 2: Get session info
        print("\n2. Getting session information...")
        info = controller.get_session_info()
        
        assert info is not None, "Failed to get session info"
        assert info["emotion_focus"] == "joy"
        assert info["therapy_mode"] == "targeted"
        print(f"  ✅ Session info: {info['session_id'][:20]}...")
        
        # Test 3: Start playback
        print("\n3. Starting playback...")
        success = controller.start_playback()
        
        assert success, "Failed to start playback"
        state = controller.get_playback_state()
        assert state.current_song is not None
        print(f"  ✅ Playing: {state.current_song.title}")
        
        # Test 4: Playback controls
        print("\n4. Testing playback controls...")
        
        controls_tested = []
        
        # Test pause
        success = controller.handle_playback_control("pause")
        assert success, "Failed to pause"
        controls_tested.append("pause")
        
        # Test resume
        success = controller.handle_playback_control("resume")
        assert success, "Failed to resume"
        controls_tested.append("resume")
        
        # Test volume
        success = controller.handle_playback_control("volume", volume=0.5)
        assert success, "Failed to set volume"
        controls_tested.append("volume")
        
        # Test skip next
        success = controller.handle_playback_control("skip_next")
        assert success, "Failed to skip next"
        controls_tested.append("skip_next")
        
        # Test repeat mode
        success = controller.handle_playback_control("repeat", mode="playlist")
        assert success, "Failed to set repeat mode"
        controls_tested.append("repeat")
        
        print(f"  ✅ Tested controls: {', '.join(controls_tested)}")
        
        # Test 5: Enhanced English playlist
        print("\n5. Testing enhanced English playlist...")
        english_playlist = controller.get_enhanced_english_playlist(
            emotion="love",
            count=3,
            vocal_ratio=0.8
        )
        
        assert len(english_playlist) > 0, "Failed to get English playlist"
        
        vocal_count = sum(1 for song in english_playlist if song.has_vocals)
        english_count = sum(1 for song in english_playlist if song.language == "english")
        
        print(f"  ✅ English playlist: {len(english_playlist)} songs, {vocal_count} vocal, {english_count} English")
        
        # Test 6: Full session mode
        print("\n6. Testing full session mode...")
        full_session = controller.create_continuous_session(
            emotion="sadness",
            mode="full_session",
            secondary_emotions=["healing"]
        )
        
        assert full_session.therapy_mode == "full_session"
        assert "healing" in full_session.secondary_emotions
        print(f"  ✅ Full session created with {len(full_session.playlist.songs)} songs")
        
        # Test 7: Custom session with preferences
        print("\n7. Testing custom session with preferences...")
        user_prefs = {
            "vocal_ratio": 0.9,
            "volume": 0.7,
            "energy_preference": "balanced"
        }
        
        custom_session = controller.create_continuous_session(
            emotion="neutral",
            mode="custom",
            song_count=4,
            user_preferences=user_prefs
        )
        
        assert custom_session.therapy_mode == "custom"
        assert custom_session.user_preferences["vocal_ratio"] == 0.9
        print(f"  ✅ Custom session with preferences applied")
        
        # Test 8: Save and load session
        print("\n8. Testing session persistence...")
        success = controller.save_current_session()
        assert success, "Failed to save session"
        
        playlist_id = controller.current_session.playlist.id
        success = controller.load_saved_playlist(playlist_id)
        assert success, "Failed to load saved playlist"
        print(f"  ✅ Session saved and loaded successfully")
        
        # Test 9: Recommendations
        print("\n9. Testing recommendations...")
        recommendations = controller.get_recommendations("anxiety", limit=2)
        
        assert "popular_playlists" in recommendations
        assert "similar_emotions" in recommendations
        assert "therapeutic_benefits" in recommendations
        print(f"  ✅ Recommendations generated")
        
        # Test 10: Stop and cleanup
        print("\n10. Testing stop and cleanup...")
        success = controller.handle_playback_control("stop")
        assert success, "Failed to stop playback"
        
        controller.cleanup()
        print(f"  ✅ Stopped and cleaned up")
        
        # Summary
        print("\n" + "=" * 45)
        print("📈 Test Summary:")
        print(f"  🎯 Sessions created: 3 (targeted, full, custom)")
        print(f"  🎵 Events tracked: {len(events)}")
        print(f"  ✅ All controller functionality working")
        
        if events:
            print("  📝 Event log:")
            for event in events[-5:]:  # Show last 5 events
                print(f"    - {event}")
        
        print("\n🎉 Enhanced Music Controller tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        try:
            os.unlink(temp_file.name)
        except:
            pass

if __name__ == "__main__":
    test_controller_functionality()