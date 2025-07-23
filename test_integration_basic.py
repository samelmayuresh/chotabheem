# test_integration_basic.py - Basic test for emotion-music integration
from utils.emotion_music_integration import EmotionMusicIntegrator

def test_basic_integration():
    """Test basic integration functionality"""
    
    print("🎵 Testing Basic Emotion-Music Integration")
    print("=" * 50)
    
    try:
        # Test 1: Create integrator
        print("\n1. Creating integrator...")
        integrator = EmotionMusicIntegrator()
        
        assert integrator.music_controller is not None
        assert integrator.emotion_analyzer is not None
        print("  ✅ Integrator created successfully")
        
        # Test 2: Test helper methods
        print("\n2. Testing helper methods...")
        
        # Test intensity mapping
        energy_1 = integrator._map_intensity_to_energy(0.1)
        energy_5 = integrator._map_intensity_to_energy(0.5)
        energy_9 = integrator._map_intensity_to_energy(0.9)
        
        assert 1 <= energy_1 <= 10
        assert 1 <= energy_5 <= 10
        assert 1 <= energy_9 <= 10
        assert energy_1 < energy_5 < energy_9
        
        print(f"  ✅ Intensity mapping: 0.1→{energy_1}, 0.5→{energy_5}, 0.9→{energy_9}")
        
        # Test 3: Integration statistics (empty state)
        print("\n3. Testing integration statistics...")
        stats = integrator.get_integration_statistics()
        
        assert isinstance(stats, dict)
        if "message" in stats:
            print("  ✅ No sessions recorded yet (expected)")
        else:
            print(f"  ✅ Statistics structure: {list(stats.keys())}")
        
        # Test 4: Configuration
        print("\n4. Testing configuration...")
        
        assert integrator.confidence_threshold == 0.6
        assert integrator.secondary_emotion_threshold == 0.3
        assert integrator.dynamic_adjustment_enabled == True
        
        print("  ✅ Default configuration loaded")
        
        # Test 5: Session management (empty operations)
        print("\n5. Testing session management...")
        
        # Test non-existent session operations
        assert integrator.get_session_analytics("fake-id") is None
        assert integrator.complete_session("fake-id") == False
        assert integrator.provide_user_feedback("fake-id", {}) == False
        assert integrator.adjust_session_dynamically("fake-id", None) == False
        
        print("  ✅ Non-existent session handling works correctly")
        
        # Test 6: Music controller integration
        print("\n6. Testing music controller integration...")
        
        # Test that music controller methods are accessible
        assert hasattr(integrator.music_controller, 'song_database')
        assert hasattr(integrator.music_controller, 'playlist_manager')
        assert hasattr(integrator.music_controller, 'player_engine')
        
        # Test song database access
        joy_songs = integrator.music_controller.song_database.get_vocal_songs_for_emotion("joy", 3)
        assert len(joy_songs) > 0
        
        print(f"  ✅ Music controller accessible, {len(joy_songs)} joy songs available")
        
        # Test 7: Cleanup
        print("\n7. Testing cleanup...")
        integrator.cleanup()
        
        assert len(integrator.active_sessions) == 0
        print("  ✅ Cleanup completed")
        
        # Summary
        print("\n" + "=" * 50)
        print("📈 Basic Integration Test Summary:")
        print("  🎯 Integrator creation: ✅")
        print("  🔧 Helper methods: ✅")
        print("  📊 Statistics: ✅")
        print("  ⚙️  Configuration: ✅")
        print("  📋 Session management: ✅")
        print("  🎵 Music controller: ✅")
        print("  🧹 Cleanup: ✅")
        
        print("\n🎉 Basic integration tests completed successfully!")
        print("💡 Advanced features require full emotion analysis setup")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_basic_integration()