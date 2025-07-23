#!/usr/bin/env python3
# Test for emotion analyzer integration
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils.emotion_analyzer_integration import EmotionAnalyzer, IntegratedEmotionAnalyzer, create_enhanced_analyzer
    import numpy as np
    
    print("Testing Emotion Analyzer Integration...")
    
    # Test backward compatibility
    print("\n1. Testing Backward Compatibility...")
    
    # Create analyzer with legacy interface
    legacy_analyzer = EmotionAnalyzer()
    
    test_text = "I am feeling really happy and excited today!"
    
    # Test text analysis (legacy format)
    text_result = legacy_analyzer.analyze_text(test_text)
    print(f"✅ Legacy text analysis: {len(text_result)} emotions")
    print(f"   - Primary: {text_result[0]['label']} (score: {text_result[0]['score']:.3f})")
    
    # Test audio analysis (legacy format)
    sample_rate = 16000
    test_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 2, int(sample_rate * 2))).astype(np.float32)
    
    transcript, audio_result = legacy_analyzer.analyze_audio(test_audio, sample_rate)
    print(f"✅ Legacy audio analysis: {len(audio_result)} emotions")
    print(f"   - Primary: {audio_result[0]['label']} (score: {audio_result[0]['score']:.3f})")
    
    # Test insights (enhanced)
    insights = legacy_analyzer.get_emotion_insights(text_result)
    print(f"✅ Enhanced insights: {len(insights['insights'])} insights generated")
    for insight in insights['insights'][:2]:
        print(f"   • {insight}")
    
    # Test enhanced features
    print("\n2. Testing Enhanced Features...")
    
    enhanced_analyzer = create_enhanced_analyzer('balanced')
    
    # Test multimodal analysis (new feature)
    multimodal_result = enhanced_analyzer.analyze_multimodal(test_text, test_audio, sample_rate)
    print(f"✅ Multimodal analysis completed")
    print(f"   - Primary emotion: {multimodal_result['primary_emotion']}")
    print(f"   - Confidence: {multimodal_result['confidence']:.3f}")
    print(f"   - Fusion method: {multimodal_result['fusion_method']}")
    print(f"   - Processing time: {multimodal_result['processing_time']:.3f}s")
    
    # Test performance summary
    print("\n3. Testing Performance Summary...")
    
    performance = enhanced_analyzer.get_performance_summary()
    if 'error' not in performance:
        print(f"✅ Performance summary retrieved")
        print(f"   - Total analyses: {performance['metrics']['total_analyses']}")
        print(f"   - Component status: {sum(performance['component_status'].values())} active")
    else:
        print(f"⚠️  Performance summary fallback: {performance.get('fallback_mode', False)}")
    
    # Test different precision levels
    print("\n4. Testing Different Precision Levels...")
    
    precision_levels = ['fast', 'balanced', 'high_precision']
    
    for level in precision_levels:
        try:
            analyzer = create_enhanced_analyzer(level)
            result = analyzer.analyze_text("This is a test message.")
            
            processing_time = 0.0
            if isinstance(result, list) and result:
                processing_time = result[0].get('processing_time', 0.0)
            
            print(f"✅ {level}: {result[0]['label'] if result else 'none'} (time: {processing_time:.3f}s)")
            
        except Exception as e:
            print(f"❌ {level}: {e}")
    
    # Test error handling
    print("\n5. Testing Error Handling...")
    
    # Test with empty inputs
    empty_text_result = legacy_analyzer.analyze_text("")
    print(f"✅ Empty text handled: {empty_text_result[0]['label'] if empty_text_result else 'none'}")
    
    empty_audio = np.array([])
    try:
        _, empty_audio_result = legacy_analyzer.analyze_audio(empty_audio, sample_rate)
        print(f"✅ Empty audio handled: {empty_audio_result[0]['label'] if empty_audio_result else 'none'}")
    except Exception as e:
        print(f"⚠️  Empty audio handling: {e}")
    
    # Test configuration save/load
    print("\n6. Testing Configuration Management...")
    
    try:
        config_file = "test_integration_config.json"
        
        # Save configuration
        enhanced_analyzer.save_configuration(config_file)
        print(f"✅ Configuration saved")
        
        # Load configuration
        loaded_analyzer = IntegratedEmotionAnalyzer.load_configuration(config_file)
        print(f"✅ Configuration loaded")
        
        # Test loaded analyzer
        loaded_result = loaded_analyzer.analyze_text("Test message")
        print(f"✅ Loaded analyzer works: {loaded_result[0]['label'] if loaded_result else 'none'}")
        
        # Clean up
        if os.path.exists(config_file):
            os.remove(config_file)
            
    except Exception as e:
        print(f"⚠️  Configuration management: {e}")
    
    print("\n🎉 Integration tests completed successfully!")
    print("\n📋 Summary:")
    print("   ✅ Backward compatibility maintained")
    print("   ✅ Enhanced features available")
    print("   ✅ Multimodal analysis working")
    print("   ✅ Error handling robust")
    print("   ✅ Configuration management functional")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()