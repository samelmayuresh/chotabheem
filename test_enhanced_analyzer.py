#!/usr/bin/env python3
# Test for enhanced emotion analyzer
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils.enhanced_emotion_analyzer import EnhancedEmotionAnalyzer, EmotionConfig, EmotionResult
    import numpy as np
    
    print("Testing Enhanced Emotion Analyzer...")
    
    # Test different configurations
    print("\n1. Testing Different Configurations...")
    
    configs = [
        ('Fast', EmotionConfig(precision_level='fast', enable_ensemble=True)),
        ('Balanced', EmotionConfig(precision_level='balanced', enable_ensemble=True)),
        ('High Precision', EmotionConfig(precision_level='high_precision', enable_ensemble=True))
    ]
    
    analyzers = {}
    
    for name, config in configs:
        print(f"\n1.{configs.index((name, config)) + 1} Testing {name} Configuration...")
        try:
            analyzer = EnhancedEmotionAnalyzer(config)
            analyzers[name] = analyzer
            
            print(f"✅ {name} analyzer created successfully")
            
            # Check component initialization
            components = analyzer.get_performance_summary()['component_status']
            print(f"   - Text ensemble: {components['text_ensemble']}")
            print(f"   - Voice ensemble: {components['voice_ensemble']}")
            print(f"   - Fusion engine: {components['fusion_engine']}")
            print(f"   - Quality assurance: {components['quality_assurance']}")
            
        except Exception as e:
            print(f"❌ {name} analyzer failed: {e}")
    
    # Test text analysis
    print("\n2. Testing Text Analysis...")
    
    test_texts = [
        "I am absolutely thrilled and excited about this amazing opportunity!",
        "I feel really sad and disappointed about what happened today.",
        "This situation is making me quite angry and frustrated.",
        "I'm feeling pretty neutral about the whole thing."
    ]
    
    if 'Balanced' in analyzers:
        analyzer = analyzers['Balanced']
        
        for i, text in enumerate(test_texts, 1):
            print(f"\n2.{i} Testing: '{text[:50]}...'")
            
            try:
                result = analyzer.analyze_text(text)
                
                print(f"✅ Text analysis completed")
                print(f"   - Primary emotion: {result.primary_emotion.label}")
                print(f"   - Confidence: {result.primary_emotion.confidence:.3f}")
                print(f"   - Confidence level: {result.confidence_level}")
                print(f"   - Processing time: {result.processing_metadata['processing_time']:.3f}s")
                print(f"   - Insights: {len(result.insights)}")
                print(f"   - Recommendations: {len(result.recommendations)}")
                
                # Show top insights
                for insight in result.insights[:2]:
                    print(f"     • {insight}")
                
            except Exception as e:
                print(f"❌ Text analysis failed: {e}")
    
    # Test audio analysis
    print("\n3. Testing Audio Analysis...")
    
    # Create different types of test audio
    sample_rate = 16000
    duration = 2.0
    
    test_audios = {
        "Happy tone": np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration))) * 0.7,
        "Sad tone": np.sin(2 * np.pi * 220 * np.linspace(0, duration, int(sample_rate * duration))) * 0.3,
        "Energetic": np.random.randn(int(sample_rate * duration)) * 0.5,
        "Quiet": np.sin(2 * np.pi * 330 * np.linspace(0, duration, int(sample_rate * duration))) * 0.1
    }
    
    if 'Balanced' in analyzers:
        analyzer = analyzers['Balanced']
        
        for i, (description, audio) in enumerate(test_audios.items(), 1):
            print(f"\n3.{i} Testing: {description}")
            
            try:
                result = analyzer.analyze_audio(audio.astype(np.float32), sample_rate)
                
                print(f"✅ Audio analysis completed")
                print(f"   - Primary emotion: {result.primary_emotion.label}")
                print(f"   - Confidence: {result.primary_emotion.confidence:.3f}")
                print(f"   - Confidence level: {result.confidence_level}")
                print(f"   - Processing time: {result.processing_metadata['processing_time']:.3f}s")
                
                # Show validation result if available
                if result.validation_result:
                    print(f"   - Validation: {'✅ Valid' if result.validation_result.is_valid else '❌ Invalid'}")
                
            except Exception as e:
                print(f"❌ Audio analysis failed: {e}")
    
    # Test multimodal analysis
    print("\n4. Testing Multimodal Analysis...")
    
    if 'Balanced' in analyzers:
        analyzer = analyzers['Balanced']
        
        multimodal_tests = [
            ("Happy text + Happy audio", test_texts[0], test_audios["Happy tone"]),
            ("Sad text + Sad audio", test_texts[1], test_audios["Sad tone"]),
            ("Happy text + Sad audio", test_texts[0], test_audios["Sad tone"]),  # Conflicting
        ]
        
        for i, (description, text, audio) in enumerate(multimodal_tests, 1):
            print(f"\n4.{i} Testing: {description}")
            
            try:
                result = analyzer.analyze_multimodal(text, audio.astype(np.float32), sample_rate)
                
                print(f"✅ Multimodal analysis completed")
                print(f"   - Primary emotion: {result.primary_emotion.label}")
                print(f"   - Confidence: {result.primary_emotion.confidence:.3f}")
                print(f"   - Confidence level: {result.confidence_level}")
                print(f"   - Processing time: {result.processing_metadata['processing_time']:.3f}s")
                
                # Show fusion information if available
                if result.fusion_result:
                    print(f"   - Fusion method: {result.fusion_result.fusion_method}")
                    print(f"   - Modality contributions: {result.fusion_result.modality_contributions}")
                
                # Show some insights
                for insight in result.insights[:3]:
                    print(f"     • {insight}")
                
            except Exception as e:
                print(f"❌ Multimodal analysis failed: {e}")
    
    # Test performance tracking
    print("\n5. Testing Performance Tracking...")
    
    if 'Balanced' in analyzers:
        analyzer = analyzers['Balanced']
        
        # Perform multiple analyses to build performance data
        for i in range(5):
            analyzer.analyze_text(f"Test text number {i}")
        
        performance = analyzer.get_performance_summary()
        
        print(f"✅ Performance summary generated")
        print(f"   - Total analyses: {performance['metrics']['total_analyses']}")
        print(f"   - Text analyses: {performance['metrics']['text_analyses']}")
        print(f"   - Voice analyses: {performance['metrics']['voice_analyses']}")
        print(f"   - Multimodal analyses: {performance['metrics']['multimodal_analyses']}")
        print(f"   - Average processing time: {performance['metrics']['average_processing_time']:.3f}s")
        print(f"   - Average confidence: {performance['metrics']['average_confidence']:.3f}")
        print(f"   - Recent analyses: {performance['recent_analyses']}")
    
    # Test configuration save/load
    print("\n6. Testing Configuration Save/Load...")
    
    try:
        # Create custom configuration
        custom_config = EmotionConfig(
            precision_level='high_precision',
            enable_ensemble=True,
            enable_multimodal=True,
            fusion_strategy='attention_based',
            confidence_threshold=0.7
        )
        
        custom_analyzer = EnhancedEmotionAnalyzer(custom_config)
        
        # Save configuration
        config_file = "test_emotion_config.json"
        custom_analyzer.save_configuration(config_file)
        print(f"✅ Configuration saved to {config_file}")
        
        # Load configuration
        loaded_analyzer = EnhancedEmotionAnalyzer.load_configuration(config_file)
        print(f"✅ Configuration loaded from {config_file}")
        
        # Verify configuration
        loaded_summary = loaded_analyzer.get_performance_summary()
        print(f"   - Precision level: {loaded_summary['configuration']['precision_level']}")
        print(f"   - Fusion strategy: {loaded_summary['configuration']['fusion_strategy']}")
        
        # Clean up
        if os.path.exists(config_file):
            os.remove(config_file)
        
    except Exception as e:
        print(f"❌ Configuration save/load failed: {e}")
    
    # Test error handling
    print("\n7. Testing Error Handling...")
    
    if 'Fast' in analyzers:
        analyzer = analyzers['Fast']
        
        # Test with empty inputs
        try:
            empty_text_result = analyzer.analyze_text("")
            print(f"✅ Empty text handled: {empty_text_result.primary_emotion.label}")
        except Exception as e:
            print(f"⚠️  Empty text handling: {e}")
        
        # Test with invalid audio
        try:
            empty_audio = np.array([])
            empty_audio_result = analyzer.analyze_audio(empty_audio, sample_rate)
            print(f"✅ Empty audio handled: {empty_audio_result.primary_emotion.label}")
        except Exception as e:
            print(f"⚠️  Empty audio handling: {e}")
    
    # Test different precision levels comparison
    print("\n8. Testing Precision Level Comparison...")
    
    test_text = "I'm feeling incredibly happy and excited about this wonderful news!"
    
    for name, analyzer in analyzers.items():
        try:
            result = analyzer.analyze_text(test_text)
            print(f"✅ {name}: {result.primary_emotion.label} (conf: {result.primary_emotion.confidence:.3f}, time: {result.processing_metadata['processing_time']:.3f}s)")
        except Exception as e:
            print(f"❌ {name}: {e}")
    
    print("\n🎉 Enhanced emotion analyzer tests completed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()