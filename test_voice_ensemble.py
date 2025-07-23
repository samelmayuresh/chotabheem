#!/usr/bin/env python3
# Test for voice emotion ensemble system
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils.voice_emotion_ensemble import (
        VoiceEmotionEnsemble, ModelConfig, create_default_voice_ensemble,
        SpectralEmotionModel, ProsodicEmotionModel, VoiceFeatureExtractor
    )
    import numpy as np
    
    print("Testing Voice Emotion Ensemble System...")
    
    # Test voice feature extraction
    print("\n1. Testing Voice Feature Extraction...")
    feature_extractor = VoiceFeatureExtractor()
    
    # Create dummy audio data
    sample_rate = 16000
    duration = 3.0
    dummy_audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)
    
    try:
        features = feature_extractor.extract_features(dummy_audio, sample_rate)
        print(f"✅ Features extracted successfully")
        print(f"   - MFCC shape: {features.mfcc.shape}")
        print(f"   - Spectral centroid shape: {features.spectral_centroid.shape}")
        print(f"   - F0 shape: {features.f0.shape}")
        print(f"   - Tempo: {features.tempo:.1f} BPM")
    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
    
    # Test individual voice models
    print("\n2. Testing Individual Voice Models...")
    
    # Test Spectral Model
    print("\n2.1 Testing Spectral Emotion Model...")
    spectral_config = ModelConfig(
        name="test_spectral",
        model_path="",
        model_type="spectral",
        weight=1.0,
        enabled=True
    )
    
    try:
        spectral_model = SpectralEmotionModel(spectral_config)
        spectral_model.load_model()
        
        predictions = spectral_model.predict(dummy_audio, sample_rate)
        print(f"✅ Spectral model predictions: {len(predictions)} emotions")
        
        for pred in predictions[:3]:
            print(f"   - {pred.label}: {pred.score:.3f} (conf: {pred.confidence:.3f})")
    
    except Exception as e:
        print(f"❌ Spectral model test failed: {e}")
    
    # Test Prosodic Model
    print("\n2.2 Testing Prosodic Emotion Model...")
    prosodic_config = ModelConfig(
        name="test_prosodic",
        model_path="",
        model_type="prosodic",
        weight=1.0,
        enabled=True
    )
    
    try:
        prosodic_model = ProsodicEmotionModel(prosodic_config)
        prosodic_model.load_model()
        
        predictions = prosodic_model.predict(dummy_audio, sample_rate)
        print(f"✅ Prosodic model predictions: {len(predictions)} emotions")
        
        for pred in predictions[:3]:
            print(f"   - {pred.label}: {pred.score:.3f} (conf: {pred.confidence:.3f})")
    
    except Exception as e:
        print(f"❌ Prosodic model test failed: {e}")
    
    # Test Voice Ensemble
    print("\n3. Testing Voice Emotion Ensemble...")
    
    # Create ensemble with rule-based models (to avoid model download issues)
    test_configs = [
        ModelConfig(
            name="spectral_primary",
            model_path="",
            model_type="spectral",
            weight=1.0,
            enabled=True
        ),
        ModelConfig(
            name="prosodic_secondary",
            model_path="",
            model_type="prosodic",
            weight=0.8,
            enabled=True
        )
    ]
    
    try:
        ensemble = VoiceEmotionEnsemble(test_configs, voting_strategy="weighted_average")
        print(f"✅ Voice ensemble created with {len(ensemble.models)} models")
        print(f"✅ Model types: {[model.config.model_type for model in ensemble.models.values()]}")
        
        # Test prediction with different audio types
        test_audios = [
            ("Happy tone", np.sin(2 * np.pi * 440 * np.linspace(0, 2, int(sample_rate * 2))) * 0.5),
            ("Sad tone", np.sin(2 * np.pi * 220 * np.linspace(0, 2, int(sample_rate * 2))) * 0.3),
            ("Random noise", np.random.randn(int(sample_rate * 2)) * 0.2)
        ]
        
        for i, (description, audio) in enumerate(test_audios, 1):
            print(f"\n   Test {i}: {description}")
            
            try:
                predictions = ensemble.predict(audio.astype(np.float32), sample_rate)
                print(f"   ✅ Predictions: {len(predictions)} emotions")
                
                for pred in predictions[:3]:
                    voting_method = pred.metadata.get('voting_method', 'unknown')
                    agreement = pred.metadata.get('agreement_score', 0.0)
                    audio_duration = pred.metadata.get('audio_duration', 0.0)
                    
                    print(f"      - {pred.label}: {pred.score:.3f} (conf: {pred.confidence:.3f})")
                    print(f"        Method: {voting_method}, Agreement: {agreement:.3f}, Duration: {audio_duration:.1f}s")
            
            except Exception as e:
                print(f"   ❌ Prediction failed: {e}")
    
    except Exception as e:
        print(f"❌ Voice ensemble test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test ensemble performance tracking
    print("\n4. Testing Voice Ensemble Performance...")
    try:
        if 'ensemble' in locals():
            performance = ensemble.get_ensemble_performance()
            print(f"✅ Performance tracking:")
            print(f"   - Predictions made: {performance['prediction_count']}")
            print(f"   - Active models: {performance['model_count']}")
            print(f"   - Loaded models: {performance['loaded_models']}")
            print(f"   - Model weights: {performance['weights']}")
            
            # Test model status
            status = ensemble.get_model_status()
            print(f"\n✅ Model status:")
            for name, model_status in status.items():
                print(f"   - {name}: loaded={model_status['loaded']}, weight={model_status['weight']:.2f}")
                print(f"     Type: {model_status['config']['model_type']}")
    
    except Exception as e:
        print(f"❌ Performance tracking test failed: {e}")
    
    # Test edge cases
    print("\n5. Testing Edge Cases...")
    try:
        if 'ensemble' in locals():
            # Test with empty audio
            empty_result = ensemble.predict(np.array([]), sample_rate)
            print(f"✅ Empty audio handled: {len(empty_result)} predictions")
            
            # Test with very short audio
            short_audio = np.random.randn(int(sample_rate * 0.1)).astype(np.float32)
            short_result = ensemble.predict(short_audio, sample_rate)
            print(f"✅ Short audio handled: {len(short_result)} predictions")
            
            # Test with different sample rates
            high_sr_audio = np.random.randn(int(44100 * 1)).astype(np.float32)
            high_sr_result = ensemble.predict(high_sr_audio, 44100)
            print(f"✅ High sample rate handled: {len(high_sr_result)} predictions")
    
    except Exception as e:
        print(f"❌ Edge case test failed: {e}")
    
    print("\n🎉 Voice emotion ensemble tests completed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()