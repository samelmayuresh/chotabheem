#!/usr/bin/env python3
# Test for specialized voice emotion models
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils.specialized_voice_models import (
        HuBERTEmotionModel, DimensionalEmotionModel, NoiseRobustEmotionModel,
        create_specialized_voice_model
    )
    from utils.voice_emotion_ensemble import VoiceEmotionEnsemble, ModelConfig
    import numpy as np
    
    print("Testing Specialized Voice Emotion Models...")
    
    # Create test audio samples
    sample_rate = 16000
    duration = 2.0
    
    # Different types of test audio
    test_audios = {
        "clean_speech": np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration))) * 0.5,
        "noisy_speech": np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration))) * 0.5 + np.random.randn(int(sample_rate * duration)) * 0.2,
        "low_pitch": np.sin(2 * np.pi * 220 * np.linspace(0, duration, int(sample_rate * duration))) * 0.3,
        "high_energy": np.sin(2 * np.pi * 880 * np.linspace(0, duration, int(sample_rate * duration))) * 0.8
    }
    
    # Test individual specialized models
    print("\n1. Testing Individual Specialized Models...")
    
    # Test Dimensional Model
    print("\n1.1 Testing Dimensional Emotion Model...")
    dimensional_config = ModelConfig(
        name="test_dimensional",
        model_path="",
        model_type="dimensional",
        weight=1.0,
        enabled=True
    )
    
    try:
        dimensional_model = DimensionalEmotionModel(dimensional_config)
        dimensional_model.load_model()
        
        for audio_name, audio in test_audios.items():
            print(f"\n   Testing {audio_name}:")
            predictions = dimensional_model.predict(audio.astype(np.float32), sample_rate)
            
            print(f"   ✅ Predictions: {len(predictions)} emotions")
            for pred in predictions[:3]:
                valence = pred.metadata.get('valence', 0)
                arousal = pred.metadata.get('arousal', 0)
                print(f"      - {pred.label}: {pred.score:.3f} (V:{valence:.2f}, A:{arousal:.2f})")
    
    except Exception as e:
        print(f"❌ Dimensional model test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Noise Robust Model
    print("\n1.2 Testing Noise Robust Emotion Model...")
    noise_robust_config = ModelConfig(
        name="test_noise_robust",
        model_path="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        model_type="noise_robust",
        weight=1.0,
        enabled=True
    )
    
    try:
        noise_model = NoiseRobustEmotionModel(noise_robust_config)
        
        # Test noise assessment
        clean_audio = test_audios["clean_speech"]
        noisy_audio = test_audios["noisy_speech"]
        
        clean_noise_level = noise_model._assess_noise_level(clean_audio)
        noisy_noise_level = noise_model._assess_noise_level(noisy_audio)
        
        print(f"   ✅ Noise level assessment:")
        print(f"      - Clean audio: {clean_noise_level:.3f}")
        print(f"      - Noisy audio: {noisy_noise_level:.3f}")
        
        # Test noise reduction
        enhanced_audio = noise_model._advanced_noise_reduction(noisy_audio, sample_rate, noisy_noise_level)
        print(f"   ✅ Noise reduction applied: {len(enhanced_audio)} samples")
    
    except Exception as e:
        print(f"❌ Noise robust model test failed: {e}")
    
    # Test model factory
    print("\n2. Testing Specialized Model Factory...")
    
    model_types = ["dimensional", "noise_robust", "spectral", "prosodic"]
    
    for model_type in model_types:
        try:
            config = ModelConfig(
                name=f"test_{model_type}",
                model_path="",
                model_type=model_type,
                weight=1.0,
                enabled=True
            )
            
            model = create_specialized_voice_model(config)
            print(f"✅ Created {model_type} model: {model.__class__.__name__}")
        
        except Exception as e:
            print(f"❌ Failed to create {model_type} model: {e}")
    
    # Test enhanced voice ensemble
    print("\n3. Testing Enhanced Voice Ensemble...")
    
    enhanced_configs = [
        ModelConfig(
            name="dimensional_primary",
            model_path="",
            model_type="dimensional",
            weight=1.0,
            enabled=True
        ),
        ModelConfig(
            name="spectral_secondary",
            model_path="",
            model_type="spectral",
            weight=0.8,
            enabled=True
        ),
        ModelConfig(
            name="prosodic_tertiary",
            model_path="",
            model_type="prosodic",
            weight=0.7,
            enabled=True
        )
    ]
    
    try:
        enhanced_ensemble = VoiceEmotionEnsemble(enhanced_configs, voting_strategy="adaptive")
        print(f"✅ Enhanced ensemble created with {len(enhanced_ensemble.models)} models")
        
        # Test with different audio types
        for audio_name, audio in test_audios.items():
            print(f"\n   Testing {audio_name}:")
            
            try:
                predictions = enhanced_ensemble.predict(audio.astype(np.float32), sample_rate)
                print(f"   ✅ Predictions: {len(predictions)} emotions")
                
                for pred in predictions[:3]:
                    voting_method = pred.metadata.get('voting_method', 'unknown')
                    agreement = pred.metadata.get('agreement_score', 0.0)
                    
                    print(f"      - {pred.label}: {pred.score:.3f} (conf: {pred.confidence:.3f})")
                    print(f"        Method: {voting_method}, Agreement: {agreement:.3f}")
                    
                    # Show dimensional info if available
                    if 'valence' in pred.metadata:
                        valence = pred.metadata['valence']
                        arousal = pred.metadata['arousal']
                        print(f"        Dimensional: V={valence:.2f}, A={arousal:.2f}")
            
            except Exception as e:
                print(f"   ❌ Prediction failed: {e}")
    
    except Exception as e:
        print(f"❌ Enhanced ensemble test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test performance comparison
    print("\n4. Testing Performance Comparison...")
    
    try:
        if 'enhanced_ensemble' in locals():
            # Test multiple predictions to see performance
            test_audio = test_audios["clean_speech"]
            
            for i in range(3):
                predictions = enhanced_ensemble.predict(test_audio.astype(np.float32), sample_rate)
                print(f"   Prediction {i+1}: {len(predictions)} emotions, top: {predictions[0].label}")
            
            # Check performance metrics
            performance = enhanced_ensemble.get_ensemble_performance()
            print(f"\n✅ Performance metrics:")
            print(f"   - Total predictions: {performance['prediction_count']}")
            print(f"   - Active models: {performance['model_count']}")
            print(f"   - Confidence calibration: {performance['confidence_calibration']:.3f}")
            
            # Check individual model performance
            for name, perf in performance['individual_performance'].items():
                print(f"   - {name}: {perf['sample_count']} samples, {perf['processing_time']:.3f}s avg")
    
    except Exception as e:
        print(f"❌ Performance comparison failed: {e}")
    
    # Test robustness with edge cases
    print("\n5. Testing Robustness...")
    
    edge_cases = {
        "very_short": np.random.randn(int(sample_rate * 0.1)).astype(np.float32),
        "very_quiet": np.sin(2 * np.pi * 440 * np.linspace(0, 1, sample_rate)) * 0.01,
        "very_loud": np.sin(2 * np.pi * 440 * np.linspace(0, 1, sample_rate)) * 0.99,
        "silence": np.zeros(sample_rate).astype(np.float32)
    }
    
    try:
        if 'enhanced_ensemble' in locals():
            for case_name, audio in edge_cases.items():
                try:
                    predictions = enhanced_ensemble.predict(audio, sample_rate)
                    print(f"   ✅ {case_name}: {len(predictions)} predictions, top: {predictions[0].label}")
                except Exception as e:
                    print(f"   ⚠️  {case_name}: {e}")
    
    except Exception as e:
        print(f"❌ Robustness test failed: {e}")
    
    print("\n🎉 Specialized voice model tests completed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()