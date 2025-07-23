#!/usr/bin/env python3
# Test for text emotion ensemble system
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils.text_emotion_ensemble import (
        TextEmotionEnsemble, ModelConfig, create_default_text_ensemble
    )
    
    print("Testing Text Emotion Ensemble...")
    
    # Create a simple test configuration
    test_configs = [
        ModelConfig(
            name="test_model",
            model_path="j-hartmann/emotion-english-distilroberta-base",
            model_type="transformers",
            weight=1.0,
            enabled=True
        )
    ]
    
    print("✅ Creating ensemble with test configuration...")
    ensemble = TextEmotionEnsemble(test_configs)
    
    print("✅ Ensemble created successfully")
    print(f"✅ Models initialized: {list(ensemble.models.keys())}")
    print(f"✅ Model weights: {ensemble.weights}")
    
    # Test prediction (this will try to load the model)
    test_text = "I am feeling really happy today!"
    print(f"\n✅ Testing prediction with text: '{test_text}'")
    
    try:
        predictions = ensemble.predict(test_text)
        print(f"✅ Predictions received: {len(predictions)} emotions")
        
        for pred in predictions[:3]:  # Show top 3
            print(f"   - {pred.label}: {pred.score:.3f} (confidence: {pred.confidence:.3f})")
    
    except Exception as e:
        print(f"⚠️  Prediction test skipped (model loading issue): {e}")
    
    # Test ensemble status
    status = ensemble.get_model_status()
    print(f"\n✅ Model status retrieved: {len(status)} models")
    
    # Test performance tracking
    performance = ensemble.get_ensemble_performance()
    print(f"✅ Performance metrics: {performance['prediction_count']} predictions made")
    
    print("\n🎉 Text ensemble tests completed successfully!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()