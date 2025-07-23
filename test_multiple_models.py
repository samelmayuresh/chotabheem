#!/usr/bin/env python3
# Test for multiple specialized emotion models
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils.text_emotion_ensemble import create_default_text_ensemble, ModelConfig, TextEmotionEnsemble
    from utils.specialized_emotion_models import BERTEmotionModel, RoBERTaEmotionModel
    
    print("Testing Multiple Specialized Emotion Models...")
    
    # Test individual specialized models
    print("\n1. Testing BERT Emotion Model...")
    bert_config = ModelConfig(
        name="test_bert",
        model_path="j-hartmann/emotion-english-distilroberta-base",
        model_type="bert",
        weight=1.0,
        enabled=True
    )
    
    bert_model = BERTEmotionModel(bert_config)
    print(f"✅ BERT model created: {bert_model.config.name}")
    
    # Test RoBERTa model
    print("\n2. Testing RoBERTa Emotion Model...")
    roberta_config = ModelConfig(
        name="test_roberta",
        model_path="cardiffnlp/twitter-roberta-base-emotion",
        model_type="roberta",
        weight=1.0,
        enabled=True
    )
    
    roberta_model = RoBERTaEmotionModel(roberta_config)
    print(f"✅ RoBERTa model created: {roberta_model.config.name}")
    
    # Test ensemble with multiple models
    print("\n3. Testing Multi-Model Ensemble...")
    multi_configs = [
        ModelConfig(
            name="bert_primary",
            model_path="j-hartmann/emotion-english-distilroberta-base",
            model_type="bert",
            weight=1.0,
            enabled=True
        ),
        ModelConfig(
            name="context_aware",
            model_path="j-hartmann/emotion-english-distilroberta-base",
            model_type="context_aware",
            weight=0.8,
            enabled=True
        )
    ]
    
    ensemble = TextEmotionEnsemble(multi_configs)
    print(f"✅ Multi-model ensemble created with {len(ensemble.models)} models")
    print(f"✅ Model types: {[model.config.model_type for model in ensemble.models.values()]}")
    
    # Test predictions with different text types
    test_texts = [
        "I am extremely happy and excited about this amazing opportunity!",
        "Work has been really stressful lately and I'm feeling overwhelmed.",
        "The meeting with my boss went well and I feel satisfied with the outcome."
    ]
    
    print("\n4. Testing Predictions...")
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}: '{text[:50]}...'")
        
        try:
            # Test with context
            context = {'primary_context': 'work' if 'work' in text.lower() else 'personal'}
            predictions = ensemble.predict(text, context)
            
            print(f"✅ Predictions: {len(predictions)} emotions detected")
            for pred in predictions[:3]:  # Show top 3
                print(f"   - {pred.label}: {pred.score:.3f} (confidence: {pred.confidence:.3f})")
                print(f"     Model: {pred.model_name}, Type: {pred.metadata.get('model_type', 'unknown')}")
        
        except Exception as e:
            print(f"⚠️  Prediction failed: {e}")
    
    # Test ensemble performance tracking
    print("\n5. Testing Performance Tracking...")
    performance = ensemble.get_ensemble_performance()
    print(f"✅ Ensemble predictions made: {performance['prediction_count']}")
    print(f"✅ Active models: {performance['model_count']}")
    print(f"✅ Model weights: {performance['weights']}")
    
    # Test model status
    status = ensemble.get_model_status()
    print(f"\n✅ Model status retrieved for {len(status)} models")
    for name, model_status in status.items():
        print(f"   - {name}: loaded={model_status['loaded']}, weight={model_status['weight']:.2f}")
    
    print("\n🎉 Multiple model integration tests completed successfully!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()