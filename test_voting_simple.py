#!/usr/bin/env python3
# Simple test for voting system
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils.ensemble_voting import WeightedAverageVoting, create_voting_system
    from utils.text_emotion_ensemble import EmotionScore
    
    print("Testing Simple Voting System...")
    
    # Test voting system creation
    print("\n1. Testing Voting System Creation...")
    voting_system = create_voting_system("weighted_average")
    print(f"✅ Created voting system: {voting_system.name}")
    
    # Create mock predictions
    print("\n2. Testing Mock Predictions...")
    mock_predictions = {
        "model1": [
            EmotionScore(
                label="joy",
                score=0.8,
                confidence=0.8,
                source="text",
                model_name="model1",
                processing_time=0.1,
                metadata={}
            ),
            EmotionScore(
                label="sadness",
                score=0.2,
                confidence=0.6,
                source="text",
                model_name="model1",
                processing_time=0.1,
                metadata={}
            )
        ],
        "model2": [
            EmotionScore(
                label="joy",
                score=0.7,
                confidence=0.7,
                source="text",
                model_name="model2",
                processing_time=0.1,
                metadata={}
            ),
            EmotionScore(
                label="anger",
                score=0.3,
                confidence=0.5,
                source="text",
                model_name="model2",
                processing_time=0.1,
                metadata={}
            )
        ]
    }
    
    model_weights = {"model1": 1.0, "model2": 0.8}
    
    print("✅ Created mock predictions")
    
    # Test voting
    print("\n3. Testing Voting...")
    try:
        result = voting_system.vote(mock_predictions, model_weights)
        print(f"✅ Voting completed: {len(result.emotion_scores)} emotions")
        print(f"✅ Voting method: {result.voting_method}")
        print(f"✅ Confidence level: {result.confidence_level}")
        print(f"✅ Agreement score: {result.agreement_score:.3f}")
        
        print("\nTop emotions:")
        for i, emotion in enumerate(result.emotion_scores[:3], 1):
            print(f"  {i}. {emotion.label}: {emotion.score:.3f} (conf: {emotion.confidence:.3f})")
    
    except Exception as e:
        print(f"❌ Voting failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 Simple voting test completed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()