#!/usr/bin/env python3
# Test for multimodal fusion system
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils.multimodal_fusion import (
        MultimodalFusionEngine, WeightedAverageFusion, AttentionBasedFusion,
        ConflictResolver, ModalityInput, FusionResult
    )
    from utils.text_emotion_ensemble import EmotionScore
    import numpy as np
    
    print("Testing Multimodal Fusion System...")
    
    # Create mock emotion scores for testing
    def create_mock_text_emotions():
        return [
            EmotionScore(
                label="joy",
                score=0.8,
                confidence=0.9,
                source="text",
                model_name="text_ensemble",
                processing_time=0.1,
                metadata={"original_length": 100, "model_type": "bert"}
            ),
            EmotionScore(
                label="surprise",
                score=0.2,
                confidence=0.6,
                source="text",
                model_name="text_ensemble",
                processing_time=0.1,
                metadata={"original_length": 100, "model_type": "bert"}
            )
        ]
    
    def create_mock_voice_emotions():
        return [
            EmotionScore(
                label="joy",
                score=0.6,
                confidence=0.7,
                source="voice",
                model_name="voice_ensemble",
                processing_time=0.2,
                metadata={"audio_duration": 3.0, "model_type": "wav2vec2"}
            ),
            EmotionScore(
                label="neutral",
                score=0.4,
                confidence=0.5,
                source="voice",
                model_name="voice_ensemble",
                processing_time=0.2,
                metadata={"audio_duration": 3.0, "model_type": "wav2vec2"}
            )
        ]
    
    def create_conflicting_voice_emotions():
        return [
            EmotionScore(
                label="anger",
                score=0.9,
                confidence=0.8,
                source="voice",
                model_name="voice_ensemble",
                processing_time=0.2,
                metadata={"audio_duration": 2.5, "model_type": "prosodic"}
            ),
            EmotionScore(
                label="joy",
                score=0.1,
                confidence=0.3,
                source="voice",
                model_name="voice_ensemble",
                processing_time=0.2,
                metadata={"audio_duration": 2.5, "model_type": "prosodic"}
            )
        ]
    
    # Test individual fusion strategies
    print("\n1. Testing Individual Fusion Strategies...")
    
    # Test Weighted Average Fusion
    print("\n1.1 Testing Weighted Average Fusion...")
    try:
        weighted_fusion = WeightedAverageFusion()
        
        # Create modality inputs
        text_input = ModalityInput(
            modality='text',
            emotion_scores=create_mock_text_emotions(),
            confidence_level='high',
            quality_score=0.8,
            processing_time=0.1,
            metadata={'emotion_count': 2}
        )
        
        voice_input = ModalityInput(
            modality='voice',
            emotion_scores=create_mock_voice_emotions(),
            confidence_level='medium',
            quality_score=0.7,
            processing_time=0.2,
            metadata={'emotion_count': 2}
        )
        
        fusion_result = weighted_fusion.fuse([text_input, voice_input])
        
        print(f"✅ Weighted fusion completed")
        print(f"   - Method: {fusion_result.fusion_method}")
        print(f"   - Confidence: {fusion_result.confidence_level}")
        print(f"   - Emotions: {len(fusion_result.emotion_scores)}")
        print(f"   - Modality contributions: {fusion_result.modality_contributions}")
        
        for i, emotion in enumerate(fusion_result.emotion_scores[:3], 1):
            print(f"   {i}. {emotion.label}: {emotion.score:.3f} (conf: {emotion.confidence:.3f})")
    
    except Exception as e:
        print(f"❌ Weighted fusion test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Attention-Based Fusion
    print("\n1.2 Testing Attention-Based Fusion...")
    try:
        attention_fusion = AttentionBasedFusion()
        
        fusion_result = attention_fusion.fuse([text_input, voice_input])
        
        print(f"✅ Attention fusion completed")
        print(f"   - Method: {fusion_result.fusion_method}")
        print(f"   - Confidence: {fusion_result.confidence_level}")
        print(f"   - Attention weights: {fusion_result.modality_contributions}")
        print(f"   - Uncertainty metrics: {list(fusion_result.uncertainty_metrics.keys())}")
        
        for i, emotion in enumerate(fusion_result.emotion_scores[:3], 1):
            attention_weight = emotion.metadata.get('attention_weights', {})
            print(f"   {i}. {emotion.label}: {emotion.score:.3f} (attention: {attention_weight})")
    
    except Exception as e:
        print(f"❌ Attention fusion test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Conflict Resolution
    print("\n2. Testing Conflict Resolution...")
    try:
        conflict_resolver = ConflictResolver()
        
        # Create conflicting inputs
        text_input_conflict = ModalityInput(
            modality='text',
            emotion_scores=create_mock_text_emotions(),  # Joy dominant
            confidence_level='high',
            quality_score=0.9,
            processing_time=0.1,
            metadata={'emotion_count': 2}
        )
        
        voice_input_conflict = ModalityInput(
            modality='voice',
            emotion_scores=create_conflicting_voice_emotions(),  # Anger dominant
            confidence_level='medium',
            quality_score=0.6,
            processing_time=0.2,
            metadata={'emotion_count': 2}
        )
        
        # Test weighted fusion with conflicts
        fusion_result = weighted_fusion.fuse([text_input_conflict, voice_input_conflict])
        
        print(f"✅ Conflict detection completed")
        print(f"   - Conflicts detected: {len(fusion_result.conflict_resolution)}")
        print(f"   - Conflict emotions: {list(fusion_result.conflict_resolution.keys())}")
        
        for emotion, conflict_info in fusion_result.conflict_resolution.items():
            print(f"   - {emotion}: conflict level {conflict_info.get('conflict_level', 0):.3f}")
    
    except Exception as e:
        print(f"❌ Conflict resolution test failed: {e}")
    
    # Test Multimodal Fusion Engine
    print("\n3. Testing Multimodal Fusion Engine...")
    try:
        fusion_engine = MultimodalFusionEngine({
            'default_strategy': 'weighted_average',
            'conflict_resolution_strategy': 'confidence_based'
        })
        
        # Test with agreeing modalities
        print("\n3.1 Testing with agreeing modalities...")
        text_emotions = create_mock_text_emotions()
        voice_emotions = create_mock_voice_emotions()
        
        result = fusion_engine.fuse_predictions(text_emotions, voice_emotions)
        
        print(f"✅ Fusion engine test completed")
        print(f"   - Method: {result.fusion_method}")
        print(f"   - Confidence: {result.confidence_level}")
        print(f"   - Top emotion: {result.emotion_scores[0].label}")
        print(f"   - Modality contributions: {result.modality_contributions}")
        
        # Test with conflicting modalities
        print("\n3.2 Testing with conflicting modalities...")
        conflicting_voice = create_conflicting_voice_emotions()
        
        conflict_result = fusion_engine.fuse_predictions(text_emotions, conflicting_voice)
        
        print(f"✅ Conflict fusion completed")
        print(f"   - Conflicts: {len(conflict_result.conflict_resolution)}")
        print(f"   - Top emotion: {conflict_result.emotion_scores[0].label}")
        print(f"   - Confidence: {conflict_result.confidence_level}")
        
        # Test single modality
        print("\n3.3 Testing with single modality...")
        single_result = fusion_engine.fuse_predictions(text_emotions, [])
        
        print(f"✅ Single modality fusion completed")
        print(f"   - Method: {single_result.fusion_method}")
        print(f"   - Top emotion: {single_result.emotion_scores[0].label}")
    
    except Exception as e:
        print(f"❌ Fusion engine test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Context-Aware Fusion
    print("\n4. Testing Context-Aware Fusion...")
    try:
        # Test with different contexts
        contexts = [
            {'text_length': 200, 'has_complex_language': True},
            {'audio_quality': 0.9, 'has_prosodic_cues': True},
            {'uncertainty_tolerance': 'high'}
        ]
        
        for i, context in enumerate(contexts, 1):
            print(f"\n4.{i} Testing context: {context}")
            
            result = fusion_engine.fuse_predictions(
                create_mock_text_emotions(), 
                create_mock_voice_emotions(),
                context=context
            )
            
            print(f"   ✅ Context fusion completed")
            print(f"   - Strategy selected: {result.fusion_method}")
            print(f"   - Modality weights: {result.modality_contributions}")
            print(f"   - Top emotion: {result.emotion_scores[0].label}")
    
    except Exception as e:
        print(f"❌ Context-aware fusion test failed: {e}")
    
    # Test Fusion Statistics
    print("\n5. Testing Fusion Statistics...")
    try:
        # Perform multiple fusions to build history
        for i in range(10):
            if i % 2 == 0:
                fusion_engine.fuse_predictions(
                    create_mock_text_emotions(), 
                    create_mock_voice_emotions()
                )
            else:
                fusion_engine.fuse_predictions(
                    create_mock_text_emotions(), 
                    create_conflicting_voice_emotions()
                )
        
        stats = fusion_engine.get_fusion_statistics()
        
        print(f"✅ Fusion statistics generated")
        print(f"   - Total fusions: {stats['total_fusions']}")
        print(f"   - Recent fusions: {stats['recent_fusions']}")
        print(f"   - Fusion methods: {stats['fusion_methods']}")
        print(f"   - Confidence distribution: {stats['confidence_distribution']}")
        print(f"   - Average conflicts: {stats['average_conflicts']:.2f}")
        print(f"   - Modality usage: {stats['modality_usage']}")
    
    except Exception as e:
        print(f"❌ Fusion statistics test failed: {e}")
    
    # Test Edge Cases
    print("\n6. Testing Edge Cases...")
    try:
        # Test with empty inputs
        empty_result = fusion_engine.fuse_predictions([], [])
        print(f"✅ Empty input handled: {empty_result.emotion_scores[0].label}")
        
        # Test with very low confidence
        low_conf_text = [
            EmotionScore("neutral", 0.1, 0.1, "text", "test", 0.1, {})
        ]
        low_conf_voice = [
            EmotionScore("neutral", 0.1, 0.1, "voice", "test", 0.1, {})
        ]
        
        low_conf_result = fusion_engine.fuse_predictions(low_conf_text, low_conf_voice)
        print(f"✅ Low confidence handled: {low_conf_result.confidence_level}")
        
        # Test with many emotions
        many_emotions_text = [
            EmotionScore(f"emotion_{i}", 0.1, 0.5, "text", "test", 0.1, {})
            for i in range(10)
        ]
        
        many_result = fusion_engine.fuse_predictions(many_emotions_text, [])
        print(f"✅ Many emotions handled: {len(many_result.emotion_scores)} emotions")
    
    except Exception as e:
        print(f"❌ Edge cases test failed: {e}")
    
    print("\n🎉 Multimodal fusion tests completed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()