# utils/multimodal_fusion.py - Multimodal emotion fusion system
import numpy as np
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
from datetime import datetime
import json
from utils.text_emotion_ensemble import EmotionScore
from utils.ensemble_voting import VotingResult
import scipy.stats as stats

@dataclass
class FusionResult:
    """Result of multimodal fusion with detailed metadata"""
    emotion_scores: List[EmotionScore]
    fusion_method: str
    confidence_level: str
    modality_contributions: Dict[str, float]
    conflict_resolution: Dict[str, Any]
    uncertainty_metrics: Dict[str, float]
    processing_metadata: Dict[str, Any]

@dataclass
class ModalityInput:
    """Input from a single modality"""
    modality: str  # 'text', 'voice'
    emotion_scores: List[EmotionScore]
    confidence_level: str
    quality_score: float
    processing_time: float
    metadata: Dict[str, Any]

class BaseFusionStrategy(ABC):
    """Abstract base class for fusion strategies"""
    
    def __init__(self, name: str):
        self.name = name
        self.fusion_history: List[Dict] = []
        
    @abstractmethod
    def fuse(self, modality_inputs: List[ModalityInput], 
             context: Dict = None) -> FusionResult:
        """Fuse predictions from multiple modalities"""
        pass
    
    def update_history(self, fusion_result: FusionResult):
        """Update fusion history for learning"""
        self.fusion_history.append({
            'timestamp': datetime.now(),
            'method': fusion_result.fusion_method,
            'confidence': fusion_result.confidence_level,
            'modalities': len([m for m in fusion_result.modality_contributions.keys()]),
            'top_emotion': fusion_result.emotion_scores[0].label if fusion_result.emotion_scores else 'none'
        })

class WeightedAverageFusion(BaseFusionStrategy):
    """Weighted average fusion with quality-based weighting"""
    
    def __init__(self):
        super().__init__("weighted_average")
        self.modality_weights = {
            'text': 0.6,
            'voice': 0.4
        }
        self.quality_weight_factor = 0.3
        
    def fuse(self, modality_inputs: List[ModalityInput], 
             context: Dict = None) -> FusionResult:
        """Fuse using weighted average with quality adjustment"""
        
        if not modality_inputs:
            return self._create_empty_result()
        
        # Calculate dynamic weights based on quality and confidence
        dynamic_weights = self._calculate_dynamic_weights(modality_inputs)
        
        # Aggregate emotions across modalities
        emotion_aggregates = {}
        modality_contributions = {}
        
        for modality_input in modality_inputs:
            weight = dynamic_weights.get(modality_input.modality, 0.0)
            modality_contributions[modality_input.modality] = weight
            
            for emotion_score in modality_input.emotion_scores:
                if emotion_score.label not in emotion_aggregates:
                    emotion_aggregates[emotion_score.label] = {
                        'scores': [],
                        'weights': [],
                        'confidences': [],
                        'sources': []
                    }
                
                emotion_aggregates[emotion_score.label]['scores'].append(emotion_score.score)
                emotion_aggregates[emotion_score.label]['weights'].append(weight)
                emotion_aggregates[emotion_score.label]['confidences'].append(emotion_score.confidence)
                emotion_aggregates[emotion_score.label]['sources'].append(modality_input.modality)
        
        # Calculate fused emotion scores
        fused_emotions = []
        conflict_info = {}
        
        for emotion, data in emotion_aggregates.items():
            scores = np.array(data['scores'])
            weights = np.array(data['weights'])
            confidences = np.array(data['confidences'])
            
            # Weighted average score
            if np.sum(weights) > 0:
                fused_score = np.average(scores, weights=weights)
                fused_confidence = np.average(confidences, weights=weights)
            else:
                fused_score = np.mean(scores)
                fused_confidence = np.mean(confidences)
            
            # Detect and measure conflicts
            if len(scores) > 1:
                score_variance = np.var(scores)
                conflict_level = min(score_variance * 4, 1.0)  # Normalize to [0,1]
                conflict_info[emotion] = {
                    'variance': score_variance,
                    'conflict_level': conflict_level,
                    'sources': data['sources']
                }
            
            # Agreement-based confidence adjustment
            agreement_factor = self._calculate_agreement_factor(scores, weights)
            adjusted_confidence = fused_confidence * agreement_factor
            
            fused_emotions.append(EmotionScore(
                label=emotion,
                score=fused_score,
                confidence=adjusted_confidence,
                source='multimodal',
                model_name='fusion_weighted_average',
                processing_time=sum(mi.processing_time for mi in modality_inputs),
                metadata={
                    'fusion_method': 'weighted_average',
                    'contributing_modalities': data['sources'],
                    'agreement_factor': agreement_factor,
                    'conflict_level': conflict_info.get(emotion, {}).get('conflict_level', 0.0),
                    'weight_distribution': dict(zip(data['sources'], weights))
                }
            ))
        
        # Sort by fused score
        fused_emotions.sort(key=lambda x: x.score, reverse=True)
        
        # Calculate overall metrics
        confidence_level = self._determine_confidence_level(fused_emotions, conflict_info)
        uncertainty_metrics = self._calculate_uncertainty_metrics(emotion_aggregates, conflict_info)
        
        result = FusionResult(
            emotion_scores=fused_emotions,
            fusion_method="weighted_average",
            confidence_level=confidence_level,
            modality_contributions=modality_contributions,
            conflict_resolution=conflict_info,
            uncertainty_metrics=uncertainty_metrics,
            processing_metadata={
                'modality_count': len(modality_inputs),
                'total_emotions': len(emotion_aggregates),
                'context': context or {}
            }
        )
        
        self.update_history(result)
        return result
    
    def _calculate_dynamic_weights(self, modality_inputs: List[ModalityInput]) -> Dict[str, float]:
        """Calculate dynamic weights based on quality and confidence"""
        weights = {}
        
        for modality_input in modality_inputs:
            base_weight = self.modality_weights.get(modality_input.modality, 0.5)
            
            # Adjust based on quality
            quality_adjustment = (modality_input.quality_score - 0.5) * self.quality_weight_factor
            
            # Adjust based on confidence level
            confidence_mapping = {'high': 0.2, 'medium': 0.0, 'low': -0.2}
            confidence_adjustment = confidence_mapping.get(modality_input.confidence_level, 0.0)
            
            # Calculate final weight
            final_weight = base_weight + quality_adjustment + confidence_adjustment
            weights[modality_input.modality] = max(0.1, min(1.0, final_weight))
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def _calculate_agreement_factor(self, scores: np.ndarray, weights: np.ndarray) -> float:
        """Calculate agreement factor based on score consistency"""
        if len(scores) <= 1:
            return 1.0
        
        # Weighted variance
        weighted_mean = np.average(scores, weights=weights)
        weighted_variance = np.average((scores - weighted_mean) ** 2, weights=weights)
        
        # Convert to agreement factor (lower variance = higher agreement)
        agreement_factor = max(0.3, 1.0 - weighted_variance * 2)
        return agreement_factor
    
    def _determine_confidence_level(self, emotions: List[EmotionScore], 
                                  conflict_info: Dict) -> str:
        """Determine overall confidence level"""
        if not emotions:
            return "low"
        
        # Average confidence of top emotions
        top_emotions = emotions[:3]
        avg_confidence = np.mean([e.confidence for e in top_emotions])
        
        # Average conflict level
        avg_conflict = np.mean([info.get('conflict_level', 0) for info in conflict_info.values()])
        
        # Combined assessment
        if avg_confidence >= 0.8 and avg_conflict <= 0.3:
            return "high"
        elif avg_confidence >= 0.6 and avg_conflict <= 0.5:
            return "medium"
        else:
            return "low"
    
    def _calculate_uncertainty_metrics(self, emotion_aggregates: Dict, 
                                     conflict_info: Dict) -> Dict[str, float]:
        """Calculate comprehensive uncertainty metrics"""
        if not emotion_aggregates:
            return {}
        
        all_scores = []
        all_conflicts = []
        
        for emotion, data in emotion_aggregates.items():
            all_scores.extend(data['scores'])
            if emotion in conflict_info:
                all_conflicts.append(conflict_info[emotion]['conflict_level'])
        
        return {
            'overall_variance': np.var(all_scores) if all_scores else 0.0,
            'average_conflict': np.mean(all_conflicts) if all_conflicts else 0.0,
            'prediction_entropy': self._calculate_entropy(all_scores) if all_scores else 0.0,
            'modality_agreement': 1.0 - np.mean(all_conflicts) if all_conflicts else 1.0
        }
    
    def _calculate_entropy(self, scores: List[float]) -> float:
        """Calculate entropy of prediction scores"""
        if not scores:
            return 0.0
        
        # Normalize to probabilities
        scores = np.array(scores)
        scores = scores / np.sum(scores) if np.sum(scores) > 0 else scores
        
        # Calculate entropy
        entropy = -np.sum(scores * np.log(scores + 1e-10))
        return entropy
    
    def _create_empty_result(self) -> FusionResult:
        """Create empty result for edge cases"""
        return FusionResult(
            emotion_scores=[EmotionScore(
                label="neutral",
                score=1.0,
                confidence=0.5,
                source="multimodal",
                model_name="fusion_weighted_average",
                processing_time=0.0,
                metadata={"reason": "no_modality_inputs"}
            )],
            fusion_method="weighted_average",
            confidence_level="low",
            modality_contributions={},
            conflict_resolution={},
            uncertainty_metrics={},
            processing_metadata={"reason": "no_modality_inputs"}
        )

class AttentionBasedFusion(BaseFusionStrategy):
    """Attention-based fusion that learns to focus on reliable modalities"""
    
    def __init__(self):
        super().__init__("attention_based")
        self.attention_weights = {}
        self.learning_rate = 0.1
        
    def fuse(self, modality_inputs: List[ModalityInput], 
             context: Dict = None) -> FusionResult:
        """Fuse using attention mechanism"""
        
        if not modality_inputs:
            return self._create_empty_result()
        
        # Calculate attention weights
        attention_weights = self._calculate_attention_weights(modality_inputs, context)
        
        # Apply attention to modality inputs
        attended_inputs = self._apply_attention(modality_inputs, attention_weights)
        
        # Aggregate with attention weighting
        fused_emotions = self._aggregate_with_attention(attended_inputs, attention_weights)
        
        # Calculate conflict resolution
        conflict_info = self._analyze_conflicts(modality_inputs, attention_weights)
        
        # Determine confidence and uncertainty
        confidence_level = self._determine_attention_confidence(fused_emotions, attention_weights)
        uncertainty_metrics = self._calculate_attention_uncertainty(modality_inputs, attention_weights)
        
        result = FusionResult(
            emotion_scores=fused_emotions,
            fusion_method="attention_based",
            confidence_level=confidence_level,
            modality_contributions=attention_weights,
            conflict_resolution=conflict_info,
            uncertainty_metrics=uncertainty_metrics,
            processing_metadata={
                'attention_weights': attention_weights,
                'modality_count': len(modality_inputs),
                'context': context or {}
            }
        )
        
        self.update_history(result)
        return result
    
    def _calculate_attention_weights(self, modality_inputs: List[ModalityInput], 
                                   context: Dict = None) -> Dict[str, float]:
        """Calculate attention weights for each modality"""
        weights = {}
        
        for modality_input in modality_inputs:
            # Base attention score
            attention_score = 0.5
            
            # Quality-based attention
            attention_score += (modality_input.quality_score - 0.5) * 0.3
            
            # Confidence-based attention
            confidence_mapping = {'high': 0.3, 'medium': 0.0, 'low': -0.3}
            attention_score += confidence_mapping.get(modality_input.confidence_level, 0.0)
            
            # Context-based attention
            if context:
                context_boost = self._get_context_attention_boost(modality_input.modality, context)
                attention_score += context_boost
            
            # Historical performance (if available)
            if modality_input.modality in self.attention_weights:
                historical_weight = self.attention_weights[modality_input.modality]
                attention_score = (attention_score + historical_weight) / 2
            
            weights[modality_input.modality] = max(0.1, min(1.0, attention_score))
        
        # Apply softmax normalization
        weights = self._softmax_normalize(weights)
        
        # Update historical weights
        for modality, weight in weights.items():
            if modality in self.attention_weights:
                self.attention_weights[modality] = (
                    (1 - self.learning_rate) * self.attention_weights[modality] +
                    self.learning_rate * weight
                )
            else:
                self.attention_weights[modality] = weight
        
        return weights
    
    def _get_context_attention_boost(self, modality: str, context: Dict) -> float:
        """Get attention boost based on context"""
        boost = 0.0
        
        # Text modality gets boost for text-heavy contexts
        if modality == 'text':
            if context.get('text_length', 0) > 50:
                boost += 0.1
            if context.get('has_complex_language', False):
                boost += 0.1
        
        # Voice modality gets boost for audio-rich contexts
        elif modality == 'voice':
            if context.get('audio_quality', 0.5) > 0.7:
                boost += 0.1
            if context.get('has_prosodic_cues', False):
                boost += 0.1
        
        return boost
    
    def _softmax_normalize(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply softmax normalization to weights"""
        values = np.array(list(weights.values()))
        softmax_values = np.exp(values) / np.sum(np.exp(values))
        
        return {k: float(v) for k, v in zip(weights.keys(), softmax_values)}
    
    def _apply_attention(self, modality_inputs: List[ModalityInput], 
                        attention_weights: Dict[str, float]) -> List[ModalityInput]:
        """Apply attention weights to modality inputs"""
        attended_inputs = []
        
        for modality_input in modality_inputs:
            attention_weight = attention_weights.get(modality_input.modality, 0.5)
            
            # Scale emotion scores by attention weight
            attended_scores = []
            for emotion_score in modality_input.emotion_scores:
                attended_score = EmotionScore(
                    label=emotion_score.label,
                    score=emotion_score.score * attention_weight,
                    confidence=emotion_score.confidence * attention_weight,
                    source=emotion_score.source,
                    model_name=emotion_score.model_name,
                    processing_time=emotion_score.processing_time,
                    metadata={
                        **emotion_score.metadata,
                        'attention_weight': attention_weight,
                        'original_score': emotion_score.score,
                        'original_confidence': emotion_score.confidence
                    }
                )
                attended_scores.append(attended_score)
            
            attended_input = ModalityInput(
                modality=modality_input.modality,
                emotion_scores=attended_scores,
                confidence_level=modality_input.confidence_level,
                quality_score=modality_input.quality_score * attention_weight,
                processing_time=modality_input.processing_time,
                metadata={
                    **modality_input.metadata,
                    'attention_weight': attention_weight
                }
            )
            attended_inputs.append(attended_input)
        
        return attended_inputs
    
    def _aggregate_with_attention(self, attended_inputs: List[ModalityInput], 
                                 attention_weights: Dict[str, float]) -> List[EmotionScore]:
        """Aggregate emotion scores with attention weighting"""
        emotion_aggregates = {}
        
        for modality_input in attended_inputs:
            for emotion_score in modality_input.emotion_scores:
                if emotion_score.label not in emotion_aggregates:
                    emotion_aggregates[emotion_score.label] = {
                        'scores': [],
                        'confidences': [],
                        'attention_weights': [],
                        'sources': []
                    }
                
                emotion_aggregates[emotion_score.label]['scores'].append(emotion_score.score)
                emotion_aggregates[emotion_score.label]['confidences'].append(emotion_score.confidence)
                emotion_aggregates[emotion_score.label]['attention_weights'].append(
                    attention_weights.get(modality_input.modality, 0.5)
                )
                emotion_aggregates[emotion_score.label]['sources'].append(modality_input.modality)
        
        # Calculate final scores
        fused_emotions = []
        for emotion, data in emotion_aggregates.items():
            # Attention-weighted average
            scores = np.array(data['scores'])
            weights = np.array(data['attention_weights'])
            confidences = np.array(data['confidences'])
            
            final_score = np.sum(scores)  # Scores already weighted by attention
            final_confidence = np.average(confidences, weights=weights)
            
            fused_emotions.append(EmotionScore(
                label=emotion,
                score=final_score,
                confidence=final_confidence,
                source='multimodal',
                model_name='fusion_attention',
                processing_time=sum(mi.processing_time for mi in attended_inputs),
                metadata={
                    'fusion_method': 'attention_based',
                    'contributing_modalities': data['sources'],
                    'attention_weights': dict(zip(data['sources'], weights)),
                    'aggregation_method': 'attention_weighted_sum'
                }
            ))
        
        fused_emotions.sort(key=lambda x: x.score, reverse=True)
        return fused_emotions
    
    def _analyze_conflicts(self, modality_inputs: List[ModalityInput], 
                          attention_weights: Dict[str, float]) -> Dict[str, Any]:
        """Analyze conflicts between modalities"""
        conflicts = {}
        
        # Find emotions present in multiple modalities
        emotion_by_modality = {}
        for modality_input in modality_inputs:
            emotion_by_modality[modality_input.modality] = {
                e.label: e.score for e in modality_input.emotion_scores
            }
        
        # Check for conflicts
        all_emotions = set()
        for emotions in emotion_by_modality.values():
            all_emotions.update(emotions.keys())
        
        for emotion in all_emotions:
            modality_scores = []
            modality_names = []
            
            for modality, emotions in emotion_by_modality.items():
                if emotion in emotions:
                    modality_scores.append(emotions[emotion])
                    modality_names.append(modality)
            
            if len(modality_scores) > 1:
                score_variance = np.var(modality_scores)
                if score_variance > 0.1:  # Significant conflict
                    conflicts[emotion] = {
                        'conflict_level': min(score_variance * 4, 1.0),
                        'conflicting_modalities': modality_names,
                        'scores': dict(zip(modality_names, modality_scores)),
                        'resolution_method': 'attention_weighting',
                        'dominant_modality': max(modality_names, 
                                               key=lambda m: attention_weights.get(m, 0))
                    }
        
        return conflicts
    
    def _determine_attention_confidence(self, emotions: List[EmotionScore], 
                                      attention_weights: Dict[str, float]) -> str:
        """Determine confidence level for attention-based fusion"""
        if not emotions:
            return "low"
        
        # Average confidence of top emotions
        top_emotions = emotions[:3]
        avg_confidence = np.mean([e.confidence for e in top_emotions])
        
        # Attention distribution (more focused = higher confidence)
        attention_values = list(attention_weights.values())
        attention_entropy = stats.entropy(attention_values)
        max_entropy = np.log(len(attention_values))
        attention_focus = 1.0 - (attention_entropy / max_entropy if max_entropy > 0 else 0)
        
        # Combined confidence
        combined_confidence = (avg_confidence + attention_focus) / 2
        
        if combined_confidence >= 0.8:
            return "high"
        elif combined_confidence >= 0.6:
            return "medium"
        else:
            return "low"
    
    def _calculate_attention_uncertainty(self, modality_inputs: List[ModalityInput], 
                                       attention_weights: Dict[str, float]) -> Dict[str, float]:
        """Calculate uncertainty metrics for attention-based fusion"""
        attention_values = list(attention_weights.values())
        
        return {
            'attention_entropy': stats.entropy(attention_values),
            'attention_variance': np.var(attention_values),
            'dominant_modality_weight': max(attention_values),
            'modality_balance': 1.0 - abs(max(attention_values) - min(attention_values))
        }
    
    def _create_empty_result(self) -> FusionResult:
        """Create empty result for edge cases"""
        return FusionResult(
            emotion_scores=[EmotionScore(
                label="neutral",
                score=1.0,
                confidence=0.5,
                source="multimodal",
                model_name="fusion_attention",
                processing_time=0.0,
                metadata={"reason": "no_modality_inputs"}
            )],
            fusion_method="attention_based",
            confidence_level="low",
            modality_contributions={},
            conflict_resolution={},
            uncertainty_metrics={},
            processing_metadata={"reason": "no_modality_inputs"}
        )

class ConflictResolver:
    """Specialized conflict resolution for multimodal fusion"""
    
    def __init__(self):
        self.resolution_strategies = {
            'confidence_based': self._resolve_by_confidence,
            'quality_based': self._resolve_by_quality,
            'context_based': self._resolve_by_context,
            'ensemble_voting': self._resolve_by_voting
        }
    
    def resolve_conflicts(self, modality_inputs: List[ModalityInput], 
                         conflicts: Dict[str, Any],
                         strategy: str = 'confidence_based') -> Dict[str, Any]:
        """Resolve conflicts between modalities"""
        
        if not conflicts:
            return {}
        
        resolution_method = self.resolution_strategies.get(
            strategy, self._resolve_by_confidence
        )
        
        resolved_conflicts = {}
        for emotion, conflict_info in conflicts.items():
            resolution = resolution_method(emotion, conflict_info, modality_inputs)
            resolved_conflicts[emotion] = {
                **conflict_info,
                'resolution': resolution,
                'resolution_strategy': strategy
            }
        
        return resolved_conflicts
    
    def _resolve_by_confidence(self, emotion: str, conflict_info: Dict, 
                              modality_inputs: List[ModalityInput]) -> Dict[str, Any]:
        """Resolve conflict by choosing highest confidence prediction"""
        best_modality = None
        best_confidence = 0.0
        
        for modality_input in modality_inputs:
            for emotion_score in modality_input.emotion_scores:
                if emotion_score.label == emotion and emotion_score.confidence > best_confidence:
                    best_confidence = emotion_score.confidence
                    best_modality = modality_input.modality
        
        return {
            'chosen_modality': best_modality,
            'confidence': best_confidence,
            'reason': 'highest_confidence'
        }
    
    def _resolve_by_quality(self, emotion: str, conflict_info: Dict, 
                           modality_inputs: List[ModalityInput]) -> Dict[str, Any]:
        """Resolve conflict by choosing highest quality modality"""
        best_modality = None
        best_quality = 0.0
        
        for modality_input in modality_inputs:
            if modality_input.quality_score > best_quality:
                # Check if this modality has the conflicting emotion
                has_emotion = any(e.label == emotion for e in modality_input.emotion_scores)
                if has_emotion:
                    best_quality = modality_input.quality_score
                    best_modality = modality_input.modality
        
        return {
            'chosen_modality': best_modality,
            'quality_score': best_quality,
            'reason': 'highest_quality'
        }
    
    def _resolve_by_context(self, emotion: str, conflict_info: Dict, 
                           modality_inputs: List[ModalityInput]) -> Dict[str, Any]:
        """Resolve conflict based on contextual appropriateness"""
        # Simple context-based resolution
        # In practice, this would use more sophisticated context analysis
        
        context_preferences = {
            'joy': 'text',      # Text often better for positive emotions
            'sadness': 'voice', # Voice often better for negative emotions
            'anger': 'voice',   # Voice captures anger well
            'fear': 'voice',    # Voice captures fear well
            'surprise': 'text', # Text context helps with surprise
            'neutral': 'text'   # Text baseline for neutral
        }
        
        preferred_modality = context_preferences.get(emotion, 'text')
        
        return {
            'chosen_modality': preferred_modality,
            'reason': 'contextual_preference',
            'context_rule': f"{emotion}_prefers_{preferred_modality}"
        }
    
    def _resolve_by_voting(self, emotion: str, conflict_info: Dict, 
                          modality_inputs: List[ModalityInput]) -> Dict[str, Any]:
        """Resolve conflict using ensemble voting principles"""
        # Collect all predictions for this emotion
        predictions = []
        
        for modality_input in modality_inputs:
            for emotion_score in modality_input.emotion_scores:
                if emotion_score.label == emotion:
                    predictions.append({
                        'modality': modality_input.modality,
                        'score': emotion_score.score,
                        'confidence': emotion_score.confidence,
                        'quality': modality_input.quality_score
                    })
        
        if not predictions:
            return {'chosen_modality': None, 'reason': 'no_predictions'}
        
        # Weighted voting
        total_weight = 0.0
        weighted_sum = 0.0
        
        for pred in predictions:
            weight = pred['confidence'] * pred['quality']
            weighted_sum += pred['score'] * weight
            total_weight += weight
        
        final_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # Choose modality with score closest to ensemble result
        best_modality = min(predictions, 
                           key=lambda p: abs(p['score'] - final_score))['modality']
        
        return {
            'chosen_modality': best_modality,
            'ensemble_score': final_score,
            'reason': 'ensemble_voting'
        }

class MultimodalFusionEngine:
    """Main engine for multimodal emotion fusion"""
    
    def __init__(self, fusion_config: Dict = None):
        self.config = fusion_config or {}
        self.fusion_strategies = {
            'weighted_average': WeightedAverageFusion(),
            'attention_based': AttentionBasedFusion()
        }
        self.conflict_resolver = ConflictResolver()
        self.fusion_history: List[Dict] = []
        
    def fuse_predictions(self, text_emotions: List[EmotionScore], 
                        voice_emotions: List[EmotionScore],
                        context: Dict = None) -> FusionResult:
        """Main fusion method combining text and voice predictions"""
        
        # Prepare modality inputs
        modality_inputs = []
        
        if text_emotions:
            text_input = ModalityInput(
                modality='text',
                emotion_scores=text_emotions,
                confidence_level=self._assess_confidence_level(text_emotions),
                quality_score=self._assess_quality_score(text_emotions, 'text'),
                processing_time=sum(e.processing_time for e in text_emotions) / len(text_emotions),
                metadata={'emotion_count': len(text_emotions)}
            )
            modality_inputs.append(text_input)
        
        if voice_emotions:
            voice_input = ModalityInput(
                modality='voice',
                emotion_scores=voice_emotions,
                confidence_level=self._assess_confidence_level(voice_emotions),
                quality_score=self._assess_quality_score(voice_emotions, 'voice'),
                processing_time=sum(e.processing_time for e in voice_emotions) / len(voice_emotions),
                metadata={'emotion_count': len(voice_emotions)}
            )
            modality_inputs.append(voice_input)
        
        if not modality_inputs:
            return self._create_empty_fusion_result()
        
        # Select fusion strategy
        fusion_strategy = self._select_fusion_strategy(modality_inputs, context)
        
        # Perform fusion
        fusion_result = self.fusion_strategies[fusion_strategy].fuse(modality_inputs, context)
        
        # Resolve conflicts if present
        if fusion_result.conflict_resolution:
            resolved_conflicts = self.conflict_resolver.resolve_conflicts(
                modality_inputs, 
                fusion_result.conflict_resolution,
                self.config.get('conflict_resolution_strategy', 'confidence_based')
            )
            fusion_result.conflict_resolution = resolved_conflicts
        
        # Update history
        self._update_fusion_history(fusion_result, modality_inputs)
        
        return fusion_result
    
    def _select_fusion_strategy(self, modality_inputs: List[ModalityInput], 
                               context: Dict = None) -> str:
        """Select the best fusion strategy based on inputs and context"""
        
        # Default strategy
        default_strategy = self.config.get('default_strategy', 'weighted_average')
        
        # Strategy selection logic
        if len(modality_inputs) == 1:
            return 'weighted_average'  # Simple case
        
        # Check quality differences
        qualities = [mi.quality_score for mi in modality_inputs]
        quality_variance = np.var(qualities)
        
        # If quality varies significantly, use attention-based fusion
        if quality_variance > 0.2:
            return 'attention_based'
        
        # Check confidence differences
        confidence_levels = [mi.confidence_level for mi in modality_inputs]
        if len(set(confidence_levels)) > 1:  # Different confidence levels
            return 'attention_based'
        
        return default_strategy
    
    def _assess_confidence_level(self, emotions: List[EmotionScore]) -> str:
        """Assess overall confidence level of emotion predictions"""
        if not emotions:
            return 'low'
        
        avg_confidence = np.mean([e.confidence for e in emotions[:3]])
        
        if avg_confidence >= 0.8:
            return 'high'
        elif avg_confidence >= 0.6:
            return 'medium'
        else:
            return 'low'
    
    def _assess_quality_score(self, emotions: List[EmotionScore], modality: str) -> float:
        """Assess quality score based on emotion predictions and modality"""
        if not emotions:
            return 0.5
        
        # Base quality from confidence
        base_quality = np.mean([e.confidence for e in emotions])
        
        # Modality-specific adjustments
        if modality == 'text':
            # Text quality factors
            text_length = emotions[0].metadata.get('original_length', 0)
            if text_length > 50:
                base_quality += 0.1
            elif text_length < 10:
                base_quality -= 0.1
        
        elif modality == 'voice':
            # Voice quality factors
            audio_duration = emotions[0].metadata.get('audio_duration', 0)
            if audio_duration > 2.0:
                base_quality += 0.1
            elif audio_duration < 0.5:
                base_quality -= 0.1
        
        return min(max(base_quality, 0.1), 1.0)
    
    def _update_fusion_history(self, fusion_result: FusionResult, 
                              modality_inputs: List[ModalityInput]):
        """Update fusion history for learning and analysis"""
        history_entry = {
            'timestamp': datetime.now(),
            'fusion_method': fusion_result.fusion_method,
            'confidence_level': fusion_result.confidence_level,
            'modality_count': len(modality_inputs),
            'modalities': [mi.modality for mi in modality_inputs],
            'top_emotion': fusion_result.emotion_scores[0].label if fusion_result.emotion_scores else 'none',
            'conflict_count': len(fusion_result.conflict_resolution),
            'uncertainty_score': fusion_result.uncertainty_metrics.get('overall_variance', 0.0)
        }
        
        self.fusion_history.append(history_entry)
        
        # Keep only recent history
        if len(self.fusion_history) > 1000:
            self.fusion_history = self.fusion_history[-1000:]
    
    def _create_empty_fusion_result(self) -> FusionResult:
        """Create empty fusion result for edge cases"""
        return FusionResult(
            emotion_scores=[EmotionScore(
                label="neutral",
                score=1.0,
                confidence=0.5,
                source="multimodal",
                model_name="fusion_engine",
                processing_time=0.0,
                metadata={"reason": "no_modality_inputs"}
            )],
            fusion_method="none",
            confidence_level="low",
            modality_contributions={},
            conflict_resolution={},
            uncertainty_metrics={},
            processing_metadata={"reason": "no_modality_inputs"}
        )
    
    def get_fusion_statistics(self) -> Dict[str, Any]:
        """Get statistics about fusion performance"""
        if not self.fusion_history:
            return {}
        
        recent_history = self.fusion_history[-100:]  # Last 100 fusions
        
        return {
            'total_fusions': len(self.fusion_history),
            'recent_fusions': len(recent_history),
            'fusion_methods': {
                method: sum(1 for h in recent_history if h['fusion_method'] == method)
                for method in ['weighted_average', 'attention_based']
            },
            'confidence_distribution': {
                level: sum(1 for h in recent_history if h['confidence_level'] == level)
                for level in ['high', 'medium', 'low']
            },
            'average_conflicts': np.mean([h['conflict_count'] for h in recent_history]),
            'modality_usage': {
                'text_only': sum(1 for h in recent_history if h['modalities'] == ['text']),
                'voice_only': sum(1 for h in recent_history if h['modalities'] == ['voice']),
                'multimodal': sum(1 for h in recent_history if len(h['modalities']) > 1)
            }
        }