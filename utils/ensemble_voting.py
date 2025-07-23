# utils/ensemble_voting.py - Advanced ensemble voting and prediction systems
import numpy as np
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
from datetime import datetime
import scipy.stats as stats
from utils.text_emotion_ensemble import EmotionScore

@dataclass
class VotingResult:
    """Result of ensemble voting with detailed metadata"""
    emotion_scores: List[EmotionScore]
    voting_method: str
    confidence_level: str
    agreement_score: float
    model_contributions: Dict[str, float]
    uncertainty_metrics: Dict[str, float]
    metadata: Dict[str, Any]

class BaseVotingStrategy(ABC):
    """Abstract base class for voting strategies"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def vote(self, model_predictions: Dict[str, List[EmotionScore]], 
             model_weights: Dict[str, float],
             context: Dict = None) -> VotingResult:
        """Perform ensemble voting"""
        pass

class WeightedAverageVoting(BaseVotingStrategy):
    """Weighted average voting with confidence calibration"""
    
    def __init__(self):
        super().__init__("weighted_average")
    
    def vote(self, model_predictions: Dict[str, List[EmotionScore]], 
             model_weights: Dict[str, float],
             context: Dict = None) -> VotingResult:
        """Weighted average voting with enhanced confidence calculation"""
        
        if not model_predictions:
            return self._create_empty_result()
        
        # Aggregate predictions by emotion label
        emotion_aggregates = {}
        model_contributions = {}
        
        for model_name, predictions in model_predictions.items():
            weight = model_weights.get(model_name, 1.0)
            model_contributions[model_name] = 0.0
            
            for pred in predictions:
                if pred.label not in emotion_aggregates:
                    emotion_aggregates[pred.label] = {
                        'scores': [],
                        'weights': [],
                        'confidences': [],
                        'models': []
                    }
                
                emotion_aggregates[pred.label]['scores'].append(pred.score)
                emotion_aggregates[pred.label]['weights'].append(weight)
                emotion_aggregates[pred.label]['confidences'].append(pred.confidence)
                emotion_aggregates[pred.label]['models'].append(model_name)
                
                model_contributions[model_name] += weight * pred.score
        
        # Calculate weighted averages
        final_emotions = []
        total_contribution = sum(model_contributions.values())
        
        for emotion, data in emotion_aggregates.items():
            scores = np.array(data['scores'])
            weights = np.array(data['weights'])
            confidences = np.array(data['confidences'])
            
            # Weighted average score
            weighted_score = np.average(scores, weights=weights)
            
            # Enhanced confidence calculation
            confidence = self._calculate_enhanced_confidence(
                scores, weights, confidences, len(model_predictions)
            )
            
            # Agreement score (how much models agree)
            agreement = self._calculate_agreement(scores, weights)
            
            final_emotions.append(EmotionScore(
                label=emotion,
                score=weighted_score,
                confidence=confidence,
                source='text',
                model_name='ensemble_weighted',
                processing_time=0.0,
                metadata={
                    'voting_method': 'weighted_average',
                    'agreement_score': agreement,
                    'contributing_models': data['models'],
                    'model_count': len(data['models']),
                    'score_variance': np.var(scores),
                    'weight_sum': np.sum(weights)
                }
            ))
        
        # Sort by score
        final_emotions.sort(key=lambda x: x.score, reverse=True)
        
        # Calculate overall metrics
        agreement_score = self._calculate_overall_agreement(emotion_aggregates)
        confidence_level = self._determine_confidence_level(final_emotions)
        uncertainty_metrics = self._calculate_uncertainty_metrics(emotion_aggregates)
        
        # Normalize model contributions
        if total_contribution > 0:
            model_contributions = {
                k: v / total_contribution for k, v in model_contributions.items()
            }
        
        return VotingResult(
            emotion_scores=final_emotions,
            voting_method="weighted_average",
            confidence_level=confidence_level,
            agreement_score=agreement_score,
            model_contributions=model_contributions,
            uncertainty_metrics=uncertainty_metrics,
            metadata={
                'total_models': len(model_predictions),
                'total_predictions': sum(len(preds) for preds in model_predictions.values()),
                'context': context or {}
            }
        )
    
    def _calculate_enhanced_confidence(self, scores: np.ndarray, weights: np.ndarray,
                                     confidences: np.ndarray, model_count: int) -> float:
        """Calculate enhanced confidence based on multiple factors"""
        if len(scores) == 0:
            return 0.5
        
        # Base confidence from individual model confidences
        base_confidence = np.average(confidences, weights=weights)
        
        # Agreement factor (lower variance = higher confidence)
        if len(scores) > 1:
            score_variance = np.var(scores)
            agreement_factor = max(0.0, 1.0 - score_variance * 2)
        else:
            agreement_factor = 0.8  # Single model gets moderate agreement
        
        # Model count factor (more models = higher confidence if they agree)
        model_factor = min(1.0, len(scores) / max(model_count, 1) + 0.2)
        
        # Weight distribution factor (more balanced weights = higher confidence)
        if len(weights) > 1:
            weight_entropy = stats.entropy(weights / np.sum(weights))
            max_entropy = np.log(len(weights))
            weight_factor = weight_entropy / max_entropy if max_entropy > 0 else 0.5
        else:
            weight_factor = 0.5
        
        # Combined confidence
        confidence = (
            base_confidence * 0.4 +
            agreement_factor * 0.3 +
            model_factor * 0.2 +
            weight_factor * 0.1
        )
        
        return min(max(confidence, 0.1), 1.0)
    
    def _calculate_agreement(self, scores: np.ndarray, weights: np.ndarray) -> float:
        """Calculate agreement score for a specific emotion"""
        if len(scores) <= 1:
            return 1.0
        
        # Weighted standard deviation
        weighted_mean = np.average(scores, weights=weights)
        weighted_variance = np.average((scores - weighted_mean) ** 2, weights=weights)
        
        # Convert to agreement score (lower variance = higher agreement)
        agreement = max(0.0, 1.0 - weighted_variance * 4)
        return agreement
    
    def _calculate_overall_agreement(self, emotion_aggregates: Dict) -> float:
        """Calculate overall agreement across all emotions"""
        if not emotion_aggregates:
            return 0.0
        
        agreements = []
        for emotion, data in emotion_aggregates.items():
            scores = np.array(data['scores'])
            weights = np.array(data['weights'])
            agreement = self._calculate_agreement(scores, weights)
            agreements.append(agreement)
        
        return np.mean(agreements)
    
    def _determine_confidence_level(self, emotions: List[EmotionScore]) -> str:
        """Determine overall confidence level"""
        if not emotions:
            return "low"
        
        avg_confidence = np.mean([e.confidence for e in emotions[:3]])  # Top 3 emotions
        
        if avg_confidence >= 0.8:
            return "high"
        elif avg_confidence >= 0.6:
            return "medium"
        else:
            return "low"
    
    def _calculate_uncertainty_metrics(self, emotion_aggregates: Dict) -> Dict[str, float]:
        """Calculate various uncertainty metrics"""
        if not emotion_aggregates:
            return {}
        
        all_scores = []
        all_variances = []
        
        for emotion, data in emotion_aggregates.items():
            scores = np.array(data['scores'])
            all_scores.extend(scores)
            if len(scores) > 1:
                all_variances.append(np.var(scores))
        
        return {
            'overall_variance': np.var(all_scores) if all_scores else 0.0,
            'average_emotion_variance': np.mean(all_variances) if all_variances else 0.0,
            'prediction_entropy': self._calculate_entropy(all_scores) if all_scores else 0.0,
            'emotion_count': len(emotion_aggregates)
        }
    
    def _calculate_entropy(self, scores: List[float]) -> float:
        """Calculate entropy of prediction scores"""
        if not scores:
            return 0.0
        
        # Normalize scores to probabilities
        scores = np.array(scores)
        scores = scores / np.sum(scores) if np.sum(scores) > 0 else scores
        
        # Calculate entropy
        entropy = -np.sum(scores * np.log(scores + 1e-10))
        return entropy
    
    def _create_empty_result(self) -> VotingResult:
        """Create empty result for edge cases"""
        return VotingResult(
            emotion_scores=[EmotionScore(
                label="neutral",
                score=1.0,
                confidence=0.5,
                source="text",
                model_name="ensemble_weighted",
                processing_time=0.0,
                metadata={"reason": "no_predictions"}
            )],
            voting_method="weighted_average",
            confidence_level="low",
            agreement_score=0.0,
            model_contributions={},
            uncertainty_metrics={},
            metadata={"reason": "no_predictions"}
        )

class MajorityVoting(BaseVotingStrategy):
    """Majority voting with confidence weighting"""
    
    def __init__(self):
        super().__init__("majority")
    
    def vote(self, model_predictions: Dict[str, List[EmotionScore]], 
             model_weights: Dict[str, float],
             context: Dict = None) -> VotingResult:
        """Majority voting based on top predictions from each model"""
        
        if not model_predictions:
            return self._create_empty_result()
        
        # Get top prediction from each model
        top_predictions = {}
        for model_name, predictions in model_predictions.items():
            if predictions:
                top_pred = max(predictions, key=lambda x: x.score)
                top_predictions[model_name] = top_pred
        
        # Count votes for each emotion
        emotion_votes = {}
        model_contributions = {}
        
        for model_name, pred in top_predictions.items():
            weight = model_weights.get(model_name, 1.0)
            
            if pred.label not in emotion_votes:
                emotion_votes[pred.label] = {
                    'vote_count': 0,
                    'weighted_votes': 0.0,
                    'scores': [],
                    'confidences': [],
                    'models': []
                }
            
            emotion_votes[pred.label]['vote_count'] += 1
            emotion_votes[pred.label]['weighted_votes'] += weight
            emotion_votes[pred.label]['scores'].append(pred.score)
            emotion_votes[pred.label]['confidences'].append(pred.confidence)
            emotion_votes[pred.label]['models'].append(model_name)
            
            model_contributions[model_name] = weight
        
        # Create final emotion scores
        final_emotions = []
        total_votes = sum(data['weighted_votes'] for data in emotion_votes.values())
        
        for emotion, data in emotion_votes.items():
            # Score based on weighted vote proportion
            vote_score = data['weighted_votes'] / max(total_votes, 1.0)
            
            # Average confidence from contributing models
            avg_confidence = np.mean(data['confidences'])
            
            # Agreement based on how many models voted for this emotion
            agreement = data['vote_count'] / len(model_predictions)
            
            final_emotions.append(EmotionScore(
                label=emotion,
                score=vote_score,
                confidence=avg_confidence * agreement,  # Adjust confidence by agreement
                source='text',
                model_name='ensemble_majority',
                processing_time=0.0,
                metadata={
                    'voting_method': 'majority',
                    'vote_count': data['vote_count'],
                    'weighted_votes': data['weighted_votes'],
                    'agreement_score': agreement,
                    'contributing_models': data['models'],
                    'avg_model_score': np.mean(data['scores'])
                }
            ))
        
        # Sort by vote score
        final_emotions.sort(key=lambda x: x.score, reverse=True)
        
        # Calculate overall metrics
        agreement_score = self._calculate_majority_agreement(emotion_votes, len(model_predictions))
        confidence_level = self._determine_confidence_level(final_emotions)
        
        return VotingResult(
            emotion_scores=final_emotions,
            voting_method="majority",
            confidence_level=confidence_level,
            agreement_score=agreement_score,
            model_contributions=model_contributions,
            uncertainty_metrics={
                'vote_distribution_entropy': self._calculate_vote_entropy(emotion_votes),
                'consensus_strength': max(data['vote_count'] for data in emotion_votes.values()) / len(model_predictions)
            },
            metadata={
                'total_models': len(model_predictions),
                'voting_emotions': len(emotion_votes),
                'context': context or {}
            }
        )
    
    def _calculate_majority_agreement(self, emotion_votes: Dict, total_models: int) -> float:
        """Calculate agreement score for majority voting"""
        if not emotion_votes:
            return 0.0
        
        # Agreement based on how concentrated the votes are
        vote_counts = [data['vote_count'] for data in emotion_votes.values()]
        max_votes = max(vote_counts)
        
        # Higher agreement if one emotion gets most votes
        agreement = max_votes / total_models
        return agreement
    
    def _calculate_vote_entropy(self, emotion_votes: Dict) -> float:
        """Calculate entropy of vote distribution"""
        if not emotion_votes:
            return 0.0
        
        vote_counts = [data['vote_count'] for data in emotion_votes.values()]
        total_votes = sum(vote_counts)
        
        if total_votes == 0:
            return 0.0
        
        probabilities = [count / total_votes for count in vote_counts]
        entropy = -sum(p * np.log(p + 1e-10) for p in probabilities)
        
        return entropy
    
    def _determine_confidence_level(self, emotions: List[EmotionScore]) -> str:
        """Determine confidence level for majority voting"""
        if not emotions:
            return "low"
        
        top_emotion = emotions[0]
        
        # High confidence if clear majority
        if top_emotion.score >= 0.6:
            return "high"
        elif top_emotion.score >= 0.4:
            return "medium"
        else:
            return "low"
    
    def _create_empty_result(self) -> VotingResult:
        """Create empty result for edge cases"""
        return VotingResult(
            emotion_scores=[EmotionScore(
                label="neutral",
                score=1.0,
                confidence=0.5,
                source="text",
                model_name="ensemble_majority",
                processing_time=0.0,
                metadata={"reason": "no_predictions"}
            )],
            voting_method="majority",
            confidence_level="low",
            agreement_score=0.0,
            model_contributions={},
            uncertainty_metrics={},
            metadata={"reason": "no_predictions"}
        )

class BayesianVoting(BaseVotingStrategy):
    """Bayesian ensemble voting with uncertainty quantification"""
    
    def __init__(self):
        super().__init__("bayesian")
        self.prior_emotions = {
            'joy': 0.2, 'sadness': 0.15, 'anger': 0.1, 'fear': 0.1,
            'surprise': 0.1, 'disgust': 0.05, 'neutral': 0.3
        }
    
    def vote(self, model_predictions: Dict[str, List[EmotionScore]], 
             model_weights: Dict[str, float],
             context: Dict = None) -> VotingResult:
        """Bayesian voting with prior knowledge and uncertainty"""
        
        if not model_predictions:
            return self._create_empty_result()
        
        # Collect all unique emotions
        all_emotions = set()
        for predictions in model_predictions.values():
            for pred in predictions:
                all_emotions.add(pred.label)
        
        # Calculate Bayesian posterior for each emotion
        emotion_posteriors = {}
        model_contributions = {}
        
        for emotion in all_emotions:
            # Prior probability
            prior = self.prior_emotions.get(emotion, 0.1)
            
            # Likelihood from each model
            likelihoods = []
            contributing_models = []
            
            for model_name, predictions in model_predictions.items():
                weight = model_weights.get(model_name, 1.0)
                
                # Find prediction for this emotion
                emotion_pred = next((p for p in predictions if p.label == emotion), None)
                
                if emotion_pred:
                    # Use model's confidence as likelihood
                    likelihood = emotion_pred.confidence * weight
                    likelihoods.append(likelihood)
                    contributing_models.append(model_name)
                    
                    if model_name not in model_contributions:
                        model_contributions[model_name] = 0.0
                    model_contributions[model_name] += likelihood
            
            # Calculate posterior (simplified Bayesian update)
            if likelihoods:
                avg_likelihood = np.mean(likelihoods)
                posterior = (prior * avg_likelihood) / (prior * avg_likelihood + (1 - prior) * (1 - avg_likelihood))
                
                # Uncertainty based on likelihood variance
                uncertainty = np.var(likelihoods) if len(likelihoods) > 1 else 0.1
                confidence = max(0.1, posterior * (1 - uncertainty))
            else:
                posterior = prior
                confidence = 0.1
                uncertainty = 0.5
            
            emotion_posteriors[emotion] = {
                'posterior': posterior,
                'confidence': confidence,
                'uncertainty': uncertainty,
                'contributing_models': contributing_models,
                'likelihood_count': len(likelihoods)
            }
        
        # Create final emotion scores
        final_emotions = []
        total_posterior = sum(data['posterior'] for data in emotion_posteriors.values())
        
        for emotion, data in emotion_posteriors.items():
            # Normalize posterior
            normalized_score = data['posterior'] / max(total_posterior, 1.0)
            
            final_emotions.append(EmotionScore(
                label=emotion,
                score=normalized_score,
                confidence=data['confidence'],
                source='text',
                model_name='ensemble_bayesian',
                processing_time=0.0,
                metadata={
                    'voting_method': 'bayesian',
                    'posterior': data['posterior'],
                    'uncertainty': data['uncertainty'],
                    'prior': self.prior_emotions.get(emotion, 0.1),
                    'contributing_models': data['contributing_models'],
                    'likelihood_count': data['likelihood_count']
                }
            ))
        
        # Sort by posterior probability
        final_emotions.sort(key=lambda x: x.score, reverse=True)
        
        # Calculate overall metrics
        agreement_score = self._calculate_bayesian_agreement(emotion_posteriors)
        confidence_level = self._determine_bayesian_confidence(final_emotions)
        uncertainty_metrics = self._calculate_bayesian_uncertainty(emotion_posteriors)
        
        return VotingResult(
            emotion_scores=final_emotions,
            voting_method="bayesian",
            confidence_level=confidence_level,
            agreement_score=agreement_score,
            model_contributions=model_contributions,
            uncertainty_metrics=uncertainty_metrics,
            metadata={
                'total_models': len(model_predictions),
                'total_emotions': len(all_emotions),
                'context': context or {}
            }
        )
    
    def _calculate_bayesian_agreement(self, emotion_posteriors: Dict) -> float:
        """Calculate agreement for Bayesian voting"""
        if not emotion_posteriors:
            return 0.0
        
        # Agreement based on how concentrated the posterior distribution is
        posteriors = [data['posterior'] for data in emotion_posteriors.values()]
        max_posterior = max(posteriors)
        total_posterior = sum(posteriors)
        
        # Higher agreement if one emotion has high posterior
        agreement = max_posterior / max(total_posterior, 1.0)
        return agreement
    
    def _determine_bayesian_confidence(self, emotions: List[EmotionScore]) -> str:
        """Determine confidence level for Bayesian voting"""
        if not emotions:
            return "low"
        
        top_emotion = emotions[0]
        avg_uncertainty = np.mean([e.metadata.get('uncertainty', 0.5) for e in emotions[:3]])
        
        # High confidence if high posterior and low uncertainty
        if top_emotion.score >= 0.6 and avg_uncertainty <= 0.2:
            return "high"
        elif top_emotion.score >= 0.4 and avg_uncertainty <= 0.4:
            return "medium"
        else:
            return "low"
    
    def _calculate_bayesian_uncertainty(self, emotion_posteriors: Dict) -> Dict[str, float]:
        """Calculate Bayesian uncertainty metrics"""
        if not emotion_posteriors:
            return {}
        
        posteriors = [data['posterior'] for data in emotion_posteriors.values()]
        uncertainties = [data['uncertainty'] for data in emotion_posteriors.values()]
        
        return {
            'posterior_entropy': -sum(p * np.log(p + 1e-10) for p in posteriors),
            'average_uncertainty': np.mean(uncertainties),
            'max_uncertainty': max(uncertainties),
            'uncertainty_variance': np.var(uncertainties)
        }
    
    def _create_empty_result(self) -> VotingResult:
        """Create empty result for edge cases"""
        return VotingResult(
            emotion_scores=[EmotionScore(
                label="neutral",
                score=1.0,
                confidence=0.5,
                source="text",
                model_name="ensemble_bayesian",
                processing_time=0.0,
                metadata={"reason": "no_predictions"}
            )],
            voting_method="bayesian",
            confidence_level="low",
            agreement_score=0.0,
            model_contributions={},
            uncertainty_metrics={},
            metadata={"reason": "no_predictions"}
        )

class AdaptiveVoting(BaseVotingStrategy):
    """Adaptive voting that selects the best strategy based on context"""
    
    def __init__(self):
        super().__init__("adaptive")
        self.strategies = {
            'weighted_average': WeightedAverageVoting(),
            'majority': MajorityVoting(),
            'bayesian': BayesianVoting()
        }
        self.strategy_performance = {
            'weighted_average': {'accuracy': 0.85, 'confidence': 0.8},
            'majority': {'accuracy': 0.80, 'confidence': 0.75},
            'bayesian': {'accuracy': 0.82, 'confidence': 0.85}
        }
    
    def vote(self, model_predictions: Dict[str, List[EmotionScore]], 
             model_weights: Dict[str, float],
             context: Dict = None) -> VotingResult:
        """Adaptive voting that selects the best strategy"""
        
        if not model_predictions:
            return self.strategies['weighted_average']._create_empty_result()
        
        # Select best strategy based on context and model characteristics
        selected_strategy = self._select_strategy(model_predictions, model_weights, context)
        
        # Perform voting with selected strategy
        result = self.strategies[selected_strategy].vote(model_predictions, model_weights, context)
        
        # Update metadata to indicate adaptive selection
        result.voting_method = f"adaptive_{selected_strategy}"
        result.metadata['selected_strategy'] = selected_strategy
        result.metadata['strategy_reason'] = self._get_strategy_reason(
            selected_strategy, model_predictions, context
        )
        
        return result
    
    def _select_strategy(self, model_predictions: Dict[str, List[EmotionScore]], 
                        model_weights: Dict[str, float],
                        context: Dict = None) -> str:
        """Select the best voting strategy based on current conditions"""
        
        model_count = len(model_predictions)
        total_predictions = sum(len(preds) for preds in model_predictions.values())
        
        # Strategy selection logic
        if model_count <= 2:
            # Few models: use weighted average
            return 'weighted_average'
        elif model_count >= 5:
            # Many models: use majority voting
            return 'majority'
        elif context and context.get('uncertainty_tolerance', 'medium') == 'high':
            # High uncertainty tolerance: use Bayesian
            return 'bayesian'
        else:
            # Default: weighted average
            return 'weighted_average'
    
    def _get_strategy_reason(self, strategy: str, model_predictions: Dict, 
                           context: Dict = None) -> str:
        """Get reason for strategy selection"""
        model_count = len(model_predictions)
        
        reasons = {
            'weighted_average': f"Selected for {model_count} models - optimal for balanced weighting",
            'majority': f"Selected for {model_count} models - optimal for consensus building",
            'bayesian': f"Selected for uncertainty quantification with {model_count} models"
        }
        
        return reasons.get(strategy, f"Selected {strategy} strategy")

def create_voting_system(strategy: str = "adaptive") -> BaseVotingStrategy:
    """Factory function to create voting systems"""
    strategies = {
        'weighted_average': WeightedAverageVoting,
        'majority': MajorityVoting,
        'bayesian': BayesianVoting,
        'adaptive': AdaptiveVoting
    }
    
    strategy_class = strategies.get(strategy, AdaptiveVoting)
    return strategy_class()