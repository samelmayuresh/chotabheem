# utils/text_emotion_ensemble.py - Text emotion ensemble system
import numpy as np
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
import os
from datetime import datetime
import streamlit as st
from transformers import pipeline
import torch

@dataclass
class EmotionScore:
    """Enhanced emotion score with metadata"""
    label: str
    score: float
    confidence: float
    source: str  # 'text', 'voice', 'fused'
    model_name: str
    processing_time: float
    metadata: Dict[str, Any]

@dataclass
class ModelConfig:
    """Configuration for individual emotion models"""
    name: str
    model_path: str
    model_type: str  # 'transformers', 'custom', etc.
    weight: float = 1.0
    enabled: bool = True
    device: str = 'auto'
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}

@dataclass
class ModelPerformance:
    """Track individual model performance"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    processing_time: float
    memory_usage: float
    last_updated: datetime
    sample_count: int = 0

class BaseEmotionModel(ABC):
    """Abstract base class for emotion models"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.is_loaded = False
        self.performance = ModelPerformance(
            model_name=config.name,
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            processing_time=0.0,
            memory_usage=0.0,
            last_updated=datetime.now()
        )
    
    @abstractmethod
    def load_model(self) -> bool:
        """Load the emotion model"""
        pass
    
    @abstractmethod
    def predict(self, text: str, context: Dict = None) -> List[EmotionScore]:
        """Predict emotions from text"""
        pass
    
    def unload_model(self):
        """Unload model to free memory"""
        self.model = None
        self.is_loaded = False
    
    def update_performance(self, metrics: Dict[str, float]):
        """Update model performance metrics"""
        for key, value in metrics.items():
            if hasattr(self.performance, key):
                setattr(self.performance, key, value)
        self.performance.last_updated = datetime.now()

class TransformersEmotionModel(BaseEmotionModel):
    """Transformers-based emotion model wrapper"""
    
    def load_model(self) -> bool:
        """Load transformers model with caching"""
        try:
            if self.is_loaded:
                return True
            
            device = self._get_device()
            
            # Use Streamlit caching for model loading
            @st.cache_resource
            def _load_transformers_model(model_path: str, device: str):
                return pipeline(
                    "text-classification",
                    model=model_path,
                    top_k=None,
                    device=device,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
            
            self.model = _load_transformers_model(self.config.model_path, device)
            self.is_loaded = True
            logging.info(f"Loaded model {self.config.name} on device {device}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to load model {self.config.name}: {e}")
            return False
    
    def predict(self, text: str, context: Dict = None) -> List[EmotionScore]:
        """Predict emotions using transformers model"""
        if not self.is_loaded and not self.load_model():
            return []
        
        try:
            start_time = datetime.now()
            
            # Get predictions from model
            predictions = self.model(text)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Convert to EmotionScore format
            emotion_scores = []
            
            # Handle different prediction formats
            if isinstance(predictions, list) and len(predictions) > 0:
                if isinstance(predictions[0], dict):
                    # Standard format: [{'label': 'joy', 'score': 0.9}, ...]
                    for pred in predictions:
                        emotion_scores.append(EmotionScore(
                            label=pred['label'],
                            score=pred['score'],
                            confidence=pred['score'],
                            source='text',
                            model_name=self.config.name,
                            processing_time=processing_time,
                            metadata={
                                'model_type': 'transformers',
                                'context': context or {}
                            }
                        ))
                elif isinstance(predictions[0], list):
                    # Nested format: [[{'label': 'joy', 'score': 0.9}, ...]]
                    for pred in predictions[0]:
                        emotion_scores.append(EmotionScore(
                            label=pred['label'],
                            score=pred['score'],
                            confidence=pred['score'],
                            source='text',
                            model_name=self.config.name,
                            processing_time=processing_time,
                            metadata={
                                'model_type': 'transformers',
                                'context': context or {}
                            }
                        ))
            
            # Update performance tracking
            self.performance.processing_time = processing_time
            self.performance.sample_count += 1
            
            return emotion_scores
            
        except Exception as e:
            logging.error(f"Prediction failed for model {self.config.name}: {e}")
            return []
    
    def _get_device(self) -> str:
        """Determine the best device for the model"""
        if self.config.device == 'auto':
            return 0 if torch.cuda.is_available() else -1
        return self.config.device

class TextEmotionEnsemble:
    """Ensemble of specialized text emotion models"""
    
    def __init__(self, model_configs: List[ModelConfig], voting_strategy: str = "adaptive"):
        self.model_configs = model_configs
        self.models: Dict[str, BaseEmotionModel] = {}
        self.weights: Dict[str, float] = {}
        self.performance_history: List[Dict] = []
        self.ensemble_performance = {
            'accuracy': 0.0,
            'confidence_calibration': 0.0,
            'prediction_count': 0
        }
        self.voting_strategy = voting_strategy
        self.voting_system = None
        
        self._initialize_models()
        self._calculate_model_weights()
        self._initialize_voting_system()
    
    def _initialize_voting_system(self):
        """Initialize the voting system"""
        # Simply set to None - will be initialized on first use
        self.voting_system = None
    
    def _initialize_models(self):
        """Initialize all emotion models"""
        from utils.specialized_emotion_models import create_specialized_model
        
        for config in self.model_configs:
            if not config.enabled:
                continue
                
            try:
                # Use specialized model factory
                model = create_specialized_model(config)
                self.models[config.name] = model
                logging.info(f"Initialized model: {config.name} (type: {config.model_type})")
                
            except Exception as e:
                logging.error(f"Failed to initialize model {config.name}: {e}")
    
    def _calculate_model_weights(self) -> Dict[str, float]:
        """Dynamic weight calculation based on model performance"""
        if not self.models:
            return {}
        
        weights = {}
        
        # Initial weights based on configuration
        for name, model in self.models.items():
            base_weight = model.config.weight
            
            # Adjust based on performance if available
            if model.performance.sample_count > 0:
                # Weight based on accuracy and processing speed
                accuracy_factor = model.performance.accuracy if model.performance.accuracy > 0 else 0.5
                speed_factor = max(0.1, 1.0 / max(model.performance.processing_time, 0.001))
                
                # Normalize speed factor
                speed_factor = min(speed_factor, 10.0) / 10.0
                
                # Combined weight
                performance_weight = (accuracy_factor * 0.7 + speed_factor * 0.3)
                weights[name] = base_weight * performance_weight
            else:
                weights[name] = base_weight
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {name: weight / total_weight for name, weight in weights.items()}
        
        self.weights = weights
        return weights
    
    def predict(self, text: str, context: Dict = None) -> List[EmotionScore]:
        """Enhanced ensemble prediction with advanced voting"""
        if not text or not text.strip():
            return [EmotionScore(
                label="neutral",
                score=1.0,
                confidence=1.0,
                source="text",
                model_name="ensemble",
                processing_time=0.0,
                metadata={"reason": "empty_text"}
            )]
        
        model_predictions = {}
        total_processing_time = 0.0
        
        # Get predictions from all models
        for name, model in self.models.items():
            try:
                predictions = model.predict(text, context)
                if predictions:
                    model_predictions[name] = predictions
                    total_processing_time += predictions[0].processing_time
                        
            except Exception as e:
                logging.error(f"Model {name} prediction failed: {e}")
        
        if not model_predictions:
            return [EmotionScore(
                label="neutral",
                score=1.0,
                confidence=0.5,
                source="text",
                model_name="ensemble",
                processing_time=0.0,
                metadata={"reason": "no_predictions"}
            )]
        
        # Use advanced voting system if available
        if self.voting_system is None and self.voting_strategy != "simple":
            try:
                # Lazy initialization of voting system
                from utils.ensemble_voting import create_voting_system
                self.voting_system = create_voting_system(self.voting_strategy)
            except Exception as e:
                logging.error(f"Failed to initialize voting system: {e}")
                self.voting_system = False  # Mark as failed to avoid retrying
        
        if self.voting_system and self.voting_system is not False:
            try:
                voting_result = self.voting_system.vote(
                    model_predictions, self.weights, context
                )
                
                # Update processing time for all scores
                for score in voting_result.emotion_scores:
                    score.processing_time = total_processing_time
                    score.metadata.update({
                        'voting_method': voting_result.voting_method,
                        'agreement_score': voting_result.agreement_score,
                        'confidence_level': voting_result.confidence_level,
                        'model_contributions': voting_result.model_contributions
                    })
                
                # Update ensemble performance tracking
                self.ensemble_performance['prediction_count'] += 1
                self.ensemble_performance['confidence_calibration'] = voting_result.agreement_score
                
                return voting_result.emotion_scores
                
            except Exception as e:
                logging.error(f"Advanced voting failed: {e}")
                # Fall back to simple weighted average
        
        # Fallback to simple weighted average if voting system fails
        return self._simple_weighted_prediction(model_predictions, total_processing_time)
    
    def _simple_weighted_prediction(self, model_predictions: Dict[str, List[EmotionScore]], 
                                  total_processing_time: float) -> List[EmotionScore]:
        """Fallback simple weighted prediction"""
        all_predictions = {}
        
        # Aggregate predictions by emotion label
        for name, predictions in model_predictions.items():
            for pred in predictions:
                if pred.label not in all_predictions:
                    all_predictions[pred.label] = []
                all_predictions[pred.label].append({
                    'score': pred.score,
                    'weight': self.weights.get(name, 0.0),
                    'model': name,
                    'confidence': pred.confidence
                })
        
        # Calculate weighted ensemble scores
        ensemble_scores = []
        for emotion, predictions in all_predictions.items():
            # Weighted average of scores
            weighted_score = sum(p['score'] * p['weight'] for p in predictions)
            total_weight = sum(p['weight'] for p in predictions)
            
            if total_weight > 0:
                final_score = weighted_score / total_weight
            else:
                final_score = np.mean([p['score'] for p in predictions])
            
            # Calculate confidence based on agreement between models
            scores = [p['score'] for p in predictions]
            confidence = self._calculate_confidence(scores, len(model_predictions))
            
            ensemble_scores.append(EmotionScore(
                label=emotion,
                score=final_score,
                confidence=confidence,
                source="text",
                model_name="ensemble_fallback",
                processing_time=total_processing_time,
                metadata={
                    'contributing_models': [p['model'] for p in predictions],
                    'model_count': len(predictions),
                    'weight_sum': total_weight,
                    'score_variance': np.var(scores) if len(scores) > 1 else 0.0,
                    'voting_method': 'simple_weighted'
                }
            ))
        
        # Sort by score and return
        ensemble_scores.sort(key=lambda x: x.score, reverse=True)
        
        # Update ensemble performance tracking
        self.ensemble_performance['prediction_count'] += 1
        
        return ensemble_scores
    
    def _calculate_confidence(self, scores: List[float], model_count: int) -> float:
        """Calculate confidence based on model agreement"""
        if len(scores) <= 1:
            return scores[0] if scores else 0.5
        
        # Base confidence on score variance (lower variance = higher confidence)
        score_variance = np.var(scores)
        variance_confidence = max(0.0, 1.0 - score_variance * 2)
        
        # Boost confidence if more models agree
        model_agreement = len(scores) / max(model_count, 1)
        
        # Combined confidence
        confidence = (variance_confidence * 0.7 + model_agreement * 0.3)
        
        return min(max(confidence, 0.1), 1.0)
    
    def get_model_status(self) -> Dict[str, Dict]:
        """Get status of all models in the ensemble"""
        status = {}
        for name, model in self.models.items():
            status[name] = {
                'loaded': model.is_loaded,
                'weight': self.weights.get(name, 0.0),
                'performance': {
                    'accuracy': model.performance.accuracy,
                    'processing_time': model.performance.processing_time,
                    'sample_count': model.performance.sample_count
                },
                'config': {
                    'model_path': model.config.model_path,
                    'enabled': model.config.enabled
                }
            }
        return status
    
    def update_model_performance(self, model_name: str, metrics: Dict[str, float]):
        """Update performance metrics for a specific model"""
        if model_name in self.models:
            self.models[model_name].update_performance(metrics)
            self._calculate_model_weights()  # Recalculate weights
    
    def enable_model(self, model_name: str, enabled: bool = True):
        """Enable or disable a specific model"""
        if model_name in self.models:
            self.models[model_name].config.enabled = enabled
            if not enabled:
                self.models[model_name].unload_model()
            self._calculate_model_weights()
    
    def get_ensemble_performance(self) -> Dict[str, Any]:
        """Get overall ensemble performance metrics"""
        return {
            **self.ensemble_performance,
            'model_count': len([m for m in self.models.values() if m.config.enabled]),
            'loaded_models': len([m for m in self.models.values() if m.is_loaded]),
            'weights': self.weights.copy(),
            'individual_performance': {
                name: {
                    'accuracy': model.performance.accuracy,
                    'processing_time': model.performance.processing_time,
                    'sample_count': model.performance.sample_count
                }
                for name, model in self.models.items()
            }
        }
    
    def save_configuration(self, filepath: str):
        """Save ensemble configuration to file"""
        config_data = {
            'model_configs': [
                {
                    'name': config.name,
                    'model_path': config.model_path,
                    'model_type': config.model_type,
                    'weight': config.weight,
                    'enabled': config.enabled,
                    'device': config.device,
                    'parameters': config.parameters
                }
                for config in self.model_configs
            ],
            'weights': self.weights,
            'performance': self.ensemble_performance
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(config_data, f, indent=2, default=str)
            logging.info(f"Configuration saved to {filepath}")
        except Exception as e:
            logging.error(f"Failed to save configuration: {e}")
    
    @classmethod
    def load_configuration(cls, filepath: str) -> 'TextEmotionEnsemble':
        """Load ensemble configuration from file"""
        try:
            with open(filepath, 'r') as f:
                config_data = json.load(f)
            
            model_configs = []
            for config_dict in config_data['model_configs']:
                model_configs.append(ModelConfig(**config_dict))
            
            ensemble = cls(model_configs)
            
            # Restore weights and performance if available
            if 'weights' in config_data:
                ensemble.weights = config_data['weights']
            if 'performance' in config_data:
                ensemble.ensemble_performance.update(config_data['performance'])
            
            logging.info(f"Configuration loaded from {filepath}")
            return ensemble
            
        except Exception as e:
            logging.error(f"Failed to load configuration: {e}")
            raise

def create_default_text_ensemble() -> TextEmotionEnsemble:
    """Create a default text emotion ensemble with specialized models"""
    default_configs = [
        ModelConfig(
            name="primary_bert",
            model_path="j-hartmann/emotion-english-distilroberta-base",
            model_type="bert",
            weight=1.0,
            enabled=True
        ),
        ModelConfig(
            name="roberta_social",
            model_path="cardiffnlp/twitter-roberta-base-emotion",
            model_type="roberta",
            weight=0.9,
            enabled=True
        ),
        ModelConfig(
            name="context_aware",
            model_path="j-hartmann/emotion-english-distilroberta-base",
            model_type="context_aware",
            weight=0.8,
            enabled=True
        ),
        ModelConfig(
            name="go_emotions",
            model_path="monologg/bert-base-cased-goemotions-original",
            model_type="bert",
            weight=0.9,
            enabled=True
        ),
        ModelConfig(
            name="multilingual",
            model_path="cardiffnlp/twitter-xlm-roberta-base-sentiment",
            model_type="multilingual",
            weight=0.7,
            enabled=True
        )
    ]
    
    return TextEmotionEnsemble(default_configs)