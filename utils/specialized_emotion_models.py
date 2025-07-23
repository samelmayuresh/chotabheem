# utils/specialized_emotion_models.py - Specialized emotion model implementations
import numpy as np
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from utils.text_emotion_ensemble import BaseEmotionModel, EmotionScore, ModelConfig

class BERTEmotionModel(BaseEmotionModel):
    """BERT-based emotion model with fine-tuning capabilities"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.tokenizer = None
        self.model_instance = None
        
    def load_model(self) -> bool:
        """Load BERT model with custom configuration"""
        try:
            if self.is_loaded:
                return True
            
            device = self._get_device()
            
            @st.cache_resource
            def _load_bert_model(model_path: str, device: str):
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
                
                if device != 'cpu' and torch.cuda.is_available():
                    model = model.to(f'cuda:{device}' if isinstance(device, int) else 'cuda')
                
                return tokenizer, model
            
            self.tokenizer, self.model_instance = _load_bert_model(
                self.config.model_path, device
            )
            
            # Create pipeline
            self.model = pipeline(
                "text-classification",
                model=self.model_instance,
                tokenizer=self.tokenizer,
                top_k=None,
                device=0 if torch.cuda.is_available() and device != 'cpu' else -1
            )
            
            self.is_loaded = True
            logging.info(f"Loaded BERT model {self.config.name}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to load BERT model {self.config.name}: {e}")
            return False
    
    def predict(self, text: str, context: Dict = None) -> List[EmotionScore]:
        """Predict emotions with BERT model"""
        if not self.is_loaded and not self.load_model():
            return []
        
        try:
            start_time = datetime.now()
            
            # Preprocess text for BERT
            processed_text = self._preprocess_for_bert(text)
            
            # Get predictions
            predictions = self.model(processed_text)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Convert to EmotionScore format
            emotion_scores = []
            pred_list = predictions if isinstance(predictions, list) else [predictions]
            
            for pred_group in pred_list:
                if isinstance(pred_group, list):
                    for pred in pred_group:
                        emotion_scores.append(EmotionScore(
                            label=self._normalize_emotion_label(pred['label']),
                            score=pred['score'],
                            confidence=pred['score'],
                            source='text',
                            model_name=self.config.name,
                            processing_time=processing_time,
                            metadata={
                                'model_type': 'bert',
                                'context': context or {},
                                'preprocessing': 'bert_optimized'
                            }
                        ))
                else:
                    emotion_scores.append(EmotionScore(
                        label=self._normalize_emotion_label(pred_group['label']),
                        score=pred_group['score'],
                        confidence=pred_group['score'],
                        source='text',
                        model_name=self.config.name,
                        processing_time=processing_time,
                        metadata={
                            'model_type': 'bert',
                            'context': context or {},
                            'preprocessing': 'bert_optimized'
                        }
                    ))
            
            self.performance.processing_time = processing_time
            self.performance.sample_count += 1
            
            return emotion_scores
            
        except Exception as e:
            logging.error(f"BERT prediction failed for {self.config.name}: {e}")
            return []
    
    def _preprocess_for_bert(self, text: str) -> str:
        """Preprocess text specifically for BERT models"""
        # BERT-specific preprocessing
        text = text.strip()
        
        # Handle very long texts by truncating intelligently
        if len(text) > 500:
            sentences = text.split('. ')
            if len(sentences) > 1:
                # Keep first and last sentences for context
                text = sentences[0] + '. ' + sentences[-1]
            else:
                text = text[:500]
        
        return text
    
    def _normalize_emotion_label(self, label: str) -> str:
        """Normalize emotion labels to standard format"""
        label_mapping = {
            'LABEL_0': 'sadness',
            'LABEL_1': 'joy',
            'LABEL_2': 'love',
            'LABEL_3': 'anger',
            'LABEL_4': 'fear',
            'LABEL_5': 'surprise',
            'sadness': 'sadness',
            'joy': 'joy',
            'love': 'love',
            'anger': 'anger',
            'fear': 'fear',
            'surprise': 'surprise',
            'happy': 'joy',
            'sad': 'sadness',
            'angry': 'anger',
            'scared': 'fear',
            'surprised': 'surprise'
        }
        
        return label_mapping.get(label.lower(), label.lower())
    
    def _get_device(self) -> str:
        """Get appropriate device for BERT model"""
        if self.config.device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return self.config.device

class RoBERTaEmotionModel(BaseEmotionModel):
    """RoBERTa-based emotion model optimized for social media text"""
    
    def load_model(self) -> bool:
        """Load RoBERTa model optimized for emotion detection"""
        try:
            if self.is_loaded:
                return True
            
            device = 0 if torch.cuda.is_available() else -1
            
            @st.cache_resource
            def _load_roberta_model(model_path: str, device: int):
                return pipeline(
                    "text-classification",
                    model=model_path,
                    top_k=None,
                    device=device,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
            
            self.model = _load_roberta_model(self.config.model_path, device)
            self.is_loaded = True
            logging.info(f"Loaded RoBERTa model {self.config.name}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to load RoBERTa model {self.config.name}: {e}")
            return False
    
    def predict(self, text: str, context: Dict = None) -> List[EmotionScore]:
        """Predict emotions with RoBERTa model"""
        if not self.is_loaded and not self.load_model():
            return []
        
        try:
            start_time = datetime.now()
            
            # RoBERTa-specific preprocessing
            processed_text = self._preprocess_for_roberta(text)
            
            predictions = self.model(processed_text)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            emotion_scores = []
            pred_list = predictions if isinstance(predictions, list) else [predictions]
            
            for pred_group in pred_list:
                if isinstance(pred_group, list):
                    for pred in pred_group:
                        emotion_scores.append(EmotionScore(
                            label=self._normalize_roberta_label(pred['label']),
                            score=pred['score'],
                            confidence=pred['score'],
                            source='text',
                            model_name=self.config.name,
                            processing_time=processing_time,
                            metadata={
                                'model_type': 'roberta',
                                'context': context or {},
                                'social_media_optimized': True
                            }
                        ))
            
            self.performance.processing_time = processing_time
            self.performance.sample_count += 1
            
            return emotion_scores
            
        except Exception as e:
            logging.error(f"RoBERTa prediction failed for {self.config.name}: {e}")
            return []
    
    def _preprocess_for_roberta(self, text: str) -> str:
        """Preprocess text for RoBERTa (social media optimized)"""
        # Preserve social media elements that RoBERTa handles well
        text = text.strip()
        
        # Don't over-process - RoBERTa handles informal text well
        return text
    
    def _normalize_roberta_label(self, label: str) -> str:
        """Normalize RoBERTa emotion labels"""
        label_mapping = {
            'optimism': 'joy',
            'pessimism': 'sadness',
            'trust': 'love',
            'anticipation': 'surprise'
        }
        
        return label_mapping.get(label.lower(), label.lower())

class ContextAwareEmotionModel(BaseEmotionModel):
    """Context-aware emotion model that considers situational factors"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.base_model = None
        self.context_weights = {
            'work': {'stress': 1.2, 'frustration': 1.1, 'satisfaction': 0.9},
            'personal': {'love': 1.2, 'joy': 1.1, 'sadness': 1.1},
            'health': {'fear': 1.3, 'anxiety': 1.2, 'relief': 1.1},
            'social': {'excitement': 1.2, 'embarrassment': 1.1, 'pride': 1.1}
        }
    
    def load_model(self) -> bool:
        """Load base model for context-aware predictions"""
        try:
            if self.is_loaded:
                return True
            
            # Use a robust base model
            @st.cache_resource
            def _load_context_model(model_path: str):
                return pipeline(
                    "text-classification",
                    model=model_path,
                    top_k=None,
                    device=0 if torch.cuda.is_available() else -1
                )
            
            self.base_model = _load_context_model(self.config.model_path)
            self.is_loaded = True
            logging.info(f"Loaded context-aware model {self.config.name}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to load context-aware model {self.config.name}: {e}")
            return False
    
    def predict(self, text: str, context: Dict = None) -> List[EmotionScore]:
        """Predict emotions with context awareness"""
        if not self.is_loaded and not self.load_model():
            return []
        
        try:
            start_time = datetime.now()
            
            # Get base predictions
            base_predictions = self.base_model(text)
            
            # Apply context adjustments
            context_adjusted = self._apply_context_adjustments(
                base_predictions, context or {}
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            emotion_scores = []
            pred_list = context_adjusted if isinstance(context_adjusted, list) else [context_adjusted]
            
            for pred_group in pred_list:
                if isinstance(pred_group, list):
                    for pred in pred_group:
                        emotion_scores.append(EmotionScore(
                            label=pred['label'],
                            score=pred['score'],
                            confidence=pred.get('confidence', pred['score']),
                            source='text',
                            model_name=self.config.name,
                            processing_time=processing_time,
                            metadata={
                                'model_type': 'context_aware',
                                'context': context or {},
                                'context_adjustment': pred.get('adjustment', 1.0)
                            }
                        ))
            
            self.performance.processing_time = processing_time
            self.performance.sample_count += 1
            
            return emotion_scores
            
        except Exception as e:
            logging.error(f"Context-aware prediction failed for {self.config.name}: {e}")
            return []
    
    def _apply_context_adjustments(self, predictions: List[Dict], 
                                 context: Dict) -> List[Dict]:
        """Apply context-based adjustments to predictions"""
        if not context or not predictions:
            return predictions
        
        # Detect context from text or metadata
        detected_context = context.get('primary_context', 'general')
        
        adjusted_predictions = []
        pred_list = predictions if isinstance(predictions, list) else [predictions]
        
        for pred_group in pred_list:
            if isinstance(pred_group, list):
                adjusted_group = []
                for pred in pred_group:
                    adjusted_pred = pred.copy()
                    
                    # Apply context weight if available
                    if detected_context in self.context_weights:
                        emotion = pred['label'].lower()
                        weight = self.context_weights[detected_context].get(emotion, 1.0)
                        adjusted_pred['score'] = min(pred['score'] * weight, 1.0)
                        adjusted_pred['adjustment'] = weight
                    
                    adjusted_group.append(adjusted_pred)
                adjusted_predictions.append(adjusted_group)
            else:
                adjusted_pred = pred_group.copy()
                if detected_context in self.context_weights:
                    emotion = pred_group['label'].lower()
                    weight = self.context_weights[detected_context].get(emotion, 1.0)
                    adjusted_pred['score'] = min(pred_group['score'] * weight, 1.0)
                    adjusted_pred['adjustment'] = weight
                adjusted_predictions.append(adjusted_pred)
        
        return adjusted_predictions

class MultilingualEmotionModel(BaseEmotionModel):
    """Multilingual emotion model supporting multiple languages"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.supported_languages = ['en', 'es', 'fr', 'de', 'it', 'pt', 'nl']
        self.language_detector = None
    
    def load_model(self) -> bool:
        """Load multilingual emotion model"""
        try:
            if self.is_loaded:
                return True
            
            @st.cache_resource
            def _load_multilingual_model(model_path: str):
                return pipeline(
                    "text-classification",
                    model=model_path,
                    top_k=None,
                    device=0 if torch.cuda.is_available() else -1
                )
            
            # Use a multilingual model path or fallback
            model_path = self.config.model_path or "cardiffnlp/twitter-xlm-roberta-base-sentiment"
            self.model = _load_multilingual_model(model_path)
            
            self.is_loaded = True
            logging.info(f"Loaded multilingual model {self.config.name}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to load multilingual model {self.config.name}: {e}")
            return False
    
    def predict(self, text: str, context: Dict = None) -> List[EmotionScore]:
        """Predict emotions with multilingual support"""
        if not self.is_loaded and not self.load_model():
            return []
        
        try:
            start_time = datetime.now()
            
            # Detect language (simplified)
            detected_lang = self._detect_language(text)
            
            # Get predictions
            predictions = self.model(text)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            emotion_scores = []
            pred_list = predictions if isinstance(predictions, list) else [predictions]
            
            for pred_group in pred_list:
                if isinstance(pred_group, list):
                    for pred in pred_group:
                        emotion_scores.append(EmotionScore(
                            label=self._normalize_multilingual_label(pred['label']),
                            score=pred['score'],
                            confidence=pred['score'],
                            source='text',
                            model_name=self.config.name,
                            processing_time=processing_time,
                            metadata={
                                'model_type': 'multilingual',
                                'detected_language': detected_lang,
                                'context': context or {}
                            }
                        ))
            
            self.performance.processing_time = processing_time
            self.performance.sample_count += 1
            
            return emotion_scores
            
        except Exception as e:
            logging.error(f"Multilingual prediction failed for {self.config.name}: {e}")
            return []
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection (can be enhanced with proper language detection)"""
        # Simplified language detection based on common words
        english_words = {'the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with'}
        spanish_words = {'el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'no'}
        french_words = {'le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir'}
        
        words = set(text.lower().split())
        
        if words & english_words:
            return 'en'
        elif words & spanish_words:
            return 'es'
        elif words & french_words:
            return 'fr'
        else:
            return 'unknown'
    
    def _normalize_multilingual_label(self, label: str) -> str:
        """Normalize multilingual emotion labels to English"""
        label_mapping = {
            'positive': 'joy',
            'negative': 'sadness',
            'neutral': 'neutral',
            'positivo': 'joy',
            'negativo': 'sadness',
            'neutro': 'neutral',
            'positif': 'joy',
            'négatif': 'sadness',
            'neutre': 'neutral'
        }
        
        return label_mapping.get(label.lower(), label.lower())

def create_specialized_model(config: ModelConfig) -> BaseEmotionModel:
    """Factory function to create specialized emotion models"""
    model_type_mapping = {
        'bert': BERTEmotionModel,
        'roberta': RoBERTaEmotionModel,
        'context_aware': ContextAwareEmotionModel,
        'multilingual': MultilingualEmotionModel,
        'transformers': BERTEmotionModel  # Default to BERT for transformers
    }
    
    model_class = model_type_mapping.get(config.model_type, BERTEmotionModel)
    return model_class(config)