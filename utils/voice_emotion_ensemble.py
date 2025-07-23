# utils/voice_emotion_ensemble.py - Voice emotion ensemble system
import numpy as np
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
import os
from datetime import datetime
import streamlit as st
import librosa
import torch
from transformers import pipeline
from utils.text_emotion_ensemble import EmotionScore, ModelConfig, ModelPerformance

@dataclass
class VoiceFeatures:
    """Container for extracted voice features"""
    mfcc: np.ndarray
    spectral_centroid: np.ndarray
    spectral_rolloff: np.ndarray
    zero_crossing_rate: np.ndarray
    rms_energy: np.ndarray
    f0: np.ndarray
    voiced_flag: np.ndarray
    voiced_probs: np.ndarray
    chroma: np.ndarray
    mel_spectrogram: np.ndarray
    tempo: float
    spectral_bandwidth: np.ndarray
    spectral_contrast: np.ndarray
    tonnetz: np.ndarray

class BaseVoiceEmotionModel(ABC):
    """Abstract base class for voice emotion models"""
    
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
        """Load the voice emotion model"""
        pass
    
    @abstractmethod
    def predict(self, audio: np.ndarray, sample_rate: int, context: Dict = None) -> List[EmotionScore]:
        """Predict emotions from audio"""
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

class Wav2Vec2EmotionModel(BaseVoiceEmotionModel):
    """Wav2Vec2-based emotion model for speech emotion recognition"""
    
    def load_model(self) -> bool:
        """Load Wav2Vec2 model with caching"""
        try:
            if self.is_loaded:
                return True
            
            @st.cache_resource
            def _load_wav2vec2_model(model_path: str):
                return pipeline(
                    "audio-classification",
                    model=model_path,
                    device=0 if torch.cuda.is_available() else -1
                )
            
            # Use a robust Wav2Vec2 emotion model
            model_path = self.config.model_path or "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
            self.model = _load_wav2vec2_model(model_path)
            self.is_loaded = True
            logging.info(f"Loaded Wav2Vec2 model {self.config.name}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to load Wav2Vec2 model {self.config.name}: {e}")
            return False
    
    def predict(self, audio: np.ndarray, sample_rate: int, context: Dict = None) -> List[EmotionScore]:
        """Predict emotions using Wav2Vec2 model"""
        if not self.is_loaded and not self.load_model():
            return []
        
        try:
            start_time = datetime.now()
            
            # Ensure audio is in correct format for Wav2Vec2
            if sample_rate != 16000:
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000
            
            # Get predictions from model
            predictions = self.model(audio)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Convert to EmotionScore format
            emotion_scores = []
            pred_list = predictions if isinstance(predictions, list) else [predictions]
            
            for pred in pred_list:
                emotion_scores.append(EmotionScore(
                    label=self._normalize_emotion_label(pred['label']),
                    score=pred['score'],
                    confidence=pred['score'],
                    source='voice',
                    model_name=self.config.name,
                    processing_time=processing_time,
                    metadata={
                        'model_type': 'wav2vec2',
                        'sample_rate': sample_rate,
                        'audio_duration': len(audio) / sample_rate,
                        'context': context or {}
                    }
                ))
            
            self.performance.processing_time = processing_time
            self.performance.sample_count += 1
            
            return emotion_scores
            
        except Exception as e:
            logging.error(f"Wav2Vec2 prediction failed for {self.config.name}: {e}")
            return []
    
    def _normalize_emotion_label(self, label: str) -> str:
        """Normalize emotion labels from Wav2Vec2 models"""
        label_mapping = {
            'angry': 'anger',
            'calm': 'neutral',
            'disgust': 'disgust',
            'fearful': 'fear',
            'happy': 'joy',
            'neutral': 'neutral',
            'sad': 'sadness',
            'surprised': 'surprise'
        }
        
        return label_mapping.get(label.lower(), label.lower())

class SpectralEmotionModel(BaseVoiceEmotionModel):
    """Spectral analysis-based emotion model"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.feature_extractor = VoiceFeatureExtractor()
        
    def load_model(self) -> bool:
        """Load spectral analysis model (rule-based)"""
        try:
            # This is a rule-based model, so no external model loading required
            self.is_loaded = True
            logging.info(f"Loaded spectral model {self.config.name}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to load spectral model {self.config.name}: {e}")
            return False
    
    def predict(self, audio: np.ndarray, sample_rate: int, context: Dict = None) -> List[EmotionScore]:
        """Predict emotions using spectral analysis"""
        if not self.is_loaded and not self.load_model():
            return []
        
        try:
            start_time = datetime.now()
            
            # Extract comprehensive features
            features = self.feature_extractor.extract_features(audio, sample_rate)
            
            # Rule-based emotion classification
            emotion_scores = self._classify_emotions_from_features(features)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Update metadata
            for score in emotion_scores:
                score.model_name = self.config.name
                score.processing_time = processing_time
                score.metadata.update({
                    'model_type': 'spectral',
                    'feature_count': len(features.__dict__),
                    'context': context or {}
                })
            
            self.performance.processing_time = processing_time
            self.performance.sample_count += 1
            
            return emotion_scores
            
        except Exception as e:
            logging.error(f"Spectral prediction failed for {self.config.name}: {e}")
            return []
    
    def _classify_emotions_from_features(self, features: VoiceFeatures) -> List[EmotionScore]:
        """Rule-based emotion classification from spectral features"""
        emotion_scores = {}
        
        # Analyze pitch characteristics
        f0_valid = features.f0[~np.isnan(features.f0)]
        if len(f0_valid) > 0:
            f0_mean = np.mean(f0_valid)
            f0_std = np.std(f0_valid)
            
            # High pitch variation suggests excitement or anger
            if f0_std > 50:
                if f0_mean > 200:
                    emotion_scores['joy'] = 0.7
                    emotion_scores['surprise'] = 0.5
                else:
                    emotion_scores['anger'] = 0.6
                    emotion_scores['fear'] = 0.4
            
            # Low pitch suggests sadness or calmness
            elif f0_mean < 150:
                emotion_scores['sadness'] = 0.6
                emotion_scores['neutral'] = 0.4
            
            # Moderate pitch suggests neutral or content
            else:
                emotion_scores['neutral'] = 0.7
                emotion_scores['joy'] = 0.3
        
        # Analyze energy characteristics
        energy_mean = np.mean(features.rms_energy)
        energy_std = np.std(features.rms_energy)
        
        # High energy suggests excitement or anger
        if energy_mean > 0.1:
            emotion_scores['joy'] = emotion_scores.get('joy', 0) + 0.3
            emotion_scores['anger'] = emotion_scores.get('anger', 0) + 0.2
        
        # Low energy suggests sadness or calmness
        elif energy_mean < 0.05:
            emotion_scores['sadness'] = emotion_scores.get('sadness', 0) + 0.3
            emotion_scores['neutral'] = emotion_scores.get('neutral', 0) + 0.2
        
        # Analyze spectral characteristics
        spectral_centroid_mean = np.mean(features.spectral_centroid)
        
        # High spectral centroid suggests brightness (joy, surprise)
        if spectral_centroid_mean > 2000:
            emotion_scores['joy'] = emotion_scores.get('joy', 0) + 0.2
            emotion_scores['surprise'] = emotion_scores.get('surprise', 0) + 0.2
        
        # Low spectral centroid suggests darkness (sadness, anger)
        elif spectral_centroid_mean < 1000:
            emotion_scores['sadness'] = emotion_scores.get('sadness', 0) + 0.2
            emotion_scores['anger'] = emotion_scores.get('anger', 0) + 0.1
        
        # Convert to EmotionScore objects
        result = []
        for emotion, score in emotion_scores.items():
            # Normalize scores
            normalized_score = min(score, 1.0)
            confidence = normalized_score * 0.8  # Spectral analysis has moderate confidence
            
            result.append(EmotionScore(
                label=emotion,
                score=normalized_score,
                confidence=confidence,
                source='voice',
                model_name=self.config.name,
                processing_time=0.0,  # Will be updated by caller
                metadata={'analysis_type': 'spectral_rules'}
            ))
        
        # Ensure we have at least neutral if no emotions detected
        if not result:
            result.append(EmotionScore(
                label='neutral',
                score=0.5,
                confidence=0.4,
                source='voice',
                model_name=self.config.name,
                processing_time=0.0,
                metadata={'analysis_type': 'spectral_fallback'}
            ))
        
        # Sort by score
        result.sort(key=lambda x: x.score, reverse=True)
        return result

class ProsodicEmotionModel(BaseVoiceEmotionModel):
    """Prosodic feature-based emotion model"""
    
    def load_model(self) -> bool:
        """Load prosodic analysis model"""
        try:
            self.is_loaded = True
            logging.info(f"Loaded prosodic model {self.config.name}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to load prosodic model {self.config.name}: {e}")
            return False
    
    def predict(self, audio: np.ndarray, sample_rate: int, context: Dict = None) -> List[EmotionScore]:
        """Predict emotions using prosodic analysis"""
        if not self.is_loaded and not self.load_model():
            return []
        
        try:
            start_time = datetime.now()
            
            # Extract prosodic features
            prosodic_features = self._extract_prosodic_features(audio, sample_rate)
            
            # Classify emotions based on prosodic patterns
            emotion_scores = self._classify_prosodic_emotions(prosodic_features)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Update metadata
            for score in emotion_scores:
                score.model_name = self.config.name
                score.processing_time = processing_time
                score.metadata.update({
                    'model_type': 'prosodic',
                    'prosodic_features': prosodic_features,
                    'context': context or {}
                })
            
            self.performance.processing_time = processing_time
            self.performance.sample_count += 1
            
            return emotion_scores
            
        except Exception as e:
            logging.error(f"Prosodic prediction failed for {self.config.name}: {e}")
            return []
    
    def _extract_prosodic_features(self, audio: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Extract prosodic features from audio"""
        features = {}
        
        try:
            # Pitch-related features
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
            )
            
            f0_valid = f0[~np.isnan(f0)]
            if len(f0_valid) > 0:
                features['f0_mean'] = np.mean(f0_valid)
                features['f0_std'] = np.std(f0_valid)
                features['f0_range'] = np.max(f0_valid) - np.min(f0_valid)
                features['f0_slope'] = np.polyfit(range(len(f0_valid)), f0_valid, 1)[0]
            else:
                features.update({'f0_mean': 0, 'f0_std': 0, 'f0_range': 0, 'f0_slope': 0})
            
            # Energy-related features
            rms = librosa.feature.rms(y=audio)[0]
            features['energy_mean'] = np.mean(rms)
            features['energy_std'] = np.std(rms)
            features['energy_range'] = np.max(rms) - np.min(rms)
            
            # Rhythm-related features
            tempo, beats = librosa.beat.beat_track(y=audio, sr=sample_rate)
            features['tempo'] = tempo
            features['rhythm_regularity'] = np.std(np.diff(beats)) if len(beats) > 1 else 0
            
            # Speaking rate (approximation)
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            features['speaking_rate'] = np.mean(zcr) * sample_rate / 2
            
            # Pause analysis
            silence_threshold = np.percentile(rms, 20)
            silence_frames = rms < silence_threshold
            pause_durations = self._get_pause_durations(silence_frames, sample_rate)
            features['avg_pause_duration'] = np.mean(pause_durations) if pause_durations else 0
            features['pause_frequency'] = len(pause_durations) / (len(audio) / sample_rate)
            
        except Exception as e:
            logging.warning(f"Prosodic feature extraction failed: {e}")
            # Return default features
            features = {
                'f0_mean': 0, 'f0_std': 0, 'f0_range': 0, 'f0_slope': 0,
                'energy_mean': 0, 'energy_std': 0, 'energy_range': 0,
                'tempo': 120, 'rhythm_regularity': 0, 'speaking_rate': 0,
                'avg_pause_duration': 0, 'pause_frequency': 0
            }
        
        return features
    
    def _get_pause_durations(self, silence_frames: np.ndarray, sample_rate: int) -> List[float]:
        """Extract pause durations from silence frames"""
        pause_durations = []
        in_pause = False
        pause_start = 0
        
        hop_length = 512  # Default hop length for RMS
        frame_duration = hop_length / sample_rate
        
        for i, is_silent in enumerate(silence_frames):
            if is_silent and not in_pause:
                in_pause = True
                pause_start = i
            elif not is_silent and in_pause:
                in_pause = False
                pause_duration = (i - pause_start) * frame_duration
                if pause_duration > 0.1:  # Only count pauses longer than 100ms
                    pause_durations.append(pause_duration)
        
        return pause_durations
    
    def _classify_prosodic_emotions(self, features: Dict[str, float]) -> List[EmotionScore]:
        """Classify emotions based on prosodic features"""
        emotion_scores = {}
        
        # Analyze pitch patterns
        f0_mean = features.get('f0_mean', 0)
        f0_std = features.get('f0_std', 0)
        f0_range = features.get('f0_range', 0)
        
        # High pitch variation and range suggest excitement
        if f0_std > 30 and f0_range > 100:
            if f0_mean > 180:
                emotion_scores['joy'] = 0.8
                emotion_scores['surprise'] = 0.6
            else:
                emotion_scores['anger'] = 0.7
                emotion_scores['fear'] = 0.5
        
        # Low pitch variation suggests calmness or sadness
        elif f0_std < 15:
            if f0_mean < 140:
                emotion_scores['sadness'] = 0.7
            else:
                emotion_scores['neutral'] = 0.6
        
        # Analyze energy patterns
        energy_mean = features.get('energy_mean', 0)
        energy_std = features.get('energy_std', 0)
        
        # High energy with variation suggests excitement or anger
        if energy_mean > 0.08 and energy_std > 0.02:
            emotion_scores['joy'] = emotion_scores.get('joy', 0) + 0.3
            emotion_scores['anger'] = emotion_scores.get('anger', 0) + 0.2
        
        # Low energy suggests sadness or calmness
        elif energy_mean < 0.03:
            emotion_scores['sadness'] = emotion_scores.get('sadness', 0) + 0.4
            emotion_scores['neutral'] = emotion_scores.get('neutral', 0) + 0.2
        
        # Analyze speaking rate
        speaking_rate = features.get('speaking_rate', 0)
        
        # Fast speaking suggests excitement or nervousness
        if speaking_rate > 150:
            emotion_scores['joy'] = emotion_scores.get('joy', 0) + 0.2
            emotion_scores['fear'] = emotion_scores.get('fear', 0) + 0.2
        
        # Slow speaking suggests sadness or thoughtfulness
        elif speaking_rate < 80:
            emotion_scores['sadness'] = emotion_scores.get('sadness', 0) + 0.3
        
        # Analyze pause patterns
        pause_frequency = features.get('pause_frequency', 0)
        avg_pause_duration = features.get('avg_pause_duration', 0)
        
        # Frequent long pauses suggest hesitation or sadness
        if pause_frequency > 2 and avg_pause_duration > 0.5:
            emotion_scores['sadness'] = emotion_scores.get('sadness', 0) + 0.2
            emotion_scores['fear'] = emotion_scores.get('fear', 0) + 0.1
        
        # Convert to EmotionScore objects
        result = []
        for emotion, score in emotion_scores.items():
            normalized_score = min(score, 1.0)
            confidence = normalized_score * 0.75  # Prosodic analysis has good confidence
            
            result.append(EmotionScore(
                label=emotion,
                score=normalized_score,
                confidence=confidence,
                source='voice',
                model_name=self.config.name,
                processing_time=0.0,
                metadata={'analysis_type': 'prosodic_rules'}
            ))
        
        # Ensure we have at least neutral
        if not result:
            result.append(EmotionScore(
                label='neutral',
                score=0.5,
                confidence=0.4,
                source='voice',
                model_name=self.config.name,
                processing_time=0.0,
                metadata={'analysis_type': 'prosodic_fallback'}
            ))
        
        result.sort(key=lambda x: x.score, reverse=True)
        return result

class VoiceFeatureExtractor:
    """Comprehensive voice feature extraction"""
    
    def __init__(self):
        self.default_sr = 16000
    
    def extract_features(self, audio: np.ndarray, sample_rate: int) -> VoiceFeatures:
        """Extract comprehensive voice features"""
        try:
            # Resample if necessary
            if sample_rate != self.default_sr:
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.default_sr)
                sample_rate = self.default_sr
            
            # Basic spectral features
            mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
            rms_energy = librosa.feature.rms(y=audio)
            
            # Pitch features
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
            )
            
            # Advanced spectral features
            chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
            mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sample_rate)
            
            # Tempo
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sample_rate)
            
            return VoiceFeatures(
                mfcc=mfcc,
                spectral_centroid=spectral_centroid,
                spectral_rolloff=spectral_rolloff,
                zero_crossing_rate=zero_crossing_rate,
                rms_energy=rms_energy,
                f0=f0,
                voiced_flag=voiced_flag,
                voiced_probs=voiced_probs,
                chroma=chroma,
                mel_spectrogram=mel_spectrogram,
                tempo=tempo,
                spectral_bandwidth=spectral_bandwidth,
                spectral_contrast=spectral_contrast,
                tonnetz=tonnetz
            )
            
        except Exception as e:
            logging.error(f"Feature extraction failed: {e}")
            # Return minimal features
            return self._create_minimal_features(audio, sample_rate)
    
    def _create_minimal_features(self, audio: np.ndarray, sample_rate: int) -> VoiceFeatures:
        """Create minimal features when extraction fails"""
        duration = len(audio) / sample_rate
        frames = int(duration * 50)  # Approximate frame count
        
        return VoiceFeatures(
            mfcc=np.zeros((13, frames)),
            spectral_centroid=np.zeros((1, frames)),
            spectral_rolloff=np.zeros((1, frames)),
            zero_crossing_rate=np.zeros((1, frames)),
            rms_energy=np.zeros((1, frames)),
            f0=np.zeros(frames),
            voiced_flag=np.zeros(frames, dtype=bool),
            voiced_probs=np.zeros(frames),
            chroma=np.zeros((12, frames)),
            mel_spectrogram=np.zeros((128, frames)),
            tempo=120.0,
            spectral_bandwidth=np.zeros((1, frames)),
            spectral_contrast=np.zeros((7, frames)),
            tonnetz=np.zeros((6, frames))
        )

class VoiceEmotionEnsemble:
    """Ensemble of specialized voice emotion models"""
    
    def __init__(self, model_configs: List[ModelConfig], voting_strategy: str = "adaptive"):
        self.model_configs = model_configs
        self.models: Dict[str, BaseVoiceEmotionModel] = {}
        self.weights: Dict[str, float] = {}
        self.performance_history: List[Dict] = []
        self.ensemble_performance = {
            'accuracy': 0.0,
            'confidence_calibration': 0.0,
            'prediction_count': 0
        }
        self.voting_strategy = voting_strategy
        self.voting_system = None
        self.feature_extractor = VoiceFeatureExtractor()
        
        self._initialize_models()
        self._calculate_model_weights()
        self._initialize_voting_system()
    
    def _initialize_models(self):
        """Initialize all voice emotion models"""
        for config in self.model_configs:
            if not config.enabled:
                continue
                
            try:
                model = self._create_voice_model(config)
                self.models[config.name] = model
                logging.info(f"Initialized voice model: {config.name} (type: {config.model_type})")
                
            except Exception as e:
                logging.error(f"Failed to initialize voice model {config.name}: {e}")
    
    def _create_voice_model(self, config: ModelConfig) -> BaseVoiceEmotionModel:
        """Factory method to create voice emotion models"""
        try:
            from utils.specialized_voice_models import create_specialized_voice_model
            return create_specialized_voice_model(config)
        except ImportError:
            # Fallback to basic models if specialized models not available
            model_type_mapping = {
                'wav2vec2': Wav2Vec2EmotionModel,
                'spectral': SpectralEmotionModel,
                'prosodic': ProsodicEmotionModel
            }
            
            model_class = model_type_mapping.get(config.model_type, Wav2Vec2EmotionModel)
            return model_class(config)
    
    def _calculate_model_weights(self) -> Dict[str, float]:
        """Calculate dynamic weights for voice models"""
        if not self.models:
            return {}
        
        weights = {}
        
        for name, model in self.models.items():
            base_weight = model.config.weight
            
            # Adjust based on performance if available
            if model.performance.sample_count > 0:
                accuracy_factor = model.performance.accuracy if model.performance.accuracy > 0 else 0.5
                speed_factor = max(0.1, 1.0 / max(model.performance.processing_time, 0.001))
                speed_factor = min(speed_factor, 10.0) / 10.0
                
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
    
    def _initialize_voting_system(self):
        """Initialize the voting system"""
        self.voting_system = None  # Will be initialized on first use
    
    def predict(self, audio: np.ndarray, sample_rate: int, context: Dict = None) -> List[EmotionScore]:
        """Enhanced voice ensemble prediction"""
        if audio is None or len(audio) == 0:
            return [EmotionScore(
                label="neutral",
                score=1.0,
                confidence=0.5,
                source="voice",
                model_name="voice_ensemble",
                processing_time=0.0,
                metadata={"reason": "empty_audio"}
            )]
        
        model_predictions = {}
        total_processing_time = 0.0
        
        # Get predictions from all models
        for name, model in self.models.items():
            try:
                predictions = model.predict(audio, sample_rate, context)
                if predictions:
                    model_predictions[name] = predictions
                    total_processing_time += predictions[0].processing_time
                        
            except Exception as e:
                logging.error(f"Voice model {name} prediction failed: {e}")
        
        if not model_predictions:
            return [EmotionScore(
                label="neutral",
                score=1.0,
                confidence=0.5,
                source="voice",
                model_name="voice_ensemble",
                processing_time=0.0,
                metadata={"reason": "no_predictions"}
            )]
        
        # Use advanced voting system if available
        if self.voting_system is None and self.voting_strategy != "simple":
            try:
                from utils.ensemble_voting import create_voting_system
                self.voting_system = create_voting_system(self.voting_strategy)
            except Exception as e:
                logging.error(f"Failed to initialize voice voting system: {e}")
                self.voting_system = False
        
        if self.voting_system and self.voting_system is not False:
            try:
                voting_result = self.voting_system.vote(
                    model_predictions, self.weights, context
                )
                
                # Update processing time and metadata
                for score in voting_result.emotion_scores:
                    score.processing_time = total_processing_time
                    score.source = 'voice'
                    score.metadata.update({
                        'voting_method': voting_result.voting_method,
                        'agreement_score': voting_result.agreement_score,
                        'confidence_level': voting_result.confidence_level,
                        'model_contributions': voting_result.model_contributions,
                        'audio_duration': len(audio) / sample_rate
                    })
                
                self.ensemble_performance['prediction_count'] += 1
                self.ensemble_performance['confidence_calibration'] = voting_result.agreement_score
                
                return voting_result.emotion_scores
                
            except Exception as e:
                logging.error(f"Voice voting failed: {e}")
        
        # Fallback to simple weighted average
        return self._simple_weighted_prediction(model_predictions, total_processing_time, len(audio) / sample_rate)
    
    def _simple_weighted_prediction(self, model_predictions: Dict[str, List[EmotionScore]], 
                                  total_processing_time: float, audio_duration: float) -> List[EmotionScore]:
        """Fallback simple weighted prediction for voice"""
        all_predictions = {}
        
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
        
        ensemble_scores = []
        for emotion, predictions in all_predictions.items():
            weighted_score = sum(p['score'] * p['weight'] for p in predictions)
            total_weight = sum(p['weight'] for p in predictions)
            
            if total_weight > 0:
                final_score = weighted_score / total_weight
            else:
                final_score = np.mean([p['score'] for p in predictions])
            
            scores = [p['score'] for p in predictions]
            confidence = self._calculate_confidence(scores, len(model_predictions))
            
            ensemble_scores.append(EmotionScore(
                label=emotion,
                score=final_score,
                confidence=confidence,
                source="voice",
                model_name="voice_ensemble_fallback",
                processing_time=total_processing_time,
                metadata={
                    'contributing_models': [p['model'] for p in predictions],
                    'model_count': len(predictions),
                    'weight_sum': total_weight,
                    'score_variance': np.var(scores) if len(scores) > 1 else 0.0,
                    'voting_method': 'simple_weighted',
                    'audio_duration': audio_duration
                }
            ))
        
        ensemble_scores.sort(key=lambda x: x.score, reverse=True)
        self.ensemble_performance['prediction_count'] += 1
        
        return ensemble_scores
    
    def _calculate_confidence(self, scores: List[float], model_count: int) -> float:
        """Calculate confidence based on model agreement"""
        if len(scores) <= 1:
            return scores[0] if scores else 0.5
        
        score_variance = np.var(scores)
        variance_confidence = max(0.0, 1.0 - score_variance * 2)
        model_agreement = len(scores) / max(model_count, 1)
        
        confidence = (variance_confidence * 0.7 + model_agreement * 0.3)
        return min(max(confidence, 0.1), 1.0)
    
    def get_model_status(self) -> Dict[str, Dict]:
        """Get status of all voice models"""
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
                    'model_type': model.config.model_type,
                    'enabled': model.config.enabled
                }
            }
        return status
    
    def get_ensemble_performance(self) -> Dict[str, Any]:
        """Get overall voice ensemble performance"""
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

def create_default_voice_ensemble() -> VoiceEmotionEnsemble:
    """Create a default voice emotion ensemble with specialized models"""
    default_configs = [
        ModelConfig(
            name="wav2vec2_primary",
            model_path="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
            model_type="wav2vec2",
            weight=1.0,
            enabled=True
        ),
        ModelConfig(
            name="hubert_robust",
            model_path="superb/hubert-large-superb-er",
            model_type="hubert",
            weight=0.9,
            enabled=True
        ),
        ModelConfig(
            name="spectral_analyzer",
            model_path="",
            model_type="spectral",
            weight=0.7,
            enabled=True
        ),
        ModelConfig(
            name="prosodic_analyzer",
            model_path="",
            model_type="prosodic",
            weight=0.8,
            enabled=True
        ),
        ModelConfig(
            name="dimensional_analyzer",
            model_path="",
            model_type="dimensional",
            weight=0.6,
            enabled=True
        ),
        ModelConfig(
            name="noise_robust",
            model_path="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
            model_type="noise_robust",
            weight=0.8,
            enabled=True
        )
    ]
    
    return VoiceEmotionEnsemble(default_configs)