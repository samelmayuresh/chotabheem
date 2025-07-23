# utils/specialized_voice_models.py - Specialized voice emotion model implementations
import numpy as np
import logging
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import streamlit as st
import librosa
import torch
from transformers import pipeline, Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from utils.voice_emotion_ensemble import BaseVoiceEmotionModel, VoiceFeatures
from utils.text_emotion_ensemble import EmotionScore, ModelConfig
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
import joblib

class HuBERTEmotionModel(BaseVoiceEmotionModel):
    """HuBERT-based emotion model for robust speech emotion recognition"""
    
    def load_model(self) -> bool:
        """Load HuBERT model with caching"""
        try:
            if self.is_loaded:
                return True
            
            @st.cache_resource
            def _load_hubert_model(model_path: str):
                return pipeline(
                    "audio-classification",
                    model=model_path,
                    device=0 if torch.cuda.is_available() else -1
                )
            
            # Use HuBERT model or fallback to Wav2Vec2
            model_path = self.config.model_path or "superb/hubert-large-superb-er"
            try:
                self.model = _load_hubert_model(model_path)
            except:
                # Fallback to Wav2Vec2 if HuBERT not available
                model_path = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
                self.model = _load_hubert_model(model_path)
            
            self.is_loaded = True
            logging.info(f"Loaded HuBERT model {self.config.name}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to load HuBERT model {self.config.name}: {e}")
            return False
    
    def predict(self, audio: np.ndarray, sample_rate: int, context: Dict = None) -> List[EmotionScore]:
        """Predict emotions using HuBERT model"""
        if not self.is_loaded and not self.load_model():
            return []
        
        try:
            start_time = datetime.now()
            
            # Preprocess audio for HuBERT
            processed_audio = self._preprocess_for_hubert(audio, sample_rate)
            
            # Get predictions
            predictions = self.model(processed_audio)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Convert to EmotionScore format
            emotion_scores = []
            pred_list = predictions if isinstance(predictions, list) else [predictions]
            
            for pred in pred_list:
                emotion_scores.append(EmotionScore(
                    label=self._normalize_emotion_label(pred['label']),
                    score=pred['score'],
                    confidence=pred['score'] * 0.9,  # HuBERT has high confidence
                    source='voice',
                    model_name=self.config.name,
                    processing_time=processing_time,
                    metadata={
                        'model_type': 'hubert',
                        'sample_rate': sample_rate,
                        'audio_duration': len(audio) / sample_rate,
                        'preprocessing': 'hubert_optimized',
                        'context': context or {}
                    }
                ))
            
            self.performance.processing_time = processing_time
            self.performance.sample_count += 1
            
            return emotion_scores
            
        except Exception as e:
            logging.error(f"HuBERT prediction failed for {self.config.name}: {e}")
            return []
    
    def _preprocess_for_hubert(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Preprocess audio specifically for HuBERT models"""
        # Ensure 16kHz sample rate
        if sample_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        
        # Normalize audio
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.95
        
        # Apply mild noise reduction
        audio = self._apply_noise_reduction(audio)
        
        return audio
    
    def _apply_noise_reduction(self, audio: np.ndarray) -> np.ndarray:
        """Apply mild noise reduction for better emotion recognition"""
        try:
            # Simple spectral gating
            stft = librosa.stft(audio, n_fft=1024, hop_length=256)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Estimate noise floor
            noise_floor = np.percentile(magnitude, 15)
            
            # Apply gentle gating
            mask = magnitude > (noise_floor * 1.5)
            enhanced_magnitude = magnitude * (mask * 0.8 + 0.2)
            
            # Reconstruct
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft, hop_length=256)
            
            return enhanced_audio
            
        except Exception:
            return audio  # Return original if enhancement fails
    
    def _normalize_emotion_label(self, label: str) -> str:
        """Normalize HuBERT emotion labels"""
        label_mapping = {
            'angry': 'anger',
            'calm': 'neutral',
            'disgust': 'disgust',
            'fearful': 'fear',
            'happy': 'joy',
            'neutral': 'neutral',
            'sad': 'sadness',
            'surprised': 'surprise',
            'excitement': 'joy',
            'boredom': 'neutral'
        }
        
        return label_mapping.get(label.lower(), label.lower())

class DimensionalEmotionModel(BaseVoiceEmotionModel):
    """Dimensional emotion model using valence-arousal space"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.valence_model = None
        self.arousal_model = None
        
    def load_model(self) -> bool:
        """Load dimensional emotion models"""
        try:
            if self.is_loaded:
                return True
            
            # For now, use rule-based dimensional analysis
            # In production, this would load trained valence/arousal models
            self.is_loaded = True
            logging.info(f"Loaded dimensional model {self.config.name}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to load dimensional model {self.config.name}: {e}")
            return False
    
    def predict(self, audio: np.ndarray, sample_rate: int, context: Dict = None) -> List[EmotionScore]:
        """Predict emotions using dimensional analysis"""
        if not self.is_loaded and not self.load_model():
            return []
        
        try:
            start_time = datetime.now()
            
            # Extract features for dimensional analysis
            features = self._extract_dimensional_features(audio, sample_rate)
            
            # Predict valence and arousal
            valence, arousal = self._predict_valence_arousal(features)
            
            # Map to categorical emotions
            emotion_scores = self._map_dimensional_to_categorical(valence, arousal)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Update metadata
            for score in emotion_scores:
                score.model_name = self.config.name
                score.processing_time = processing_time
                score.metadata.update({
                    'model_type': 'dimensional',
                    'valence': valence,
                    'arousal': arousal,
                    'dimensional_space': 'valence_arousal',
                    'context': context or {}
                })
            
            self.performance.processing_time = processing_time
            self.performance.sample_count += 1
            
            return emotion_scores
            
        except Exception as e:
            logging.error(f"Dimensional prediction failed for {self.config.name}: {e}")
            return []
    
    def _extract_dimensional_features(self, audio: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Extract features relevant for dimensional emotion analysis"""
        features = {}
        
        try:
            # Pitch features (related to arousal)
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
            )
            
            f0_valid = f0[~np.isnan(f0)]
            if len(f0_valid) > 0:
                features['f0_mean'] = np.mean(f0_valid)
                features['f0_std'] = np.std(f0_valid)
                features['f0_range'] = np.max(f0_valid) - np.min(f0_valid)
            else:
                features.update({'f0_mean': 0, 'f0_std': 0, 'f0_range': 0})
            
            # Energy features (related to arousal)
            rms = librosa.feature.rms(y=audio)[0]
            features['energy_mean'] = np.mean(rms)
            features['energy_std'] = np.std(rms)
            features['energy_max'] = np.max(rms)
            
            # Spectral features (related to valence)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)[0]
            
            features['spectral_centroid_mean'] = np.mean(spectral_centroid)
            features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
            features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
            
            # MFCC features (general emotional content)
            mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
            for i in range(min(5, mfcc.shape[0])):  # Use first 5 MFCCs
                features[f'mfcc_{i}_mean'] = np.mean(mfcc[i])
                features[f'mfcc_{i}_std'] = np.std(mfcc[i])
            
            # Zero crossing rate (related to speech characteristics)
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            features['zcr_mean'] = np.mean(zcr)
            features['zcr_std'] = np.std(zcr)
            
        except Exception as e:
            logging.warning(f"Dimensional feature extraction failed: {e}")
            # Return default features
            features = {
                'f0_mean': 150, 'f0_std': 20, 'f0_range': 100,
                'energy_mean': 0.05, 'energy_std': 0.02, 'energy_max': 0.1,
                'spectral_centroid_mean': 1500, 'spectral_rolloff_mean': 3000,
                'spectral_bandwidth_mean': 1000, 'zcr_mean': 0.1, 'zcr_std': 0.05
            }
            
            # Add default MFCC features
            for i in range(5):
                features[f'mfcc_{i}_mean'] = 0.0
                features[f'mfcc_{i}_std'] = 1.0
        
        return features
    
    def _predict_valence_arousal(self, features: Dict[str, float]) -> Tuple[float, float]:
        """Predict valence and arousal from features"""
        # Rule-based valence prediction (positive/negative)
        valence_indicators = []
        
        # High spectral centroid suggests brightness (positive valence)
        if features['spectral_centroid_mean'] > 2000:
            valence_indicators.append(0.3)
        elif features['spectral_centroid_mean'] < 1000:
            valence_indicators.append(-0.3)
        
        # MFCC patterns (simplified)
        if features.get('mfcc_1_mean', 0) > 0:
            valence_indicators.append(0.2)
        else:
            valence_indicators.append(-0.2)
        
        # Energy consistency (stable energy suggests positive valence)
        energy_cv = features['energy_std'] / max(features['energy_mean'], 0.001)
        if energy_cv < 0.5:
            valence_indicators.append(0.1)
        else:
            valence_indicators.append(-0.1)
        
        valence = np.mean(valence_indicators) if valence_indicators else 0.0
        valence = np.clip(valence, -1.0, 1.0)
        
        # Rule-based arousal prediction (activation level)
        arousal_indicators = []
        
        # High pitch variation suggests high arousal
        if features['f0_std'] > 40:
            arousal_indicators.append(0.4)
        elif features['f0_std'] < 15:
            arousal_indicators.append(-0.4)
        
        # High energy suggests high arousal
        if features['energy_mean'] > 0.08:
            arousal_indicators.append(0.3)
        elif features['energy_mean'] < 0.03:
            arousal_indicators.append(-0.3)
        
        # High zero crossing rate suggests high arousal
        if features['zcr_mean'] > 0.15:
            arousal_indicators.append(0.2)
        elif features['zcr_mean'] < 0.05:
            arousal_indicators.append(-0.2)
        
        arousal = np.mean(arousal_indicators) if arousal_indicators else 0.0
        arousal = np.clip(arousal, -1.0, 1.0)
        
        return valence, arousal
    
    def _map_dimensional_to_categorical(self, valence: float, arousal: float) -> List[EmotionScore]:
        """Map valence-arousal to categorical emotions"""
        emotion_scores = []
        
        # Define emotion positions in valence-arousal space
        emotion_positions = {
            'joy': (0.7, 0.5),
            'excitement': (0.5, 0.8),
            'surprise': (0.0, 0.8),
            'anger': (-0.5, 0.7),
            'fear': (-0.6, 0.6),
            'sadness': (-0.7, -0.4),
            'disgust': (-0.6, 0.2),
            'neutral': (0.0, 0.0),
            'calm': (0.3, -0.5),
            'boredom': (-0.2, -0.7)
        }
        
        # Calculate distances and convert to scores
        for emotion, (target_v, target_a) in emotion_positions.items():
            # Euclidean distance in valence-arousal space
            distance = np.sqrt((valence - target_v)**2 + (arousal - target_a)**2)
            
            # Convert distance to similarity score (closer = higher score)
            max_distance = np.sqrt(2)  # Maximum possible distance in [-1,1] x [-1,1]
            similarity = 1.0 - (distance / max_distance)
            
            # Apply threshold to avoid very low scores
            if similarity > 0.1:
                confidence = similarity * 0.8  # Dimensional models have good confidence
                
                emotion_scores.append(EmotionScore(
                    label=emotion,
                    score=similarity,
                    confidence=confidence,
                    source='voice',
                    model_name=self.config.name,
                    processing_time=0.0,  # Will be updated by caller
                    metadata={
                        'dimensional_distance': distance,
                        'target_valence': target_v,
                        'target_arousal': target_a
                    }
                ))
        
        # Sort by score and return top emotions
        emotion_scores.sort(key=lambda x: x.score, reverse=True)
        return emotion_scores[:7]  # Return top 7 emotions

class NoiseRobustEmotionModel(BaseVoiceEmotionModel):
    """Noise-robust emotion model with advanced preprocessing"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.base_model = None
        self.noise_profiles = {}
        
    def load_model(self) -> bool:
        """Load noise-robust emotion model"""
        try:
            if self.is_loaded:
                return True
            
            @st.cache_resource
            def _load_robust_model(model_path: str):
                return pipeline(
                    "audio-classification",
                    model=model_path,
                    device=0 if torch.cuda.is_available() else -1
                )
            
            # Use a robust model or fallback
            model_path = self.config.model_path or "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
            self.base_model = _load_robust_model(model_path)
            
            self.is_loaded = True
            logging.info(f"Loaded noise-robust model {self.config.name}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to load noise-robust model {self.config.name}: {e}")
            return False
    
    def predict(self, audio: np.ndarray, sample_rate: int, context: Dict = None) -> List[EmotionScore]:
        """Predict emotions with noise robustness"""
        if not self.is_loaded and not self.load_model():
            return []
        
        try:
            start_time = datetime.now()
            
            # Assess and reduce noise
            noise_level = self._assess_noise_level(audio)
            cleaned_audio = self._advanced_noise_reduction(audio, sample_rate, noise_level)
            
            # Get predictions from base model
            predictions = self.base_model(cleaned_audio)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Adjust confidence based on noise level
            emotion_scores = []
            pred_list = predictions if isinstance(predictions, list) else [predictions]
            
            for pred in pred_list:
                # Reduce confidence for noisy audio
                confidence_adjustment = max(0.3, 1.0 - noise_level * 0.5)
                adjusted_confidence = pred['score'] * confidence_adjustment
                
                emotion_scores.append(EmotionScore(
                    label=self._normalize_emotion_label(pred['label']),
                    score=pred['score'],
                    confidence=adjusted_confidence,
                    source='voice',
                    model_name=self.config.name,
                    processing_time=processing_time,
                    metadata={
                        'model_type': 'noise_robust',
                        'noise_level': noise_level,
                        'confidence_adjustment': confidence_adjustment,
                        'noise_reduction_applied': True,
                        'context': context or {}
                    }
                ))
            
            self.performance.processing_time = processing_time
            self.performance.sample_count += 1
            
            return emotion_scores
            
        except Exception as e:
            logging.error(f"Noise-robust prediction failed for {self.config.name}: {e}")
            return []
    
    def _assess_noise_level(self, audio: np.ndarray) -> float:
        """Assess the noise level in audio"""
        try:
            # Calculate signal-to-noise ratio estimate
            # Use spectral analysis to estimate noise
            stft = librosa.stft(audio, n_fft=1024, hop_length=256)
            magnitude = np.abs(stft)
            
            # Estimate noise floor (bottom 20% of magnitudes)
            noise_floor = np.percentile(magnitude, 20)
            signal_level = np.percentile(magnitude, 80)
            
            # Calculate SNR
            snr = signal_level / max(noise_floor, 1e-10)
            
            # Convert to noise level (0 = clean, 1 = very noisy)
            noise_level = max(0.0, min(1.0, 1.0 - np.log10(snr + 1) / 3))
            
            return noise_level
            
        except Exception:
            return 0.5  # Default moderate noise level
    
    def _advanced_noise_reduction(self, audio: np.ndarray, sample_rate: int, noise_level: float) -> np.ndarray:
        """Apply advanced noise reduction based on noise level"""
        if noise_level < 0.3:
            return audio  # Clean enough, no processing needed
        
        try:
            # Multi-band spectral gating
            stft = librosa.stft(audio, n_fft=2048, hop_length=512)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Frequency-dependent noise reduction
            freq_bins = magnitude.shape[0]
            enhanced_magnitude = magnitude.copy()
            
            for freq_idx in range(freq_bins):
                freq_magnitude = magnitude[freq_idx, :]
                
                # Estimate noise floor for this frequency band
                noise_floor = np.percentile(freq_magnitude, 15)
                
                # Apply adaptive gating
                gate_threshold = noise_floor * (2.0 + noise_level)
                mask = freq_magnitude > gate_threshold
                
                # Smooth the mask to avoid artifacts
                mask_smooth = self._smooth_mask(mask)
                
                # Apply gating with smooth transitions
                enhanced_magnitude[freq_idx, :] = freq_magnitude * (mask_smooth * 0.8 + 0.2)
            
            # Reconstruct audio
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft, hop_length=512)
            
            # Apply mild low-pass filter for very noisy audio
            if noise_level > 0.7:
                enhanced_audio = self._apply_lowpass_filter(enhanced_audio, sample_rate)
            
            return enhanced_audio
            
        except Exception as e:
            logging.warning(f"Advanced noise reduction failed: {e}")
            return audio
    
    def _smooth_mask(self, mask: np.ndarray, window_size: int = 5) -> np.ndarray:
        """Smooth binary mask to reduce artifacts"""
        try:
            # Simple moving average smoothing
            mask_float = mask.astype(float)
            smoothed = np.convolve(mask_float, np.ones(window_size)/window_size, mode='same')
            return smoothed
        except Exception:
            return mask.astype(float)
    
    def _apply_lowpass_filter(self, audio: np.ndarray, sample_rate: int, cutoff: float = 8000) -> np.ndarray:
        """Apply low-pass filter to remove high-frequency noise"""
        try:
            from scipy import signal
            
            # Design Butterworth low-pass filter
            nyquist = sample_rate / 2
            normalized_cutoff = cutoff / nyquist
            b, a = signal.butter(4, normalized_cutoff, btype='low')
            
            # Apply filter
            filtered_audio = signal.filtfilt(b, a, audio)
            return filtered_audio
            
        except Exception:
            return audio  # Return original if filtering fails
    
    def _normalize_emotion_label(self, label: str) -> str:
        """Normalize emotion labels"""
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

def create_specialized_voice_model(config: ModelConfig) -> BaseVoiceEmotionModel:
    """Factory function to create specialized voice emotion models"""
    from utils.voice_emotion_ensemble import Wav2Vec2EmotionModel, SpectralEmotionModel, ProsodicEmotionModel
    
    model_type_mapping = {
        'wav2vec2': Wav2Vec2EmotionModel,
        'hubert': HuBERTEmotionModel,
        'spectral': SpectralEmotionModel,
        'prosodic': ProsodicEmotionModel,
        'dimensional': DimensionalEmotionModel,
        'noise_robust': NoiseRobustEmotionModel
    }
    
    model_class = model_type_mapping.get(config.model_type, Wav2Vec2EmotionModel)
    return model_class(config)