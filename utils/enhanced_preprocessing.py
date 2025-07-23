# utils/enhanced_preprocessing.py - Advanced preprocessing for emotion analysis
import re
import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import contractions
import unicodedata

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

@dataclass
class ProcessedText:
    """Container for processed text data"""
    original: str
    cleaned: str
    normalized: str
    tokens: List[str]
    sentences: List[str]
    quality_score: float
    metadata: Dict[str, Any]

@dataclass
class ProcessedAudio:
    """Container for processed audio data"""
    audio_data: np.ndarray
    sample_rate: int
    duration: float
    quality_score: float
    features: Dict[str, np.ndarray]
    metadata: Dict[str, Any]

class EnhancedTextProcessor:
    """Advanced text preprocessing with context awareness"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.emotion_intensifiers = {
            'very', 'extremely', 'incredibly', 'absolutely', 'completely',
            'totally', 'utterly', 'quite', 'rather', 'really', 'truly',
            'deeply', 'highly', 'tremendously', 'enormously'
        }
        self.negation_words = {
            'not', 'no', 'never', 'nothing', 'nowhere', 'nobody', 'none',
            'neither', 'nor', 'hardly', 'scarcely', 'barely', 'seldom'
        }
        
    def process(self, text: str, metadata: Dict = None) -> ProcessedText:
        """Comprehensive text preprocessing with quality assessment"""
        if not text or not text.strip():
            return ProcessedText(
                original="",
                cleaned="",
                normalized="",
                tokens=[],
                sentences=[],
                quality_score=0.0,
                metadata={"error": "Empty text"}
            )
        
        original = text
        metadata = metadata or {}
        
        # Step 1: Basic cleaning
        cleaned = self._basic_clean(text)
        
        # Step 2: Expand contractions
        expanded = self._expand_contractions(cleaned)
        
        # Step 3: Normalize text
        normalized = self._normalize_text(expanded)
        
        # Step 4: Tokenization
        tokens = self._tokenize(normalized)
        sentences = self._sentence_tokenize(normalized)
        
        # Step 5: Quality assessment
        quality_score = self._assess_text_quality(original, tokens, sentences)
        
        # Step 6: Extract metadata
        processing_metadata = self._extract_text_metadata(
            original, cleaned, normalized, tokens, sentences
        )
        processing_metadata.update(metadata)
        
        return ProcessedText(
            original=original,
            cleaned=cleaned,
            normalized=normalized,
            tokens=tokens,
            sentences=sentences,
            quality_score=quality_score,
            metadata=processing_metadata
        )
    
    def _basic_clean(self, text: str) -> str:
        """Basic text cleaning operations"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep emoticons and punctuation
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)\[\]\{\}\'\"@#\$%\^&\*\+\=\<\>\~\`]', '', text)
        
        # Normalize quotes
        text = re.sub(r'[""''`]', '"', text)
        
        # Handle repeated punctuation
        text = re.sub(r'([.!?]){2,}', r'\1', text)
        
        return text.strip()
    
    def _expand_contractions(self, text: str) -> str:
        """Expand contractions using contractions library"""
        try:
            return contractions.fix(text)
        except Exception as e:
            logging.warning(f"Contraction expansion failed: {e}")
            return text
    
    def _normalize_text(self, text: str) -> str:
        """Advanced text normalization"""
        # Unicode normalization
        text = unicodedata.normalize('NFKD', text)
        
        # Handle repeated characters (e.g., "sooooo" -> "so")
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        # Normalize case while preserving emphasis
        # Keep ALL CAPS words as they might indicate strong emotion
        words = text.split()
        normalized_words = []
        
        for word in words:
            if word.isupper() and len(word) > 2:
                # Keep emphasis but add marker
                normalized_words.append(f"EMPHASIS_{word.lower()}")
            else:
                normalized_words.append(word.lower())
        
        return ' '.join(normalized_words)
    
    def _tokenize(self, text: str) -> List[str]:
        """Advanced tokenization preserving emotion-relevant features"""
        try:
            tokens = word_tokenize(text)
            
            # Filter tokens but preserve emotion-relevant ones
            filtered_tokens = []
            for token in tokens:
                # Keep emotion intensifiers and negations
                if (token.lower() in self.emotion_intensifiers or 
                    token.lower() in self.negation_words or
                    token.startswith('EMPHASIS_') or
                    len(token) > 2):
                    filtered_tokens.append(token)
            
            return filtered_tokens
        except Exception as e:
            logging.warning(f"Tokenization failed: {e}")
            return text.split()
    
    def _sentence_tokenize(self, text: str) -> List[str]:
        """Sentence tokenization with emotion context preservation"""
        try:
            sentences = sent_tokenize(text)
            return [s.strip() for s in sentences if s.strip()]
        except Exception as e:
            logging.warning(f"Sentence tokenization failed: {e}")
            return [text]
    
    def _assess_text_quality(self, original: str, tokens: List[str], 
                           sentences: List[str]) -> float:
        """Assess text quality for emotion analysis"""
        quality_factors = []
        
        # Length factor (optimal range: 10-500 characters)
        length = len(original)
        if 10 <= length <= 500:
            length_score = 1.0
        elif length < 10:
            length_score = length / 10.0
        else:
            length_score = max(0.5, 500 / length)
        quality_factors.append(length_score)
        
        # Token diversity
        unique_tokens = len(set(tokens))
        total_tokens = len(tokens)
        diversity_score = unique_tokens / max(total_tokens, 1)
        quality_factors.append(min(diversity_score * 2, 1.0))
        
        # Sentence structure
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        structure_score = 1.0 if 5 <= avg_sentence_length <= 25 else 0.7
        quality_factors.append(structure_score)
        
        # Emotional content indicators
        emotion_indicators = sum(1 for token in tokens 
                               if token.lower() in self.emotion_intensifiers or
                                  token.startswith('EMPHASIS_'))
        emotion_score = min(emotion_indicators / max(total_tokens, 1) * 10, 1.0)
        quality_factors.append(emotion_score)
        
        return np.mean(quality_factors)
    
    def _extract_text_metadata(self, original: str, cleaned: str, normalized: str,
                              tokens: List[str], sentences: List[str]) -> Dict[str, Any]:
        """Extract comprehensive text metadata"""
        return {
            'original_length': len(original),
            'cleaned_length': len(cleaned),
            'token_count': len(tokens),
            'sentence_count': len(sentences),
            'avg_sentence_length': sum(len(s.split()) for s in sentences) / max(len(sentences), 1),
            'has_emphasis': any(token.startswith('EMPHASIS_') for token in tokens),
            'intensifier_count': sum(1 for token in tokens if token.lower() in self.emotion_intensifiers),
            'negation_count': sum(1 for token in tokens if token.lower() in self.negation_words),
            'processing_timestamp': np.datetime64('now')
        }

class EnhancedAudioProcessor:
    """Advanced audio preprocessing with quality enhancement"""
    
    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr
        self.min_duration = 0.5  # Minimum 0.5 seconds
        self.max_duration = 300  # Maximum 5 minutes
        
    def process(self, audio_data: np.ndarray, sample_rate: int, 
                metadata: Dict = None) -> ProcessedAudio:
        """Comprehensive audio preprocessing with quality enhancement"""
        metadata = metadata or {}
        
        try:
            # Step 1: Basic validation and conversion
            audio_data = self._validate_and_convert(audio_data, sample_rate)
            
            # Step 2: Resample if needed
            if sample_rate != self.target_sr:
                audio_data = librosa.resample(
                    audio_data, orig_sr=sample_rate, target_sr=self.target_sr
                )
                sample_rate = self.target_sr
            
            # Step 3: Noise reduction and enhancement
            enhanced_audio = self._enhance_audio(audio_data)
            
            # Step 4: Quality assessment
            quality_score = self._assess_audio_quality(enhanced_audio, sample_rate)
            
            # Step 5: Feature extraction
            features = self._extract_audio_features(enhanced_audio, sample_rate)
            
            # Step 6: Extract metadata
            processing_metadata = self._extract_audio_metadata(
                enhanced_audio, sample_rate, features
            )
            processing_metadata.update(metadata)
            
            duration = len(enhanced_audio) / sample_rate
            
            return ProcessedAudio(
                audio_data=enhanced_audio,
                sample_rate=sample_rate,
                duration=duration,
                quality_score=quality_score,
                features=features,
                metadata=processing_metadata
            )
            
        except Exception as e:
            logging.error(f"Audio processing failed: {e}")
            # Return minimal valid audio
            dummy_audio = np.zeros(int(self.target_sr * 0.5))
            return ProcessedAudio(
                audio_data=dummy_audio,
                sample_rate=self.target_sr,
                duration=0.5,
                quality_score=0.0,
                features={},
                metadata={"error": str(e)}
            )
    
    def _validate_and_convert(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Validate and convert audio data to proper format"""
        # Convert to numpy array if needed
        if not isinstance(audio_data, np.ndarray):
            audio_data = np.array(audio_data)
        
        # Ensure float32 format
        if audio_data.dtype != np.float32:
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            else:
                audio_data = audio_data.astype(np.float32)
        
        # Handle stereo to mono conversion
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Normalize amplitude
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data)) * 0.95
        
        return audio_data
    
    def _enhance_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply audio enhancement techniques"""
        try:
            # Simple noise reduction using spectral gating
            # This is a basic implementation - more sophisticated methods could be added
            
            # Compute short-time Fourier transform
            stft = librosa.stft(audio_data, n_fft=2048, hop_length=512)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Estimate noise floor (bottom 10% of magnitude values)
            noise_floor = np.percentile(magnitude, 10)
            
            # Apply spectral gating
            mask = magnitude > (noise_floor * 2.0)
            enhanced_magnitude = magnitude * mask
            
            # Reconstruct audio
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft, hop_length=512)
            
            return enhanced_audio
            
        except Exception as e:
            logging.warning(f"Audio enhancement failed: {e}")
            return audio_data
    
    def _assess_audio_quality(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """Assess audio quality for emotion analysis"""
        quality_factors = []
        
        # Duration factor
        duration = len(audio_data) / sample_rate
        if self.min_duration <= duration <= self.max_duration:
            duration_score = 1.0
        elif duration < self.min_duration:
            duration_score = duration / self.min_duration
        else:
            duration_score = max(0.3, self.max_duration / duration)
        quality_factors.append(duration_score)
        
        # Signal-to-noise ratio estimation
        try:
            # Simple SNR estimation using energy in different frequency bands
            stft = librosa.stft(audio_data)
            magnitude = np.abs(stft)
            
            # Voice frequency range (80-8000 Hz)
            voice_freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=2048)
            voice_mask = (voice_freqs >= 80) & (voice_freqs <= 8000)
            
            voice_energy = np.mean(magnitude[voice_mask])
            total_energy = np.mean(magnitude)
            
            snr_score = min(voice_energy / max(total_energy, 1e-10), 1.0)
            quality_factors.append(snr_score)
            
        except Exception:
            quality_factors.append(0.5)  # Default moderate quality
        
        # Dynamic range
        dynamic_range = np.max(audio_data) - np.min(audio_data)
        range_score = min(dynamic_range * 2, 1.0)
        quality_factors.append(range_score)
        
        # Zero crossing rate (indicator of speech content)
        zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
        zcr_score = min(np.mean(zcr) * 20, 1.0)
        quality_factors.append(zcr_score)
        
        return np.mean(quality_factors)
    
    def _extract_audio_features(self, audio_data: np.ndarray, 
                               sample_rate: int) -> Dict[str, np.ndarray]:
        """Extract comprehensive audio features for emotion analysis"""
        features = {}
        
        try:
            # Spectral features
            features['mfcc'] = librosa.feature.mfcc(
                y=audio_data, sr=sample_rate, n_mfcc=13
            )
            features['spectral_centroid'] = librosa.feature.spectral_centroid(
                y=audio_data, sr=sample_rate
            )
            features['spectral_rolloff'] = librosa.feature.spectral_rolloff(
                y=audio_data, sr=sample_rate
            )
            features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(
                audio_data
            )
            
            # Prosodic features
            features['rms_energy'] = librosa.feature.rms(y=audio_data)
            
            # Pitch features
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio_data, fmin=librosa.note_to_hz('C2'), 
                fmax=librosa.note_to_hz('C7')
            )
            features['f0'] = f0
            features['voiced_flag'] = voiced_flag
            features['voiced_probs'] = voiced_probs
            
        except Exception as e:
            logging.warning(f"Feature extraction failed: {e}")
            # Provide minimal features
            features['mfcc'] = np.zeros((13, 1))
            features['spectral_centroid'] = np.zeros((1, 1))
        
        return features
    
    def _extract_audio_metadata(self, audio_data: np.ndarray, sample_rate: int,
                               features: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Extract comprehensive audio metadata"""
        metadata = {
            'duration': len(audio_data) / sample_rate,
            'sample_rate': sample_rate,
            'num_samples': len(audio_data),
            'max_amplitude': float(np.max(np.abs(audio_data))),
            'rms_amplitude': float(np.sqrt(np.mean(audio_data**2))),
            'processing_timestamp': np.datetime64('now')
        }
        
        # Add feature statistics
        for feature_name, feature_data in features.items():
            if isinstance(feature_data, np.ndarray) and feature_data.size > 0:
                metadata[f'{feature_name}_mean'] = float(np.nanmean(feature_data))
                metadata[f'{feature_name}_std'] = float(np.nanstd(feature_data))
        
        return metadata

class ContextAnalyzer:
    """Analyze and integrate contextual information for emotion analysis"""
    
    def __init__(self):
        self.temporal_patterns = {}
        self.situational_contexts = {
            'work', 'personal', 'social', 'family', 'health', 'academic'
        }
        
    def analyze_context(self, text: str = None, audio_features: Dict = None,
                       metadata: Dict = None) -> Dict[str, Any]:
        """Analyze contextual factors that might influence emotion"""
        context = {
            'temporal': self._analyze_temporal_context(metadata),
            'situational': self._analyze_situational_context(text, metadata),
            'linguistic': self._analyze_linguistic_context(text),
            'acoustic': self._analyze_acoustic_context(audio_features)
        }
        
        return context
    
    def _analyze_temporal_context(self, metadata: Dict = None) -> Dict[str, Any]:
        """Analyze temporal context factors"""
        if not metadata:
            return {}
        
        # Time of day effects on emotion
        # This would be enhanced with actual timestamp analysis
        return {
            'time_factor': 1.0,  # Placeholder for time-based adjustments
            'temporal_consistency': 1.0  # Placeholder for consistency tracking
        }
    
    def _analyze_situational_context(self, text: str = None, 
                                   metadata: Dict = None) -> Dict[str, Any]:
        """Analyze situational context from text content"""
        if not text:
            return {}
        
        # Simple keyword-based context detection
        context_keywords = {
            'work': ['work', 'job', 'office', 'meeting', 'boss', 'colleague'],
            'personal': ['family', 'friend', 'relationship', 'home'],
            'health': ['doctor', 'hospital', 'sick', 'pain', 'medicine'],
            'academic': ['school', 'study', 'exam', 'grade', 'teacher']
        }
        
        detected_contexts = []
        text_lower = text.lower()
        
        for context, keywords in context_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_contexts.append(context)
        
        return {
            'detected_contexts': detected_contexts,
            'primary_context': detected_contexts[0] if detected_contexts else 'general'
        }
    
    def _analyze_linguistic_context(self, text: str = None) -> Dict[str, Any]:
        """Analyze linguistic patterns that provide emotional context"""
        if not text:
            return {}
        
        # Analyze linguistic markers
        question_count = text.count('?')
        exclamation_count = text.count('!')
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        
        return {
            'question_count': question_count,
            'exclamation_count': exclamation_count,
            'caps_ratio': caps_ratio,
            'linguistic_intensity': (exclamation_count + caps_ratio * 10) / max(len(text.split()), 1)
        }
    
    def _analyze_acoustic_context(self, audio_features: Dict = None) -> Dict[str, Any]:
        """Analyze acoustic context from audio features"""
        if not audio_features:
            return {}
        
        context = {}
        
        # Analyze speaking rate from zero crossing rate
        if 'zero_crossing_rate' in audio_features:
            zcr = audio_features['zero_crossing_rate']
            context['speaking_rate'] = float(np.mean(zcr))
        
        # Analyze pitch variation
        if 'f0' in audio_features:
            f0 = audio_features['f0']
            valid_f0 = f0[~np.isnan(f0)]
            if len(valid_f0) > 0:
                context['pitch_mean'] = float(np.mean(valid_f0))
                context['pitch_std'] = float(np.std(valid_f0))
                context['pitch_range'] = float(np.max(valid_f0) - np.min(valid_f0))
        
        # Analyze energy patterns
        if 'rms_energy' in audio_features:
            energy = audio_features['rms_energy']
            context['energy_mean'] = float(np.mean(energy))
            context['energy_variation'] = float(np.std(energy))
        
        return context