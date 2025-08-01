"""
Simple emotion detector fallback for cloud deployment
"""
import cv2
import numpy as np
import streamlit as st
from transformers import pipeline
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

class SimpleEmotionDetector:
    """Simple emotion detector using basic models for cloud deployment."""
    
    def __init__(self):
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.emotion_colors = {
            'angry': (0, 0, 255), 'disgust': (0, 128, 0), 'fear': (128, 0, 128),
            'happy': (0, 255, 0), 'neutral': (128, 128, 128), 'sad': (255, 0, 0),
            'surprise': (0, 255, 255)
        }
        
        # Initialize face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize emotion model
        self.emotion_model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize emotion detection model."""
        try:
            st.info("🚀 Loading Simple Emotion Detection Model...")
            self.emotion_model = pipeline(
                "image-classification", 
                model="trpakov/vit-face-expression",
                device=-1
            )
            st.success("✅ Simple Emotion Model loaded!")
        except Exception as e:
            st.error(f"❌ Error loading emotion model: {e}")
    
    def detect_faces(self, image):
        """Simple face detection."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return None
        
        # Get the largest face
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        
        # Add padding
        padding = int(0.2 * min(w, h))
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        return {'bbox': (x, y, w, h), 'confidence': 0.8}
    
    def predict_emotion(self, face_img):
        """Simple emotion prediction."""
        try:
            if self.emotion_model is None:
                return self._create_fallback()
            
            # Convert to PIL Image
            if len(face_img.shape) == 3:
                pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            else:
                pil_img = Image.fromarray(face_img).convert('RGB')
            
            # Get prediction
            predictions = self.emotion_model(pil_img)
            
            # Convert to our format
            emotion_scores = {}
            for pred in predictions:
                label = pred['label'].lower()
                score = pred['score']
                
                # Map to our emotions
                if label in self.emotions:
                    emotion_scores[label] = score
                elif 'happy' in label or 'joy' in label:
                    emotion_scores['happy'] = score
                elif 'sad' in label:
                    emotion_scores['sad'] = score
                elif 'angry' in label or 'anger' in label:
                    emotion_scores['angry'] = score
                elif 'fear' in label:
                    emotion_scores['fear'] = score
                elif 'surprise' in label:
                    emotion_scores['surprise'] = score
                elif 'disgust' in label:
                    emotion_scores['disgust'] = score
                else:
                    emotion_scores['neutral'] = score
            
            # Fill missing emotions with small values
            for emotion in self.emotions:
                if emotion not in emotion_scores:
                    emotion_scores[emotion] = 0.01
            
            # Normalize
            total = sum(emotion_scores.values())
            for emotion in emotion_scores:
                emotion_scores[emotion] /= total
            
            dominant_emotion = max(emotion_scores, key=emotion_scores.get)
            
            return {
                'dominant_emotion': dominant_emotion,
                'emotion_scores': emotion_scores,
                'confidence': emotion_scores[dominant_emotion]
            }
            
        except Exception as e:
            st.warning(f"⚠️ Error in emotion prediction: {e}")
            return self._create_fallback()
    
    def _create_fallback(self):
        """Create fallback emotion distribution."""
        emotion_scores = {
            'neutral': 0.4, 'happy': 0.2, 'sad': 0.15, 'surprise': 0.1,
            'angry': 0.05, 'fear': 0.05, 'disgust': 0.05
        }
        
        return {
            'dominant_emotion': 'neutral',
            'emotion_scores': emotion_scores,
            'confidence': 0.4
        }
    
    def analyze_image(self, image):
        """Analyze image for emotions."""
        try:
            # Detect face
            face_info = self.detect_faces(image)
            
            if face_info is None:
                st.warning("❌ No face detected. Please use a clear image with a visible face.")
                return None
            
            x, y, w, h = face_info['bbox']
            
            # Extract face region
            face_img = image[y:y+h, x:x+w]
            
            if face_img.size == 0:
                st.error("❌ Could not extract face region.")
                return None
            
            # Predict emotion
            emotion_result = self.predict_emotion(face_img)
            
            # Format result
            result = [{
                'region': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
                'dominant_emotion': emotion_result['dominant_emotion'],
                'emotion': emotion_result['emotion_scores'],
                'confidence': emotion_result['confidence'],
                'face_area': w * h,
                'detection_confidence': face_info['confidence'],
                'model_type': 'Simple Emotion Detector'
            }]
            
            return result
            
        except Exception as e:
            st.error(f"Error in image analysis: {str(e)}")
            return None

# Global instance
simple_detector = SimpleEmotionDetector()

def analyze_face_simple(image_input):
    """Simple face analysis function."""
    try:
        # Handle different input types
        if hasattr(image_input, 'read'):
            # BytesIO object
            image_bytes = image_input.read()
            image_input.seek(0)  # Reset for potential reuse
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif isinstance(image_input, np.ndarray):
            image = image_input
        else:
            st.error("❌ Unsupported image input type")
            return None
        
        if image is None:
            st.error("❌ Could not decode image")
            return None
        
        return simple_detector.analyze_image(image)
        
    except Exception as e:
        st.error(f"Error in face analysis: {str(e)}")
        return None

def draw_emotion_on_face_simple(image, results):
    """Draw emotion results on face."""
    try:
        if not results:
            return image
        
        result = results[0]
        region = result['region']
        emotion = result['dominant_emotion']
        confidence = result['confidence']
        
        x, y, w, h = region['x'], region['y'], region['w'], region['h']
        color = simple_detector.emotion_colors.get(emotion, (255, 255, 255))
        
        # Draw rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        
        # Draw label
        label = f"{emotion}: {confidence:.1%}"
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return image
        
    except Exception as e:
        st.error(f"Error drawing on image: {str(e)}")
        return image

def format_emotion_results_simple(results):
    """Format emotion results for display."""
    if not results:
        return []
    
    formatted = []
    for result in results:
        emotions = result['emotion']
        # Convert to list format expected by the app
        emotion_list = [{'label': emotion, 'score': score} for emotion, score in emotions.items()]
        emotion_list.sort(key=lambda x: x['score'], reverse=True)
        formatted.extend(emotion_list)
    
    return formatted